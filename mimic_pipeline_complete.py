import os
import boto3
import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from datetime import timedelta
from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_timestamp, unix_timestamp, expr, abs as spark_abs, stddev
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType, DoubleType, StringType
from pyspark.sql import Window

# ===========================================================
# Configuration Management
# ===========================================================
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    default_config = {
        'aws': {
            'raw_bucket': 's3://mimic4-project/raw-data',
            'processed_bucket': 's3://mimic4-project/processed-data'
        },
        'paths': {
            'local_raw_dir': './data/raw',
            'local_processed_dir': './data/processed',
            'hourly_bins_dir': './data/hourly_bins',
            'output_dir': './data/output'
        },
        'processing': {
            'obs_window_hours': 18,
            'pred_window_hours': 6,
            'pred_interval_hours': 8,
            'extra_hours_after_pred': 2,
            'min_obs_completeness': 0.7,
            'min_icu_los_hours': 24,
            'max_icu_los_hours': 720,
            'min_adult_age': 18
        },
        'vitals_config': {
            'rr': {'itemids': [220210], 'min_val': 0, 'max_val': 50, 'vent_aware': True},
            'hr': {'itemids': [220045], 'min_val': 20, 'max_val': 230, 'vent_aware': False},
            'temp_c': {'itemids': [223762], 'min_val': 33.0, 'max_val': 42.0, 'vent_aware': False},
            'temp_f': {'itemids': [223761], 'min_val': 91.4, 'max_val': 107.6, 'vent_aware': False},
            'spo2': {'itemids': [220277, 646], 'min_val': 70, 'max_val': 100, 'vent_aware': False},
            'gcs': {'itemids': [220739, 223900, 223901], 'min_val': 3, 'max_val': 15, 'vent_aware': False},
            'sbp_ibp': {'itemids': [220050], 'min_val': 50, 'max_val': 250, 'vent_aware': False},
            'dbp_ibp': {'itemids': [220051], 'min_val': 30, 'max_val': 150, 'vent_aware': False},
            'map_ibp': {'itemids': [220052], 'min_val': 40, 'max_val': 200, 'vent_aware': False},
            'sbp_nibp': {'itemids': [220179], 'min_val': 50, 'max_val': 250, 'vent_aware': False},
            'dbp_nibp': {'itemids': [220180], 'min_val': 30, 'max_val': 150, 'vent_aware': False},
            'map_nibp': {'itemids': [220181], 'min_val': 40, 'max_val': 200, 'vent_aware': False}
        },
        'sedative_itemids': [221319, 221668, 222168, 221744, 225942, 225972, 221320, 229420, 225150, 221195, 227212],
        'ventilator_itemids': [223848, 223849, 223870, 225792, 225794],
        'logging': {
            'level': 'INFO',
            'file': 'pipeline.log'
        }
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            # Deep merge dictionaries
            def merge_configs(default, user):
                result = default.copy()
                for key, value in user.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = merge_configs(result[key], value)
                    else:
                        result[key] = value
                return result
            return merge_configs(default_config, user_config)
    return default_config

# ===========================================================
# Logging Setup
# ===========================================================
def setup_logging(log_level='INFO', log_file='pipeline.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ===========================================================
# Path Management
# ===========================================================
class PathManager:
    def __init__(self, config):
        self.config = config
        self.paths = config['paths']
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        for path in self.paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def get_path(self, path_type, filename=None):
        """Get full path for a given path type and optional filename"""
        base_path = Path(self.paths[path_type])
        if filename:
            return str(base_path / filename)
        return str(base_path)

# ===========================================================
# Data Quality Checker
# ===========================================================
class DataQualityChecker:
    def __init__(self, logger):
        self.logger = logger
    
    def check_missing_rate(self, df, feature_cols=None, threshold=0.9):
        """Check missing rate for features"""
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns
        
        missing_stats = []
        for col in feature_cols:
            missing_rate = df[col].isnull().sum() / len(df)
            missing_stats.append({
                'feature': col,
                'missing_rate': missing_rate,
                'missing_count': df[col].isnull().sum(),
                'total_count': len(df),
                'above_threshold': missing_rate > threshold
            })
        
        missing_df = pd.DataFrame(missing_stats)
        high_missing = missing_df[missing_df['above_threshold']]
        
        self.logger.info(f"Missing rate analysis completed for {len(feature_cols)} features")
        if len(high_missing) > 0:
            self.logger.warning(f"Features with >{threshold*100}% missing: {high_missing['feature'].tolist()}")
        
        return missing_df
    
    def check_temporal_coverage(self, df, time_col='charttime', stay_id_col='stay_id'):
        """Check temporal coverage for each stay"""
        coverage_stats = []
        
        for stay_id in df[stay_id_col].unique():
            stay_data = df[df[stay_id_col] == stay_id].copy()
            stay_data[time_col] = pd.to_datetime(stay_data[time_col])
            
            if len(stay_data) > 1:
                time_span = (stay_data[time_col].max() - stay_data[time_col].min()).total_seconds() / 3600
                data_points = len(stay_data)
                avg_interval = time_span / (data_points - 1) if data_points > 1 else 0
            else:
                time_span = 0
                data_points = len(stay_data)
                avg_interval = 0
            
            coverage_stats.append({
                'stay_id': stay_id,
                'time_span_hours': time_span,
                'data_points': data_points,
                'avg_interval_hours': avg_interval
            })
        
        return pd.DataFrame(coverage_stats)

# ===========================================================
# Enhanced Spark Session Manager
# ===========================================================
class SparkManager:
    def __init__(self, logger):
        self._spark = None
        self.logger = logger
    
    def get_spark(self, app_name="MIMIC_Preprocessing", memory_fraction=0.8):
        if self._spark is None:
            self.logger.info(f"Initializing Spark session: {app_name}")
            self._spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.memory", "2g") \
                .config("spark.sql.adaptive.skewJoin.enabled", "true") \
                .config("spark.executor.memoryFraction", str(memory_fraction)) \
                .getOrCreate()
        return self._spark
    
    def stop(self):
        if self._spark:
            self.logger.info("Stopping Spark session")
            self._spark.stop()
            self._spark = None

# ===========================================================
# Enhanced AWS S3 Utilities
# ===========================================================
class S3Manager:
    def __init__(self, logger):
        self.logger = logger
        self.s3 = boto3.client("s3")
    
    def download_from_s3(self, bucket_name, s3_key, local_path, max_retries=3):
        """Download file from S3 with retry logic"""
        for attempt in range(max_retries):
            try:
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(bucket_name, s3_key, local_path)
                self.logger.info(f"Successfully downloaded: {s3_key} to {local_path}")
                return True
            except Exception as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed for {s3_key}: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to download {s3_key} after {max_retries} attempts")
                    raise
        return False
    
    def upload_to_s3(self, local_path, bucket_name, s3_key, max_retries=3):
        """Upload file to S3 with retry logic"""
        if not os.path.exists(local_path):
            self.logger.error(f"Local file not found: {local_path}")
            return False
            
        for attempt in range(max_retries):
            try:
                self.s3.upload_file(local_path, bucket_name, s3_key)
                self.logger.info(f"Successfully uploaded: {local_path} to s3://{bucket_name}/{s3_key}")
                return True
            except Exception as e:
                self.logger.warning(f"Upload attempt {attempt + 1} failed for {local_path}: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to upload {local_path} after {max_retries} attempts")
                    raise
        return False

# ===========================================================
# Enhanced Cohort Selection
# ===========================================================
class CohortSelector:
    def __init__(self, config, path_manager, logger):
        self.config = config
        self.path_manager = path_manager
        self.logger = logger
    
    def select_cohort(self, patients_path, admissions_path, icustays_path):
        """Enhanced cohort selection with better validation"""
        try:
            self.logger.info("Starting cohort selection...")
            
            # Load data
            patients = pd.read_csv(patients_path)
            admissions = pd.read_csv(admissions_path)
            icustays = pd.read_csv(icustays_path)
            
            self.logger.info(f"Loaded data - Patients: {len(patients)}, Admissions: {len(admissions)}, ICU stays: {len(icustays)}")
            
            # Date conversions
            admissions['deathtime'] = pd.to_datetime(admissions['deathtime'], errors='coerce')
            icustays['intime'] = pd.to_datetime(icustays['intime'], errors='coerce')
            icustays['outtime'] = pd.to_datetime(icustays['outtime'], errors='coerce')
            icustays['icu_los_hours'] = (icustays['outtime'] - icustays['intime']).dt.total_seconds() / 3600
            
            # Apply filters step by step with logging
            initial_count = len(icustays)
            
            # Adult patients only
            min_age = self.config['processing']['min_adult_age']
            adult_patients = patients[patients['anchor_age'] >= min_age]
            self.logger.info(f"Adult patients (>={min_age}): {len(adult_patients)}/{len(patients)}")
            
            # Merge data
            data = admissions.merge(adult_patients, on='subject_id')
            cohort = data.merge(icustays, on=['subject_id', 'hadm_id'])
            self.logger.info(f"After merging: {len(cohort)}")
            
            # First ICU stay per patient
            cohort = cohort.sort_values(['subject_id', 'intime']).groupby('subject_id').first().reset_index()
            self.logger.info(f"First ICU stays only: {len(cohort)}")
            
            # ICU length of stay filter
            min_los = self.config['processing']['min_icu_los_hours']
            max_los = self.config['processing']['max_icu_los_hours']
            los_filter = (cohort['icu_los_hours'] >= min_los) & (cohort['icu_los_hours'] <= max_los)
            cohort = cohort[los_filter]
            self.logger.info(f"After LOS filter ({min_los}-{max_los}h): {len(cohort)}")
            
            # Death time validation
            valid_death = cohort['deathtime'].isna() | (cohort['deathtime'] <= cohort['outtime'])
            cohort = cohort[valid_death]
            self.logger.info(f"After death time validation: {len(cohort)}")
            
            # Process demographics
            cohort = cohort.rename(columns={'anchor_age': 'age'})
            gender_ohe = pd.get_dummies(cohort['gender'], prefix='gender', dtype=int)
            cohort = pd.concat([cohort.drop(columns=['gender']), gender_ohe], axis=1)
            
            # Select final columns
            final_cols = ['subject_id', 'hadm_id', 'stay_id', 'age', 'gender_F', 'gender_M',
                         'intime', 'outtime', 'deathtime', 'icu_los_hours']
            
            # Ensure all gender columns exist
            for col in ['gender_F', 'gender_M']:
                if col not in cohort.columns:
                    cohort[col] = 0
            
            final_cohort = cohort[final_cols]
            
            # Save cohort
            cohort_path = self.path_manager.get_path('local_processed_dir', 'cohort.csv')
            final_cohort.to_csv(cohort_path, index=False)
            
            self.logger.info(f"Cohort selection completed. Final size: {len(final_cohort)}")
            self.logger.info(f"Mortality rate: {final_cohort['deathtime'].notna().mean():.3f}")
            
            return final_cohort
            
        except Exception as e:
            self.logger.error(f"Error in cohort selection: {str(e)}")
            raise

# ===========================================================
# Enhanced Vital Signs Processor
# ===========================================================
class VitalSignsProcessor:
    def __init__(self, config, path_manager, spark_manager, logger):
        self.config = config
        self.path_manager = path_manager
        self.spark_manager = spark_manager
        self.logger = logger
        self.quality_checker = DataQualityChecker(logger)
    
    def process_respiratory_rate(self, cohort_path, chartevents_path, procedureevents_path):
        """Enhanced RR processing with ventilator awareness"""
        try:
            self.logger.info("Processing respiratory rate with ventilator awareness...")
            spark = self.spark_manager.get_spark("RR_Processing")
            
            # Load cohort
            cohort = spark.read.csv(cohort_path, header=True, inferSchema=True) \
                          .select("subject_id", "hadm_id", "stay_id").dropDuplicates()
            
            # Load RR data
            rr_itemid = self.config['vitals_config']['rr']['itemids'][0]
            rr_df = spark.read.csv(chartevents_path, header=True, inferSchema=True) \
                     .filter(col("itemid") == rr_itemid) \
                     .withColumn("charttime", to_timestamp("charttime")) \
                     .join(cohort, on=["subject_id", "hadm_id", "stay_id"], how="inner")
            
            # Apply value range filters
            min_val = self.config['vitals_config']['rr']['min_val']
            max_val = self.config['vitals_config']['rr']['max_val']
            rr_df = rr_df.filter((col("valuenum") > min_val) & (col("valuenum") <= max_val))
            
            # Ventilator data processing
            vent_itemids = self.config['ventilator_itemids'][:3]  # Chart events
            chart_vent_df = spark.read.csv(chartevents_path, header=True, inferSchema=True) \
                                     .filter(col("itemid").isin(vent_itemids)) \
                                     .join(cohort, on=["subject_id", "hadm_id", "stay_id"], how="inner") \
                                     .withColumn("charttime", to_timestamp("charttime"))
            
            proc_vent_itemids = self.config['ventilator_itemids'][3:5]  # Procedure events
            proc_vent_df = spark.read.csv(procedureevents_path, header=True, inferSchema=True) \
                                    .filter(col("itemid").isin(proc_vent_itemids)) \
                                    .withColumn("starttime", to_timestamp("starttime")) \
                                    .join(cohort, on=["subject_id", "hadm_id", "stay_id"], how="inner")
            
            # Process low RR values (≤6) with ventilator context
            cols_keep = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"]
            rr_low = rr_df.filter(col("valuenum") <= 6)
            rr_normal = rr_df.filter(col("valuenum") > 6).select(*cols_keep)
            
            # Chart-based ventilator flag
            chart_flag = chart_vent_df.select("subject_id", "hadm_id", "stay_id", "charttime").dropDuplicates()
            rr_keep_chart = rr_low.join(chart_flag, on=["subject_id", "hadm_id", "stay_id", "charttime"], how="leftsemi").select(*cols_keep)
            
            # Procedure-based ventilator flag (within 1 hour)
            proc_flag = proc_vent_df.select("subject_id", "hadm_id", "stay_id", "starttime") \
                                   .withColumnRenamed("starttime", "charttime").dropDuplicates()
            
            rr_proc = rr_low.alias("rr").join(proc_flag.alias("pv"), on=["subject_id", "hadm_id", "stay_id"], how="inner") \
                           .filter(spark_abs(unix_timestamp("rr.charttime") - unix_timestamp("pv.charttime")) <= 3600) \
                           .select("rr.*").select(*cols_keep)
            
            # Combine all valid RR measurements
            rr_final = rr_normal.union(rr_keep_chart).union(rr_proc).dropDuplicates()
            
            # Save result
            output_path = self.path_manager.get_path('local_processed_dir', 'rr.parquet')
            rr_final.write.mode("overwrite").parquet(output_path)
            
            self.logger.info(f"RR processing completed. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error in RR processing: {str(e)}")
            raise
    
    def process_vital_sign_generic(self, vital_name, cohort_path, chartevents_path):
        """Generic vital signs processor"""
        try:
            self.logger.info(f"Processing {vital_name}...")
            spark = self.spark_manager.get_spark(f"{vital_name.upper()}_Processing")
            
            # Load cohort
            cohort = spark.read.csv(cohort_path, header=True, inferSchema=True) \
                          .select("subject_id", "hadm_id", "stay_id").dropDuplicates()
            
            # Get vital sign configuration
            if vital_name == 'temp':
                # Special handling for temperature (both C and F)
                temp_c_config = self.config['vitals_config']['temp_c']
                temp_f_config = self.config['vitals_config']['temp_f']
                
                itemids = temp_c_config['itemids'] + temp_f_config['itemids']
                
                vital_df = spark.read.csv(chartevents_path, header=True, inferSchema=True) \
                            .filter(col("itemid").isin(itemids)) \
                            .join(cohort, on=["subject_id", "hadm_id", "stay_id"], how="inner") \
                            .filter(col("valuenum").isNotNull())
                
                # Convert temperature to Celsius
                temp_c_itemid = temp_c_config['itemids'][0]
                temp_f_itemid = temp_f_config['itemids'][0]
                
                vital_df = vital_df.withColumn("temperature_celsius",
                    when(col("itemid") == temp_c_itemid, col("valuenum"))
                    .when(col("itemid") == temp_f_itemid, (col("valuenum") - 32) / 1.8)
                    .otherwise(col("valuenum"))
                ).filter((col("temperature_celsius") >= temp_c_config['min_val']) & 
                        (col("temperature_celsius") <= temp_c_config['max_val']))
            else:
                # Standard processing for other vitals
                vital_config = self.config['vitals_config'][vital_name]
                itemids = vital_config['itemids']
                min_val = vital_config['min_val']
                max_val = vital_config['max_val']
                
                vital_df = spark.read.csv(chartevents_path, header=True, inferSchema=True) \
                            .filter(col("itemid").isin(itemids)) \
                            .join(cohort, on=["subject_id", "hadm_id", "stay_id"], how="inner") \
                            .filter((col("valuenum") >= min_val) & (col("valuenum") <= max_val)) \
                            .filter(col("valuenum").isNotNull())
            
            # Save result
            output_path = self.path_manager.get_path('local_processed_dir', f'{vital_name}.parquet')
            vital_df.write.mode("overwrite").parquet(output_path)
            
            self.logger.info(f"{vital_name} processing completed. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error in {vital_name} processing: {str(e)}")
            raise
    
    def process_gcs_with_sedation(self, cohort_path, inputevents_path, chartevents_path):
        """Process GCS with sedation flag"""
        try:
            self.logger.info("Processing GCS with sedation awareness...")
            spark = self.spark_manager.get_spark("GCS_Sedation")
            
            # Load cohort
            cohort = spark.read.csv(cohort_path, header=True, inferSchema=True) \
                          .select("stay_id").dropDuplicates()
            
            # Load sedation data
            sedative_itemids = self.config['sedative_itemids']
            inputevents = spark.read.csv(inputevents_path, header=True, inferSchema=True) \
                .filter(col("itemid").isin(sedative_itemids)) \
                .join(cohort, on="stay_id", how="inner") \
                .withColumn("starttime", to_timestamp("starttime"))
            
            # Load GCS data
            gcs_itemids = self.config['vitals_config']['gcs']['itemids']
            gcs = spark.read.csv(chartevents_path, header=True, inferSchema=True) \
                .filter(col("itemid").isin(gcs_itemids)) \
                .join(cohort, on="stay_id", how="inner") \
                .withColumn("charttime", to_timestamp("charttime")) \
                .filter(col("valuenum").isNotNull())
            
            # Apply value range filter
            min_val = self.config['vitals_config']['gcs']['min_val']
            max_val = self.config['vitals_config']['gcs']['max_val']
            gcs = gcs.filter((col("valuenum") >= min_val) & (col("valuenum") <= max_val))
            
            # Join GCS with sedation data and flag
            sedated_flagged = gcs.join(inputevents, on="stay_id", how="left") \
                                 .withColumn("time_diff", 
                                           expr("abs(unix_timestamp(charttime) - unix_timestamp(starttime))")) \
                                 .withColumn("sedated_flag",
                                           when(col("time_diff") <= 14400, 1).otherwise(0)) \
                                 .select("stay_id", "charttime", "itemid", "valuenum", "sedated_flag")
            
            # Save result
            output_path = self.path_manager.get_path('local_processed_dir', 'gcs.parquet')
            sedated_flagged.write.mode("overwrite").parquet(output_path)
            
            self.logger.info(f"GCS processing completed. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error in GCS processing: {str(e)}")
            raise
    
    def process_bp_cleaning(self, cohort_path, chartevents_path):
        """Process blood pressure with IBP/NIBP priority"""
        try:
            self.logger.info("Processing blood pressure with IBP/NIBP priority...")
            spark = self.spark_manager.get_spark("BP_Processing")
            
            # Load cohort
            cohort = spark.read.csv(cohort_path, header=True, inferSchema=True) \
                          .select("stay_id").dropDuplicates()
            
            # BP item IDs
            bp_itemids = [220050, 220051, 220052, 220179, 220180, 220181]  # IBP + NIBP
            chartevents_df = spark.read.csv(chartevents_path, header=True, inferSchema=True) \
                .filter(col("itemid").isin(bp_itemids)) \
                .select("stay_id", "charttime", "itemid", "valuenum", "valueuom") \
                .join(cohort, on="stay_id", how="inner")
            
            # BP configuration
            ITEMS = {
                "SBP": {"IBP": [220050], "NIBP": [220179]},
                "DBP": {"IBP": [220051], "NIBP": [220180]},
                "MAP": {"IBP": [220052], "NIBP": [220181]}
            }
            
            EXTREME = {
                "SBP": (50, 250),
                "DBP": (30, 150),
                "MAP": (40, 200)
            }
            
            FLAT_STD_THRESH = 2.0
            FLAT_WINDOW_SEC = 300
            
            def process_component(comp):
                value_col = comp.lower()
                lo, hi = EXTREME[comp]
                
                # IBP data
                ibp = chartevents_df.filter(col("itemid").isin(ITEMS[comp]["IBP"])) \
                    .select("stay_id", "charttime", col("valuenum").alias(value_col)) \
                    .withColumn("ts", unix_timestamp("charttime"))
                
                # NIBP data
                nibp = chartevents_df.filter(col("itemid").isin(ITEMS[comp]["NIBP"])) \
                    .select("stay_id", "charttime", col("valuenum").alias(f"nibp_{value_col}"))
                
                # IBP validity checks
                ibp = ibp.withColumn("is_valid_ext", col(value_col).between(lo, hi))
                
                # 5-minute rolling std for flat line detection
                w5 = Window.partitionBy("stay_id").orderBy("ts").rangeBetween(-FLAT_WINDOW_SEC, 0)
                ibp = ibp.withColumn(f"std5_{value_col}", stddev(value_col).over(w5))
                ibp = ibp.withColumn("flat_artifact", col(f"std5_{value_col}") < FLAT_STD_THRESH)
                ibp = ibp.withColumn("is_valid_ibp", col("is_valid_ext") & (~col("flat_artifact")))
                
                # Merge IBP and NIBP
                merged = ibp.join(nibp, ["stay_id", "charttime"], "outer")
                merged = merged.withColumn(
                    f"final_{value_col}",
                    when(col("is_valid_ibp"), col(value_col)).otherwise(col(f"nibp_{value_col}"))
                ).withColumn(
                    f"{value_col}_source",
                    when(col("is_valid_ibp") & col(value_col).isNotNull(), "IBP")
                    .when(col(f"nibp_{value_col}").isNotNull(), "NIBP")
                    .otherwise("NA")
                )
                
                return merged.select("stay_id", "charttime", f"final_{value_col}", f"{value_col}_source")
            
            # Process each BP component
            sbp_res = process_component("SBP")
            dbp_res = process_component("DBP")
            map_res = process_component("MAP")
            
            # Merge all components
            final_bp = sbp_res.join(dbp_res, ["stay_id", "charttime"], "outer") \
                             .join(map_res, ["stay_id", "charttime"], "outer")
            
            # MAP correction (calculate if missing)
            final_bp = final_bp.withColumn(
                "final_map",
                when(col("final_map").isNotNull(), col("final_map"))
                .otherwise((col("final_dbp") * 2 + col("final_sbp")) / 3.0)
            )
            
            # Save result
            output_path = self.path_manager.get_path('local_processed_dir', 'bp.parquet')
            final_bp.write.mode("overwrite").parquet(output_path)
            
            self.logger.info(f"BP processing completed. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error in BP processing: {str(e)}")
            raise

# ===========================================================
# Enhanced Hourly Binning
# ===========================================================
class HourlyBinner:
    def __init__(self, config, path_manager, logger):
        self.config = config
        self.path_manager = path_manager
        self.logger = logger
        self.quality_checker = DataQualityChecker(logger)
    
    def create_hourly_bins_for_vitals(self, cohort_path):
        """Create hourly bins for all vital signs with enhanced missing data analysis"""
        try:
            self.logger.info("Starting hourly binning for all vitals...")
            
            # Load cohort
            cohort = pd.read_csv(cohort_path)
            cohort['intime'] = pd.to_datetime(cohort['intime'])
            cohort['outtime'] = pd.to_datetime(cohort['outtime'])
            cohort['deathtime'] = pd.to_datetime(cohort['deathtime'])
            
            # Process each vital sign
            processed_vitals = []
            vitals_to_process = ['rr', 'hr', 'temp', 'spo2', 'gcs', 'bp']
            
            for vital_name in vitals_to_process:
                try:
                    self.logger.info(f"Processing hourly bins for {vital_name}...")
                    vital_result = self._process_single_vital_hourly_bins(vital_name, cohort)
                    
                    if vital_result is not None:
                        # Save individual vital bins
                        output_path = self.path_manager.get_path('hourly_bins_dir', f'{vital_name}_hourly_bins.csv')
                        vital_result.to_csv(output_path, index=False)
                        
                        # Check data quality
                        feature_cols = [col for col in vital_result.columns 
                                      if col not in ['subject_id', 'hadm_id', 'stay_id', 'hour_from_intime', 'bin_start', 'bin_end']]
                        missing_analysis = self.quality_checker.check_missing_rate(vital_result, feature_cols)
                        
                        # Save missing analysis
                        missing_path = self.path_manager.get_path('hourly_bins_dir', f'{vital_name}_missing_analysis.csv')
                        missing_analysis.to_csv(missing_path, index=False)
                        
                        processed_vitals.append(vital_name)
                        self.logger.info(f"Successfully processed {vital_name} - shape: {vital_result.shape}")
                    else:
                        self.logger.warning(f"Failed to process {vital_name}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {vital_name}: {str(e)}")
                    continue
            
            self.logger.info(f"Hourly binning completed for {len(processed_vitals)} vitals: {processed_vitals}")
            return processed_vitals
            
        except Exception as e:
            self.logger.error(f"Error in hourly binning: {str(e)}")
            raise
    
    def _process_single_vital_hourly_bins(self, vital_name, cohort):
        """Process hourly bins for a single vital sign"""
        try:
            # Define file path based on vital name
            if vital_name == 'temp':
                vital_file = self.path_manager.get_path('local_processed_dir', 'temp.parquet')
                value_col = 'temperature_celsius'
            else:
                vital_file = self.path_manager.get_path('local_processed_dir', f'{vital_name}.parquet')
                value_col = 'valuenum'
            
            # Check if file exists
            if not os.path.exists(vital_file):
                self.logger.warning(f"Vital file not found: {vital_file}")
                return None
            
            # Load vital data
            vital_data = pd.read_parquet(vital_file)
            vital_data['charttime'] = pd.to_datetime(vital_data['charttime'])
            
            # Merge with cohort
            vital_merged = vital_data.merge(
                cohort[['stay_id', 'intime', 'outtime', 'deathtime', 'subject_id', 'hadm_id', 'icu_los_hours']], 
                on='stay_id', how='inner'
            )
            
            # Filter to ICU stay period
            mask = (vital_merged['charttime'] >= vital_merged['intime']) & \
                   (vital_merged['charttime'] <= vital_merged['outtime'])
            vital_filtered = vital_merged[mask].copy()
            
            if len(vital_filtered) == 0:
                self.logger.warning(f"No data after filtering for {vital_name}")
                return None
            
            # Process hourly bins
            result_list = []
            
            for stay_id in vital_filtered['stay_id'].unique():
                patient_data = vital_filtered[vital_filtered['stay_id'] == stay_id].copy()
                patient_cohort = cohort[cohort['stay_id'] == stay_id].iloc[0]
                
                patient_bins = self._create_patient_hourly_bins(
                    patient_data, patient_cohort, vital_name, value_col
                )
                result_list.extend(patient_bins)
            
            if len(result_list) == 0:
                self.logger.warning(f"No hourly bins created for {vital_name}")
                return None
                
            return pd.DataFrame(result_list)
            
        except Exception as e:
            self.logger.error(f"Error processing {vital_name} hourly bins: {str(e)}")
            return None
    
    def _create_patient_hourly_bins(self, patient_data, patient_cohort, vital_name, value_col):
        """Create hourly bins for a single patient"""
        result_list = []
        
        intime = patient_cohort['intime']
        outtime = patient_cohort['outtime']
        deathtime = patient_cohort['deathtime']
        icu_los_hours = patient_cohort['icu_los_hours']
        
        # Determine end time for binning
        if pd.notna(deathtime):
            end_time = deathtime
        else:
            end_time = outtime
            
        total_hours = int(np.ceil(icu_los_hours))
        
        for hour in range(total_hours):
            bin_start = intime + timedelta(hours=hour)
            bin_end = intime + timedelta(hours=hour+1)
            
            if bin_end > end_time:
                bin_end = end_time
            if bin_start >= end_time:
                break
                
            # Filter data for this hour bin
            hour_mask = (patient_data['charttime'] >= bin_start) & (patient_data['charttime'] < bin_end)
            hour_data = patient_data[hour_mask]
            
            result_row = {
                'subject_id': patient_cohort['subject_id'],
                'hadm_id': patient_cohort['hadm_id'],
                'stay_id': patient_cohort['stay_id'],
                'hour_from_intime': hour,
                'bin_start': bin_start,
                'bin_end': bin_end
            }
            
            # Process based on vital type
            if vital_name == 'bp':
                # Blood pressure has multiple columns
                bp_cols = ['final_sbp', 'final_dbp', 'final_map']
                for col_name in bp_cols:
                    if len(hour_data) > 0 and col_name in hour_data.columns:
                        col_short = col_name.replace('final_', '')
                        result_row.update({
                            f'{col_short}_mean': hour_data[col_name].mean(),
                            f'{col_short}_last': hour_data[col_name].iloc[-1],
                            f'{col_short}_min': hour_data[col_name].min(),
                            f'{col_short}_max': hour_data[col_name].max()
                        })
                    else:
                        col_short = col_name.replace('final_', '')
                        result_row.update({
                            f'{col_short}_mean': np.nan,
                            f'{col_short}_last': np.nan,
                            f'{col_short}_min': np.nan,
                            f'{col_short}_max': np.nan
                        })
                        
            elif vital_name == 'gcs':
                # GCS includes sedation flag
                if len(hour_data) > 0:
                    result_row.update({
                        f'gcs_mean': hour_data[value_col].mean(),
                        f'gcs_last': hour_data[value_col].iloc[-1],
                        f'gcs_min': hour_data[value_col].min(),
                        f'gcs_max': hour_data[value_col].max(),
                        'sedated_flag': 1 if (hour_data.get('sedated_flag', 0) == 1).any() else 0
                    })
                else:
                    result_row.update({
                        f'gcs_mean': np.nan,
                        f'gcs_last': np.nan,
                        f'gcs_min': np.nan,
                        f'gcs_max': np.nan,
                        'sedated_flag': np.nan
                    })
                    
            else:
                # Standard vital signs
                prefix = vital_name
                if len(hour_data) > 0:
                    result_row.update({
                        f'{prefix}_mean': hour_data[value_col].mean(),
                        f'{prefix}_last': hour_data[value_col].iloc[-1],
                        f'{prefix}_min': hour_data[value_col].min(),
                        f'{prefix}_max': hour_data[value_col].max()
                    })
                else:
                    result_row.update({
                        f'{prefix}_mean': np.nan,
                        f'{prefix}_last': np.nan,
                        f'{prefix}_min': np.nan,
                        f'{prefix}_max': np.nan
                    })
            
            result_list.append(result_row)
        
        return result_list

# ===========================================================
# Enhanced Data Merger
# ===========================================================
class DataMerger:
    def __init__(self, config, path_manager, logger):
        self.config = config
        self.path_manager = path_manager
        self.logger = logger
        self.quality_checker = DataQualityChecker(logger)
    
    def merge_all_vitals(self):
        """Merge all vital signs hourly bins with comprehensive quality checks"""
        try:
            self.logger.info("Starting to merge all vital signs...")
            
            # Define files to merge
            file_info = [
                ('bp_hourly_bins.csv', ['sbp', 'dbp', 'map']),
                ('hr_hourly_bins.csv', ['hr']),
                ('temp_hourly_bins.csv', ['temp']),
                ('spo2_hourly_bins.csv', ['spo2']),
                ('rr_hourly_bins.csv', ['rr']),
                ('gcs_hourly_bins.csv', ['gcs'])
            ]
            
            merge_keys = ['subject_id', 'hadm_id', 'stay_id', 'hour_from_intime', 'bin_start', 'bin_end']
            
            dfs = []
            available_vitals = []
            
            for filename, vital_names in file_info:
                file_path = self.path_manager.get_path('hourly_bins_dir', filename)
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    if len(df) > 0:
                        dfs.append(df)
                        available_vitals.extend(vital_names)
                        self.logger.info(f"Loaded {filename}: {df.shape}")
                    else:
                        self.logger.warning(f"Empty file: {filename}")
                else:
                    self.logger.warning(f"File not found: {filename}")
            
            if not dfs:
                raise ValueError("No hourly bin files found for merging!")
            
            self.logger.info(f"Available vitals for merging: {available_vitals}")
            
            # Perform iterative inner join
            merged = dfs[0]
            for i, df in enumerate(dfs[1:], 1):
                before_shape = merged.shape
                merged = pd.merge(merged, df, on=merge_keys, how='inner')
                after_shape = merged.shape
                self.logger.info(f"After merge {i}: {before_shape} -> {after_shape}")
            
            # Quality assessment
            self.logger.info(f"Final merged dataset shape: {merged.shape}")
            
            # Check temporal coverage per patient
            coverage_stats = merged.groupby('stay_id').agg({
                'hour_from_intime': ['min', 'max', 'count'],
                'subject_id': 'first'
            }).reset_index()
            
            coverage_stats.columns = ['stay_id', 'min_hour', 'max_hour', 'total_hours', 'subject_id']
            coverage_stats['hour_range'] = coverage_stats['max_hour'] - coverage_stats['min_hour'] + 1
            coverage_stats['coverage_ratio'] = coverage_stats['total_hours'] / coverage_stats['hour_range']
            
            self.logger.info(f"Temporal coverage stats:")
            self.logger.info(f"  Mean coverage ratio: {coverage_stats['coverage_ratio'].mean():.3f}")
            self.logger.info(f"  Median total hours: {coverage_stats['total_hours'].median():.1f}")
            
            # Save merged data and quality reports
            output_path = self.path_manager.get_path('hourly_bins_dir', 'all_features_hourly_bins_inner.csv')
            merged.to_csv(output_path, index=False)
            
            coverage_path = self.path_manager.get_path('hourly_bins_dir', 'temporal_coverage_analysis.csv')
            coverage_stats.to_csv(coverage_path, index=False)
            
            # Overall missing data analysis
            feature_cols = [col for col in merged.columns if col not in merge_keys]
            missing_analysis = self.quality_checker.check_missing_rate(merged, feature_cols)
            missing_path = self.path_manager.get_path('hourly_bins_dir', 'overall_missing_analysis.csv')
            missing_analysis.to_csv(missing_path, index=False)
            
            self.logger.info(f"Vitals merge completed successfully!")
            self.logger.info(f"Output saved to: {output_path}")
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error in vitals merging: {str(e)}")
            raise

# ===========================================================
# Enhanced Sliding Window Creator
# ===========================================================
class SlidingWindowCreator:
    def __init__(self, config, path_manager, logger):
        self.config = config
        self.path_manager = path_manager
        self.logger = logger
        self.quality_checker = DataQualityChecker(logger)
    
    def create_sliding_windows(self):
        """Create sliding windows with enhanced quality control"""
        try:
            # Get parameters from config
            obs_window_hours = self.config['processing']['obs_window_hours']
            pred_window_hours = self.config['processing']['pred_window_hours']
            pred_interval_hours = self.config['processing']['pred_interval_hours']
            extra_hours_after_pred = self.config['processing']['extra_hours_after_pred']
            min_completeness = self.config['processing']['min_obs_completeness']
            
            self.logger.info(f"Creating sliding windows with parameters:")
            self.logger.info(f"  Observation window: {obs_window_hours}h")
            self.logger.info(f"  Prediction window: {pred_window_hours}h")
            self.logger.info(f"  Interval: {pred_interval_hours}h")
            self.logger.info(f"  Min completeness: {min_completeness}")
            
            # Load data
            data_path = self.path_manager.get_path('hourly_bins_dir', 'all_features_hourly_bins_inner.csv')
            cohort_path = self.path_manager.get_path('local_processed_dir', 'cohort.csv')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Merged data not found: {data_path}")
            if not os.path.exists(cohort_path):
                raise FileNotFoundError(f"Cohort data not found: {cohort_path}")
            
            data = pd.read_csv(data_path)
            cohort = pd.read_csv(cohort_path)
            
            self.logger.info(f"Loaded merged data: {data.shape}")
            self.logger.info(f"Loaded cohort: {cohort.shape}")
            
            # Prepare data
            cohort['intime'] = pd.to_datetime(cohort['intime'])
            cohort['outtime'] = pd.to_datetime(cohort['outtime'])
            cohort['deathtime'] = pd.to_datetime(cohort['deathtime'])
            
            data_with_cohort = data.merge(
                cohort[['subject_id', 'stay_id', 'intime', 'outtime', 'deathtime']],
                on=['subject_id', 'stay_id'], how='inner'
            )
            
            # Define feature columns
            exclude_cols = ['subject_id', 'stay_id', 'hour_from_intime', 'hadm_id', 'bin_start', 'bin_end']
            feature_cols = [c for c in data.columns if c not in exclude_cols]
            
            self.logger.info(f"Feature columns: {len(feature_cols)}")
            
            # Process windows
            windows, sequences = self._process_all_windows(
                data_with_cohort, feature_cols, obs_window_hours, 
                pred_window_hours, pred_interval_hours, 
                extra_hours_after_pred, min_completeness
            )
            
            # Save results
            windows_df = pd.DataFrame(windows)
            sequences_array = np.array(sequences)
            
            metadata_path = self.path_manager.get_path('output_dir', 'sliding_windows_metadata.csv')
            sequences_path = self.path_manager.get_path('output_dir', 'sliding_windows_sequences.npy')
            
            windows_df.to_csv(metadata_path, index=False)
            np.save(sequences_path, sequences_array)
            
            # Quality analysis
            pos_labels = windows_df['death_in_pred_window_new'].sum()
            total_windows = len(windows_df)
            
            self.logger.info(f"Sliding windows creation completed:")
            self.logger.info(f"  Total windows: {total_windows}")
            self.logger.info(f"  Positive labels: {pos_labels}")
            self.logger.info(f"  Label ratio: {pos_labels/total_windows:.4f}")
            self.logger.info(f"  Sequence shape: {sequences_array.shape}")
            
            # Save completeness analysis
            completeness_stats = windows_df['obs_completeness'].describe()
            self.logger.info(f"Observation completeness statistics:")
            for stat, value in completeness_stats.items():
                self.logger.info(f"  {stat}: {value:.3f}")
            
            return windows_df, sequences_array
            
        except Exception as e:
            self.logger.error(f"Error in sliding windows creation: {str(e)}")
            raise
    
    def _process_all_windows(self, data_with_cohort, feature_cols, obs_window_hours, 
                           pred_window_hours, pred_interval_hours, extra_hours_after_pred, 
                           min_completeness):
        """Process all sliding windows for all patients"""
        
        data_grouped = data_with_cohort.groupby('subject_id')
        
        windows = []
        sequences = []
        stats = {'total_possible': 0, 'death_during_obs': 0, 'low_completeness': 0, 'created': 0, 'positive': 0}
        
        for subject_id, subject_data in data_grouped:
            subject_windows, subject_sequences, subject_stats = self._process_subject_windows(
                subject_id, subject_data, feature_cols, obs_window_hours,
                pred_window_hours, pred_interval_hours, extra_hours_after_pred, min_completeness
            )
            
            windows.extend(subject_windows)
            sequences.extend(subject_sequences)
            
            # Update statistics
            for key in stats:
                stats[key] += subject_stats.get(key, 0)
        
        self.logger.info(f"Window processing statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
        
        return windows, sequences
    
    def _process_subject_windows(self, subject_id, subject_data, feature_cols, obs_window_hours,
                               pred_window_hours, pred_interval_hours, extra_hours_after_pred, 
                               min_completeness):
        """Process sliding windows for a single subject"""
        
        subject_data = subject_data.sort_values('hour_from_intime').reset_index(drop=True)
        
        stay_id = subject_data['stay_id'].iloc[0]
        intime = subject_data['intime'].iloc[0]
        deathtime = subject_data['deathtime'].iloc[0]
        max_hour = subject_data['hour_from_intime'].max()
        
        # Calculate possible window starts
        min_required_hours = obs_window_hours + pred_window_hours
        possible_starts = np.arange(0, max_hour + 1 - min_required_hours + 1, pred_interval_hours)
        
        windows = []
        sequences = []
        stats = {'total_possible': len(possible_starts), 'death_during_obs': 0, 
                'low_completeness': 0, 'created': 0, 'positive': 0}
        
        for window_id, obs_start in enumerate(possible_starts):
            obs_end = obs_start + obs_window_hours
            pred_start = obs_end
            pred_end = obs_end + pred_window_hours
            
            obs_start_time = intime + pd.Timedelta(hours=obs_start)
            obs_end_time = intime + pd.Timedelta(hours=obs_end)
            pred_start_time = intime + pd.Timedelta(hours=pred_start)
            pred_end_time = intime + pd.Timedelta(hours=pred_end)
            
            # Enhanced labeling logic
            remove_window, new_label = self._determine_window_label(
                deathtime, obs_start_time, obs_end_time, pred_start_time, pred_end_time, 
                extra_hours_after_pred
            )
            
            if remove_window:
                stats['death_during_obs'] += 1
                continue
            
            # Create observation sequence
            obs_sequence, obs_completeness = self._create_observation_sequence(
                subject_data, feature_cols, obs_start, obs_end
            )
            
            # Apply completeness filter only for negative cases
            if (new_label == 0) and (obs_completeness < min_completeness):
                stats['low_completeness'] += 1
                continue
            
            # Store window and sequence
            window_info = {
                'subject_id': subject_id,
                'stay_id': stay_id,
                'window_id': window_id,
                'obs_start_hour': obs_start,
                'obs_end_hour': obs_end,
                'pred_start_hour': pred_start,
                'pred_end_hour': pred_end,
                'obs_completeness': obs_completeness,
                'death_in_pred_window_new': new_label
            }
            
            windows.append(window_info)
            sequences.append(obs_sequence)
            
            stats['created'] += 1
            stats['positive'] += new_label
        
        return windows, sequences, stats
    
    def _determine_window_label(self, deathtime, obs_start_time, obs_end_time, 
                              pred_start_time, pred_end_time, extra_hours_after_pred):
        """Determine if window should be removed and what label to assign"""
        remove_window = False
        new_label = 0
        
        if pd.notna(deathtime):
            # Remove window if death occurs during observation
            if obs_start_time <= deathtime <= obs_end_time:
                remove_window = True
            # Positive label if death occurs in prediction window or shortly after
            elif pred_start_time <= deathtime <= pred_end_time:
                new_label = 1
            elif pred_end_time < deathtime <= pred_end_time + pd.Timedelta(hours=extra_hours_after_pred):
                new_label = 1
        
        return remove_window, new_label
    
    def _create_observation_sequence(self, subject_data, feature_cols, obs_start, obs_end):
        """Create observation sequence for the specified time window"""
        
        # Filter data for observation window
        obs_mask = (subject_data['hour_from_intime'] >= obs_start) & \
                  (subject_data['hour_from_intime'] < obs_end)
        obs_window = subject_data[obs_mask]
        
        # Create hour range and index data
        hour_range = np.arange(obs_start, obs_end)
        obs_window_indexed = obs_window.set_index('hour_from_intime')
        
        # Build sequence
        sequence_data = []
        total_values = 0
        missing_values = 0
        
        for hour in hour_range:
            if hour in obs_window_indexed.index:
                hour_features = obs_window_indexed.loc[hour, feature_cols].values
                missing_in_hour = pd.isna(hour_features).sum()
            else:
                hour_features = np.full(len(feature_cols), np.nan)
                missing_in_hour = len(feature_cols)
                
            sequence_data.append(hour_features.tolist())
            missing_values += missing_in_hour
            total_values += len(feature_cols)
        
        # Calculate completeness
        obs_completeness = 1 - (missing_values / total_values) if total_values > 0 else 0
        
        return np.array(sequence_data), obs_completeness

# ===========================================================
# Enhanced Pipeline Orchestrator
# ===========================================================
class MIMICPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging(
            log_level=self.config.get('logging', {}).get('level', 'INFO'),
            log_file=self.config.get('logging', {}).get('file', 'pipeline.log')
        )
        
        self.path_manager = PathManager(self.config)
        self.s3_manager = S3Manager(self.logger)
        self.spark_manager = SparkManager(self.logger)
        
        # Initialize processors
        self.cohort_selector = CohortSelector(self.config, self.path_manager, self.logger)
        self.vitals_processor = VitalSignsProcessor(self.config, self.path_manager, self.spark_manager, self.logger)
        self.hourly_binner = HourlyBinner(self.config, self.path_manager, self.logger)
        self.data_merger = DataMerger(self.config, self.path_manager, self.logger)
        self.window_creator = SlidingWindowCreator(self.config, self.path_manager, self.logger)
        
    def run_full_pipeline(self):
        """Execute the complete preprocessing pipeline"""
        try:
            self.logger.info("=== Starting MIMIC-IV Preprocessing Pipeline ===")
            
            # Step 1: Download raw data
            self.logger.info("=== Step 1: Downloading raw data ===")
            raw_file_paths = self._download_raw_data()
            
            # Step 2: Select cohort
            self.logger.info("=== Step 2: Selecting cohort ===")
            cohort_df = self.cohort_selector.select_cohort(
                raw_file_paths["patients"], 
                raw_file_paths["admissions"], 
                raw_file_paths["icustays"]
            )
            
            # Step 3: Process vital signs
            self.logger.info("=== Step 3: Processing vital signs ===")
            cohort_path = self.path_manager.get_path('local_processed_dir', 'cohort.csv')
            self._process_all_vitals(cohort_path, raw_file_paths)
            
            # Step 4: Create hourly bins
            self.logger.info("=== Step 4: Creating hourly bins ===")
            processed_vitals = self.hourly_binner.create_hourly_bins_for_vitals(cohort_path)
            
            # Step 5: Merge all vitals
            self.logger.info("=== Step 5: Merging all vitals ===")
            merged_data = self.data_merger.merge_all_vitals()
            
            # Step 6: Create sliding windows
            self.logger.info("=== Step 6: Creating sliding windows ===")
            windows_df, sequences_array = self.window_creator.create_sliding_windows()
            
            # Step 7: Final processing and feature combination
            self.logger.info("=== Step 7: Final processing ===")
            final_features = self._finalize_features(windows_df, sequences_array)
            
            # Step 8: Upload results to S3
            self.logger.info("=== Step 8: Uploading results ===")
            self._upload_final_results()
            
            self.logger.info("=== Pipeline completed successfully! ===")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            self.spark_manager.stop()
    
    def _download_raw_data(self):
        """Download raw data from S3"""
        raw_files = {
            "patients": "hosp/patients.csv.gz",
            "admissions": "hosp/admissions.csv.gz", 
            "icustays": "icu/icustays.csv.gz",
            "chartevents": "icu/chartevents.csv.gz",
            "inputevents": "icu/inputevents.csv.gz",
            "procedureevents": "icu/procedureevents.csv.gz"
        }
        
        raw_bucket_url = self.config['aws']['raw_bucket']
        bucket_name = raw_bucket_url.replace('s3://', '').split('/')[0]
        s3_prefix = '/'.join(raw_bucket_url.replace('s3://', '').split('/')[1:])
        
        local_paths = {}
        for name, s3_key in raw_files.items():
            full_s3_key = f"{s3_prefix}/{s3_key}" if s3_prefix else s3_key
            local_path = self.path_manager.get_path('local_raw_dir', os.path.basename(s3_key))
            
            self.s3_manager.download_from_s3(bucket_name, full_s3_key, local_path)
            local_paths[name] = local_path
        
        return local_paths
    
    def _process_all_vitals(self, cohort_path, raw_file_paths):
        """Process all vital signs"""
        
        # Process respiratory rate with ventilator awareness
        self.vitals_processor.process_respiratory_rate(
            cohort_path, 
            raw_file_paths["chartevents"], 
            raw_file_paths["procedureevents"]
        )
        
        # Process other vitals
        vitals_to_process = ['hr', 'temp', 'spo2']
        
        for vital in vitals_to_process:
            self.vitals_processor.process_vital_sign_generic(
                vital, cohort_path, raw_file_paths["chartevents"]
            )
        
        # Process GCS with sedation
        self.vitals_processor.process_gcs_with_sedation(
            cohort_path, 
            raw_file_paths["inputevents"], 
            raw_file_paths["chartevents"]
        )
        
        # Process BP
        self.vitals_processor.process_bp_cleaning(
            cohort_path, 
            raw_file_paths["chartevents"]
        )
    
    def _finalize_features(self, windows_df, sequences_array):
        """Finalize feature processing and combination"""
        
        # Load cohort for static features
        cohort_path = self.path_manager.get_path('local_processed_dir', 'cohort.csv')
        cohort_df = pd.read_csv(cohort_path)
        
        static_cols = ['age', 'gender_M', 'gender_F']
        static_df = cohort_df[['subject_id', 'stay_id'] + static_cols]
        
        # Merge with metadata
        meta_with_static = windows_df.merge(static_df, on=['subject_id', 'stay_id'], how='left')
        
        # Expand static features to all time steps
        N, T, F_temporal = sequences_array.shape
        X_static = meta_with_static[static_cols].to_numpy()
        X_static_expanded = np.repeat(X_static[:, np.newaxis, :], T, axis=1)
        
        # Combine temporal and static features
        X_combined = np.concatenate([sequences_array, X_static_expanded], axis=2)
        y = meta_with_static['death_in_pred_window_new'].to_numpy()
        
        # Save final results
        output_dir = self.path_manager.get_path('output_dir')
        
        np.save(os.path.join(output_dir, 'tcn_input_combined.npy'), X_combined)
        np.save(os.path.join(output_dir, 'tcn_labels.npy'), y)
        meta_with_static.to_csv(os.path.join(output_dir, 'tcn_metadata_with_static.csv'), index=False)
        
        # Save feature names
        # Get temporal feature names from original data
        data_path = self.path_manager.get_path('hourly_bins_dir', 'all_features_hourly_bins_inner.csv')
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            exclude_cols = ['subject_id', 'stay_id', 'hour_from_intime', 'hadm_id', 'bin_start', 'bin_end']
            temporal_features = [c for c in data.columns if c not in exclude_cols]
        else:
            temporal_features = [f'feature_{i}' for i in range(F_temporal)]
        
        combined_features = temporal_features + static_cols
        
        with open(os.path.join(output_dir, 'tcn_feature_names.txt'), 'w') as f:
            f.write("\n".join(combined_features))
        
        self.logger.info(f"Final features combined - Shape: {X_combined.shape}")
        return X_combined, y, combined_features
    
    def _upload_final_results(self):
        """Upload final results to S3"""
        
        processed_bucket_url = self.config['aws']['processed_bucket']
        bucket_name = processed_bucket_url.replace('s3://', '').split('/')[0]
        s3_prefix = '/'.join(processed_bucket_url.replace('s3://', '').split('/')[1:])
        
        output_dir = self.path_manager.get_path('output_dir')
        
        final_files = [
            'tcn_input_combined.npy',
            'tcn_labels.npy', 
            'tcn_metadata_with_static.csv',
            'tcn_feature_names.txt',
            'sliding_windows_metadata.csv'
        ]
        
        for filename in final_files:
            local_path = os.path.join(output_dir, filename)
            if os.path.exists(local_path):
                s3_key = f"{s3_prefix}/final_dataset/{filename}" if s3_prefix else f"final_dataset/{filename}"
                self.s3_manager.upload_to_s3(local_path, bucket_name, s3_key)

# ===========================================================
# Data Cleaning and Processing Utilities
# ===========================================================
class DataCleaner:
    def __init__(self, config, path_manager, logger):
        self.config = config
        self.path_manager = path_manager
        self.logger = logger
        self.quality_checker = DataQualityChecker(logger)
    
    def clean_and_process_data(self):
        """Clean data and remove unnecessary columns"""
        try:
            self.logger.info("Starting data cleaning and processing...")
            
            # Load metadata and data
            metadata_path = self.path_manager.get_path('output_dir', 'sliding_windows_metadata.csv')
            data_path = self.path_manager.get_path('hourly_bins_dir', 'all_features_hourly_bins_inner.csv')
            cohort_path = self.path_manager.get_path('local_processed_dir', 'cohort.csv')
            
            if not all(os.path.exists(p) for p in [metadata_path, data_path, cohort_path]):
                raise FileNotFoundError("Required files not found for data cleaning")
            
            metadata = pd.read_csv(metadata_path)
            original_data = pd.read_csv(data_path)
            cohort = pd.read_csv(cohort_path)
            
            # Remove problematic columns
            cols_to_remove = ['temp_min', 'temp_max', 'gcs_min', 'gcs_max']
            existing_cols_to_remove = [col for col in cols_to_remove if col in original_data.columns]
            
            if existing_cols_to_remove:
                original_data = original_data.drop(columns=existing_cols_to_remove)
                self.logger.info(f"Removed columns: {existing_cols_to_remove}")
            
            # Update feature columns
            exclude_cols = ['subject_id', 'stay_id', 'hour_from_intime', 'hadm_id', 'bin_start', 'bin_end']
            new_feature_cols = [c for c in original_data.columns if c not in exclude_cols]
            
            # Prepare cohort data
            cohort['intime'] = pd.to_datetime(cohort['intime'])
            cohort['outtime'] = pd.to_datetime(cohort['outtime'])
            cohort['deathtime'] = pd.to_datetime(cohort['deathtime'])
            
            processed_data_with_cohort = original_data.merge(
                cohort[['subject_id', 'stay_id', 'intime', 'outtime', 'deathtime']],
                on=['subject_id', 'stay_id'], how='inner'
            )
            
            # Recreate sequences with cleaned data
            obs_window_hours = self.config['processing']['obs_window_hours']
            new_sequences = []
            
            processed_grouped = processed_data_with_cohort.groupby('subject_id')
            
            for _, row in metadata.iterrows():
                subject_id = row['subject_id']
                obs_start = row['obs_start_hour']
                obs_end = row['obs_end_hour']
                
                if subject_id not in processed_grouped.groups:
                    self.logger.warning(f"Subject {subject_id} not found in processed data")
                    empty_sequence = np.full((obs_window_hours, len(new_feature_cols)), np.nan)
                    new_sequences.append(empty_sequence)
                    continue
                
                subject_data = processed_grouped.get_group(subject_id).sort_values('hour_from_intime').reset_index(drop=True)
                
                obs_mask = (subject_data['hour_from_intime'] >= obs_start) & \
                          (subject_data['hour_from_intime'] < obs_end)
                obs_window = subject_data[obs_mask]
                
                hour_range = np.arange(obs_start, obs_end)
                obs_window_indexed = obs_window.set_index('hour_from_intime')
                
                sequence_data = []
                for hour in hour_range:
                    if hour in obs_window_indexed.index:
                        hour_features = obs_window_indexed.loc[hour, new_feature_cols].values
                    else:
                        hour_features = np.full(len(new_feature_cols), np.nan)
                    sequence_data.append(hour_features.tolist())
                
                new_sequences.append(np.array(sequence_data))
            
            new_sequences_array = np.array(new_sequences)
            
            # Save cleaned results
            output_dir = self.path_manager.get_path('output_dir')
            
            original_data.to_csv(os.path.join(output_dir, 'processed_hourly_data.csv'), index=False)
            metadata.to_csv(os.path.join(output_dir, 'cleaned_sliding_windows_metadata.csv'), index=False)
            np.save(os.path.join(output_dir, 'processed_sliding_windows_sequences.npy'), new_sequences_array)
            
            with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
                f.write('\n'.join(new_feature_cols))
            
            self.logger.info(f"Data cleaning completed. Final sequence shape: {new_sequences_array.shape}")
            return new_sequences_array, new_feature_cols
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise

# ===========================================================
# Configuration File Generator
# ===========================================================
def generate_default_config(output_path="config.yaml"):
    """Generate a default configuration file"""
    default_config = {
        'aws': {
            'raw_bucket': 's3://mimic4-project/raw-data',
            'processed_bucket': 's3://mimic4-project/processed-data'
        },
        'paths': {
            'local_raw_dir': './data/raw',
            'local_processed_dir': './data/processed',
            'hourly_bins_dir': './data/hourly_bins',
            'output_dir': './data/output'
        },
        'processing': {
            'obs_window_hours': 18,
            'pred_window_hours': 6,
            'pred_interval_hours': 8,
            'extra_hours_after_pred': 2,
            'min_obs_completeness': 0.7,
            'min_icu_los_hours': 24,
            'max_icu_los_hours': 720,
            'min_adult_age': 18
        },
        'vitals_config': {
            'rr': {'itemids': [220210], 'min_val': 0, 'max_val': 50, 'vent_aware': True},
            'hr': {'itemids': [220045], 'min_val': 20, 'max_val': 230, 'vent_aware': False},
            'temp_c': {'itemids': [223762], 'min_val': 33.0, 'max_val': 42.0, 'vent_aware': False},
            'temp_f': {'itemids': [223761], 'min_val': 91.4, 'max_val': 107.6, 'vent_aware': False},
            'spo2': {'itemids': [220277, 646], 'min_val': 70, 'max_val': 100, 'vent_aware': False},
            'gcs': {'itemids': [220739, 223900, 223901], 'min_val': 3, 'max_val': 15, 'vent_aware': False}
        },
        'sedative_itemids': [221319, 221668, 222168, 221744, 225942, 225972, 221320, 229420, 225150, 221195, 227212],
        'ventilator_itemids': [223848, 223849, 223870, 225792, 225794],
        'logging': {
            'level': 'INFO',
            'file': 'pipeline.log'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration saved to {output_path}")

# ===========================================================
# Main Execution Functions
# ===========================================================
def main():
    """Main execution function"""
    try:
        # Generate default config if it doesn't exist
        if not os.path.exists("config.yaml"):
            print("No config.yaml found. Generating default configuration...")
            generate_default_config()
            print("Please review and modify config.yaml as needed, then run again.")
            return
        
        # Initialize and run pipeline
        pipeline = MIMICPipeline()
        pipeline.run_full_pipeline()
        
        # Optional: Run data cleaning
        data_cleaner = DataCleaner(pipeline.config, pipeline.path_manager, pipeline.logger)
        cleaned_sequences, feature_names = data_cleaner.clean_and_process_data()
        
        print("Pipeline completed successfully!")
        print(f"Final dataset shape: {cleaned_sequences.shape}")
        print(f"Number of features: {len(feature_names)}")
        
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        raise

def run_pipeline_step(step_name, config_path="config.yaml"):
    """Run a specific step of the pipeline"""
    pipeline = MIMICPipeline(config_path)
    
    if step_name == "cohort":
        print("Running cohort selection...")
        raw_files = pipeline._download_raw_data()
        cohort_df = pipeline.cohort_selector.select_cohort(
            raw_files["patients"], raw_files["admissions"], raw_files["icustays"]
        )
        print(f"Cohort selection completed. Size: {len(cohort_df)}")
        
    elif step_name == "vitals":
        print("Running vital signs processing...")
        cohort_path = pipeline.path_manager.get_path('local_processed_dir', 'cohort.csv')
        raw_files = pipeline._download_raw_data()
        pipeline._process_all_vitals(cohort_path, raw_files)
        print("Vital signs processing completed.")
        
    elif step_name == "bins":
        print("Running hourly binning...")
        cohort_path = pipeline.path_manager.get_path('local_processed_dir', 'cohort.csv')
        processed_vitals = pipeline.hourly_binner.create_hourly_bins_for_vitals(cohort_path)
        print(f"Hourly binning completed for {len(processed_vitals)} vitals.")
        
    elif step_name == "merge":
        print("Running data merging...")
        merged_data = pipeline.data_merger.merge_all_vitals()
        print(f"Data merging completed. Shape: {merged_data.shape}")
        
    elif step_name == "windows":
        print("Running sliding window creation...")
        windows_df, sequences_array = pipeline.window_creator.create_sliding_windows()
        print(f"Sliding windows created. Count: {len(windows_df)}, Shape: {sequences_array.shape}")
        
    else:
        print(f"Unknown step: {step_name}")
        print("Available steps: cohort, vitals, bins, merge, windows")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        step_name = sys.argv[1]
        config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        run_pipeline_step(step_name, config_path)
    else:
        main()