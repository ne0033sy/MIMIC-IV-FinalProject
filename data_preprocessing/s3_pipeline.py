import os
import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from datetime import timedelta
from functools import reduce
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_timestamp, unix_timestamp, expr, abs as spark_abs, stddev, broadcast
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType, DoubleType, StringType, LongType
from pyspark.sql import Window
import gc
import psutil
import boto3

# ===========================================================
# Predefined Spark Schemas for Performance
# ===========================================================
class SparkSchemas:
    """Predefined schemas to avoid inference bottlenecks"""
    
    @staticmethod
    def get_chartevents_schema():
        return StructType([
            StructField("subject_id", LongType(), True),
            StructField("hadm_id", LongType(), True),
            StructField("stay_id", LongType(), True),
            StructField("caregiver_id", LongType(), True),
            StructField("charttime", StringType(), True),
            StructField("storetime", StringType(), True),
            StructField("itemid", IntegerType(), True),
            StructField("value", StringType(), True),
            StructField("valuenum", DoubleType(), True),
            StructField("valueuom", StringType(), True),
            StructField("warning", StringType(), True)
        ])
    
    @staticmethod
    def get_inputevents_schema():
        return StructType([
            StructField("subject_id", LongType(), True),
            StructField("hadm_id", LongType(), True),
            StructField("stay_id", LongType(), True),
            StructField("caregiver_id", LongType(), True),
            StructField("starttime", StringType(), True),
            StructField("endtime", StringType(), True),
            StructField("storetime", StringType(), True),
            StructField("itemid", IntegerType(), True),
            StructField("amount", DoubleType(), True),
            StructField("amountuom", StringType(), True),
            StructField("rate", DoubleType(), True),
            StructField("rateuom", StringType(), True),
            StructField("orderid", LongType(), True),
            StructField("linkorderid", LongType(), True),
            StructField("ordercategoryname", StringType(), True),
            StructField("secondaryordercategoryname", StringType(), True),
            StructField("ordercomponenttypedescription", StringType(), True),
            StructField("ordercategorydescription", StringType(), True),
            StructField("patientweight", DoubleType(), True),
            StructField("totalamount", DoubleType(), True),
            StructField("totalamountuom", StringType(), True),
            StructField("isopenbag", IntegerType(), True),
            StructField("continueinnextdept", IntegerType(), True),
            StructField("statusdescription", StringType(), True),
            StructField("originalamount", DoubleType(), True),
            StructField("originalrate", DoubleType(), True)
        ])
    
    @staticmethod
    def get_procedureevents_schema():
        return StructType([
            StructField("subject_id", LongType(), True),
            StructField("hadm_id", LongType(), True),
            StructField("stay_id", LongType(), True),
            StructField("caregiver_id", LongType(), True),
            StructField("starttime", StringType(), True),
            StructField("endtime", StringType(), True),
            StructField("storetime", StringType(), True),
            StructField("itemid", IntegerType(), True),
            StructField("value", StringType(), True),
            StructField("valueuom", StringType(), True),
            StructField("location", StringType(), True),
            StructField("locationcategory", StringType(), True),
            StructField("orderid", LongType(), True),
            StructField("linkorderid", LongType(), True),
            StructField("ordercategoryname", StringType(), True),
            StructField("ordercategorydescription", StringType(), True),
            StructField("patientweight", DoubleType(), True),
            StructField("isopenbag", IntegerType(), True),
            StructField("continueinnextdept", IntegerType(), True),
            StructField("statusdescription", StringType(), True),
            StructField("originalamount", DoubleType(), True),
            StructField("originalrate", DoubleType(), True)
        ])

# ===========================================================
# Memory Management Utilities
# ===========================================================
class MemoryManager:
    """Memory management utilities for large-scale processing"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024  # GB
    
    @staticmethod
    def optimize_garbage_collection():
        """Force garbage collection and return freed memory"""
        before = MemoryManager.get_memory_usage()
        gc.collect()
        after = MemoryManager.get_memory_usage()
        return before - after
    
    @staticmethod
    def get_optimal_chunk_size(total_rows, available_memory_gb=4):
        """Calculate optimal chunk size based on available memory"""
        # Estimate ~100 bytes per row for typical MIMIC data
        bytes_per_row = 100
        max_rows_in_memory = int((available_memory_gb * 1024 * 1024 * 1024) / bytes_per_row)
        return min(max_rows_in_memory, max(10000, total_rows // 10))

# ===========================================================
# Configuration Management
# ===========================================================
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    default_config = {
        'aws': {
            'raw_bucket': 's3://your-mimic4-raw-bucket/raw-data',
            'processed_bucket': 's3://your-mimic4-processed-bucket/processed-data',
            'region': 'us-east-1'  # AWS 리전 설정
        },
        'paths': {
            'local_raw_dir': './data/raw',  # S3에서 다운로드할 임시 로컬 디렉토리
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
            'min_adult_age': 18,
            'chunk_size': 50000,  # Add chunk size for memory optimization
            'max_memory_gb': 8     # Maximum memory usage
        },
        'vitals_config': {
            'rr': {'itemids': [220210], 'min_val': 0, 'max_val': 100, 'vent_aware': True},
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
# Enhanced AWS S3 Manager - 새로 추가
# ===========================================================
class S3Manager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        # AWS 설정 from config
        aws_config = self.config.get('aws', {})
        region = aws_config.get('region', 'us-east-1')
        
        # S3 클라이언트 초기화
        try:
            self.s3 = boto3.client('s3', region_name=region)
            # 연결 테스트
            self.s3.list_buckets()
            self.logger.info(f"AWS S3 client initialized successfully (region: {region})")
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS S3 client: {str(e)}")
            self.logger.error("Please ensure AWS credentials are configured properly")
            raise
    
    def download_from_s3(self, bucket_name, s3_key, local_path, max_retries=3):
        """Download file from S3 with retry logic"""
        for attempt in range(max_retries):
            try:
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                
                # 파일 크기 확인 (선택적)
                try:
                    response = self.s3.head_object(Bucket=bucket_name, Key=s3_key)
                    file_size_mb = response['ContentLength'] / (1024 * 1024)
                    self.logger.info(f"Downloading {s3_key} ({file_size_mb:.1f} MB)...")
                except:
                    self.logger.info(f"Downloading {s3_key}...")
                
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
        
        # 파일 크기 확인
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        self.logger.info(f"Uploading {local_path} ({file_size_mb:.1f} MB) to s3://{bucket_name}/{s3_key}")
        
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
    
    def parse_s3_path(self, s3_path):
        """S3 경로를 버킷과 키로 파싱"""
        if not s3_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path format: {s3_path}")
        
        s3_path = s3_path.replace('s3://', '')
        parts = s3_path.split('/', 1)
        bucket_name = parts[0]
        s3_prefix = parts[1] if len(parts) > 1 else ''
        return bucket_name, s3_prefix
    
    def list_s3_objects(self, bucket_name, prefix=''):
        """List objects in S3 bucket with given prefix"""
        try:
            response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            else:
                return []
        except Exception as e:
            self.logger.error(f"Failed to list objects in s3://{bucket_name}/{prefix}: {str(e)}")
            raise

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
        for path_key, path_value in self.paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def get_path(self, path_type, filename=None):
        """Get full path for a given path type and optional filename"""
        base_path = Path(self.paths[path_type])
        if filename:
            return str(base_path / filename)
        return str(base_path)
    
    def get_raw_file_path(self, table_name):
        """Get path to downloaded raw MIMIC-IV files"""
        # Map table names to their expected local filenames after download
        table_map = {
            'patients': 'patients.csv.gz',
            'admissions': 'admissions.csv.gz',
            'icustays': 'icustays.csv.gz',
            'chartevents': 'chartevents.csv.gz',
            'inputevents': 'inputevents.csv.gz',
            'procedureevents': 'procedureevents.csv.gz'
        }
        
        if table_name in table_map:
            return os.path.join(self.paths['local_raw_dir'], table_map[table_name])
        else:
            raise ValueError(f"Unknown table name: {table_name}")

# ===========================================================
# Data Quality Checker (Memory Optimized) - 기존 코드 그대로
# ===========================================================
class DataQualityChecker:
    def __init__(self, logger):
        self.logger = logger
    
    def check_missing_rate(self, df, feature_cols=None, threshold=0.9, chunk_size=10000):
        """Memory-optimized missing rate check with chunking"""
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns
        
        total_rows = len(df)
        missing_stats = []
        
        # Process in chunks to manage memory
        for col in feature_cols:
            total_missing = 0
            
            for chunk_start in range(0, total_rows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_rows)
                chunk = df.iloc[chunk_start:chunk_end]
                total_missing += chunk[col].isnull().sum()
                
                # Force garbage collection every 10 chunks
                if (chunk_start // chunk_size) % 10 == 0:
                    gc.collect()
            
            missing_rate = total_missing / total_rows
            missing_stats.append({
                'feature': col,
                'missing_rate': missing_rate,
                'missing_count': total_missing,
                'total_count': total_rows,
                'above_threshold': missing_rate > threshold
            })
        
        missing_df = pd.DataFrame(missing_stats)
        high_missing = missing_df[missing_df['above_threshold']]
        
        self.logger.info(f"Missing rate analysis completed for {len(feature_cols)} features")
        if len(high_missing) > 0:
            self.logger.warning(f"Features with >{threshold*100}% missing: {high_missing['feature'].tolist()}")
        
        return missing_df

# ===========================================================
# Optimized Spark Session Manager - 기존 코드 그대로
# ===========================================================
class SparkManager:
    def __init__(self, logger):
        self._spark = None
        self.logger = logger
    
    def get_spark(self, app_name="MIMIC_Preprocessing", memory_fraction=0.6):
        if self._spark is None:
            self.logger.info(f"Initializing optimized Spark session: {app_name}")
            self._spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.adaptive.skewJoin.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.executor.memory", "6g") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memoryFraction", str(memory_fraction)) \
                .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "1") \
                .config("spark.sql.adaptive.coalescePartitions.parallelismFirst", "false") \
                .config("spark.sql.files.maxPartitionBytes", "268435456") \
                .config("spark.sql.shuffle.partitions", "200") \
                .config("spark.default.parallelism", "100") \
                .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
                .getOrCreate()
            
            # Set log level to reduce verbosity
            self._spark.sparkContext.setLogLevel("WARN")
            
        return self._spark
    
    def stop(self):
        if self._spark:
            self.logger.info("Stopping Spark session")
            self._spark.stop()
            self._spark = None

# ===========================================================
# Memory-Optimized Cohort Selection - 기존 코드와 동일 (S3 연동 제외)
# ===========================================================
class CohortSelector:
    def __init__(self, config, path_manager, logger):
        self.config = config
        self.path_manager = path_manager
        self.logger = logger
        self.memory_manager = MemoryManager()
    
    def select_cohort(self):
        """Memory-optimized cohort selection"""
        try:
            self.logger.info("Starting memory-optimized cohort selection...")
            initial_memory = self.memory_manager.get_memory_usage()
            
            # Get file paths (이제 다운로드된 로컬 파일들을 사용)
            patients_path = self.path_manager.get_raw_file_path('patients')
            admissions_path = self.path_manager.get_raw_file_path('admissions')
            icustays_path = self.path_manager.get_raw_file_path('icustays')
            
            # Check if files exist
            for path_name, path in [('patients', patients_path), ('admissions', admissions_path), ('icustays', icustays_path)]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path_name} file not found: {path}")
            
            # Load data with chunking for large files
            chunk_size = self.config['processing']['chunk_size']
            
            # Load patients in chunks if large
            patients = self._load_csv_chunked(patients_path, chunk_size)
            admissions = self._load_csv_chunked(admissions_path, chunk_size)
            icustays = self._load_csv_chunked(icustays_path, chunk_size)
            
            self.logger.info(f"Loaded data - Patients: {len(patients)}, Admissions: {len(admissions)}, ICU stays: {len(icustays)}")
            
            # Process with memory optimization
            final_cohort = self._process_cohort_memory_optimized(patients, admissions, icustays)
            
            # Cleanup
            del patients, admissions, icustays
            freed_memory = self.memory_manager.optimize_garbage_collection()
            final_memory = self.memory_manager.get_memory_usage()
            
            self.logger.info(f"Memory usage: {initial_memory:.2f}GB -> {final_memory:.2f}GB (freed: {freed_memory:.2f}GB)")
            
            return final_cohort
            
        except Exception as e:
            self.logger.error(f"Error in cohort selection: {str(e)}")
            raise
    
    def _load_csv_chunked(self, file_path, chunk_size):
        """Load large CSV files in chunks to manage memory"""
        try:
            # First, read a small sample to estimate size
            sample = pd.read_csv(file_path, nrows=1000)
            
            # Estimate total rows (rough approximation)
            file_size = os.path.getsize(file_path)
            sample_size = sample.memory_usage(deep=True).sum()
            estimated_rows = int((file_size / sample_size) * 1000)
            
            if estimated_rows < chunk_size:
                # Small file, load normally
                return pd.read_csv(file_path)
            else:
                # Large file, load in chunks
                self.logger.info(f"Loading large file in chunks: {file_path}")
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    chunks.append(chunk)
                    if len(chunks) % 10 == 0:  # GC every 10 chunks
                        gc.collect()
                
                result = pd.concat(chunks, ignore_index=True)
                del chunks
                gc.collect()
                return result
                
        except Exception as e:
            self.logger.warning(f"Chunked loading failed, attempting normal load: {str(e)}")
            return pd.read_csv(file_path)
    
    def _process_cohort_memory_optimized(self, patients, admissions, icustays):
        """Process cohort with memory optimization"""
        # Date conversions with memory optimization
        admissions['deathtime'] = pd.to_datetime(admissions['deathtime'], errors='coerce')
        icustays['intime'] = pd.to_datetime(icustays['intime'], errors='coerce')
        icustays['outtime'] = pd.to_datetime(icustays['outtime'], errors='coerce')
        icustays['icu_los_hours'] = (icustays['outtime'] - icustays['intime']).dt.total_seconds() / 3600
        
        # Apply filters step by step with memory cleanup
        min_age = self.config['processing']['min_adult_age']
        adult_patients = patients[patients['anchor_age'] >= min_age].copy()
        del patients  # Free memory immediately
        gc.collect()
        
        self.logger.info(f"Adult patients (>={min_age}): {len(adult_patients)}")
        
        # Merge with memory optimization - use suffixes to avoid column conflicts
        data = admissions.merge(adult_patients, on='subject_id', suffixes=('', '_pat'))
        del adult_patients
        gc.collect()
        
        cohort = data.merge(icustays, on=['subject_id', 'hadm_id'], suffixes=('', '_icu'))
        del data, icustays
        gc.collect()
        
        self.logger.info(f"After merging: {len(cohort)}")
        
        # Apply remaining filters
        cohort = cohort.sort_values(['subject_id', 'intime']).groupby('subject_id').first().reset_index()
        self.logger.info(f"First ICU stays only: {len(cohort)}")
        
        min_los = self.config['processing']['min_icu_los_hours']
        max_los = self.config['processing']['max_icu_los_hours']
        los_filter = (cohort['icu_los_hours'] >= min_los) & (cohort['icu_los_hours'] <= max_los)
        cohort = cohort[los_filter]
        self.logger.info(f"After LOS filter ({min_los}-{max_los}h): {len(cohort)}")
        
        valid_death = cohort['deathtime'].isna() | (cohort['deathtime'] <= cohort['outtime'])
        cohort = cohort[valid_death]
        self.logger.info(f"After death time validation: {len(cohort)}")
        
        # Process demographics efficiently
        cohort = cohort.rename(columns={'anchor_age': 'age'})
        gender_ohe = pd.get_dummies(cohort['gender'], prefix='gender', dtype=int)
        cohort = pd.concat([cohort.drop(columns=['gender']), gender_ohe], axis=1)
        
        # Select final columns
        final_cols = ['subject_id', 'hadm_id', 'stay_id', 'age', 'gender_F', 'gender_M',
                     'intime', 'outtime', 'deathtime', 'icu_los_hours']
        
        for col in ['gender_F', 'gender_M']:
            if col not in cohort.columns:
                cohort[col] = 0
        
        final_cohort = cohort[final_cols].copy()
        del cohort
        gc.collect()
        
        # Save cohort
        cohort_path = self.path_manager.get_path('local_processed_dir', 'cohort.csv')
        final_cohort.to_csv(cohort_path, index=False)
        
        self.logger.info(f"Cohort selection completed. Final size: {len(final_cohort)}")
        self.logger.info(f"Mortality rate: {final_cohort['deathtime'].notna().mean():.3f}")
        self.logger.info(f"Cohort saved to: {cohort_path}")
        
        return final_cohort

# ===========================================================
# Memory-Optimized Vital Signs Processor - 기존 로직 그대로 유지
# ===========================================================
class VitalSignsProcessor:
    def __init__(self, config, path_manager, spark_manager, logger):
        self.config = config
        self.path_manager = path_manager
        self.spark_manager = spark_manager
        self.logger = logger
        self.quality_checker = DataQualityChecker(logger)
        self.schemas = SparkSchemas()
        self.memory_manager = MemoryManager()
    
    def process_respiratory_rate(self, cohort_path):
        """Memory-optimized RR processing with advanced outlier and ventilator logic"""
        try:
            self.logger.info("Processing respiratory rate with memory optimization and advanced outlier filtering...")
            spark = self.spark_manager.get_spark("RR_Processing")

            # Load cohort with broadcast for join optimization
            cohort = spark.read.csv(cohort_path, header=True, inferSchema=True) \
                .select("subject_id", "hadm_id", "stay_id").dropDuplicates()
            cohort_broadcast = broadcast(cohort)

            # Load RR data with predefined schema
            chartevents_path = self.path_manager.get_raw_file_path('chartevents')
            procedureevents_path = self.path_manager.get_raw_file_path('procedureevents')
            
            rr_itemid = self.config['vitals_config']['rr']['itemids'][0]
            
            # Load RR data with schema inference to avoid mismatch
            self.logger.info("Loading chartevents with schema inference...")
            rr_df = spark.read.csv(chartevents_path, header=True, inferSchema=True) \
                .filter(col("itemid") == rr_itemid) \
                .withColumn("charttime", to_timestamp("charttime")) \
                .join(cohort_broadcast, on=["subject_id", "hadm_id", "stay_id"], how="inner")
            
            self.logger.info(f"RR data after join count: {rr_df.count()}")

            # 1차 필터: (valuenum > 0) & (valuenum < 100)
            rr_df = rr_df.filter((col("valuenum") > 0) & (col("valuenum") < 100))

            # Check if we have data after initial filtering
            rr_count = rr_df.count()
            self.logger.info(f"RR data count after initial filtering: {rr_count}")
            
            if rr_count == 0:
                self.logger.error("CRITICAL: No RR data found after initial filtering. This indicates a serious data issue.")
                self.logger.error("Possible causes:")
                self.logger.error("1. Incorrect itemid for RR in config")
                self.logger.error("2. Schema mismatch between expected and actual chartevents structure")
                self.logger.error("3. No overlap between cohort and chartevents data")
                raise ValueError("No respiratory rate data found - pipeline cannot continue without core vital signs")

            # 2차 필터: 1~99th percentile(approxQuantile)
            try:
                quantiles = rr_df.approxQuantile("valuenum", [0.01, 0.99], 0.001)
                if len(quantiles) == 2:
                    lower, upper = quantiles
                    self.logger.info(f"RR quantiles: lower={lower}, upper={upper}")
                    rr_df = rr_df.filter((col("valuenum") >= lower) & (col("valuenum") <= upper)).cache()
                else:
                    self.logger.warning(f"Unexpected quantile result: {quantiles}, skipping percentile filtering")
                    rr_df = rr_df.cache()
            except Exception as e:
                self.logger.warning(f"Error calculating quantiles: {e}, skipping percentile filtering")
                rr_df = rr_df.cache()

            # Process ventilator data with schemas
            vent_itemids = self.config['ventilator_itemids'][:3]
            chart_vent_df = spark.read.csv(chartevents_path, header=True, schema=self.schemas.get_chartevents_schema()) \
                .filter(col("itemid").isin(vent_itemids)) \
                .join(cohort_broadcast, on=["subject_id", "hadm_id", "stay_id"], how="inner") \
                .withColumn("charttime", to_timestamp("charttime")) \
                .select("subject_id", "hadm_id", "stay_id", "charttime") \
                .dropDuplicates()

            proc_vent_itemids = self.config['ventilator_itemids'][3:5]
            proc_vent_df = spark.read.csv(procedureevents_path, header=True, schema=self.schemas.get_procedureevents_schema()) \
                .filter(col("itemid").isin(proc_vent_itemids)) \
                .withColumn("starttime", to_timestamp("starttime")) \
                .join(cohort_broadcast, on=["subject_id", "hadm_id", "stay_id"], how="inner") \
                .select("subject_id", "hadm_id", "stay_id", col("starttime").alias("charttime")) \
                .dropDuplicates()

            # RR=0~6 값은 벤틸레이터 사용 시에만 남기고, 나머지는 제거
            cols_keep = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"]
            rr_low = rr_df.filter(col("valuenum") <= 6)
            rr_normal = rr_df.filter(col("valuenum") > 6).select(*cols_keep)

            # Optimized joins for ventilator flagging
            rr_keep_chart = rr_low.join(chart_vent_df, on=["subject_id", "hadm_id", "stay_id", "charttime"], how="leftsemi") \
                .select(*cols_keep)

            rr_proc = rr_low.alias("rr").join(proc_vent_df.alias("pv"), on=["subject_id", "hadm_id", "stay_id"], how="inner") \
                .filter(spark_abs(unix_timestamp("rr.charttime") - unix_timestamp("pv.charttime")) <= 3600) \
                .select("rr.*").select(*cols_keep)

            # Union and optimize output
            rr_final = rr_normal.union(rr_keep_chart).union(rr_proc).dropDuplicates() \
                .coalesce(10)  # Optimize partition count

            # Save with optimization
            output_path = self.path_manager.get_path('local_processed_dir', 'rr.parquet')
            rr_final.write.mode("overwrite").option("compression", "snappy").parquet(output_path)

            # Cleanup cache
            rr_df.unpersist()

            self.logger.info(f"RR processing completed (outlier/ventilator logic applied). Output saved to: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error in RR processing: {str(e)}")
            raise
    
    def process_vital_sign_generic(self, vital_name, cohort_path):
        """Memory-optimized generic vital processor"""
        try:
            self.logger.info(f"Processing {vital_name} with memory optimization...")
            spark = self.spark_manager.get_spark(f"{vital_name.upper()}_Processing")
            
            # Load cohort with broadcast
            cohort = spark.read.csv(cohort_path, header=True, inferSchema=True) \
                          .select("subject_id", "hadm_id", "stay_id").dropDuplicates()
            cohort_broadcast = broadcast(cohort)
            
            # Get chartevents path
            chartevents_path = self.path_manager.get_raw_file_path('chartevents')
            
            # Process based on vital type with schema
            if vital_name == 'temp':
                temp_c_config = self.config['vitals_config']['temp_c']
                temp_f_config = self.config['vitals_config']['temp_f']
                itemids = temp_c_config['itemids'] + temp_f_config['itemids']
                
                vital_df = spark.read.csv(chartevents_path, header=True, schema=self.schemas.get_chartevents_schema()) \
                            .filter(col("itemid").isin(itemids)) \
                            .join(cohort_broadcast, on=["subject_id", "hadm_id", "stay_id"], how="inner") \
                            .filter(col("valuenum").isNotNull())
                
                # Convert temperature to Celsius efficiently
                temp_c_itemid = temp_c_config['itemids'][0]
                temp_f_itemid = temp_f_config['itemids'][0]
                
                vital_df = vital_df.withColumn("temperature_celsius",
                    when(col("itemid") == temp_c_itemid, col("valuenum"))
                    .when(col("itemid") == temp_f_itemid, (col("valuenum") - 32) / 1.8)
                    .otherwise(col("valuenum"))
                ).filter((col("temperature_celsius") >= temp_c_config['min_val']) & 
                        (col("temperature_celsius") <= temp_c_config['max_val']))
                
                # Check if we have data
                temp_count = vital_df.count()
                self.logger.info(f"Temperature data count: {temp_count}")
                
                if temp_count == 0:
                    self.logger.error("CRITICAL: No temperature data found. Core vital signs missing.")
                    raise ValueError("No temperature data found - pipeline cannot continue without core vital signs")
            else:
                # Standard processing with schema
                vital_config = self.config['vitals_config'][vital_name]
                itemids = vital_config['itemids']
                min_val = vital_config['min_val']
                max_val = vital_config['max_val']
                
                vital_df = spark.read.csv(chartevents_path, header=True, schema=self.schemas.get_chartevents_schema()) \
                            .filter(col("itemid").isin(itemids)) \
                            .join(cohort_broadcast, on=["subject_id", "hadm_id", "stay_id"], how="inner") \
                            .filter((col("valuenum") >= min_val) & (col("valuenum") <= max_val)) \
                            .filter(col("valuenum").isNotNull()) \
                            .coalesce(10)
                
                # Check if we have data
                vital_count = vital_df.count()
                self.logger.info(f"{vital_name} data count: {vital_count}")
                
                if vital_count == 0:
                    self.logger.error(f"CRITICAL: No {vital_name} data found. Core vital signs missing.")
                    raise ValueError(f"No {vital_name} data found - pipeline cannot continue without core vital signs")
            
            # Save with compression
            output_path = self.path_manager.get_path('local_processed_dir', f'{vital_name}.parquet')
            vital_df.write.mode("overwrite").option("compression", "snappy").parquet(output_path)
            
            self.logger.info(f"{vital_name} processing completed. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error in {vital_name} processing: {str(e)}")
            raise
    
    def process_gcs_with_sedation(self, cohort_path):
        """Memory-optimized GCS processing with sedation flag"""
        try:
            self.logger.info("Processing GCS with sedation awareness...")
            spark = self.spark_manager.get_spark("GCS_Sedation")
            
            # Load cohort with broadcast
            cohort = spark.read.csv(cohort_path, header=True, inferSchema=True) \
                          .select("stay_id").dropDuplicates()
            cohort_broadcast = broadcast(cohort)
            
            # Get file paths
            inputevents_path = self.path_manager.get_raw_file_path('inputevents')
            chartevents_path = self.path_manager.get_raw_file_path('chartevents')
            
            # Load sedation data with schema
            sedative_itemids = self.config['sedative_itemids']
            inputevents = spark.read.csv(inputevents_path, header=True, schema=self.schemas.get_inputevents_schema()) \
                .filter(col("itemid").isin(sedative_itemids)) \
                .join(cohort_broadcast, on="stay_id", how="inner") \
                .withColumn("starttime", to_timestamp("starttime")) \
                .select("stay_id", "starttime") \
                .dropDuplicates()
            
            # Load GCS data with schema
            gcs_itemids = self.config['vitals_config']['gcs']['itemids']
            min_val = self.config['vitals_config']['gcs']['min_val']
            max_val = self.config['vitals_config']['gcs']['max_val']
            
            gcs = spark.read.csv(chartevents_path, header=True, schema=self.schemas.get_chartevents_schema()) \
                .filter(col("itemid").isin(gcs_itemids)) \
                .join(cohort_broadcast, on="stay_id", how="inner") \
                .withColumn("charttime", to_timestamp("charttime")) \
                .filter(col("valuenum").isNotNull()) \
                .filter((col("valuenum") >= min_val) & (col("valuenum") <= max_val))
            
            # Optimized join with sedation data
            sedated_flagged = gcs.join(inputevents, on="stay_id", how="left") \
                                 .withColumn("time_diff", 
                                           expr("abs(unix_timestamp(charttime) - unix_timestamp(starttime))")) \
                                 .withColumn("sedated_flag",
                                           when(col("time_diff") <= 14400, 1).otherwise(0)) \
                                 .select("stay_id", "charttime", "itemid", "valuenum", "sedated_flag") \
                                 .coalesce(10)
            
            # Save with compression
            output_path = self.path_manager.get_path('local_processed_dir', 'gcs.parquet')
            sedated_flagged.write.mode("overwrite").option("compression", "snappy").parquet(output_path)
            
            self.logger.info(f"GCS processing completed. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error in GCS processing: {str(e)}")
            raise
    
    def process_bp_cleaning(self, cohort_path):
        """Memory-optimized blood pressure processing"""
        try:
            self.logger.info("Processing blood pressure with memory optimization...")
            spark = self.spark_manager.get_spark("BP_Processing")
            
            # Load cohort with broadcast
            cohort = spark.read.csv(cohort_path, header=True, inferSchema=True) \
                          .select("stay_id").dropDuplicates()
            cohort_broadcast = broadcast(cohort)
            
            # Get chartevents path
            chartevents_path = self.path_manager.get_raw_file_path('chartevents')
            
            # Load BP data with schema
            bp_itemids = [220050, 220051, 220052, 220179, 220180, 220181]
            chartevents_df = spark.read.csv(chartevents_path, header=True, schema=self.schemas.get_chartevents_schema()) \
                .filter(col("itemid").isin(bp_itemids)) \
                .join(cohort_broadcast, on="stay_id", how="inner") \
                .select("stay_id", "charttime", "itemid", "valuenum", "valueuom") \
                .withColumn("charttime", to_timestamp("charttime")) \
                .cache()  # Cache for multiple access
            
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
            
            def process_component_optimized(comp):
                """Optimized BP component processing"""
                value_col = comp.lower()
                lo, hi = EXTREME[comp]
                
                # IBP data with early filtering
                ibp = chartevents_df.filter(col("itemid").isin(ITEMS[comp]["IBP"])) \
                    .select("stay_id", "charttime", col("valuenum").alias(value_col)) \
                    .filter(col(value_col).between(lo, hi)) \
                    .withColumn("ts", unix_timestamp("charttime"))
                
                # NIBP data with early filtering  
                nibp = chartevents_df.filter(col("itemid").isin(ITEMS[comp]["NIBP"])) \
                    .select("stay_id", "charttime", col("valuenum").alias(f"nibp_{value_col}")) \
                    .filter(col(f"nibp_{value_col}").between(lo, hi))
                
                # 5-minute rolling std for flat line detection
                w5 = Window.partitionBy("stay_id").orderBy("ts").rangeBetween(-300, 0)
                ibp = ibp.withColumn(f"std5_{value_col}", stddev(value_col).over(w5)) \
                        .withColumn("flat_artifact", col(f"std5_{value_col}") < 2.0) \
                        .withColumn("is_valid_ibp", ~col("flat_artifact"))
                
                # Efficient merge
                merged = ibp.join(nibp, ["stay_id", "charttime"], "outer") \
                           .withColumn(f"final_{value_col}",
                               when(col("is_valid_ibp") & col(value_col).isNotNull(), col(value_col))
                               .otherwise(col(f"nibp_{value_col}"))
                           ).withColumn(f"{value_col}_source",
                               when(col("is_valid_ibp") & col(value_col).isNotNull(), "IBP")
                               .when(col(f"nibp_{value_col}").isNotNull(), "NIBP")
                               .otherwise("NA")
                           )
                
                return merged.select("stay_id", "charttime", f"final_{value_col}", f"{value_col}_source")
            
            # Process each component
            sbp_res = process_component_optimized("SBP")
            dbp_res = process_component_optimized("DBP") 
            map_res = process_component_optimized("MAP")
            
            # Merge all components efficiently
            final_bp = sbp_res.join(dbp_res, ["stay_id", "charttime"], "outer") \
                             .join(map_res, ["stay_id", "charttime"], "outer") \
                             .withColumn("final_map",
                                 when(col("final_map").isNotNull(), col("final_map"))
                                 .otherwise((col("final_dbp") * 2 + col("final_sbp")) / 3.0)
                             ).coalesce(10)
            
            # Cleanup cache
            chartevents_df.unpersist()
            
            # Save with compression
            output_path = self.path_manager.get_path('local_processed_dir', 'bp.parquet')
            final_bp.write.mode("overwrite").option("compression", "snappy").parquet(output_path)
            
            self.logger.info(f"BP processing completed. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error in BP processing: {str(e)}")
            raise

# ===========================================================
# Memory-Optimized Hourly Binning - 기존 로직 그대로 유지
# ===========================================================
class HourlyBinner:
    def __init__(self, config, path_manager, logger):
        self.config = config
        self.path_manager = path_manager
        self.logger = logger
        self.quality_checker = DataQualityChecker(logger)
        self.memory_manager = MemoryManager()
    
    def create_hourly_bins_for_vitals(self, cohort_path):
        """Memory-optimized hourly binning with chunked processing"""
        try:
            self.logger.info("Starting memory-optimized hourly binning...")
            initial_memory = self.memory_manager.get_memory_usage()
            
            # Load cohort efficiently
            cohort = pd.read_csv(cohort_path, usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'deathtime', 'icu_los_hours'])
            cohort['intime'] = pd.to_datetime(cohort['intime'])
            cohort['outtime'] = pd.to_datetime(cohort['outtime'])
            cohort['deathtime'] = pd.to_datetime(cohort['deathtime'])
            
            # Process vitals in chunks to manage memory
            processed_vitals = []
            vitals_to_process = ['rr', 'hr', 'temp', 'spo2', 'gcs', 'bp']
            chunk_size = self.config['processing']['chunk_size'] // 10  # Smaller chunks for hourly binning
            
            for vital_name in vitals_to_process:
                try:
                    self.logger.info(f"Processing hourly bins for {vital_name}...")
                    vital_result = self._process_vital_chunked(vital_name, cohort, chunk_size)
                    
                    if vital_result is not None:
                        # Save and analyze
                        output_path = self.path_manager.get_path('hourly_bins_dir', f'{vital_name}_hourly_bins.csv')
                        vital_result.to_csv(output_path, index=False)
                        
                        # Quality check with chunking
                        feature_cols = [col for col in vital_result.columns 
                                      if col not in ['subject_id', 'hadm_id', 'stay_id', 'hour_from_intime', 'bin_start', 'bin_end']]
                        missing_analysis = self.quality_checker.check_missing_rate(vital_result, feature_cols, chunk_size=1000)
                        missing_path = self.path_manager.get_path('hourly_bins_dir', f'{vital_name}_missing_analysis.csv')
                        missing_analysis.to_csv(missing_path, index=False)
                        
                        processed_vitals.append(vital_name)
                        self.logger.info(f"Successfully processed {vital_name} - shape: {vital_result.shape}")
                        
                        # Memory cleanup
                        del vital_result
                        self.memory_manager.optimize_garbage_collection()
                    else:
                        self.logger.warning(f"Failed to process {vital_name}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {vital_name}: {str(e)}")
                    continue
            
            final_memory = self.memory_manager.get_memory_usage()
            self.logger.info(f"Memory usage: {initial_memory:.2f}GB -> {final_memory:.2f}GB")
            self.logger.info(f"Hourly binning completed for {len(processed_vitals)} vitals: {processed_vitals}")
            
            return processed_vitals
            
        except Exception as e:
            self.logger.error(f"Error in hourly binning: {str(e)}")
            raise
    
    def _process_vital_chunked(self, vital_name, cohort, chunk_size):
        """Process vital with chunked patient processing"""
        try:
            # Load vital data
            vital_file = self.path_manager.get_path('local_processed_dir', 
                f'{"temp" if vital_name == "temp" else vital_name}.parquet')
            
            if not os.path.exists(vital_file):
                self.logger.warning(f"Vital file not found: {vital_file}")
                return None
            
            # Load vital data in chunks if large
            vital_data = pd.read_parquet(vital_file)
            vital_data['charttime'] = pd.to_datetime(vital_data['charttime'])
            
            if len(vital_data) == 0:
                return None
            
            # Merge with cohort
            vital_merged = vital_data.merge(
                cohort[['stay_id', 'intime', 'outtime', 'deathtime', 'subject_id', 'hadm_id', 'icu_los_hours']], 
                on='stay_id', how='inner'
            )
            
            # Filter to ICU stay period
            mask = (vital_merged['charttime'] >= vital_merged['intime']) & \
                   (vital_merged['charttime'] <= vital_merged['outtime'])
            vital_filtered = vital_merged[mask].copy()
            del vital_merged  # Free memory
            gc.collect()
            
            if len(vital_filtered) == 0:
                return None
            
            # Process patients in chunks
            unique_stays = vital_filtered['stay_id'].unique()
            result_chunks = []
            
            for i in range(0, len(unique_stays), chunk_size):
                stay_chunk = unique_stays[i:i + chunk_size]
                chunk_data = vital_filtered[vital_filtered['stay_id'].isin(stay_chunk)]
                chunk_cohort = cohort[cohort['stay_id'].isin(stay_chunk)]
                
                chunk_results = []
                for stay_id in stay_chunk:
                    patient_data = chunk_data[chunk_data['stay_id'] == stay_id]
                    patient_cohort = chunk_cohort[chunk_cohort['stay_id'] == stay_id].iloc[0]
                    
                    patient_bins = self._create_patient_hourly_bins_optimized(
                        patient_data, patient_cohort, vital_name
                    )
                    chunk_results.extend(patient_bins)
                
                if chunk_results:
                    result_chunks.append(pd.DataFrame(chunk_results))
                
                # Memory cleanup every few chunks
                if i % (chunk_size * 5) == 0:
                    gc.collect()
            
            # Combine all chunks
            if result_chunks:
                final_result = pd.concat(result_chunks, ignore_index=True)
                del result_chunks
                gc.collect()
                return final_result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing {vital_name} chunked: {str(e)}")
            return None
    
    def _create_patient_hourly_bins_optimized(self, patient_data, patient_cohort, vital_name):
        """Memory-optimized patient hourly bin creation"""
        result_list = []
        
        intime = patient_cohort['intime']
        outtime = patient_cohort['outtime']
        deathtime = patient_cohort['deathtime']
        icu_los_hours = patient_cohort['icu_los_hours']
        
        end_time = deathtime if pd.notna(deathtime) else outtime
        total_hours = min(int(np.ceil(icu_los_hours)), 168)  # Cap at 1 week to manage memory
        
        # Determine value column and processing type
        if vital_name == 'temp':
            value_col = 'temperature_celsius'
        elif vital_name == 'bp':
            # Will be handled specially
            value_col = None
        else:
            value_col = 'valuenum'
        
        # Pre-sort and index patient data for efficiency
        patient_data_sorted = patient_data.sort_values('charttime').reset_index(drop=True)
        
        for hour in range(total_hours):
            bin_start = intime + timedelta(hours=hour)
            bin_end = intime + timedelta(hours=hour+1)
            
            if bin_end > end_time:
                bin_end = end_time
            if bin_start >= end_time:
                break
            
            # Efficient hour filtering using binary search concept
            hour_mask = (patient_data_sorted['charttime'] >= bin_start) & \
                       (patient_data_sorted['charttime'] < bin_end)
            hour_data = patient_data_sorted[hour_mask]
            
            result_row = {
                'subject_id': patient_cohort['subject_id'],
                'hadm_id': patient_cohort['hadm_id'], 
                'stay_id': patient_cohort['stay_id'],
                'hour_from_intime': hour,
                'bin_start': bin_start,
                'bin_end': bin_end
            }
            
            # Process based on vital type efficiently
            if vital_name == 'bp':
                bp_cols = ['final_sbp', 'final_dbp', 'final_map']
                for col_name in bp_cols:
                    if len(hour_data) > 0 and col_name in hour_data.columns:
                        col_short = col_name.replace('final_', '')
                        values = hour_data[col_name].dropna()
                        if len(values) > 0:
                            result_row.update({
                                f'{col_short}_mean': values.mean(),
                                f'{col_short}_last': values.iloc[-1],
                                f'{col_short}_min': values.min(),
                                f'{col_short}_max': values.max()
                            })
                        else:
                            result_row.update({f'{col_short}_mean': np.nan, f'{col_short}_last': np.nan,
                                             f'{col_short}_min': np.nan, f'{col_short}_max': np.nan})
                    else:
                        col_short = col_name.replace('final_', '')
                        result_row.update({f'{col_short}_mean': np.nan, f'{col_short}_last': np.nan,
                                         f'{col_short}_min': np.nan, f'{col_short}_max': np.nan})
                        
            elif vital_name == 'gcs':
                if len(hour_data) > 0:
                    values = hour_data[value_col].dropna()
                    if len(values) > 0:
                        result_row.update({
                            'gcs_mean': values.mean(),
                            'gcs_last': values.iloc[-1],
                            'gcs_min': values.min(),
                            'gcs_max': values.max(),
                            'sedated_flag': 1 if (hour_data.get('sedated_flag', 0) == 1).any() else 0
                        })
                    else:
                        result_row.update({
                            'gcs_mean': np.nan, 'gcs_last': np.nan, 'gcs_min': np.nan, 'gcs_max': np.nan,
                            'sedated_flag': np.nan
                        })
                else:
                    result_row.update({
                        'gcs_mean': np.nan, 'gcs_last': np.nan, 'gcs_min': np.nan, 'gcs_max': np.nan,
                        'sedated_flag': np.nan
                    })
                    
            else:
                # Standard vital signs
                prefix = vital_name
                if len(hour_data) > 0:
                    values = hour_data[value_col].dropna()
                    if len(values) > 0:
                        result_row.update({
                            f'{prefix}_mean': values.mean(),
                            f'{prefix}_last': values.iloc[-1],
                            f'{prefix}_min': values.min(),
                            f'{prefix}_max': values.max()
                        })
                    else:
                        result_row.update({f'{prefix}_mean': np.nan, f'{prefix}_last': np.nan,
                                         f'{prefix}_min': np.nan, f'{prefix}_max': np.nan})
                else:
                    result_row.update({f'{prefix}_mean': np.nan, f'{prefix}_last': np.nan,
                                     f'{prefix}_min': np.nan, f'{prefix}_max': np.nan})
            
            result_list.append(result_row)
        
        return result_list

# ===========================================================
# Memory-Optimized Data Merger - 기존 로직 그대로 유지
# ===========================================================
class DataMerger:
    def __init__(self, config, path_manager, logger):
        self.config = config
        self.path_manager = path_manager
        self.logger = logger
        self.quality_checker = DataQualityChecker(logger)
        self.memory_manager = MemoryManager()
    
    def merge_all_vitals(self):
        """Memory-optimized vital signs merging"""
        try:
            self.logger.info("Starting memory-optimized vitals merge...")
            initial_memory = self.memory_manager.get_memory_usage()
            
            file_info = [
                ('bp_hourly_bins.csv', ['sbp', 'dbp', 'map']),
                ('hr_hourly_bins.csv', ['hr']),
                ('temp_hourly_bins.csv', ['temp']),
                ('spo2_hourly_bins.csv', ['spo2']),
                ('rr_hourly_bins.csv', ['rr']),
                ('gcs_hourly_bins.csv', ['gcs'])
            ]
            
            merge_keys = ['subject_id', 'hadm_id', 'stay_id', 'hour_from_intime', 'bin_start', 'bin_end']
            
            # Load files with memory optimization
            dfs = []
            available_vitals = []
            
            for filename, vital_names in file_info:
                file_path = self.path_manager.get_path('hourly_bins_dir', filename)
                
                if os.path.exists(file_path):
                    # Load with specific dtypes to save memory
                    dtype_dict = {
                        'subject_id': 'int64',
                        'hadm_id': 'int64', 
                        'stay_id': 'int64',
                        'hour_from_intime': 'int16'
                    }
                    
                    df = pd.read_csv(file_path, dtype=dtype_dict)
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
            
            # Memory-optimized iterative merge
            merged = dfs[0].copy()
            del dfs[0]
            
            for i, df in enumerate(dfs):
                before_shape = merged.shape
                
                # Use memory-efficient merge
                merged = pd.merge(merged, df, on=merge_keys, how='inner', copy=False)
                
                # Cleanup
                del df
                if i % 2 == 0:  # GC every 2 merges
                    gc.collect()
                
                after_shape = merged.shape
                self.logger.info(f"After merge {i+1}: {before_shape} -> {after_shape}")
            
            # Memory cleanup and optimization
            del dfs
            self.memory_manager.optimize_garbage_collection()
            
            # Quality assessment with chunking
            self.logger.info(f"Final merged dataset shape: {merged.shape}")
            
            # Save results
            output_path = self.path_manager.get_path('hourly_bins_dir', 'all_features_hourly_bins_inner.csv')
            merged.to_csv(output_path, index=False)
            
            final_memory = self.memory_manager.get_memory_usage()
            self.logger.info(f"Memory usage: {initial_memory:.2f}GB -> {final_memory:.2f}GB")
            self.logger.info(f"Vitals merge completed successfully! Output: {output_path}")
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error in vitals merging: {str(e)}")
            raise

# ===========================================================
# Memory-Optimized Sliding Window Creator - 기존 로직 그대로 유지
# ===========================================================  
class SlidingWindowCreator:
    def __init__(self, config, path_manager, logger):
        self.config = config
        self.path_manager = path_manager
        self.logger = logger
        self.quality_checker = DataQualityChecker(logger)
        self.memory_manager = MemoryManager()
    
    def create_sliding_windows(self):
        """Memory-optimized sliding window creation with chunked processing"""
        try:
            # Get parameters
            obs_window_hours = self.config['processing']['obs_window_hours']
            pred_window_hours = self.config['processing']['pred_window_hours'] 
            pred_interval_hours = self.config['processing']['pred_interval_hours']
            extra_hours_after_pred = self.config['processing']['extra_hours_after_pred']
            min_completeness = self.config['processing']['min_obs_completeness']
            
            self.logger.info(f"Creating memory-optimized sliding windows...")
            self.logger.info(f"Parameters: obs={obs_window_hours}h, pred={pred_window_hours}h, interval={pred_interval_hours}h")
            
            initial_memory = self.memory_manager.get_memory_usage()
            
            # Load data with memory optimization
            data_path = self.path_manager.get_path('hourly_bins_dir', 'all_features_hourly_bins_inner.csv')
            cohort_path = self.path_manager.get_path('local_processed_dir', 'cohort.csv')
            
            if not os.path.exists(data_path) or not os.path.exists(cohort_path):
                raise FileNotFoundError("Required files not found for sliding windows")
            
            # Load cohort with minimal columns
            cohort = pd.read_csv(cohort_path, usecols=['subject_id', 'stay_id', 'intime', 'outtime', 'deathtime'])
            cohort['intime'] = pd.to_datetime(cohort['intime'])
            cohort['outtime'] = pd.to_datetime(cohort['outtime'])
            cohort['deathtime'] = pd.to_datetime(cohort['deathtime'])
            
            # Load data with optimized dtypes
            dtype_dict = {
                'subject_id': 'int64',
                'hadm_id': 'int64',
                'stay_id': 'int64', 
                'hour_from_intime': 'int16'
            }
            
            data = pd.read_csv(data_path, dtype=dtype_dict)
            self.logger.info(f"Loaded data: {data.shape}, cohort: {cohort.shape}")
            
            # Merge with cohort
            data_with_cohort = data.merge(
                cohort[['subject_id', 'stay_id', 'intime', 'outtime', 'deathtime']],
                on=['subject_id', 'stay_id'], how='inner'
            )
            del data, cohort  # Free memory
            gc.collect()
            
            # Define feature columns
            exclude_cols = ['subject_id', 'stay_id', 'hour_from_intime', 'hadm_id', 'bin_start', 'bin_end']
            feature_cols = [c for c in data_with_cohort.columns if c not in exclude_cols and c not in ['intime', 'outtime', 'deathtime']]
            self.logger.info(f"Feature columns: {len(feature_cols)}")
            
            # Process windows in chunks
            chunk_size = self.config['processing']['chunk_size'] // 20  # Smaller chunks for memory
            windows, sequences = self._process_windows_chunked(
                data_with_cohort, feature_cols, obs_window_hours, pred_window_hours,
                pred_interval_hours, extra_hours_after_pred, min_completeness, chunk_size
            )
            
            # Convert to arrays with memory management
            windows_df = pd.DataFrame(windows)
            sequences_array = np.array(sequences)
            del windows, sequences
            gc.collect()
            
            # Save results
            metadata_path = self.path_manager.get_path('output_dir', 'sliding_windows_metadata.csv')
            sequences_path = self.path_manager.get_path('output_dir', 'sliding_windows_sequences.npy')
            
            windows_df.to_csv(metadata_path, index=False)
            np.save(sequences_path, sequences_array)
            
            # Quality analysis
            pos_labels = windows_df['death_in_pred_window_new'].sum()
            total_windows = len(windows_df)
            
            final_memory = self.memory_manager.get_memory_usage()
            self.logger.info(f"Memory usage: {initial_memory:.2f}GB -> {final_memory:.2f}GB")
            self.logger.info(f"Sliding windows completed: {total_windows} windows, {pos_labels} positive ({pos_labels/total_windows:.4f})")
            self.logger.info(f"Sequence shape: {sequences_array.shape}")
            
            return windows_df, sequences_array
            
        except Exception as e:
            self.logger.error(f"Error in sliding windows creation: {str(e)}")
            raise
    
    def _process_windows_chunked(self, data_with_cohort, feature_cols, obs_window_hours, 
                               pred_window_hours, pred_interval_hours, extra_hours_after_pred, 
                               min_completeness, chunk_size):
        """Process sliding windows in chunks to manage memory"""
        
        unique_subjects = data_with_cohort['subject_id'].unique()
        total_subjects = len(unique_subjects)
        
        windows = []
        sequences = []
        stats = {'total_possible': 0, 'death_during_obs': 0, 'low_completeness': 0, 'created': 0, 'positive': 0}
        
        # Process subjects in chunks
        for i in range(0, total_subjects, chunk_size):
            chunk_subjects = unique_subjects[i:i + chunk_size]
            chunk_data = data_with_cohort[data_with_cohort['subject_id'].isin(chunk_subjects)]
            
            # Process each subject in chunk
            for subject_id in chunk_subjects:
                subject_data = chunk_data[chunk_data['subject_id'] == subject_id].sort_values('hour_from_intime').reset_index(drop=True)
                
                if len(subject_data) == 0:
                    continue
                
                subject_windows, subject_sequences, subject_stats = self._process_subject_windows_optimized(
                    subject_id, subject_data, feature_cols, obs_window_hours,
                    pred_window_hours, pred_interval_hours, extra_hours_after_pred, min_completeness
                )
                
                windows.extend(subject_windows)
                sequences.extend(subject_sequences)
                
                # Update statistics
                for key in stats:
                    stats[key] += subject_stats.get(key, 0)
            
            # Memory cleanup every few chunks
            if i % (chunk_size * 3) == 0:
                gc.collect()
                current_memory = self.memory_manager.get_memory_usage()
                self.logger.info(f"Processed {i + chunk_size}/{total_subjects} subjects, Memory: {current_memory:.2f}GB")
        
        self.logger.info(f"Window processing statistics: {stats}")
        return windows, sequences
    
    def _process_subject_windows_optimized(self, subject_id, subject_data, feature_cols, obs_window_hours,
                                         pred_window_hours, pred_interval_hours, extra_hours_after_pred, 
                                         min_completeness):
        """Memory-optimized subject window processing"""
        
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
        
        # Pre-index subject data for efficient lookups
        subject_indexed = subject_data.set_index('hour_from_intime')
        
        for window_id, obs_start in enumerate(possible_starts):
            obs_end = obs_start + obs_window_hours
            pred_start = obs_end
            pred_end = obs_end + pred_window_hours
            
            obs_start_time = intime + pd.Timedelta(hours=obs_start)
            obs_end_time = intime + pd.Timedelta(hours=obs_end)
            pred_start_time = intime + pd.Timedelta(hours=pred_start)
            pred_end_time = intime + pd.Timedelta(hours=pred_end)
            
            # Determine window label
            remove_window, new_label = self._determine_window_label(
                deathtime, obs_start_time, obs_end_time, pred_start_time, pred_end_time, 
                extra_hours_after_pred
            )
            
            if remove_window:
                stats['death_during_obs'] += 1
                continue
            
            # Create observation sequence efficiently
            obs_sequence, obs_completeness = self._create_observation_sequence_optimized(
                subject_indexed, feature_cols, obs_start, obs_end
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
        """Determine window label"""
        remove_window = False
        new_label = 0
        
        if pd.notna(deathtime):
            if obs_start_time <= deathtime <= obs_end_time:
                remove_window = True
            elif pred_start_time <= deathtime <= pred_end_time:
                new_label = 1
            elif pred_end_time < deathtime <= pred_end_time + pd.Timedelta(hours=extra_hours_after_pred):
                new_label = 1
        
        return remove_window, new_label
    
    def _create_observation_sequence_optimized(self, subject_indexed, feature_cols, obs_start, obs_end):
        """Memory-optimized observation sequence creation"""
        
        hour_range = np.arange(obs_start, obs_end)
        sequence_data = []
        total_values = len(hour_range) * len(feature_cols)
        missing_values = 0
        
        # Vectorized processing where possible
        for hour in hour_range:
            if hour in subject_indexed.index:
                hour_features = subject_indexed.loc[hour, feature_cols].values
                missing_in_hour = pd.isna(hour_features).sum()
            else:
                hour_features = np.full(len(feature_cols), np.nan)
                missing_in_hour = len(feature_cols)
                
            sequence_data.append(hour_features)
            missing_values += missing_in_hour
        
        # Calculate completeness efficiently
        obs_completeness = 1 - (missing_values / total_values) if total_values > 0 else 0
        
        return np.array(sequence_data), obs_completeness

# ===========================================================
# Memory-Optimized Pipeline Orchestrator with S3 Integration
# ===========================================================
class MIMICPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging(
            log_level=self.config.get('logging', {}).get('level', 'INFO'),
            log_file=self.config.get('logging', {}).get('file', 'pipeline.log')
        )
        
        self.path_manager = PathManager(self.config)
        self.s3_manager = S3Manager(self.config, self.logger)  # S3 매니저 추가
        self.spark_manager = SparkManager(self.logger)
        self.memory_manager = MemoryManager()
        
        # Initialize processors
        self.cohort_selector = CohortSelector(self.config, self.path_manager, self.logger)
        self.vitals_processor = VitalSignsProcessor(self.config, self.path_manager, self.spark_manager, self.logger)
        self.hourly_binner = HourlyBinner(self.config, self.path_manager, self.logger)
        self.data_merger = DataMerger(self.config, self.path_manager, self.logger)
        self.window_creator = SlidingWindowCreator(self.config, self.path_manager, self.logger)
    
    def _download_raw_data_from_s3(self):
        """Download raw data from S3 bucket"""
        try:
            self.logger.info("Starting raw data download from S3...")
            
            # S3 file mapping for MIMIC-IV structure
            raw_files = {
                "patients": "hosp/patients.csv.gz",
                "admissions": "hosp/admissions.csv.gz", 
                "icustays": "icu/icustays.csv.gz",
                "chartevents": "icu/chartevents.csv.gz",
                "inputevents": "icu/inputevents.csv.gz",
                "procedureevents": "icu/procedureevents.csv.gz"
            }
            
            # Parse S3 bucket configuration
            raw_bucket_url = self.config['aws']['raw_bucket']
            bucket_name, s3_prefix = self.s3_manager.parse_s3_path(raw_bucket_url)
            
            self.logger.info(f"Downloading from bucket: {bucket_name}, prefix: {s3_prefix}")
            
            local_paths = {}
            total_size_mb = 0
            
            # Download each file
            for name, s3_key in raw_files.items():
                # Construct full S3 key
                full_s3_key = f"{s3_prefix}/{s3_key}" if s3_prefix else s3_key
                
                # Local path for downloaded file
                local_filename = os.path.basename(s3_key)
                local_path = self.path_manager.get_path('local_raw_dir', local_filename)
                
                # Download with retry logic
                self.logger.info(f"Downloading {name} from s3://{bucket_name}/{full_s3_key}")
                success = self.s3_manager.download_from_s3(bucket_name, full_s3_key, local_path)
                
                if success:
                    local_paths[name] = local_path
                    # Log file size for verification
                    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)  # MB
                    total_size_mb += file_size_mb
                    self.logger.info(f"Downloaded {name}: {file_size_mb:.1f} MB")
                else:
                    raise Exception(f"Failed to download {name}")
            
            self.logger.info(f"Successfully downloaded {len(local_paths)} files from S3 (total: {total_size_mb:.1f} MB)")
            return local_paths
            
        except Exception as e:
            self.logger.error(f"Error downloading raw data from S3: {str(e)}")
            raise
    
    def _upload_final_results_to_s3(self):
        """Upload final processed results to S3 bucket"""
        try:
            self.logger.info("Starting upload of final results to S3...")
            
            # Parse processed bucket configuration
            processed_bucket_url = self.config['aws']['processed_bucket']
            bucket_name, s3_prefix = self.s3_manager.parse_s3_path(processed_bucket_url)
            
            # List of final output files to upload
            output_dir = self.path_manager.get_path('output_dir')
            
            final_files = [
                'tcn_input_combined.npy',
                'tcn_labels.npy', 
                'tcn_metadata_with_static.csv',
                'tcn_feature_names.txt',
                'sliding_windows_metadata.csv',
                'sliding_windows_sequences.npy'
            ]
            
            # Upload each file
            uploaded_count = 0
            total_upload_size_mb = 0
            
            for filename in final_files:
                local_path = os.path.join(output_dir, filename)
                
                if os.path.exists(local_path):
                    # Construct S3 key
                    s3_key = f"{s3_prefix}/final_dataset/{filename}" if s3_prefix else f"final_dataset/{filename}"
                    
                    # Upload with retry logic
                    success = self.s3_manager.upload_to_s3(local_path, bucket_name, s3_key)
                    
                    if success:
                        uploaded_count += 1
                        # Log file size for verification
                        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)  # MB
                        total_upload_size_mb += file_size_mb
                    else:
                        self.logger.error(f"Failed to upload {filename}")
                else:
                    self.logger.warning(f"Output file not found: {local_path}")
            
            # Upload intermediate results (optional but useful for debugging)
            self.logger.info("Uploading intermediate results...")
            intermediate_files = [
                ('local_processed_dir', 'cohort.csv'),
                ('hourly_bins_dir', 'all_features_hourly_bins_inner.csv')
            ]
            
            for dir_type, filename in intermediate_files:
                local_path = self.path_manager.get_path(dir_type, filename)
                
                if os.path.exists(local_path):
                    s3_key = f"{s3_prefix}/intermediate/{filename}" if s3_prefix else f"intermediate/{filename}"
                    
                    success = self.s3_manager.upload_to_s3(local_path, bucket_name, s3_key)
                    
                    if success:
                        uploaded_count += 1
                        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                        total_upload_size_mb += file_size_mb
            
            self.logger.info(f"Successfully uploaded {uploaded_count} files to S3 (total: {total_upload_size_mb:.1f} MB)")
            self.logger.info(f"Final processed data available at: s3://{bucket_name}/{s3_prefix}/final_dataset/")
            
        except Exception as e:
            self.logger.error(f"Error uploading results to S3: {str(e)}")
            raise
    
    def run_full_pipeline(self):
        """Execute the memory-optimized preprocessing pipeline with S3 integration"""
        try:
            self.logger.info("=== Starting Memory-Optimized MIMIC-IV Pipeline with S3 Integration ===")
            initial_memory = self.memory_manager.get_memory_usage()
            start_time = pd.Timestamp.now()
            
            # Step 1: Download raw data from S3
            self.logger.info("=== Step 1: Downloading raw data from S3 ===")
            raw_file_paths = self._download_raw_data_from_s3()
            
            # Step 2: Select cohort
            self.logger.info("=== Step 2: Selecting cohort ===")
            cohort_df = self.cohort_selector.select_cohort()
            
            # Step 3: Process vital signs
            self.logger.info("=== Step 3: Processing vital signs ===")
            cohort_path = self.path_manager.get_path('local_processed_dir', 'cohort.csv')
            self._process_all_vitals(cohort_path)
            
            # Step 4: Create hourly bins
            self.logger.info("=== Step 4: Creating hourly bins ===")
            processed_vitals = self.hourly_binner.create_hourly_bins_for_vitals(cohort_path)
            
            # Step 5: Merge all vitals
            self.logger.info("=== Step 5: Merging all vitals ===")
            merged_data = self.data_merger.merge_all_vitals()
            
            # Memory cleanup before intensive operations
            del merged_data
            self.memory_manager.optimize_garbage_collection()
            
            # Step 6: Create sliding windows
            self.logger.info("=== Step 6: Creating sliding windows ===")
            windows_df, sequences_array = self.window_creator.create_sliding_windows()
            
            # Step 7: Final processing
            self.logger.info("=== Step 7: Final processing ===")
            final_features = self._finalize_features(windows_df, sequences_array)
            
            # Step 8: Upload results to S3
            self.logger.info("=== Step 8: Uploading results to S3 ===")
            self._upload_final_results_to_s3()
            
            # Final statistics
            end_time = pd.Timestamp.now()
            total_time = (end_time - start_time).total_seconds() / 60  # minutes
            final_memory = self.memory_manager.get_memory_usage()
            
            self.logger.info(f"=== Pipeline completed successfully! ===")
            self.logger.info(f"Total execution time: {total_time:.1f} minutes")
            self.logger.info(f"Total memory usage: {initial_memory:.2f}GB -> {final_memory:.2f}GB")
            self.logger.info(f"Final dataset shape: {final_features[0].shape}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            self.spark_manager.stop()
            # Optional: Clean up temporary files
            self._cleanup_temp_files()
    
    def _process_all_vitals(self, cohort_path):
        """Process all vital signs"""
        
        self.vitals_processor.process_respiratory_rate(cohort_path)
        
        vitals_to_process = ['hr', 'temp', 'spo2']
        for vital in vitals_to_process:
            self.vitals_processor.process_vital_sign_generic(vital, cohort_path)
        
        self.vitals_processor.process_gcs_with_sedation(cohort_path)
        
        self.vitals_processor.process_bp_cleaning(cohort_path)
    
    def _finalize_features(self, windows_df, sequences_array):
        """Memory-optimized feature finalization"""
        try:
            self.logger.info("Finalizing features with memory optimization...")
        
            # Load cohort efficiently
            cohort_path = self.path_manager.get_path('local_processed_dir', 'cohort.csv')
            cohort_df = pd.read_csv(cohort_path, usecols=['subject_id', 'stay_id', 'age', 'gender_M', 'gender_F'])
        
            static_cols = ['age', 'gender_M', 'gender_F']
            static_df = cohort_df[['subject_id', 'stay_id'] + static_cols]
            
            # Merge efficiently
            meta_with_static = windows_df.merge(static_df, on=['subject_id', 'stay_id'], how='left')
            del cohort_df, static_df
            gc.collect()
            
            # Load feature names and exclude unwanted columns
            data_path = self.path_manager.get_path('hourly_bins_dir', 'all_features_hourly_bins_inner.csv')
            if os.path.exists(data_path):
                sample_data = pd.read_csv(data_path, nrows=1)
                exclude_cols = ['subject_id', 'stay_id', 'hour_from_intime', 'hadm_id', 'bin_start', 'bin_end']
                temporal_features = [c for c in sample_data.columns if c not in exclude_cols]
                
                # Remove unwanted columns from temporal features
                unwanted_cols = ['temp_min', 'temp_max', 'gcs_min', 'gcs_max']
                temporal_features = [f for f in temporal_features if f not in unwanted_cols]
                self.logger.info(f"Excluded columns: {unwanted_cols}")
                
                # Find indices of unwanted columns in the sequences array
                all_temporal_features = [c for c in sample_data.columns if c not in exclude_cols]
                indices_to_keep = [i for i, f in enumerate(all_temporal_features) if f not in unwanted_cols]
                
                # Filter sequences array to remove unwanted columns
                sequences_array = sequences_array[:, :, indices_to_keep]
                self.logger.info(f"Filtered sequences shape: {sequences_array.shape}")
            else:
                temporal_features = [f'feature_{i}' for i in range(sequences_array.shape[2])]
            
            # Clean sequences array to ensure float dtype while preserving NaN
            self.logger.info("Converting sequences to float format while preserving NaN...")
            sequences_array = sequences_array.astype(np.float32)  # NaN 보존
            
            # Expand static features efficiently
            N, T, F_temporal = sequences_array.shape
            X_static = meta_with_static[static_cols].to_numpy().astype(np.float32)
            X_static_expanded = np.repeat(X_static[:, np.newaxis, :], T, axis=1)
            
            # Combine features
            X_combined = np.concatenate([sequences_array, X_static_expanded], axis=2)
            y = meta_with_static['death_in_pred_window_new'].to_numpy().astype(np.int32)
            
            # Cleanup
            del X_static, X_static_expanded, sequences_array
            gc.collect()
            
            # Save results with clean dtypes
            output_dir = self.path_manager.get_path('output_dir')
            
            np.save(os.path.join(output_dir, 'tcn_input_combined.npy'), X_combined)
            np.save(os.path.join(output_dir, 'tcn_labels.npy'), y)
            meta_with_static.to_csv(os.path.join(output_dir, 'tcn_metadata_with_static.csv'), index=False)
            
            # Save filtered feature names
            combined_features = temporal_features + static_cols
            
            with open(os.path.join(output_dir, 'tcn_feature_names.txt'), 'w') as f:
                f.write("\n".join(combined_features))
            
            self.logger.info(f"Features finalized - Shape: {X_combined.shape}")
            self.logger.info(f"Total features: {len(combined_features)} (temporal: {len(temporal_features)}, static: {len(static_cols)})")
            return X_combined, y, combined_features
            
        except Exception as e:
            self.logger.error(f"Error in feature finalization: {str(e)}")
            raise
    
    def _cleanup_temp_files(self):
        """Optional cleanup of temporary downloaded files to save disk space"""
        try:
            if self.config.get('cleanup_temp_files', False):
                self.logger.info("Cleaning up temporary raw data files...")
                
                raw_dir = self.path_manager.get_path('local_raw_dir')
                if os.path.exists(raw_dir):
                    import shutil
                    shutil.rmtree(raw_dir)
                    self.logger.info(f"Removed temporary directory: {raw_dir}")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {str(e)}")

# ===========================================================
# Configuration File Generator
# ===========================================================
def generate_default_config(output_path="config.yaml"):
    """Generate a default configuration file for S3 integration"""
    default_config = {
        'aws': {
            'raw_bucket': 's3://your-mimic4-raw-bucket/raw-data',
            'processed_bucket': 's3://your-mimic4-processed-bucket/processed-data',
            'region': 'us-east-1'  # AWS 리전 설정
        },
        'paths': {
            'local_raw_dir': './data/raw',  # S3에서 다운로드할 임시 로컬 디렉토리
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
            'min_adult_age': 18,
            'chunk_size': 50000,
            'max_memory_gb': 8
        },
        'vitals_config': {
            'rr': {'itemids': [220210], 'min_val': 0, 'max_val': 100, 'vent_aware': True},
            'hr': {'itemids': [220045], 'min_val': 20, 'max_val': 230, 'vent_aware': False},
            'temp_c': {'itemids': [223762], 'min_val': 33.0, 'max_val': 42.0, 'vent_aware': False},
            'temp_f': {'itemids': [223761], 'min_val': 91.4, 'max_val': 107.6, 'vent_aware': False},
            'spo2': {'itemids': [220277, 646], 'min_val': 70, 'max_val': 100, 'vent_aware': False},
            'gcs': {'itemids': [220739, 223900, 223901], 'min_val': 3, 'max_val': 15, 'vent_aware': False}
        },
        'sedative_itemids': [221319, 221668, 222168, 221744, 225942, 225972, 221320, 229420, 225150, 221195, 227212],
        'ventilator_itemids': [223848, 223849, 223870, 225792, 225794],
        'cleanup_temp_files': False,  # 임시 파일 정리 여부
        'logging': {
            'level': 'INFO',
            'file': 'pipeline.log'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Default S3 configuration saved to {output_path}")
    print("\nIMPORTANT: Please update the following settings in config.yaml:")
    print("1. Set 'raw_bucket' to your S3 bucket containing MIMIC-IV raw data")
    print("2. Set 'processed_bucket' to your S3 bucket for processed results")
    print("3. Ensure AWS credentials are configured (AWS CLI, IAM roles, or environment variables)")
    print("4. Verify the AWS region matches your S3 buckets")

# ===========================================================
# Utility Functions for S3 Operations
# ===========================================================
def test_s3_connection(config_path="config.yaml"):
    """Test S3 connection and permissions"""
    try:
        config = load_config(config_path)
        logger = setup_logging()
        s3_manager = S3Manager(config, logger)
        
        # Test raw bucket access
        raw_bucket_url = config['aws']['raw_bucket']
        bucket_name, s3_prefix = s3_manager.parse_s3_path(raw_bucket_url)
        
        logger.info(f"Testing access to raw bucket: {bucket_name}")
        objects = s3_manager.list_s3_objects(bucket_name, s3_prefix)
        logger.info(f"Found {len(objects)} objects in raw bucket")
        
        # Test processed bucket access
        processed_bucket_url = config['aws']['processed_bucket']
        bucket_name, s3_prefix = s3_manager.parse_s3_path(processed_bucket_url)
        
        logger.info(f"Testing access to processed bucket: {bucket_name}")
        objects = s3_manager.list_s3_objects(bucket_name, s3_prefix)
        logger.info(f"Found {len(objects)} objects in processed bucket")
        
        logger.info("S3 connection test successful!")
        return True
        
    except Exception as e:
        logger.error(f"S3 connection test failed: {str(e)}")
        return False

def list_s3_raw_files(config_path="config.yaml"):
    """List available raw files in S3 bucket"""
    try:
        config = load_config(config_path)
        logger = setup_logging()
        s3_manager = S3Manager(config, logger)
        
        raw_bucket_url = config['aws']['raw_bucket']
        bucket_name, s3_prefix = s3_manager.parse_s3_path(raw_bucket_url)
        
        logger.info(f"Listing files in s3://{bucket_name}/{s3_prefix}")
        objects = s3_manager.list_s3_objects(bucket_name, s3_prefix)
        
        # Filter for MIMIC-IV structure
        mimic_files = {}
        expected_files = {
            'patients': 'hosp/patients.csv.gz',
            'admissions': 'hosp/admissions.csv.gz',
            'icustays': 'icu/icustays.csv.gz',
            'chartevents': 'icu/chartevents.csv.gz',
            'inputevents': 'icu/inputevents.csv.gz',
            'procedureevents': 'icu/procedureevents.csv.gz'
        }
        
        for name, expected_path in expected_files.items():
            full_expected_path = f"{s3_prefix}/{expected_path}" if s3_prefix else expected_path
            if full_expected_path in objects:
                mimic_files[name] = full_expected_path
                logger.info(f"✓ Found {name}: {full_expected_path}")
            else:
                logger.warning(f"✗ Missing {name}: {full_expected_path}")
        
        logger.info(f"Found {len(mimic_files)}/{len(expected_files)} required MIMIC-IV files")
        return mimic_files
        
    except Exception as e:
        logger.error(f"Error listing S3 files: {str(e)}")
        return {}

# ===========================================================
# Main Execution Functions
# ===========================================================
def main():
    """Main execution function with S3 integration"""
    try:
        # Generate default config if it doesn't exist
        if not os.path.exists("config.yaml"):
            print("No config.yaml found. Generating default S3 configuration...")
            generate_default_config()
            print("Please review and modify config.yaml before running the pipeline.")
            return
        
        # Test S3 connection first
        print("Testing S3 connection...")
        if not test_s3_connection():
            print("S3 connection test failed. Please check your AWS credentials and configuration.")
            return
        
        # List available files
        print("Checking available raw files in S3...")
        available_files = list_s3_raw_files()
        if len(available_files) < 6:
            print("Warning: Not all required MIMIC-IV files found in S3 bucket.")
            print("Please ensure all required files are uploaded to your S3 bucket.")
        
        # Initialize and run pipeline
        pipeline = MIMICPipeline()
        pipeline.run_full_pipeline()
        
        print("Pipeline completed successfully!")
        print("Processed data has been uploaded to your S3 processed bucket.")
        
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        raise

def run_pipeline_step(step_name, config_path="config.yaml"):
    """Run a specific step of the pipeline"""
    pipeline = MIMICPipeline(config_path)
    
    if step_name == "test_s3":
        print("Testing S3 connection...")
        success = test_s3_connection(config_path)
        if success:
            print("S3 connection successful!")
        else:
            print("S3 connection failed!")
        
    elif step_name == "list_files":
        print("Listing available files in S3...")
        files = list_s3_raw_files(config_path)
        print(f"Found {len(files)} MIMIC-IV files")
        
    elif step_name == "download":
        print("Downloading raw data from S3...")
        raw_file_paths = pipeline._download_raw_data_from_s3()
        print(f"Downloaded {len(raw_file_paths)} files successfully")
        
    elif step_name == "upload":
        print("Uploading results to S3...")
        pipeline._upload_final_results_to_s3()
        print("Upload completed successfully")
        
    elif step_name == "cohort":
        print("Running cohort selection...")
        # Download first if needed
        if not os.path.exists(pipeline.path_manager.get_raw_file_path('patients')):
            pipeline._download_raw_data_from_s3()
        cohort_df = pipeline.cohort_selector.select_cohort()
        print(f"Cohort selection completed. Size: {len(cohort_df)}")
        
    elif step_name == "vitals":
        print("Running vital signs processing...")
        cohort_path = pipeline.path_manager.get_path('local_processed_dir', 'cohort.csv')
        pipeline._process_all_vitals(cohort_path)
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
    
    elif step_name == "finalize":
        print("Running feature finalization...")
        # Load sliding windows data
        metadata_path = pipeline.path_manager.get_path('output_dir', 'sliding_windows_metadata.csv')
        sequences_path = pipeline.path_manager.get_path('output_dir', 'sliding_windows_sequences.npy')
        
        if not os.path.exists(metadata_path) or not os.path.exists(sequences_path):
            print("Error: Sliding window files not found. Please run 'windows' step first.")
            return
            
        windows_df = pd.read_csv(metadata_path)
        sequences_array = np.load(sequences_path, allow_pickle=True)
        
        X_combined, y, combined_features = pipeline._finalize_features(windows_df, sequences_array)
        print(f"Feature finalization completed. Final shape: {X_combined.shape}")
        print(f"Total features: {len(combined_features)}")
        
    else:
        print(f"Unknown step: {step_name}")
        print("Available steps: test_s3, list_files, download, upload, cohort, vitals, bins, merge, windows, finalize")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        step_name = sys.argv[1]
        config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        run_pipeline_step(step_name, config_path)
    else:
        main()