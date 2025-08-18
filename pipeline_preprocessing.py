import os
import boto3
import pandas as pd
import numpy as np
from datetime import timedelta
from functools import reduce
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_timestamp, unix_timestamp, expr, abs as spark_abs, stddev
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType, DoubleType, StringType
from pyspark.sql import Window

# ===========================================================
# Spark Session Manager (메모리 효율성 개선)
# ===========================================================
class SparkManager:
    def __init__(self):
        self._spark = None
    
    def get_spark(self, app_name="MIMIC_Preprocessing"):
        if self._spark is None:
            self._spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .getOrCreate()
        return self._spark
    
    def stop(self):
        if self._spark:
            self._spark.stop()
            self._spark = None

# 전역 Spark 매니저
spark_manager = SparkManager()

# ===========================================================
# AWS S3 유틸 (에러 핸들링 추가)
# ===========================================================
def download_from_s3(bucket_name, s3_key, local_path):
    try:
        s3 = boto3.client("s3")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"Successfully downloaded: {s3_key}")
    except Exception as e:
        print(f"Error downloading {s3_key}: {str(e)}")
        raise

def upload_to_s3(local_path, bucket_name, s3_key):
    try:
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket_name, s3_key)
        print(f"Successfully uploaded: {s3_key}")
    except Exception as e:
        print(f"Error uploading {local_path}: {str(e)}")
        raise

# ===========================================================
# Cohort 선정
# ===========================================================
def select_cohort(patients_path, admissions_path, icustays_path):
    try:
        patients = pd.read_csv(patients_path)
        admissions = pd.read_csv(admissions_path)
        icustays = pd.read_csv(icustays_path)

        admissions['deathtime'] = pd.to_datetime(admissions['deathtime'], errors='coerce')
        icustays['intime'] = pd.to_datetime(icustays['intime'], errors='coerce')
        icustays['outtime'] = pd.to_datetime(icustays['outtime'], errors='coerce')
        icustays['icu_los_hours'] = (icustays['outtime'] - icustays['intime']).dt.total_seconds() / 3600

        adult_patients = patients[patients['anchor_age'] >= 18]
        data = admissions.merge(adult_patients, on='subject_id')
        cohort = data.merge(icustays, on=['subject_id', 'hadm_id'])
        cohort = cohort.sort_values(['subject_id', 'intime']).groupby('subject_id').first().reset_index()
        cohort = cohort[(cohort['icu_los_hours'] >= 24) & (cohort['icu_los_hours'] <= 720)]
        cohort = cohort[(cohort['deathtime'].isna()) | (cohort['deathtime'] <= cohort['outtime'])]
        cohort = cohort.rename(columns={'anchor_age': 'age'})

        gender_ohe = pd.get_dummies(cohort['gender'], prefix='gender', dtype=int)
        cohort = pd.concat([cohort.drop(columns=['gender']), gender_ohe], axis=1)

        final_cohort = cohort[['subject_id', 'hadm_id', 'stay_id', 'age', 'gender_F', 'gender_M',
                              'intime', 'outtime', 'deathtime', 'icu_los_hours']]
        
        print(f"Cohort selection completed. Final size: {len(final_cohort)}")
        return final_cohort
    
    except Exception as e:
        print(f"Error in cohort selection: {str(e)}")
        raise

# ===========================================================
# RR 이상치 처리 (Ventilator 고려) - 메모리 최적화
# ===========================================================
def process_rr_with_ventilator(cohort_csv, chartevents_csv, procedureevents_csv, output_parquet):
    try:
        spark = spark_manager.get_spark("RR_Preprocessing")
        
        # 코호트만 먼저 로드
        cohort = spark.read.csv(cohort_csv, header=True, inferSchema=True)\
                          .select("subject_id", "hadm_id", "stay_id").dropDuplicates()

        # RR 데이터만 필터링하여 로드 (메모리 효율성)
        rr_df = spark.read.csv(chartevents_csv, header=True, inferSchema=True)\
                 .filter(col("itemid") == 220210)\
                 .withColumn("charttime", to_timestamp("charttime"))\
                 .join(cohort, on=["subject_id","hadm_id","stay_id"], how="inner")

        # Ventilator 관련 데이터만 필터링하여 로드
        vent_itemids = [223848, 223849, 223870]
        chart_vent_df = spark.read.csv(chartevents_csv, header=True, inferSchema=True)\
                                  .filter(col("itemid").isin(vent_itemids))\
                                  .join(cohort, on=["subject_id","hadm_id","stay_id"], how="inner")\
                                  .withColumn("charttime", to_timestamp("charttime"))

        proc_vent_df = spark.read.csv(procedureevents_csv, header=True, inferSchema=True)\
                                 .filter(col("itemid").isin([225792, 225794]))\
                                 .withColumn("starttime", to_timestamp("starttime"))\
                                 .join(cohort, on=["subject_id","hadm_id","stay_id"], how="inner")

        cols_keep = ["subject_id","hadm_id","stay_id","charttime","itemid","valuenum"]
        rr_low = rr_df.filter((col("valuenum") > 0) & (col("valuenum") <= 6))
        chart_flag = chart_vent_df.select("subject_id","hadm_id","stay_id","charttime").dropDuplicates()
        proc_flag = proc_vent_df.select("subject_id","hadm_id","stay_id","starttime")\
                                 .withColumnRenamed("starttime","charttime").dropDuplicates()

        rr_keep_chart = rr_low.join(chart_flag, on=["subject_id","hadm_id","stay_id","charttime"], how="leftsemi").select(*cols_keep)
        rr_proc = rr_low.alias("rr").join(proc_flag.alias("pv"), on=["subject_id","hadm_id","stay_id"], how="inner")\
                         .filter(spark_abs(unix_timestamp("rr.charttime") - unix_timestamp("pv.charttime")) <= 3600)\
                         .select("rr.*").select(*cols_keep)

        rr_final = rr_df.filter(col("valuenum") > 6).select(*cols_keep)\
                        .union(rr_keep_chart).union(rr_proc).dropDuplicates()
        
        rr_final.write.mode("overwrite").parquet(output_parquet)
        print(f"RR processing completed. Output: {output_parquet}")
        return output_parquet
        
    except Exception as e:
        print(f"Error in RR processing: {str(e)}")
        raise

# ===========================================================
# HR 이상치 처리 - 메모리 최적화
# ===========================================================
def process_hr_cleaning(cohort_csv, chartevents_csv, output_parquet):
    try:
        spark = spark_manager.get_spark("HR_Cleaning")
        cohort = spark.read.csv(cohort_csv, header=True, inferSchema=True)\
                        .select("subject_id","hadm_id","stay_id").dropDuplicates()
        
        # HR 데이터만 필터링하여 로드
        hr_df = spark.read.csv(chartevents_csv, header=True, inferSchema=True)\
                .filter(col("itemid") == 220045)\
                .join(cohort, on=["subject_id","hadm_id","stay_id"], how="inner")\
                .filter((col("valuenum") >= 20) & (col("valuenum") <= 230))\
                .filter(col("valuenum").isNotNull())
        
        hr_df.write.mode("overwrite").parquet(output_parquet)
        print(f"HR processing completed. Output: {output_parquet}")
        return output_parquet
        
    except Exception as e:
        print(f"Error in HR processing: {str(e)}")
        raise

# ===========================================================
# Temperature 이상치 처리 - 메모리 최적화
# ===========================================================
def process_temp_cleaning(cohort_csv, chartevents_csv, output_parquet):
    try:
        spark = spark_manager.get_spark("Temp_Cleaning")
        cohort = spark.read.csv(cohort_csv, header=True, inferSchema=True)\
                        .select("stay_id").dropDuplicates()
        
        # Temperature 데이터만 필터링하여 로드
        temp_df = spark.read.csv(chartevents_csv, header=True, inferSchema=True)\
            .filter(col("itemid").isin([223762, 223761]))\
            .join(cohort, on="stay_id", how="inner")\
            .withColumn("temperature_celsius",
                when(col("itemid") == 223762, col("valuenum"))
                .when(col("itemid") == 223761, (col("valuenum") - 32) / 1.8))
        
        temp_filtered = temp_df.filter((col("temperature_celsius") >= 33.0) &
                                       (col("temperature_celsius") <= 42.0))\
                               .filter(col("temperature_celsius").isNotNull())
        
        temp_filtered.write.mode("overwrite").parquet(output_parquet)
        print(f"Temperature processing completed. Output: {output_parquet}")
        return output_parquet
        
    except Exception as e:
        print(f"Error in Temperature processing: {str(e)}")
        raise

# ===========================================================
# SpO2 이상치 처리 - 메모리 최적화
# ===========================================================
def process_spo2_cleaning(cohort_csv, chartevents_csv, output_parquet):
    try:
        spark = spark_manager.get_spark("SpO2_Cleaning")
        cohort = spark.read.csv(cohort_csv, header=True, inferSchema=True)\
                        .select("stay_id").dropDuplicates()
        
        # SpO2 데이터만 필터링하여 로드
        spo2_df = spark.read.csv(chartevents_csv, header=True, inferSchema=True)\
            .filter(col("itemid").isin([220277, 646]))\
            .join(cohort, on="stay_id", how="inner")\
            .filter((col("valuenum") >= 70) & (col("valuenum") <= 100))\
            .filter(col("valuenum").isNotNull())
        
        spo2_df.write.mode("overwrite").parquet(output_parquet)
        print(f"SpO2 processing completed. Output: {output_parquet}")
        return output_parquet
        
    except Exception as e:
        print(f"Error in SpO2 processing: {str(e)}")
        raise

# ===========================================================
# GCS 이상치 처리 (진정제 플래그 포함) - 메모리 최적화
# ===========================================================
def process_gcs_with_sedation(cohort_csv, inputevents_csv, chartevents_csv, output_parquet):
    try:
        spark = spark_manager.get_spark("GCS_Sedation")
        cohort = spark.read.csv(cohort_csv, header=True, inferSchema=True)\
                        .select("stay_id").dropDuplicates()
        
        # 진정제 투약 정보만 필터링하여 로드
        sedative_itemids = [221319, 221668, 222168, 221744, 225942, 225972, 221320, 229420, 225150, 221195, 227212]
        inputevents = spark.read.csv(inputevents_csv, header=True, inferSchema=True)\
            .filter(col("itemid").isin(sedative_itemids))\
            .join(cohort, on="stay_id", how="inner")\
            .withColumn("starttime", to_timestamp("starttime"))
        
        # GCS 관측 정보만 필터링하여 로드
        gcs_itemids = [220739, 223900, 223901]
        gcs = spark.read.csv(chartevents_csv, header=True, inferSchema=True)\
            .filter(col("itemid").isin(gcs_itemids))\
            .join(cohort, on="stay_id", how="inner")\
            .withColumn("charttime", to_timestamp("charttime"))\
            .filter(col("valuenum").isNotNull())
        
        # GCS와 진정제 투약 정보 join
        sedated_flagged = gcs.join(inputevents, on="stay_id", how="left")\
                             .withColumn("time_diff", 
                                       expr("abs(unix_timestamp(charttime) - unix_timestamp(starttime))"))\
                             .withColumn("sedated_flag",
                                       when(col("time_diff") <= 14400, 1).otherwise(0))\
                             .select("stay_id", "charttime", "itemid", "valuenum", "sedated_flag")
        
        sedated_flagged.write.mode("overwrite").parquet(output_parquet)
        print(f"GCS processing completed. Output: {output_parquet}")
        return output_parquet
        
    except Exception as e:
        print(f"Error in GCS processing: {str(e)}")
        raise

# ===========================================================
# BP 이상치 처리 (IBP 우선, NIBP fallback) - 메모리 최적화
# ===========================================================
def process_bp_cleaning(cohort_csv, chartevents_csv, output_parquet):
    try:
        spark = spark_manager.get_spark("BP_Cleaning")
        cohort = spark.read.csv(cohort_csv, header=True, inferSchema=True)\
                        .select("stay_id").dropDuplicates()
        
        # BP 관련 itemid들만 필터링하여 로드
        bp_itemids = [220050, 220051, 220052, 220179, 220180, 220181]
        chartevents_df = spark.read.csv(chartevents_csv, header=True, inferSchema=True)\
            .filter(col("itemid").isin(bp_itemids))\
            .select("stay_id","charttime","itemid","valuenum","valueuom")\
            .join(cohort, on="stay_id", how="inner")
        
        # BP 설정
        ITEMS = {
            "SBP": {
                "IBP": [220050],
                "NIBP": [220179]
            },
            "DBP": {
                "IBP": [220051], 
                "NIBP": [220180]
            },
            "MAP": {
                "IBP": [220052],
                "NIBP": [220181]
            }
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
            
            # IBP 데이터
            ibp = chartevents_df.filter(col("itemid").isin(ITEMS[comp]["IBP"]))\
                .select("stay_id", "charttime", col("valuenum").alias(value_col))\
                .withColumn("ts", unix_timestamp("charttime"))
            
            # NIBP 데이터
            nibp = chartevents_df.filter(col("itemid").isin(ITEMS[comp]["NIBP"]))\
                .select("stay_id", "charttime", col("valuenum").alias(f"nibp_{value_col}"))
            
            # IBP 유효성 검증
            ibp = ibp.withColumn("is_valid_ext", col(value_col).between(lo, hi))
            
            # 5분 rolling std
            w5 = Window.partitionBy("stay_id").orderBy("ts").rangeBetween(-FLAT_WINDOW_SEC, 0)
            ibp = ibp.withColumn(f"std5_{value_col}", stddev(value_col).over(w5))
            ibp = ibp.withColumn("flat_artifact", col(f"std5_{value_col}") < FLAT_STD_THRESH)
            ibp = ibp.withColumn("is_valid_ibp", col("is_valid_ext") & (~col("flat_artifact")))
            
            # IBP와 NIBP 병합
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
        
        # 각 컴포넌트 처리
        sbp_res = process_component("SBP")
        dbp_res = process_component("DBP") 
        map_res = process_component("MAP")
        
        # 최종 병합
        final_bp = sbp_res.join(dbp_res, ["stay_id","charttime"], "outer")\
                          .join(map_res, ["stay_id","charttime"], "outer")
        
        # MAP 보정 (필요시 계산값으로 대체)
        final_bp = final_bp.withColumn(
            "final_map",
            when(col("final_map").isNotNull(), col("final_map"))
            .otherwise((col("final_dbp")*2 + col("final_sbp"))/3.0)
        )
        
        final_bp.write.mode("overwrite").parquet(output_parquet)
        print(f"BP processing completed. Output: {output_parquet}")
        return output_parquet
        
    except Exception as e:
        print(f"Error in BP processing: {str(e)}")
        raise

# ===========================================================
# 균등 시계열 생성 (1시간 단위 bin)
# ===========================================================
def create_hourly_bins_for_vitals(cohort_path, processed_dir):
    """각 바이탈 피처에 대해 1시간 단위 bin 생성"""
    try:
        cohort = pd.read_csv(cohort_path)
        cohort['intime'] = pd.to_datetime(cohort['intime'])
        cohort['outtime'] = pd.to_datetime(cohort['outtime'])
        cohort['deathtime'] = pd.to_datetime(cohort['deathtime'])
        
        os.makedirs('./hourly_bins', exist_ok=True)
        
        # 각 피처별 처리
        vitals_config = {
            'rr': {'file': f'{processed_dir}/rr.parquet', 'value_col': 'valuenum', 'prefix': 'rr'},
            'hr': {'file': f'{processed_dir}/hr.parquet', 'value_col': 'valuenum', 'prefix': 'hr'},
            'temp': {'file': f'{processed_dir}/temp.parquet', 'value_col': 'temperature_celsius', 'prefix': 'temp'},
            'spo2': {'file': f'{processed_dir}/spo2.parquet', 'value_col': 'valuenum', 'prefix': 'spo2'},
            'gcs': {'file': f'{processed_dir}/gcs.parquet', 'value_col': 'valuenum', 'prefix': 'gcs'},
            'bp': {'file': f'{processed_dir}/bp.parquet', 'value_cols': ['final_sbp', 'final_dbp', 'final_map'], 'prefix': 'bp'}
        }
        
        for vital_name, config in vitals_config.items():
            print(f"Processing {vital_name} hourly bins...")
            
            # 파일 존재 확인
            if not os.path.exists(config['file']):
                print(f"Warning: {config['file']} not found. Skipping {vital_name}")
                continue
            
            # 데이터 로드
            vital_data = pd.read_parquet(config['file'])
            vital_data['charttime'] = pd.to_datetime(vital_data['charttime'])
            
            # 코호트와 병합
            vital_merged = vital_data.merge(
                cohort[['stay_id', 'intime', 'outtime', 'deathtime', 'subject_id', 'hadm_id', 'icu_los_hours']], 
                on='stay_id', how='inner'
            )
            
            # ICU 재원 기간 내 데이터만 필터링
            mask = (vital_merged['charttime'] >= vital_merged['intime']) & (vital_merged['charttime'] <= vital_merged['outtime'])
            vital_filtered = vital_merged[mask].copy()
            
            # 1시간 단위 bin 생성
            result_list = []
            
            for stay_id in vital_filtered['stay_id'].unique():
                patient_data = vital_filtered[vital_filtered['stay_id'] == stay_id].copy()
                patient_cohort = cohort[cohort['stay_id'] == stay_id].iloc[0]
                
                intime = patient_cohort['intime']
                outtime = patient_cohort['outtime']
                deathtime = patient_cohort['deathtime']
                icu_los_hours = patient_cohort['icu_los_hours']
                
                # bin 생성 종료 시간 결정
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
                        
                    hour_mask = (patient_data['charttime'] >= bin_start) & (patient_data['charttime'] < bin_end)
                    hour_data = patient_data[hour_mask]
                    
                    result_row = {
                        'subject_id': patient_cohort['subject_id'],
                        'hadm_id': patient_cohort['hadm_id'],
                        'stay_id': stay_id,
                        'hour_from_intime': hour,
                        'bin_start': bin_start,
                        'bin_end': bin_end
                    }
                    
                    # BP는 특별 처리 (여러 컬럼)
                    if vital_name == 'bp':
                        for col_name in config['value_cols']:
                            if len(hour_data) > 0 and col_name in hour_data.columns:
                                col_short = col_name.replace('final_', '')
                                result_row[f'{col_short}_mean'] = hour_data[col_name].mean()
                                result_row[f'{col_short}_last'] = hour_data[col_name].iloc[-1]
                                result_row[f'{col_short}_min'] = hour_data[col_name].min()
                                result_row[f'{col_short}_max'] = hour_data[col_name].max()
                            else:
                                col_short = col_name.replace('final_', '')
                                result_row[f'{col_short}_mean'] = np.nan
                                result_row[f'{col_short}_last'] = np.nan
                                result_row[f'{col_short}_min'] = np.nan
                                result_row[f'{col_short}_max'] = np.nan
                                
                    elif vital_name == 'gcs':
                        # GCS는 진정제 플래그도 포함
                        if len(hour_data) > 0:
                            result_row[f'{config["prefix"]}_mean'] = hour_data[config['value_col']].mean()
                            result_row[f'{config["prefix"]}_last'] = hour_data[config['value_col']].iloc[-1]
                            result_row[f'{config["prefix"]}_min'] = hour_data[config['value_col']].min()
                            result_row[f'{config["prefix"]}_max'] = hour_data[config['value_col']].max()
                            result_row['sedated_flag'] = 1 if (hour_data['sedated_flag'] == 1).any() else 0
                        else:
                            result_row[f'{config["prefix"]}_mean'] = np.nan
                            result_row[f'{config["prefix"]}_last'] = np.nan
                            result_row[f'{config["prefix"]}_min'] = np.nan
                            result_row[f'{config["prefix"]}_max'] = np.nan
                            result_row['sedated_flag'] = np.nan
                            
                    else:
                        # 일반 피처들
                        if len(hour_data) > 0:
                            result_row[f'{config["prefix"]}_mean'] = hour_data[config['value_col']].mean()
                            result_row[f'{config["prefix"]}_last'] = hour_data[config['value_col']].iloc[-1]
                            result_row[f'{config["prefix"]}_min'] = hour_data[config['value_col']].min()
                            result_row[f'{config["prefix"]}_max'] = hour_data[config['value_col']].max()
                        else:
                            result_row[f'{config["prefix"]}_mean'] = np.nan
                            result_row[f'{config["prefix"]}_last'] = np.nan
                            result_row[f'{config["prefix"]}_min'] = np.nan
                            result_row[f'{config["prefix"]}_max'] = np.nan
                    
                    result_list.append(result_row)
            
            # 저장
            result_df = pd.DataFrame(result_list)
            result_df.to_csv(f'./hourly_bins/{vital_name}_hourly_bins.csv', index=False)
        
        print("All hourly bins created successfully!")
        
    except Exception as e:
        print(f"Error in hourly bins creation: {str(e)}")
        raise

# ===========================================================
# 모든 바이탈 피처 병합
# ===========================================================
def merge_all_vitals():
    """모든 바이탈 피처의 hourly bin을 병합"""
    try:
        file_info = [
            ('hourly_bins/bp_hourly_bins.csv', ['sbp', 'dbp', 'map']),
            ('hourly_bins/hr_hourly_bins.csv', ['hr']),
            ('hourly_bins/temp_hourly_bins.csv', ['temp']),
            ('hourly_bins/spo2_hourly_bins.csv', ['spo2']),
            ('hourly_bins/rr_hourly_bins.csv', ['rr']),
            ('hourly_bins/gcs_hourly_bins.csv', ['gcs'])
        ]
        
        merge_keys = ['subject_id', 'hadm_id', 'stay_id', 'hour_from_intime', 'bin_start', 'bin_end']
        
        dfs = []
        for path, _ in file_info:
            if os.path.exists(path):
                df = pd.read_csv(path)
                dfs.append(df)
            else:
                print(f"Warning: {path} not found. Skipping...")
        
        if not dfs:
            raise ValueError("No hourly bin files found!")
        
        merged = reduce(lambda left, right: pd.merge(left, right, on=merge_keys, how='inner'), dfs)
        merged.to_csv('hourly_bins/all_features_hourly_bins_inner.csv', index=False)
        
        print(f"Vitals merge completed! Final shape: {merged.shape}")
        return merged
        
    except Exception as e:
        print(f"Error in vitals merging: {str(e)}")
        raise

# ===========================================================
# 슬라이딩 윈도우 생성
# ===========================================================
def create_sliding_windows():
    """슬라이딩 윈도우 생성 (관찰 18h, 예측 6h, 간격 8h)"""
    try:
        obs_window_hours = 18
        pred_window_hours = 6
        pred_interval_hours = 8
        extra_hours_after_pred = 2
        
        print("Loading data for sliding windows...")
        
        # 파일 존재 확인
        if not os.path.exists("hourly_bins/all_features_hourly_bins_inner.csv"):
            raise FileNotFoundError("all_features_hourly_bins_inner.csv not found!")
        if not os.path.exists("./data/processed/cohort.csv"):
            raise FileNotFoundError("cohort.csv not found!")
            
        data = pd.read_csv("hourly_bins/all_features_hourly_bins_inner.csv")
        cohort = pd.read_csv("./data/processed/cohort.csv")
        
        cohort['intime'] = pd.to_datetime(cohort['intime'])
        cohort['outtime'] = pd.to_datetime(cohort['outtime'])
        cohort['deathtime'] = pd.to_datetime(cohort['deathtime'])
        
        data_with_cohort = data.merge(
            cohort[['subject_id','stay_id','intime','outtime','deathtime']],
            on=['subject_id','stay_id'], how='inner'
        )
        
        exclude_cols = ['subject_id', 'stay_id', 'hour_from_intime', 'hadm_id', 'bin_start', 'bin_end']
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        data_grouped = data_with_cohort.groupby('subject_id')
        
        windows = []
        sequences = []
        new_label_count = 0
        
        for subject_id, subject_data in data_grouped:
            subject_data = subject_data.sort_values('hour_from_intime').reset_index(drop=True)
            
            stay_id = subject_data['stay_id'].iloc[0]
            intime = subject_data['intime'].iloc[0]
            deathtime = subject_data['deathtime'].iloc[0]
            max_hour = subject_data['hour_from_intime'].max()
            
            possible_starts = np.arange(0, max_hour + 1 - obs_window_hours - pred_window_hours + 1,
                                      pred_interval_hours)
            
            for window_id, obs_start in enumerate(possible_starts):
                obs_end = obs_start + obs_window_hours
                pred_start = obs_end
                pred_end = obs_end + pred_window_hours
                
                obs_start_time = intime + pd.Timedelta(hours=obs_start)
                obs_end_time = intime + pd.Timedelta(hours=obs_end)
                pred_start_time = intime + pd.Timedelta(hours=pred_start)
                pred_end_time = intime + pd.Timedelta(hours=pred_end)
                
                # 새로운 라벨링 로직
                remove_window = False
                new_label = 0
                if pd.notna(deathtime):
                    if obs_start_time <= deathtime <= obs_end_time:
                        remove_window = True
                    elif pred_start_time <= deathtime <= pred_end_time:
                        new_label = 1
                    elif pred_end_time < deathtime <= pred_end_time + pd.Timedelta(hours=extra_hours_after_pred):
                        new_label = 1
                
                if remove_window:
                    continue
                
                # 관찰 데이터 처리
                obs_mask = (subject_data['hour_from_intime'] >= obs_start) & \
                          (subject_data['hour_from_intime'] < obs_end)
                obs_window = subject_data[obs_mask]
                
                hour_range = np.arange(obs_start, obs_end)
                obs_window_indexed = obs_window.set_index('hour_from_intime')
                
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
                
                obs_completeness = 1 - (missing_values / total_values) if total_values > 0 else 0
                
                # 라벨 0인 경우에만 completeness 기준 적용
                if (new_label == 0) and (obs_completeness < 0.7):
                    continue
                
                windows.append({
                    'subject_id': subject_id,
                    'stay_id': stay_id,
                    'window_id': window_id,
                    'obs_start_hour': obs_start,
                    'obs_end_hour': obs_end,
                    'pred_start_hour': pred_start,
                    'pred_end_hour': pred_end,
                    'obs_completeness': obs_completeness,
                    'death_in_pred_window_new': new_label
                })
                sequences.append(np.array(sequence_data))
                new_label_count += new_label
        
        windows_df = pd.DataFrame(windows)
        sequences_array = np.array(sequences)
        
        windows_df.to_csv('sliding_windows_metadata.csv', index=False)
        np.save('sliding_windows_sequences.npy', sequences_array)
        
        print(f"Sliding windows created: {len(windows_df)} windows, {new_label_count} positive labels")
        return windows_df, sequences_array
        
    except Exception as e:
        print(f"Error in sliding windows creation: {str(e)}")
        raise

# ===========================================================
# 결측값 확인 및 컬럼 정리
# ===========================================================
def clean_and_process_data():
    """결측값 확인 및 불필요한 컬럼 정리"""
    try:
        print("Loading and cleaning data...")
        
        # 파일 존재 확인
        if not os.path.exists('sliding_windows_metadata.csv'):
            raise FileNotFoundError("sliding_windows_metadata.csv not found!")
        if not os.path.exists("hourly_bins/all_features_hourly_bins_inner.csv"):
            raise FileNotFoundError("all_features_hourly_bins_inner.csv not found!")
        if not os.path.exists("./data/processed/cohort.csv"):
            raise FileNotFoundError("cohort.csv not found!")
        
        metadata = pd.read_csv('sliding_windows_metadata.csv')
        original_data = pd.read_csv("hourly_bins/all_features_hourly_bins_inner.csv")
        
        exclude_cols = ['subject_id', 'stay_id', 'hour_from_intime', 'hadm_id', 'bin_start', 'bin_end']
        feature_cols = [c for c in original_data.columns if c not in exclude_cols]
        
        processed_data = original_data.copy()
        
        # 제거할 컬럼들
        cols_to_remove = ['temp_min', 'temp_max', 'gcs_min', 'gcs_max']
        existing_cols_to_remove = [col for col in cols_to_remove if col in processed_data.columns]
        
        if existing_cols_to_remove:
            processed_data = processed_data.drop(columns=existing_cols_to_remove)
            print(f"Removed columns: {existing_cols_to_remove}")
        
        # 새로운 피처 컬럼 목록
        new_exclude_cols = ['subject_id', 'stay_id', 'hour_from_intime', 'hadm_id', 'bin_start', 'bin_end']
        new_feature_cols = [c for c in processed_data.columns if c not in new_exclude_cols]
        
        # 코호트 데이터 로드
        cohort = pd.read_csv("./data/processed/cohort.csv")
        cohort['intime'] = pd.to_datetime(cohort['intime'])
        cohort['outtime'] = pd.to_datetime(cohort['outtime'])
        cohort['deathtime'] = pd.to_datetime(cohort['deathtime'])
        
        processed_data_with_cohort = processed_data.merge(
            cohort[['subject_id','stay_id','intime','outtime','deathtime']],
            on=['subject_id','stay_id'], how='inner'
        )
        
        # 새로운 시퀀스 생성
        obs_window_hours = 18
        new_sequences = []
        
        processed_grouped = processed_data_with_cohort.groupby('subject_id')
        
        for _, row in metadata.iterrows():
            subject_id = row['subject_id']
            obs_start = row['obs_start_hour']
            obs_end = row['obs_end_hour']
            
            if subject_id not in processed_grouped.groups:
                print(f"Warning: subject_id {subject_id} not found in processed data")
                # 빈 시퀀스로 채움
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
        
        # 파일 저장
        processed_data.to_csv('processed_hourly_data.csv', index=False)
        metadata.to_csv('cleaned_sliding_windows_metadata.csv', index=False)
        np.save('processed_sliding_windows_sequences.npy', new_sequences_array)
        
        with open('feature_names.txt', 'w') as f:
            for feature in new_feature_cols:
                f.write(feature + '\n')
        
        print(f"Data cleaning completed. Final sequence shape: {new_sequences_array.shape}")
        return new_sequences_array, new_feature_cols
        
    except Exception as e:
        print(f"Error in data cleaning: {str(e)}")
        raise

# ===========================================================
# 시계열 + 정적 피처 통합
# ===========================================================
def combine_temporal_static_features():
    """시계열 피처와 정적 피처(나이, 성별) 통합"""
    try:
        print("Combining temporal and static features...")
        
        # 파라미터 설정
        seq_file = 'processed_sliding_windows_sequences.npy'
        meta_file = 'cleaned_sliding_windows_metadata.csv'
        cohort_file = './data/processed/cohort.csv'
        static_cols = ['age', 'gender_M', 'gender_F']
        
        # 파일 존재 확인
        if not os.path.exists(seq_file):
            raise FileNotFoundError(f"{seq_file} not found!")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"{meta_file} not found!")
        if not os.path.exists(cohort_file):
            raise FileNotFoundError(f"{cohort_file} not found!")
        
        # 데이터 로드
        X_seq = np.load(seq_file)
        N, T, F_temporal = X_seq.shape
        
        meta_df = pd.read_csv(meta_file)
        cohort_df = pd.read_csv(cohort_file)
        
        # 정적 피처 추출
        static_df = cohort_df[['subject_id', 'stay_id'] + static_cols]
        
        # 병합
        meta_with_static = meta_df.merge(
            static_df,
            on=['subject_id', 'stay_id'],
            how='left'
        )
        
        # 정적 피처 행렬 생성
        X_static = meta_with_static[static_cols].to_numpy()
        F_static = X_static.shape[1]
        
        # 모든 시간스텝에 복사
        X_static_expanded = np.repeat(X_static[:, np.newaxis, :], T, axis=1)
        
        # 시계열 + 정적 concat
        X_combined = np.concatenate([X_seq, X_static_expanded], axis=2)
        
        # 라벨 추출
        y = meta_with_static['death_in_pred_window_new'].to_numpy()
        
        # 저장
        np.save('tcn_input_combined.npy', X_combined)
        np.save('tcn_labels.npy', y)
        meta_with_static.to_csv('tcn_metadata_with_static.csv', index=False)
        
        # 피처 이름 저장
        if os.path.exists('feature_names.txt'):
            with open('feature_names.txt', 'r') as f:
                temporal_feature_names = [line.strip() for line in f.readlines() if line.strip()]
        else:
            print("Warning: feature_names.txt not found. Using default names.")
            temporal_feature_names = [f'feature_{i}' for i in range(F_temporal)]
        
        combined_feature_names = temporal_feature_names + static_cols
        
        with open('tcn_feature_names.txt', 'w') as f:
            f.write("\n".join(combined_feature_names))
        
        print(f"Feature combination completed!")
        print(f"Final input shape: {X_combined.shape}")
        print(f"Label shape: {y.shape}")
        print(f"Total features: {len(combined_feature_names)}")
        
        return X_combined, y, combined_feature_names
        
    except Exception as e:
        print(f"Error in feature combination: {str(e)}")
        raise

# ===========================================================
# Main Pipeline Function
# ===========================================================
def main():
    """전체 파이프라인 실행"""
    try:
        raw_bucket = "s3://mimic4-project/raw-data"
        processed_bucket = "s3://mimic4-project/processed-data"
        local_raw_dir = "./data/raw"
        local_processed_dir = "./data/processed"
        
        raw_files = {
            "patients": "hosp/patients.csv.gz",
            "admissions": "hosp/admissions.csv.gz",
            "icustays": "icu/icustays.csv.gz",
            "chartevents": "icu/chartevents.csv.gz",
            "inputevents": "icu/inputevents.csv.gz",
            "procedureevents": "icu/procedureevents.csv.gz"
        }
        
        # 1. S3에서 raw 데이터 다운로드
        print("=== Step 1: Downloading raw data from S3 ===")
        for name, key in raw_files.items():
            bucket_name = raw_bucket.replace('s3://', '').split('/')[0]
            s3_key = '/'.join(raw_bucket.replace('s3://', '').split('/')[1:] + [key])
            local_path = f"{local_raw_dir}/{os.path.basename(key)}"
            download_from_s3(bucket_name, s3_key, local_path)
            raw_files[name] = local_path
        
        # 2. Cohort 선정
        print("=== Step 2: Selecting cohort ===")
        cohort_df = select_cohort(raw_files["patients"], raw_files["admissions"], raw_files["icustays"])
        os.makedirs(local_processed_dir, exist_ok=True)
        cohort_df.to_csv(f"{local_processed_dir}/cohort.csv", index=False)
        print(f"Cohort size: {len(cohort_df)}")
        
        # 3. 이상치 처리 (Spark 기반)
        print("=== Step 3: Processing outliers with Spark ===")
        process_rr_with_ventilator(
            f"{local_processed_dir}/cohort.csv", 
            raw_files["chartevents"], 
            raw_files["procedureevents"], 
            f"{local_processed_dir}/rr.parquet"
        )
        print("RR outlier processing completed")
        
        process_hr_cleaning(
            f"{local_processed_dir}/cohort.csv", 
            raw_files["chartevents"], 
            f"{local_processed_dir}/hr.parquet"
        )
        print("HR outlier processing completed")
        
        process_temp_cleaning(
            f"{local_processed_dir}/cohort.csv", 
            raw_files["chartevents"], 
            f"{local_processed_dir}/temp.parquet"
        )
        print("Temperature outlier processing completed")
        
        process_spo2_cleaning(
            f"{local_processed_dir}/cohort.csv", 
            raw_files["chartevents"], 
            f"{local_processed_dir}/spo2.parquet"
        )
        print("SpO2 outlier processing completed")
        
        process_gcs_with_sedation(
            f"{local_processed_dir}/cohort.csv", 
            raw_files["inputevents"], 
            raw_files["chartevents"], 
            f"{local_processed_dir}/gcs.parquet"
        )
        print("GCS outlier processing completed")
        
        process_bp_cleaning(
            f"{local_processed_dir}/cohort.csv", 
            raw_files["chartevents"], 
            f"{local_processed_dir}/bp.parquet"
        )
        print("BP outlier processing completed")
        
        # 4. 균등 시계열 생성 (1시간 단위 bin)
        print("=== Step 4: Creating hourly bins ===")
        create_hourly_bins_for_vitals(f"{local_processed_dir}/cohort.csv", local_processed_dir)
        
        # 5. 모든 바이탈 피처 병합
        print("=== Step 5: Merging all vitals ===")
        merge_all_vitals()
        
        # 6. 슬라이딩 윈도우 생성
        print("=== Step 6: Creating sliding windows ===")
        create_sliding_windows()
        
        # 7. 결측값 확인 및 컬럼 정리
        print("=== Step 7: Cleaning and processing data ===")
        clean_and_process_data()
        
        # 8. 시계열 + 정적 피처 통합
        print("=== Step 8: Combining temporal and static features ===")
        X_combined, y, feature_names = combine_temporal_static_features()
        
        # 9. 최종 결과 S3 업로드
        print("=== Step 9: Uploading final results to S3 ===")
        final_files = [
            'tcn_input_combined.npy',
            'tcn_labels.npy', 
            'tcn_metadata_with_static.csv',
            'tcn_feature_names.txt'
        ]
        
        bucket_name = processed_bucket.replace('s3://', '').split('/')[0]
        s3_prefix = '/'.join(processed_bucket.replace('s3://', '').split('/')[1:])
        
        for file_name in final_files:
            if os.path.exists(file_name):
                s3_key = f"{s3_prefix}/final_dataset/{file_name}"
                upload_to_s3(file_name, bucket_name, s3_key)
        
        print("=== Pipeline completed successfully! ===")
        print(f"Final dataset shape: {X_combined.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Number of features: {len(feature_names)}")
        print(f"Positive label ratio: {y.mean():.4f}")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise
    finally:
        # Spark 세션 정리
        spark_manager.stop()

if __name__ == "__main__":
    main()