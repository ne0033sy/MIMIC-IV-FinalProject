# MIMIC-IV-Preprocessing
 SKKU x AWS Project Preprocessing pipeline
 MIMIC-IV-Project

## 개요


MIMIC-IV 데이터를 활용하여 3가지 모델을 사용해 사망률 예측 및 의료진 대시보드 개발을 목적으로 하는 AWS SAY team 7 pre-final project입니다.




## MIMIC-IV 데이터 전처리 파이프라인


메모리 최적화된 MIMIC-IV 임상 데이터 전처리 파이프라인으로, ICU 환자의 사망 예측 모델을 위한 시계열 데이터를 생성합니다.




## 📦 설치
### 1. 환경 설정
```bash
# 레포지토리 클론
git clone <repository-url>
cd MIMIC-IV-Preprocessing

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 시스템 요구사항
- **Python**: 3.8+
- **Java**: 8+ (Spark 요구사항)
- **메모리**: 최소 16GB RAM (32GB 권장)
- **저장공간**: 100GB+ 여유 공간

### 3. AWS 설정 (선택사항)
```bash
# AWS CLI 설정
aws configure

# 또는 환경 설정
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

## 🏗️ 파이프라인 구조

파이프라인은 8단계로 구성되며, 각 단계별로 메모리 최적화가 적용됩니다:

1. **S3 다운로드**: 원본 MIMIC-IV 데이터 다운로드
2. **코호트 선택**: 성인 환자, 유효 ICU 체류 기간 필터링
3. **바이탈 처리**: 임상적 특성을 반영한 바이탈 사인 전처리
4. **시간별 바이닝**: 1시간 단위 통계값 계산
5. **데이터 병합**: 모든 바이탈을 하나의 테이블로 통합
6. **윈도우 생성**: 머신러닝용 슬라이딩 윈도우 생성
7. **특성 최종화**: 정적/동적 특성 결합
8. **S3 업로드**: 최종 결과물 저장

## ⚙️ 설정 (config.yaml)

첫 실행 시 기본 설정 파일이 자동 생성됩니다:

```yaml
aws:
  raw_bucket: s3://mimic4-project/raw-data
  processed_bucket: s3://mimic4-project/processed-data

paths:
  local_raw_dir: ./data/raw
  local_processed_dir: ./data/processed
  hourly_bins_dir: ./data/hourly_bins
  output_dir: ./data/output

processing:
  obs_window_hours: 18          # 관찰 윈도우 (시간)
  pred_window_hours: 6          # 예측 윈도우 (시간)
  pred_interval_hours: 8        # 예측 간격 (시간)
  min_obs_completeness: 0.7     # 최소 관찰 완성도
  min_icu_los_hours: 24         # 최소 ICU 체류시간
  max_icu_los_hours: 720        # 최대 ICU 체류시간
  chunk_size: 50000             # 메모리 청킹 크기

vitals_config:
  rr: {itemids: [220210], min_val: 0, max_val: 50}
  hr: {itemids: , min_val: 20, max_val: 230}
  # ... 기타 바이탈 설정

logging:
  level: INFO
  file: complete_pipeline.log
```

### 사용자 정의 설정
```bash
# 사용자 정의 설정 파일로 실행
python complete_pipeline.py --config custom_config.yaml 
```

## 🔧 사용법
### **전체 파이프라인 실행**
```bash
python complete_pipeline.py
```

### **단계별 실행**
```bash
# 1. 코호트 선택
python complete_pipeline.py cohort

# 2. 바이탈 사인 처리
python complete_pipeline.py vitals

# 3. 시간별 바이닝 
python complete_pipeline.py bins

# 4. 데이터 병합 
python complete_pipeline.py merge

# 5. 슬라이딩 윈도우 생성
python complete_pipeline.py windows 
```

## 📁 주요 출력 파일최종 모델 입력 데이터:
- **tcn_input_combined.npy**: `(N, T, F)` 형태의 3D 배열
- **tcn_labels.npy**: 사망 라벨 (`(N,)` 형태)
- **tcn_metadata_with_static.csv**: 윈도우 메타데이터
- **tcn_feature_names.txt**: 특성명 리스트

여기서:
- `N`: 윈도우 개수
- `T`: 시간 스텝 (기본 18시간)
- `F`: 특성 개수 (바이탈 + 인구학적 정보)

## 🔍 성능 권장사항
| 환경 | RAM | 청킹 크기 | Spark 메모리 |
|------|-----|----------|-------------|
| 개발 | 16GB | 25,000 | 6g |
| 서버 | 32GB+ | 50,000 | 12g |


## 파이프라인 요약

1. **데이터 다운로드 & 폴더 준비**
    - S3 버킷에서 원본 MIMIC-IV 테이블 자동 다운로드
    - 폴더/결과 파일 자동 생성
2. **코호트 선정**
    - 성인(18세+) ICU 재원 24~720시간 환자 선별
    - ICU 내 사망자 및 생존자 구분 (ICU 퇴원 이후 사망자는 제외)
    - 성별(원-핫), 나이 등 정적 피처 생성
3. **이상치 및 임상적 artifact 처리 (Spark 기반)**
    - **RR**: ventilator 사용 고려, 0~6 bpm 허용(ventilator check)
    - **HR**: 20~230 bpm 허용
    - **Temp**: 화씨→섭씨 변환, 33~42℃
    - **SpO₂**: 70~100% 허용
    - **GCS**: 진정제 투여 플래그 포함
    - **BP**: IBP 우선, flat-line artifact 제거, fallback NIBP, extreme outlier 제거
4. **균등 시계열(bin) 생성**
    - 모든 환자/바이탈별 1시간 단위로 mean, last, min, max 집계
    - ICU 입실~퇴실(혹은 사망) 구간에 대해 피처 생성
5. **피처 통합 및 결합**
    - 모든 바이탈 피처 데이터 병합(내부 join)
    - 불필요 컬럼 자동 제거, 결측값 관리
6. **슬라이딩 윈도우 생성 및 라벨링**
    - 관찰 윈도우 18시간, 예측 윈도우 6시간, 시작 간격 8시간
    - 사망 라벨: 예측구간 및 종료 2시간 이내 사망=1, 관찰구간 사망=window 제외
    - 관찰 completeness 70% 미만 window 자동 제외
7. **정적/시계열 피처 융합과 결과 저장**
    - 시계열+나이+성별 concatenate (모델학습용 shape)
    - numpy, csv, txt 등 다양한 실전 활용 결과 자동 저장
8. **최종 데이터 S3 업로드**
    - 모델·분석 파이프라인과 연동 가능

***

**주의**: MIMIC-IV 데이터베이스 접근 권한 필요. 의료 데이터 보안 규정 준수 필수.

