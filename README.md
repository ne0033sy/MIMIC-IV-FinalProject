# MIMIC-IV ICU 실시간 사망률 예측 프로젝트
 SKKU x AWS Project 
 MIMIC-IV-Project

## 개요


MIMIC-IV 데이터를 활용하여 3가지 모델을 사용해 사망률 예측 및 의료진 대시보드 개발을 목적으로 하는 AWS SAY team 7 final project입니다.


## 파일 구조


### data_preprocessing/

데이터 전처리 관련 파일들이 포함된 폴더입니다.

local_pipeline.py: 로컬 환경에서 데이터 전처리 파이프라인을 실행하는 파일

s3_pipeline.py: AWS S3와 연동하여 데이터 전처리 파이프라인을 실행하는 파일

### models/

모델 관련 파일이 포함된 폴더입니다.

say1-7team-model.py: 모델 훈련, 성능 확인, 예측 결과를 한번에 도출하는 통합 파일



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