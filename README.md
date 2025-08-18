# MIMIC-IV-Preprocessing
 SKKU x AWS Project Preprocessing pipeline
 MIMIC-IV-Project

## 개요

MIMIC-IV 데이터를 활용하여 3가지 모델을 사용해 사망률 예측 및 의료진 대시보드 개발을 목적으로 하는 AWS SAY team 7 pre-final porject입니다.


### 개발 환경

python 3.10.11

numpy, pandas, matplotlib, seaborn

darts, torch, scikit-learn

aws: boto3, sagemaker, awscli, jupyterlab

### 환경 구축하는 법
Vscode에서 .venv 만들기.

pip install pandas numpy boto3 scikit-learn pyarrow pyspark

### 사용하는 모델
Transformer Model

TCN(CNN)


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
