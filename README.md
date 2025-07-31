# MIMIC-IV-Preprocessing
## SKKU x AWS Project Preprocessing pipeline
## MIMIC-IV-Project

## 개요

MIMIC-IV 데이터를 활용하여 3가지 모델을 사용해 사망률 예측 및 의료진 대시보드 개발을 목적으로 하는 AWS SAY team 7 pre-final porject입니다.

## 개발 환경

python 3.10.11
numpy, pandas, matplotlib, seaborn
darts, torch, scikit-learn

aws: boto3, sagemaker, awscli, jupyterlab

## 환경 구축하는 법
Vscode에서 .venv 만들기.
pip install darts pyspark seaborn ipykernel
pip install sagemaker boto3 awscli jupyterlab
pip install lime shap

## 사용하는 모델
Transformer Model
TCN(CNN)
