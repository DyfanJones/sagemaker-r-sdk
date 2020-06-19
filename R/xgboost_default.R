# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/xgboost/defaults.py

XGBOOST_NAME = "xgboost"
XGBOOST_1P_VERSIONS = c("1", "latest")
XGBOOST_VERSION_0_90 = "0.90"
XGBOOST_VERSION_0_90_1 = "0.90-1"
XGBOOST_VERSION_0_90_2 = "0.90-2"
XGBOOST_LATEST_VERSION = "1.0-1"
# XGBOOST_SUPPORTED_VERSIONS has XGBoost Framework versions sorted from oldest to latest
XGBOOST_SUPPORTED_VERSIONS = list(
  XGBOOST_VERSION_0_90_1,
  XGBOOST_VERSION_0_90_2,
  XGBOOST_LATEST_VERSION
)
XGBOOST_VERSION_EQUIVALENTS = "-cpu-py3"
