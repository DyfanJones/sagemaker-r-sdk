# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/xgboost/estimator.py


# get XGBoost framework image URI

#' @include fw_registery.R

#' @title Get XGBoost framework image URI
#' @param region (str): The aws region in which docker image is stored.
#' @param frame_version (str):
#' @param py_version (str): python version
#' @export
get_xgboost_image_uri <- function(region, framework_version, py_version = "py3"){
  image_tag <- sprintf("%s-%s-%s",framework_version, "cpu", py_version)
  fw_default_framework_uri(XGBOOST_NAME, region, image_tag)
}

