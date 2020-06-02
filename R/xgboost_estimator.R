# get XGBoost framework image URI

#' @export
get_xgboost_image_uri <- function(region, framework_version, py_version = "py3"){
  image_tag <- sprintf("%s-%s-%s",framework_version, "cpu", py_version)
  default_framework_uri(XGBOOST_NAME, region, image_tag)
}

