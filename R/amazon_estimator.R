

#' @export
get_image_uri <- function(region_name, repo_name, repo_version = "1.0-1"){
  log_warn(
    "'get_image_uri' method will be deprecated in favor of 'ImageURIProvider' class to align with SageMaker Python SDK v2."
  )

  if(repo_name == XGBOOST_NAME){
    if(repo_version %in% XGBOOST_1P_VERSIONS){
      .warn_newer_xgboost_image()
      return(sprintf("%s/%s:%s", region_name, repo_name, repo_version))
    }

    if(!grepl("-", repo_version)) {
      xgboost_version_matches = lapply(XGBOOST_SUPPORTED_VERSIONS, function(version) split_str(version, split = "-"))
      xgboost_version_matches = xgboost_version_matches[sapply(xgboost_version_matches, function(x) x[1] == repo_version)]

      if(length(xgboost_version_matches) > 0) {
        # Assumes that XGBOOST_SUPPORTED_VERSION is sorted from oldest version to latest.
        # When SageMaker version is not specified, use the oldest one that matches
        # XGBoost version for backward compatibility.
        repo_version = xgboost_version_matches[1]}
    }

    supported_framework_versions = sapply(XGBOOST_SUPPORTED_VERSIONS, .generate_version_equivalents)
    supported_framework_versions = supported_framework_versions[sapply(supported_framework_versions, function(x) repo_version %in% x)]

    if (length(supported_framework_versions) == 0){
      stop(sprintf("SageMaker XGBoost version %s is not supported. Supported versions: %s",
                   repo_version, paste(XGBOOST_SUPPORTED_VERSIONS, collapse = ", ")), call. = F)
    }

    if (!.is_latest_xgboost_version(repo_version)) .warn_newer_xgboost_image()

    return(get_xgboost_image_uri(region_name, unlist(supported_framework_versions)[length(unlist(supported_framework_versions))]))
  }

  repo = sprintf("%s:%s", repo_name, repo_version)
  return (sprintf("%s/%s",registry(region_name), repo))
}

.is_latest_xgboost_version <- function(repo_version){
  # Compare xgboost image version with latest version
  if(repo_version %in% XGBOOST_1P_VERSIONS) return(FALSE)
  return(repo_version %in% unlist(.generate_version_equivalents(XGBOOST_LATEST_VERSION)))
}

.warn_newer_xgboost_image <- function(){
  log_warn(sprintf(paste0("There is a more up to date SageMaker XGBoost image. ",
                   "To use the newer image, please set 'repo_version'=",
                   "'%s'.\nFor example:",
                   "\tget_image_uri(region, '%s', '%s')."),XGBOOST_LATEST_VERSION , XGBOOST_NAME, XGBOOST_LATEST_VERSION))
}

.generate_version_equivalents <- function(version){
  # Returns a list of version equivalents for XGBoost
  lapply(XGBOOST_VERSION_EQUIVALENTS, function(suffix) c(paste0(version, suffix), version))
}

