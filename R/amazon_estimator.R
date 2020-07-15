# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/af7f75ae336f0481e52bb968e4cc6df91b1bac2c/src/sagemaker/amazon/amazon_estimator.py


#' @include utils.R
#' @include xgboost_estimator.R
#' @include fw_registry.R



# Return docker registry for the given AWS region
# Note: Not all the algorithms listed below have an Amazon Estimator
# implemented. For full list of pre-implemented Estimators, look at:
#   https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/amazon
# Args:
#   region_name (str): The region name for the account.
# algorithm (str): The algorithm for the account.
# Raises:
#   ValueError: If invalid algorithm passed in or if mapping does not exist for given algorithm
# and region.
registry <- function(region_name,
                     algorithm=NULL){
  region_to_accounts = list()
  if (is.null(algorithm)
      || algorithm %in% c("pca",
                       "kmeans",
                       "linear-learner",
                       "factorization-machines",
                       "ntm",
                       "randomcutforest",
                       "knn",
                       "object2vec",
                       "ipinsights")){
    region_to_accounts = list(
      "us-east-1"= "382416733822",
      "us-east-2"= "404615174143",
      "us-west-2"= "174872318107",
      "eu-west-1"= "438346466558",
      "eu-central-1"= "664544806723",
      "ap-northeast-1"= "351501993468",
      "ap-northeast-2"= "835164637446",
      "ap-southeast-2"= "712309505854",
      "us-gov-west-1"= "226302683700",
      "ap-southeast-1"= "475088953585",
      "ap-south-1"= "991648021394",
      "ca-central-1"= "469771592824",
      "eu-west-2"= "644912444149",
      "us-west-1"= "632365934929",
      "us-iso-east-1"= "490574956308",
      "ap-east-1"= "286214385809",
      "eu-north-1"= "669576153137",
      "eu-west-3"= "749696950732",
      "sa-east-1"= "855470959533",
      "me-south-1"= "249704162688",
      "cn-north-1"= "390948362332",
      "cn-northwest-1"= "387376663083")
  } else if (algorithm %in% c("lda")){
    region_to_accounts = list(
      "us-east-1"= "766337827248",
      "us-east-2"= "999911452149",
      "us-west-2"= "266724342769",
      "eu-west-1"= "999678624901",
      "eu-central-1"= "353608530281",
      "ap-northeast-1"= "258307448986",
      "ap-northeast-2"= "293181348795",
      "ap-southeast-2"= "297031611018",
      "us-gov-west-1"= "226302683700",
      "ap-southeast-1"= "475088953585",
      "ap-south-1"= "991648021394",
      "ca-central-1"= "469771592824",
      "eu-west-2"= "644912444149",
      "us-west-1"= "632365934929",
      "us-iso-east-1"= "490574956308")
  } else if (algorithm %in% c("forecasting-deepar")){
    region_to_accounts = list(
      "us-east-1"= "522234722520",
      "us-east-2"= "566113047672",
      "us-west-2"= "156387875391",
      "eu-west-1"= "224300973850",
      "eu-central-1"= "495149712605",
      "ap-northeast-1"= "633353088612",
      "ap-northeast-2"= "204372634319",
      "ap-southeast-2"= "514117268639",
      "us-gov-west-1"= "226302683700",
      "ap-southeast-1"= "475088953585",
      "ap-south-1"= "991648021394",
      "ca-central-1"= "469771592824",
      "eu-west-2"= "644912444149",
      "us-west-1"= "632365934929",
      "us-iso-east-1"= "490574956308",
      "ap-east-1"= "286214385809",
      "eu-north-1"= "669576153137",
      "eu-west-3"= "749696950732",
      "sa-east-1"= "855470959533",
      "me-south-1"= "249704162688",
      "cn-north-1"= "390948362332",
      "cn-northwest-1"= "387376663083")
  } else if (algorithm %in% c("xgboost",
                            "seq2seq",
                            "image-classification",
                            "blazingtext",
                            "object-detection",
                            "semantic-segmentation")){
    region_to_accounts = list(
      "us-east-1"= "811284229777",
      "us-east-2"= "825641698319",
      "us-west-2"= "433757028032",
      "eu-west-1"= "685385470294",
      "eu-central-1"= "813361260812",
      "ap-northeast-1"= "501404015308",
      "ap-northeast-2"= "306986355934",
      "ap-southeast-2"= "544295431143",
      "us-gov-west-1"= "226302683700",
      "ap-southeast-1"= "475088953585",
      "ap-south-1"= "991648021394",
      "ca-central-1"= "469771592824",
      "eu-west-2"= "644912444149",
      "us-west-1"= "632365934929",
      "us-iso-east-1"= "490574956308",
      "ap-east-1"= "286214385809",
      "eu-north-1"= "669576153137",
      "eu-west-3"= "749696950732",
      "sa-east-1"= "855470959533",
      "me-south-1"= "249704162688",
      "cn-north-1"= "390948362332",
      "cn-northwest-1"= "387376663083")
  } else if (algorithm %in% c("image-classification-neo", "xgboost-neo")){
    region_to_accounts = NEO_IMAGE_ACCOUNT
  } else {
    stop(sprintf("Algorithm class:%s does not have mapping to account_id with images",algorithm), call.=F)
  }

  if (region_name %in% names(region_to_accounts)){
    account_id = region_to_accounts[[region_name]]
    return (get_ecr_image_uri_prefix(account_id, region_name))
  }

  stop(sprintf("Algorithm (%s) is unsupported for region (%s).", algorithm, region_name), call. = F)
}


#' Return algorithm image URI for the given AWS region, repository name, and
#' repository version
#' @param region_name (str): The region name for the account.
#' @param repo_name (str): The repo name for the account
#' @param repo_version (str): Version fo repo to call
#' @export
get_image_uri <- function(region_name, repo_name, repo_version = "1.0-1"){
  stopifnot(is.character(region_name), is.character(repo_name), is.character(repo_version))
  log_warn(
    "'get_image_uri' method will be deprecated in favor of 'ImageURIProvider' class to align with SageMaker Python SDK v2."
  )

  if(repo_name == XGBOOST_NAME){
    if(repo_version %in% XGBOOST_1P_VERSIONS){
      .warn_newer_xgboost_image()
      return(sprintf("%s/%s:%s", registry(region_name, repo_name), repo_name, repo_version))
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
  return (sprintf("%s/%s",registry(region_name, repo_name), repo))
}

#' Return algorithm image URI for the given ecr repository
#' @description Decided to help R users integrate "bring your own R models" in sagemaker
#' @param region_name (str): The region name for the account.
#' @param repo_version (str): Version fo repo to call
#' @param sagemaker_session (sagemaker.session.Session): Session object which
#'              manages interactions with Amazon SageMaker APIs and any other
#'              AWS services needed. If not specified, the estimator creates one
#'              using the default AWS configuration chain.
#' @export
get_ecr_image_uri = function(repo_name, repo_version = NULL, sagemaker_session = NULL){
  stopifnot(is.character(repo_name),
            is.character(repo_version) || is.null(repo_version),
            is.null(sagemaker_session) || inherits(sagemaker_session, "session"))

  session = sagemaker_session %||% Session$new()

  ecr = paws::ecr(config = session$paws_credentials$credentials)

  nextToken = NULL
  repos = list()
  # get list of repositories in ecr
  while (!identical(nextToken, character(0))){
    repo_chunk = ecr$describe_repositories()
    repo_dt = rbindlist(repo_chunk$repositories)
    repos = list(repos, repo_dt)
    nextToken = repo_chunk$nextToken
  }
  repos = rbindlist(repos)

  # check if repo_name exists in registered ecr repositories
  if(nrow(repos[repositoryName == repo_name]) == 0)
    stop(sprintf("Custom repository %s doesn't exist in AWS ECR", repo_name))

  # after repo_name check only use repo_name
  repos = repos[repositoryName == repo_name]

  nextToken = NULL
  image_meta = list()
  # get all tags from repository
  while(!identical(nextToken, character(0))){
    image_chunk = ecr$describe_images(repos[, registryId], repos[, repositoryName])
    nextToken = image_chunk$nextToken
    image_chunk = lapply(image_chunk$imageDetails, function(x)
      data.table(imageTags = x$imageTags, imagePushedAt= x$imagePushedAt))
    image_meta = c(image_meta, image_chunk)
  }
  image_meta = rbindlist(image_meta)

  # check if repo_version matches existing tags
  if(!is.null(repo_version) && nrow(image_meta[imageTags == repo_version]) == 0)
    stop(sprintf("Repository version %s doesn't exist", repo_version))

  if(is.null(repo_version)) repo_version = image_meta[order(-imagePushedAt)][1,imageTags]

  file.path(repos$repositoryUri, repo_version)
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
                   "\tget_image_uri(region, '%s', '%s')."),XGBOOST_LATEST_VERSION,
                   XGBOOST_NAME, XGBOOST_LATEST_VERSION))
}

.generate_version_equivalents <- function(version){
  # Returns a list of version equivalents for XGBoost
  lapply(XGBOOST_VERSION_EQUIVALENTS, function(suffix) c(paste0(version, suffix), version))
}

