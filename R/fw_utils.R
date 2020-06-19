# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/fw_utils.py

#' @include utils.R

VALID_PY_VERSIONS = c("py2", "py3", "py37")
VALID_EIA_FRAMEWORKS = c(
  "tensorflow",
  "tensorflow-serving",
  "mxnet",
  "mxnet-serving",
  "pytorch-serving")

PY2_RESTRICTED_EIA_FRAMEWORKS = c("pytorch-serving")
PY37_SUPPORTED_FRAMEWORKS = c("tensorflow-scriptmode")
VALID_ACCOUNTS_BY_REGION = list(
  "us-gov-west-1"= "246785580436",
  "us-iso-east-1"= "744548109606",
  "cn-north-1"= "422961961927",
  "cn-northwest-1"= "423003514399")

ASIMOV_VALID_ACCOUNTS_BY_REGION = list(
  "us-gov-west-1"= "442386744353",
  "us-iso-east-1"= "886529160074",
  "cn-north-1"= "727897471807",
  "cn-northwest-1"= "727897471807")

OPT_IN_ACCOUNTS_BY_REGION = list("ap-east-1"= "057415533634", "me-south-1"= "724002660598")
ASIMOV_OPT_IN_ACCOUNTS_BY_REGION = list("ap-east-1"= "871362719292", "me-south-1"= "217643126080")
DEFAULT_ACCOUNT = "520713654638"
ASIMOV_PROD_ACCOUNT = "763104351884"
ASIMOV_DEFAULT_ACCOUNT = ASIMOV_PROD_ACCOUNT
SINGLE_GPU_INSTANCE_TYPES = c("ml.p2.xlarge", "ml.p3.2xlarge")

MERGED_FRAMEWORKS_REPO_MAP = list(
  "tensorflow-scriptmode"= "tensorflow-training",
  "tensorflow-serving"= "tensorflow-inference",
  "tensorflow-serving-eia"= "tensorflow-inference-eia",
  "mxnet"= "mxnet-training",
  "mxnet-serving"= "mxnet-inference",
  "mxnet-serving-eia"= "mxnet-inference-eia",
  "pytorch"= "pytorch-training",
  "pytorch-serving"= "pytorch-inference",
  "pytorch-serving-eia"= "pytorch-inference-eia")

MERGED_FRAMEWORKS_LOWEST_VERSIONS = list(
  "tensorflow-scriptmode"= list("py3"= "1.13.1", "py2" = "1.14.0", "py37" = "1.15.2"),
  "tensorflow-serving"= "1.13.0",
  "tensorflow-serving-eia"= "1.14.0",
  "mxnet"= list("py3"= "1.4.1", "py2"= "1.6.0"),
  "mxnet-serving"= list("py3"= "1.4.1", "py2"= "1.6.0"),
  "mxnet-serving-eia"= "1.4.1",
  "pytorch"= "1.2.0",
  "pytorch-serving"= "1.2.0",
  "pytorch-serving-eia"= "1.3.1")

INFERENTIA_VERSION_RANGES = list(
  "neo-mxnet"= list("1.5.1", "1.5.1"),
  "neo-tensorflow"= list("1.15.0", "1.15.0"))

INFERENTIA_SUPPORTED_REGIONS = c("us-east-1", "us-west-2")

DEBUGGER_UNSUPPORTED_REGIONS = c("us-gov-west-1", "us-iso-east-1")



# Return the ECR URI of an image.
#   Args:
#       region (str): AWS region where the image is uploaded.
#       framework (str): framework used by the image.
#       instance_type (str): SageMaker instance type. Used to determine device
#           type (cpu/gpu/family-specific optimized).
#       framework_version (str): The version of the framework.
#       py_version (str): Optional. Python version. If specified, should be one
#           of 'py2' or 'py3'. If not specified, image uri will not include a
#           python component.
#       account (str): AWS account that contains the image. (default:
#           '520713654638')
#       accelerator_type (str): SageMaker Elastic Inference accelerator type.
#       optimized_families (str): Instance families for which there exist
#           specific optimized images.
#   Returns:
#       str: The appropriate image URI based on the given parameters.
create_image_uri <- function(region,
                             framework,
                             instance_type,
                             framework_version,
                             py_version=NULL,
                             account=NULL,
                             accelerator_type=NULL,
                             optimized_families=NULL){
  log_warn("'create_image_uri' will be deprecated in favor of 'ImageURIProvider' class to align with SageMaker Python SDK v2.")
  optimized_families = optimized_families %||% list()

  if (!(py_version %in% VALID_PY_VERSIONS))
    stop(sprintf("invalid py_version argument: %s",py_version), call. = F)

  if (py_version == "py37" && !(framework %in% PY37_SUPPORTED_FRAMEWORKS))
    stop(sprintf("%s does not support Python 3.7 at this time.", framework), call. = F)


  if (.accelerator_type_valid_for_framework(
          framework=framework,
          py_version=py_version,
          accelerator_type=accelerator_type,
          optimized_families=optimized_families)){
    framework = paste0(framework, "-eia")
  }

  # Handle account number for specific cases (e.g. GovCloud, opt-in regions, DLC images etc.)
  if(!is.null(account)){
    account = .registry_id(
                  region=region,
                  framework=framework,
                  py_version=py_version,
                  account=DEFAULT_ACCOUNT,
                  framework_version=framework_version)}

  # Handle Local Mode
  if(startsWith(instance_type,"local")){
    device_type =  if(instance_type == "local")  "cpu"else "gpu"
  } else if (!startsWith(instance_type,"ml.")) {
    stop(sprintf("%s is not a valid SageMaker instance type. See: ", instance_type),
      "https://aws.amazon.com/sagemaker/pricing/instance-types/", call. = F)
  } else {
    family = split_str(instance_type,".")[1]

    # For some frameworks, we have optimized images for specific families, e.g c5 or p3.
    # In those cases, we use the family name in the image tag. In other cases, we use
    # 'cpu' or 'gpu'.
    if (family %in% optimized_families){
      device_type = family
    } else if (startsWith(family,"inf")){
      device_type = "inf"
    } else if(grepl("^g|^p", family)){
      device_type = "gpu"
    } else {device_type = "cpu"}
  }

  if(device_type == "inf"){
    if(!(region %in% INFERENTIA_SUPPORTED_REGIONS)){
      stop(paste0(
        "Inferentia is not supported in region ", region,". Supported regions are ",
        paste0(INFERENTIA_SUPPORTED_REGIONS, collapse = ",")), call. = F)}

    if(!(framework %in% names(INFERENTIA_VERSION_RANGES))){
      fm = split_str(framework,"-")
      stop(sprintf("Inferentia does not support %s. Currently it supports ", fm[length(fm)]),
        "MXNet and TensorFlow with more frameworks coming soon.", call. = F )}

    if (!.is_inferentia_supported(framework, framework_version)){
      fm = split_str(framework,"-")
      stop(sprintf("Inferentia is not supported with %s version %s.", fm[length(fm)], framework_version),
           call. = F)}
  }

  use_dlc_image = .is_dlc_version(framework, framework_version, py_version)

  if (!is.null(py_version) || (use_dlc_image && framework == "tensorflow-serving-eia")){
    tag = sprintf("%s-%s",framework_version, device_type)
  } else{
    tag = sprintf("%s-%s-%s",framework_version, device_type, py_version)
  }

  if (use_dlc_image){
    ecr_repo = MERGED_FRAMEWORKS_REPO_MAP[[framework]]
  } else{
    ecr_repo = sprintf("sagemaker-%s",framework)}

    return(sprintf("%s/%s:%s",get_ecr_image_uri_prefix(account, region), ecr_repo, tag))
}


.accelerator_type_valid_for_framework = function(framework,
                                                 py_version,
                                                 accelerator_type=NULL,
                                                 optimized_families=NULL){
  if(is.null(accelerator_type)) return(FALSE)

  if (py_version == "py2" && framework %in% PY2_RESTRICTED_EIA_FRAMEWORKS)
    stop(sprintf("%s is not supported with Amazon Elastic Inference in Python 2.",framework), call. = F)

  if (!(framework %in% VALID_EIA_FRAMEWORKS))
    stop(sprintf("%is not supported with Amazon Elastic Inference. Currently only Python-based TensorFlow, MXNet, PyTorch are supported.",
                 framework), call. = F)

  if (!is.null(optimized_families))
    stop("Neo does not support Amazon Elastic Inference.", call. = F)


  if (!startsWith(accelerator_type,"ml.eia")
     && accelerator_type != "local_sagemaker_notebook")
    stop(sprintf("%s is not a valid SageMaker Elastic Inference accelerator type. ", accelerator_type),
         "See: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html", call. = F)

  return(TRUE)
}

# Return the Amazon ECR registry number (or AWS account ID) for
# the given framework, framework version, Python version, and region.
#     Args:
#         region (str): The AWS region.
#         framework (str): The framework name, e.g. "tensorflow-scriptmode".
#         py_version (str): The Python version, e.g. "py3".
#         account (str): The AWS account ID to use as a default.
#         framework_version (str): The framework version.
#     Returns:
#         str: The appropriate Amazon ECR registry number. If there is no
#             specific one for the framework, framework version, Python version,
#             and region, then ``account`` is returned.
.registry_id <- function(region,
                         framework,
                         py_version,
                         account,
                         framework_version){
  if(.is_dlc_version(framework, framework_version, py_version)){
    if (region %in% names(ASIMOV_OPT_IN_ACCOUNTS_BY_REGION))
      return(ASIMOV_OPT_IN_ACCOUNTS_BY_REGION[[region]])
    if(region %in% names(ASIMOV_VALID_ACCOUNTS_BY_REGION))
      return(ASIMOV_VALID_ACCOUNTS_BY_REGION[[region]])
    return(ASIMOV_DEFAULT_ACCOUNT)
  }
  if(region %in% names(OPT_IN_ACCOUNTS_BY_REGION))
    return(OPT_IN_ACCOUNTS_BY_REGION[[region]])
  return(VALID_ACCOUNTS_BY_REGION[[region]])
}

# Return if the framework's version uses the corresponding DLC image.
#     Args:
#         framework (str): The framework name, e.g. "tensorflow-scriptmode"
#         framework_version (str): The framework version
#         py_version (str): The Python version, e.g. "py3"
#     Returns:
#         bool: Whether or not the framework's version uses the DLC image.
.is_dlc_version <- function(framework, framework_version, py_version){
  lowest_version_list = MERGED_FRAMEWORKS_LOWEST_VERSIONS[[framework]]
  if (inherits(lowest_version_list, "list")){
    lowest_version_list = lowest_version_list[[py_version]]
  }
  if (length(lowest_version_list) > 0 || !is.null(lowest_version_list))
    return(is_version_equal_or_higher(lowest_version_list, framework_version))
  return(FALSE)
}

# Determine whether the ``framework_version`` is equal to or higher than
#     ``lowest_version``
#     Args:
#         lowest_version (List[int]): lowest version represented in an integer
#             list
#         framework_version (str): framework version string
#     Returns:
#         bool: Whether or not ``framework_version`` is equal to or higher than
#             ``lowest_version``
is_version_equal_or_higher <- function(lowest_version,
                                      framework_version){
  return(package_version(framework_version) >= package_version(lowest_version))
}

# Determine whether the ``framework_version`` is equal to or lower than
# ``highest_version``
# Args:
#   highest_version (List[int]): highest version represented in an integer
# list
# framework_version (str): framework version string
# Returns:
#   bool: Whether or not ``framework_version`` is equal to or lower than
# ``highest_version``
is_version_equal_or_lower <- function(highest_version,
                                      framework_version){
  return (package_version(framework_version) <= package_version(highest_version))
}

# Return if Inferentia supports the framework and its version.
# Args:
#   framework (str): The framework name, e.g. "tensorflow"
# framework_version (str): The framework version
# Returns:
#   bool: Whether or not Inferentia supports the framework and its version.
.is_inferentia_supported <- function(framework,
                                     framework_version){
  lowest_version_list = INFERENTIA_VERSION_RANGES[[framework]][1]
  highest_version_list = INFERENTIA_VERSION_RANGES[[framework]][1]
  return(is_version_equal_or_higher(lowest_version_list, framework_version)
  && is_version_equal_or_lower(highest_version_list, framework_version))
}
