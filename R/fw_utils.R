# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/fw_utils.py

#' @include utils.R

#' @import lgr

.TAR_SOURCE_FILENAME <- ".tar.gz"

UploadedCode <- list("s3_prefix" = NULL, "script_name" = NULL)

PYTHON_2_DEPRECATION_WARNING <- paste(
  "%s is the latest version of %s that supports",
  "Python 2. Newer versions of %s will only be available for Python 3.",
  "Please set the argument \"py_version='py3'\" to use the Python 3 %s image.")

PARAMETER_SERVER_MULTI_GPU_WARNING <- paste(
  "If you have selected a multi-GPU training instance type",
  "and also enabled parameter server for distributed training,",
  "distributed training with the default parameter server configuration will not",
  "fully leverage all GPU cores; the parameter server will be configured to run",
  "only one worker per host regardless of the number of GPUs.")

DEBUGGER_UNSUPPORTED_REGIONS <- c("us-gov-west-1", "us-iso-east-1")
SINGLE_GPU_INSTANCE_TYPES <- c("ml.p2.xlarge", "ml.p3.2xlarge")

SM_DATAPARALLEL_SUPPORTED_FRAMEWORK_VERSIONS <- list(
  "tensorflow"= c("2.3.0", "2.3.1"),
  "pytorch"= "1.6.0"
)

SMDISTRIBUTED_SUPPORTED_STRATEGIES <- c("dataparallel", "modelparallel")

# Validate that the source directory exists and it contains the user script
# Args:
#   script (str): Script filename.
# directory (str): Directory containing the source file.
# Raises:
#   ValueError: If ``directory`` does not exist, is not a directory, or does
# not contain ``script``.
validate_source_dir <- function(script, directory){
  if (is.character(directory)){
    if (!file_test("-f",file.path(directory, script))){
      stop(sprintf('No file named "%s" was found in directory "%s".',script, directory), call. = F)
    }
  }
  return(TRUE)
}

# Get the model parallelism parameters provided by the user.
# Args:
#   distribution: distribution dictionary defined by the user.
# Returns:
#   params: dictionary containing model parallelism parameters
# used for training.
get_mp_parameters <- function(distribution){
  mp_dict = distribution$smdistributed$modelparallel %||% list()
  if (isTRUE(mp_dict$enabled %||% FALSE)) {
    params = mp_dict$parameters %||% list()
    validate_mp_config(params)
    return(params)
  }
  return(NULL)
}

# Validate the configuration dictionary for model parallelism.
# Args:
#   config (dict): Dictionary holding configuration keys and values.
# Raises:
#   ValueError: If any of the keys have incorrect values.
validate_mp_config <- function(config){
  if (!("partitions" %in% names(config)))
    stop("'partitions' is a required parameter.", call. = F)

  validate_positive <- function(key){
    if (!inherits(config[[key]], c("integer", "numeric")) || config[key] < 1)
      stop(sprintf("The number of %s must be a positive integer.",key), call. = F)
  }

  validate_in <- function(key, vals){
    if (!(config[[key]] %in% vals))
      stop(sprintf("%s must be a value in: [%s].",
             key, paste(vals, collapse = ", ")),
           call. = F)
  }

  validate_bool <- function(keys){
    validate_in(keys, c(TRUE, FALSE))
  }

  validate_in("pipeline", c("simple", "interleaved", "_only_forward"))
  validate_in("placement_strategy", c("spread", "cluster"))
  validate_in("optimize", c("speed", "memory"))

  for (key in c("microbatches", "partitions"))
    validate_positive(key)

  for (key in c("auto_partition", "contiguous", "load_partition", "horovod", "ddp"))
    validate_bool(key)

  if ("partition_file" %in% names(config) &&
      !inherits(config$partition_file, "character"))
    stop("'partition_file' must be a character.", call. = F)

  if (!isTRUE(config$auto_partition) && !("default_partition" %in% names(config)))
    stop("default_partition must be supplied if auto_partition is set to `FALSE`!", call. = F)

  if ("default_partition" %in% names(config) && config$default_partition >= config$partitions)
    stop("default_partition must be less than the number of partitions!", call. = F)

  if ("memory_weight" %in% names(config) && (
    config$memory_weight > 1 || config$memory_weight < 0))
    stop("memory_weight must be between 0.0 and 1.0!", call. = F)

  if ("ddp_port" %in% names(config) && "ddp" %in% names(config))
    stop("`ddp_port` needs `ddp` to be set as well", call. = F)

  if ("ddp_dist_backend" %in% names(config) && !("ddp" %in% names(config)))
    stop("`ddp_dist_backend` needs `ddp` to be set as well", call. = F)

  if ("ddp_port" %in% names(config)){
    if (!inherits(config$ddp_port, "integer") || config$ddp_port < 0){
      value = config$ddp_port
      stop(sprintf("Invalid port number %s.", value) , call. = F)
    }
  }

  if ((config$horovod %||% FALSE) && (config$ddp %||% FALSE))
    stop("'ddp' and 'horovod' cannot be simultaneously enabled.", call. = F)
}

# Package source files and upload a compress tar file to S3. The S3
# location will be ``s3://<bucket>/s3_key_prefix/sourcedir.tar.gz``.
# If directory is an S3 URI, an UploadedCode object will be returned, but
# nothing will be uploaded to S3 (this allow reuse of code already in S3).
# If directory is None, the script will be added to the archive at
# ``./<basename of script>``.
# If directory is not None, the (recursive) contents of the directory will
# be added to the archive. directory is treated as the base path of the
# archive, and the script name is assumed to be a filename or relative path
# inside the directory.
# Args:
#   sagemaker_session (sagemaker.Session): sagemaker_session session used to access S3.
# bucket (str): S3 bucket to which the compressed file is uploaded.
# s3_key_prefix (str): Prefix for the S3 key.
# script (str): Script filename or path.
# directory (str): Optional. Directory containing the source file. If it
# starts with "s3://", no action is taken.
# dependencies (List[str]): Optional. A list of paths to directories
# (absolute or relative) containing additional libraries that will be
# copied into /opt/ml/lib
# kms_key (str): Optional. KMS key ID used to upload objects to the bucket
# (default: None).
# s3_resource (boto3.resource("s3")): Optional. Pre-instantiated Boto3 Resource
# for S3 connections, can be used to customize the configuration,
# e.g. set the endpoint URL (default: None).
# Returns:
#   sagemaker.fw_utils.UserCode: An object with the S3 bucket and key (S3 prefix) and
# script name.
tar_and_upload_dir <- function(sagemaker_session,
                               bucket,
                               s3_key_prefix,
                               script,
                               directory=NULL,
                               dependencies=NULL,
                               kms_key=NULL){
  if (!is.null(directory) && startsWith(tolower(directory),"s3://")){
    UploadedCode$s3_prefix=directory
    UploadedCode$script_name= basename(script)
    return(UploadedCode)}

  script_name =  if(!is.null(directory)) script else basename(script)
  dependencies = dependencies %||% list()
  key = sprintf("%s/sourcedir.tar.gz",s3_key_prefix)
  tmp = tempfile(fileext = .TAR_SOURCE_FILENAME)

  tryCatch({source_files = unlist(c(.list_files_to_compress(script, directory), dependencies))
            tar_file = create_tar_file(source_files, tmp)})

  if (!is.null(kms_key)) {
    ServerSideEncryption = "aws:kms"
    SSEKMSKeyId =  kms_key
  } else {
    ServerSideEncryption = NULL
    SSEKMSKeyId =  NULL
    }

  if(!is.null(sagemaker_session$s3)) {
    s3_resource = sagemaker_session$s3

    obj <- readBin(tar_file, "raw", n = file.size(tar_file))
    s3_resource$put_object(Body = obj, Bucket = bucket, Key = key,
                           ServerSideEncryption = ServerSideEncryption,
                           SSEKMSKeyId = SSEKMSKeyId)
  }

  on.exit(unlink(tmp, recursive = T))

  UploadedCode$s3_prefix=sprintf("s3://%s/%s",bucket, key)
  UploadedCode$script_name=script_name

  return(UploadedCode)
}

.list_files_to_compress <- function(script, directory){
  if(is.null(directory))
    return(list(script))

  basedir = directory %||% dirname(script)

  return(list.files(basedir, full.names = T))
}


# Extract the framework and Python version from the image name.
# Args:
#   image_uri (str): Image URI, which should be one of the following forms:
#   legacy:
#   '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>-<py_ver>-<device>:<container_version>'
# legacy:
#   '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>-<py_ver>-<device>:<fw_version>-<device>-<py_ver>'
# current:
#   '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>:<fw_version>-<device>-<py_ver>'
# current:
#   '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-<fw>:<rl_toolkit><rl_version>-<device>-<py_ver>'
# current:
#   '<account>.dkr.ecr.<region>.amazonaws.com/<fw>-<image_scope>:<fw_version>-<device>-<py_ver>'
# Returns:
#   tuple: A tuple containing:
#   - str: The framework name
# - str: The Python version
# - str: The image tag
# - str: If the TensorFlow image is script mode
framework_name_from_image <- function(image_uri){
  sagemaker_pattern = ECR_URI_PATTERN
  sagemaker_match = regmatches(image_uri,regexec(ECR_URI_PATTERN,image_uri))[[1]]
  sagemaker_match = sagemaker_match[length(sagemaker_match)]
  if (is.na(sagemaker_match) || length(sagemaker_match) == 0)
    return(list(NULL, NULL, NULL, NULL))

  # extract framework, python version and image tag
  # We must support both the legacy and current image name format.
  name_pattern = "^(?:sagemaker(?:-rl)?-)?(tensorflow|mxnet|chainer|pytorch|scikit-learn|xgboost)(?:-)?(scriptmode|training)?:(.*)-(.*?)-(py2|py3[67]?)$"
  name_match = regmatches(sagemaker_match,regexec(name_pattern,sagemaker_match))[[1]]
  if (length(name_match) > 0){
    fw_pts = as.list(name_match[-1])
    fw_pts = lapply(fw_pts, function(x) if(x =="") NULL else x)
    names(fw_pts) = c("fw", "scriptmode", "ver", "device", "py")
    return(list(fw_pts$fw, fw_pts$py, sprintf("%s-%s-%s", fw_pts$ver, fw_pts$device, fw_pts$py), fw_pts$scriptmode))
  }

  legacy_name_pattern = "^sagemaker-(tensorflow|mxnet)-(py2|py3)-(cpu|gpu):(.*)$"
  legacy_match = regmatches(sagemaker_match,regexec(legacy_name_pattern,sagemaker_match))[[1]]
  if (length(legacy_match) > 0){
    lg_pts = legacy_match[-1]
    return (list(lg_pts[1], lg_pts[2], lg_pts[4], NULL))
  }
  return(list(NULL, NULL, NULL, NULL))
}

# Extract the framework version from the image tag.
# Args:
#     image_tag (str): Image tag, which should take the form
#         '<framework_version>-<device>-<py_version>'
# Returns:
#     str: The framework version.
framework_version_from_tag <- function(image_tag){
  tag_pattern = "^(.*)-(cpu|gpu)-(py2|py3[67]?)$"
  tag_match = regmatches(image_tag,regexec(tag_pattern,image_tag))[[1]]
  return(if (length(tag_match) == 0) NULL else tag_match[2])
}

# Returns the s3 key prefix for uploading code during model deployment
# The location returned is a potential concatenation of 2 parts
# 1. code_location_key_prefix if it exists
# 2. model_name or a name derived from the image
# Args:
#   code_location_key_prefix (str): the s3 key prefix from code_location
# model_name (str): the name of the model
# image (str): the image from which a default name can be extracted
# Returns:
#   str: the key prefix to be used in uploading code
model_code_key_prefix <- function(code_location_key_prefix, model_name, image){
  training_job_name = name_from_image(image)
  return(paste0(Filter(Negate(is.null), list(code_location_key_prefix, model_name %||% training_job_name)), collapse = "/"))
}


# Returns boolean indicating whether the region supports Amazon SageMaker Debugger.
# Args:
#   region_name (str): Name of the region to check against.
# Returns:
#   bool: Whether or not the region supports Amazon SageMaker Debugger.
.region_supports_debugger <- function(region_name){
  return (!(tolower(region_name) %in% DEBUGGER_UNSUPPORTED_REGIONS))
}

# Checks if version or image arguments are specified.
# Validates framework and model arguments to enforce version or image specification.
# Args:
#   framework_version (str): The version of the framework.
# py_version (str): The version of Python.
# image_uri (str): The URI of the image.
# Raises:
#   ValueError: if `image_uri` is None and either `framework_version` or `py_version` is
# None.
validate_version_or_image_args <- function(framework_version, py_version, image_uri){
  if ((is.null(framework_version) || is.null(py_version)) && is.null(image_uri))
    stop(
      "`framework_version` or `py_version` was NULL, yet `image_uri` was also NULL.",
      " Either specify both `framework_version` and `py_version`, or specify `image_uri`.",
      call. = F
    )
}

# Warn the user that training will not fully leverage all the GPU
# cores if parameter server is enabled and a multi-GPU instance is selected.
# Distributed training with the default parameter server setup doesn't
# support multi-GPU instances.
# Args:
#     training_instance_type (str): A string representing the type of training instance selected.
#     distribution (dict): A dictionary with information to enable distributed training.
#         (Defaults to None if distributed training is not enabled.) For example:
#         .. code:: python
#             {
#                 'parameter_server':
#                 {
#                     'enabled': True
#                 }
#             }
warn_if_parameter_server_with_multi_gpu <- function(training_instance_type, distribution){
  if (training_instance_type == "local" || is.null(distribution))
    return(invisible(NULL))

  is_multi_gpu_instance = (
      (training_instance_type == "local_gpu" ||
       startsWith(split_str(training_instance_type,  "\\.")[[2]],"p")) &&
      !(training_instance_type %in% SINGLE_GPU_INSTANCE_TYPES)
    )

  ps_enabled = (
    ("parameter_server" %in% names(distribution)) &&
    distribution$parameter_server$enabled %||% FALSE
  )

  if (is_multi_gpu_instance && ps_enabled)
    LOGGER$warn(PARAMETER_SERVER_MULTI_GPU_WARNING)
}

# Check if smdistributed strategy is correctly invoked by the user.
# Currently, two strategies are supported: `dataparallel` or `modelparallel`.
# Validate if the user requested strategy is supported.
# Currently, only one strategy can be specified at a time. Validate if the user has requested
# more than one strategy simultaneously.
# Validate if the smdistributed dict arg is syntactically correct.
# Additionally, perform strategy-specific validations.
# Args:
#   instance_type (str): A string representing the type of training instance selected.
# framework_name (str): A string representing the name of framework selected.
# framework_version (str): A string representing the framework version selected.
# py_version (str): A string representing the python version selected.
# distribution (dict): A dictionary with information to enable distributed training.
# (Defaults to None if distributed training is not enabled.) For example:
#   .. code:: python
# {
#   "smdistributed": {
#     "dataparallel": {
#       "enabled": True
#     }
#   }
# }
# image_uri (str): A string representing a Docker image URI.
# Raises:
#   ValueError: if distribution dictionary isn't correctly formatted or
#             multiple strategies are requested simultaneously or
#             an unsupported strategy is requested or
#             strategy-specific inputs are incorrect/unsupported
validate_smdistributed <- function(instance_type, framework_name, framework_version, py_version, distribution, image_uri=NULL){
  if (!("smdistributed" %in% names(distribution))){
    # Distribution strategy other than smdistributed is selected
    return(NULL)
  }

  # distribution contains smdistributed
  smdistributed = distribution$smdistributed
  if (!inherits(smdistributed, "list"))
    stop("smdistributed strategy requires a dictionary", call. = F)

  if (length(smdistributed) > 1){
    # more than 1 smdistributed strategy requested by the user
    err_msg = paste(
      "Cannot use more than 1 smdistributed strategy.\n",
      "Choose one of the following supported strategies:",
      paste(SMDISTRIBUTED_SUPPORTED_STRATEGIES, collapse = ", "))
    stop(err_msg, call. = F)
  }

  # validate if smdistributed strategy is supported
  # currently this for loop essentially checks for only 1 key
  for (strategy in smdistributed){
    if (!(names(strategy) %in% SMDISTRIBUTED_SUPPORTED_STRATEGIES)){
      err_msg = paste(
        sprintf("Invalid smdistributed strategy provided: %s\n", strategy),
        sprintf("Supported strategies: %s", paste(SMDISTRIBUTED_SUPPORTED_STRATEGIES, collapse = ", "))
        )
      stop(err_msg, call. = F)
    }
  }

  # smdataparallel-specific input validation
  if ("dataparallel" %in% names(smdistributed)){
    .validate_smdataparallel_args(
      instance_type, framework_name, framework_version, py_version, distribution, image_uri
    )
  }
}

# Check if request is using unsupported arguments.
# Validate if user specifies a supported instance type, framework version, and python
# version.
# Args:
#   instance_type (str): A string representing the type of training instance selected. Ex: `ml.p3.16xlarge`
# framework_name (str): A string representing the name of framework selected. Ex: `tensorflow`
# framework_version (str): A string representing the framework version selected. Ex: `2.3.1`
# py_version (str): A string representing the python version selected. Ex: `py3`
# distribution (dict): A dictionary with information to enable distributed training.
# (Defaults to None if distributed training is not enabled.) Ex:
#   .. code:: python
# {
#   "smdistributed": {
#     "dataparallel": {
#       "enabled": True
#     }
#   }
# }
# image_uri (str): A string representing a Docker image URI.
# Raises:
#   ValueError: if
# (`instance_type` is not in SM_DATAPARALLEL_SUPPORTED_INSTANCE_TYPES or
#  `py_version` is not python3 or
#  `framework_version` is not in SM_DATAPARALLEL_SUPPORTED_FRAMEWORK_VERSION
.validate_smdataparallel_args <- function(instance_type,
                                          framework_name,
                                          framework_version,
                                          py_version,
                                          distribution,
                                          image_uri=NULL){
  smdataparallel_enabled = distribution$smdistributed$dataparallel$enabled %||% FALSE

  if (!smdataparallel_enabled)
    return(NULL)

  is_instance_type_supported = instance_type %in% SM_DATAPARALLEL_SUPPORTED_INSTANCE_TYPES

  err_msg = ""

  if (!is_instance_type_supported){
    # instance_type is required
    err_msg = paste0(err_msg,
                sprintf("Provided instance_type %s is not supported by smdataparallel.\n",instance_type),
                sprintf("Please specify one of the supported instance types: %s\n",
                        paste(SMDISTRIBUTED_SUPPORTED_STRATEGIES, collapse = ", ")))
  }

  if (is.null(image_uri)){
    # ignore framework_version & py_version if image_uri is set
    # in case image_uri is not set, then both are mandatory
    supported = SM_DATAPARALLEL_SUPPORTED_FRAMEWORK_VERSIONS[[framework_name]]
    if (!(framework_version %in% supported)){
      err_msg = paste0(err_msg,
        sprintf("Provided framework_version %s is not supported by", framework_version),
        " smdataparallel.\n",
        sprintf("Please specify one of the supported framework versions: %s \n", paste(supported, collapse = ", ")))
    }
    if (!("py3" %in% py_version)){
      err_msg = paste0(err_msg,
        sprintf("Provided py_version %s is not supported by smdataparallel.\n", py_version),
        "Please specify py_version=py3")
    }
  }
  if (length(err_msg) > 0)
    stop(err_msg, call. = F)
}


python_deprecation_warning <- function(framework, latest_supported_version){
  return(sprintf(PYTHON_2_DEPRECATION_WARNING,
                 latest_supported_version, framework, framework, framework))
}


