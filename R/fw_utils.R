# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/fw_utils.py

#' @include utils.R

.TAR_SOURCE_FILENAME <- ".tar.gz"

UploadedCode <- list("s3_prefix" = NULL, "script_name" = NULL)

DEBUGGER_UNSUPPORTED_REGIONS <- c("us-gov-west-1", "us-iso-east-1")
SINGLE_GPU_INSTANCE_TYPES <- c("ml.p2.xlarge", "ml.p3.2xlarge")

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
  if (is.na(sagemaker_match))
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
      "Either specify both `framework_version` and `py_version`, or specify `image_uri`.",
      call. = F
    )
}

PYTHON_2_DEPRECATION_WARNING <- paste(
  "%s is the latest version of %s that supports",
  "Python 2. Newer versions of %s will only be available for Python 3.",
  "Please set the argument \"py_version='py3'\" to use the Python 3 %s image.")

python_deprecation_warning <- function(framework, latest_supported_version){
  return(sprintf(PYTHON_2_DEPRECATION_WARNING,
                 latest_supported_version, framework, framework, framework))
}
