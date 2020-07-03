# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/utils.py

`%||%` <- function(x, y) if (is.null(x)) return(y) else return(x)

get_aws_env <- function(x) {
    x <- Sys.getenv(x)
    if (nchar(x) == 0) return(NULL) else return(x)
  }

pkg_method <- function(fun, pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(fun,' requires the ', pkg,' package, please install it first and try again',
         call. = F)}
  fun_name <- utils::getFromNamespace(fun, pkg)
  return(fun_name)
}

get_region = pkg_method("get_region", "paws.common")
get_profile_name = pkg_method("get_profile_name", "paws.common")

.paws_cred <- function(paws_credentials) if(!inherits(paws_credentials, "PawsCredentials"))  return(paws_cred()) else return(paws_credentials)

split_str <- function(str, split = ",") unlist(strsplit(str, split = split))



name_from_image <- function(image){
  return(name_from_base(base_name_from_image(image)))
}

name_from_base <- function(base, max_length = 63, short = FALSE){
  timestamp = if(short) sagemaker_short_timestamp() else sagemaker_timestamp()
  trimmed_base = substring(base, 1,(max_length - length(timestamp) - 1))
  return(sprintf("%s-%s", trimmed_base, timestamp))
}

base_name_from_image <- function(image){
  m <- grepl("^(.+/)?([^:/]+)(:[^:]+)?$", image)
  algo_name = if(m) gsub(".*/|:.*", "", image) else image
  return(algo_name)
}

# Return a timestamp that is relatively short in length
sagemaker_short_timestamp <- function() return(format(Sys.time(), "%y%m%d-%H%M"))

# Return a timestamp with millisecond precision.
sagemaker_timestamp <- function(){
  moment = Sys.time()
  moment_ms = split_str(format(as.numeric(moment,3), nsmall = 3), "\\.")[2]
  paste0(format(Sys.time(),"%Y-%m-%d-%H-%M-%S-",tz="GMT"), moment_ms)
}

get_config_value <- function(key_path, config = NULL){
  if(is.null(config)) return(NULL)
  current_section = config

  for(key in split_str(key_path, "\\.")){
    if (key %in% current_section) current_section = current_section[key]
    else return(NULL)
  }
  return(NULL)
}

# Write large raw connections in chunks
write_bin <- function(
  value,
  filename,
  chunk_size = 2L ^ 20L) {

  # if readr is avialable then use readr::write_file else loop writeBin
  if (requireNamespace("readr", quietly = TRUE)) {
    write_file <- pkg_method("write_file", "readr")
    write_file(value, filename)
    return(invisible(TRUE))}

  total_size <- length(value)
  split_vec <- seq(1, total_size, chunk_size)

  con <- file(filename, "a+b")
  on.exit(close(con))

  if (length(split_vec) == 1) writeBin(value,con)
  else sapply(split_vec, function(x){writeBin(value[x:min(total_size,(x+chunk_size-1))],con)})
  invisible(TRUE)
}


# Returns true if training job's secondary status message has changed.
#     Args:
#         current_job_description: Current job description, returned from DescribeTrainingJob call.
#         prev_job_description: Previous job description, returned from DescribeTrainingJob call.
#     Returns:
#         boolean: Whether the secondary status message of a training job changed
#         or not.

secondary_training_status_changed <- function(current_job_description = NULL, prev_job_description = NULL){
  current_secondary_status_transitions = current_job_description$SecondaryStatusTransitions

  if(is.null(current_secondary_status_transitions) ||
     length(current_secondary_status_transitions) ==0){
    return(FALSE)
  }
  prev_job_secondary_status_transitions = if(!is.null(prev_job_description)) prev_job_description$SecondaryStatusTransitions else NULL

  last_message = (if (!is.null(prev_job_secondary_status_transitions)
                      && length(prev_job_secondary_status_transitions) > 0){
    prev_job_secondary_status_transitions[[length(prev_job_secondary_status_transitions)]]$StatusMessage
  } else {""})

  message = current_job_description$SecondaryStatusTransitions[[length(current_job_description$SecondaryStatusTransitions)]]$StatusMessage
  return(message != last_message)
}

# Returns a string contains last modified time and the secondary training
#     job status message.
#     Args:
#         job_description: Returned response from DescribeTrainingJob call
#         prev_description: Previous job description from DescribeTrainingJob call
#     Returns:
#         str: Job status string to be printed.

secondary_training_status_message <- function(job_description = NULL, prev_description = NULL){
  if (is.null(job_description)
      || is.null(job_description$SecondaryStatusTransitions)
      || length(job_description$SecondaryStatusTransitions) == 0){
    return("")}

  prev_description_secondary_transitions = if(!is.null(prev_description)) prev_description$SecondaryStatusTransitions else NULL

  prev_transitions_num = if(!is.null(prev_description_secondary_transitions)) length(prev_description$SecondaryStatusTransitions) else 0

  current_transitions = job_description$SecondaryStatusTransitions

  if (length(current_transitions) == prev_transitions_num){
    # Secondary status is not changed but the message changed.
    transitions_to_print = current_transitions[[prev_transitions_num]]
  } else{
    # Secondary status is changed we need to print all the entries.
    transitions_to_print = current_transitions[[length(current_transitions)]]
  }

  return(sprintf("%s %s - %s\n", job_description$LastModifiedTime, transitions_to_print$Status, transitions_to_print$StatusMessage))
}


# If api call fails retry call
retry_api_call <- function(expr, retry = 5){

  # if number of retries is equal to 0 then retry is skipped
  if (retry == 0) {
    resp <- tryCatch(eval.parent(substitute(expr)),
                     error = function(e) e)
  }

  for (i in seq_len(retry)) {
    resp <- tryCatch(eval.parent(substitute(expr)),
                     error = function(e) e)

    if(inherits(resp, "http_500")){

      # stop retry if statement is an invalid request
      if (grepl("InvalidRequestException", resp)) {stop(resp)}

      backoff_len <- runif(n=1, min=0, max=(2^i - 1))

      message(resp, "Request failed. Retrying in ", round(backoff_len, 1), " seconds...")

      Sys.sleep(backoff_len)
    } else {break}
  }

  if (inherits(resp, "error")) stop(resp)

  resp
}



# get prefix of ECR image URI
# Args:
#   account (str): AWS account number
# region (str): AWS region name
# Returns:
#   (str): URI prefix of ECR image

get_ecr_image_uri_prefix <- function(account,
                                     region){
  return (sprintf("%s.dkr.ecr.%s.amazonaws.com", account, region))
}

islistempty = function(obj) {(is.null(obj) || length(obj) == 0)}

# Create a tar file containing all the source_files
# Args:
#   source_files: (List[str]): List of file paths that will be contained in the tar file
# target:
#   Returns:
#   (str): path to created tar file
create_tar_file = function(source_files, target=NULL){
  if (!is.null(target)) filename = target else filename = tempfile(fileext = ".tar.gz")

  tar_subdir(filename, source_files)
  return(filename)
}

# Unpack model tarball and creates a new model tarball with the provided
# code script.
# This function does the following: - uncompresses model tarball from S3 or
# local system into a temp folder - replaces the inference code from the model
# with the new code provided - compresses the new model tarball and saves it
# in S3 or local file system
# Args:
#   inference_script (str): path or basename of the inference script that
# will be packed into the model
# source_directory (str): path including all the files that will be packed
# into the model
# dependencies (list[str]): A list of paths to directories (absolute or
#                                                           relative) with any additional libraries that will be exported to the
# container (default: []). The library folders will be copied to
# SageMaker in the same folder where the entrypoint is copied.
# Example
# The following call >>> Estimator(entry_point='train.py',
#                                  dependencies=['my/libs/common', 'virtual-env']) results in the
# following inside the container:
#   >>> $ ls
# >>> opt/ml/code
# >>>     |------ train.py
# >>>     |------ common
# >>>     |------ virtual-env
# model_uri (str): S3 or file system location of the original model tar
# repacked_model_uri (str): path or file system location where the new
# model will be saved
# sagemaker_session (sagemaker.session.Session): a sagemaker session to
# interact with S3.
# kms_key (str): KMS key ARN for encrypting the repacked model file
# Returns:
#   str: path to the new packed model
repack_model <- function(inference_script,
                         source_directory,
                         dependencies,
                         model_uri,
                         repacked_model_uri,
                         sagemaker_session,
                         kms_key=None){
  dependencies = dependencies %||% list()

  tmp = tempdir()

  # extract model from tar.gz
  model_dir = .extract_model(model_uri, sagemaker_session, tmp)

  # append file to model directory
  .create_or_update_code_dir(
    model_dir, inference_script, source_directory, dependencies, sagemaker_session, tmps)

  # repackage model_dir
  tmp_model_path = file.path(tmp, "temp-model.tar.gz")
  tar_subdir(tmp_model_path, model_dir)

  # remove temp directory/tar.gz
  on.exit(unlink(c(tmp_model_path, model_dir), recursive = T))

  # save model
  .save_model(repacked_model_uri, tmp_model_path, sagemaker_session, kms_key=kms_key)
}

is.dir <- function(directory) {(file.exists(directory) && !file_test("-f", directory))}

.create_or_update_code_dir = function(model_dir,
                                      inference_script,
                                      source_directory,
                                      dependencies,
                                      sagemaker_session,
                                      tmp) {
  code_dir = file.path(model_dir, "code")
  if (!is.null(source_directory) &&
      startsWith(tolower(source_directory), "s3://")) {
    local_code_path = file.path(tmp, "local_code.tar.gz")
    s3_parts = split_s3_uri(source_directory)
    s3 = paws::s3(config = sagemaker_session$paws_credentials$credentials)
    obj = s3$get_object(Bucket = s3_parts$bucket, Key = s3_parts$key)
    write_bin(obj$Body, local_code_path)
    untar(local_code_path, exdir = code_dir)
    on.exit(unlink(local_code_path, recursive = T))
  } else if (!is.null(source_directory)) {
    if (file.exists(code_dir)) {
      unlink(code_dir, recursive = TRUE)}
    file.copy(source_directory, code_dir, recursive = TRUE)
  } else {
    if (!file.exists(code_dir)) {
      file.copy(inference_script, code_dir, recursive = TRUE)
      if (!file.exists(file.path(code_dir, inference_script)))
        FALSE
    }
  }

  for (dependency in dependencies) {
    lib_dir = file.path(code_dir, "lib")
    if (is.dir(dependency)) {
      file.copy(dependency, file.path(lib_dir, basename(dependency)), recursive = T)
    } else {
      if (!is.dir(lib_dir)) {
        dir.create(lib_dir, recursive = TRUE)}
      file.copy(dependency, lib_dir, recursive = T)}
  }
}

.extract_model <- function(model_uri, sagemaker_session, tmp){
  tmp_model_dir = file.path(tmp, "model")
  dir.create(tmp_model_dir, showWarnings = F)
  if(startsWith(tolower(model_uri), "s3://")){
    local_model_path = file.path(tmp, "tar_file")
    s3_parts = split_s3_uri(model_uri)
    s3 = paws::s3(config = sagemaker_session$paws_credentials$credentials)
    obj = s3$get_object(Bucket = s3_parts$bucket, Key = s3_parts$key)
    write_bin(obj$Body, local_model_path)
    on.exit(unlink(local_model_path))
  } else{
    local_model_path = gsub("file://", "", model_uri)}
  untar(local_model_path, exdir = tmp_model_dir)
  return(tmp_model_dir)
}

.save_model <-
  function(repacked_model_uri,
           tmp_model_path,
           sagemaker_session,
           kms_key) {
    if (startsWith(tolower(repacked_model_uri), "s3://")) {
      s3_parts = split_s3_uri(repacked_model_uri)
      s3_parts$key = gsub(basename(s3_parts$key), basename(repacked_model_uri), s3_parts$key)
      s3 = paws::s3(config = sagemaker_session$paws_credentials$credentials)
      obj = readBin(tmp_model_path, "raw", n = file.size(tmp_model_path))
      if (!is.null(kms_key)) {
        s3$put_object(Body = obj,
                      Bucket =  s3_parts$bucket,
                      Key =  s3_parts$bucket)
      } else {
        s3$put_object(
          Body = obj,
          Bucket =  s3_parts$bucket,
          Key =  s3_parts$bucket,
          ServerSideEncryption = "aws:kms",
          SSEKMSKeyId = kms_key)}
    } else {
      file.copy(tmp_model_path,
                gsub("file://", "", repacked_model_uri.replace), recursive = T)}
  }

# tar function to use system tar
tar_subdir <- function(tarfile, srdir, compress = "gzip", ...){
  current_dir = getwd()
  setwd(srdir)
  on.exit(setwd(current_dir))
  tar(tarfile= tarfile, files=".", compression=compress, tar = "tar" , ...)
}

print_list <- function(l){
  output <- vapply(seq_along(l), function(i){paste(names(l[i]), l[[i]], sep = ": ")}, FUN.VALUE = character(1))
  paste(output, collapse = ", ")
}

IsSubR6Class <- function(subclass, cls) {
  if(is.null(subclass)) return(NULL)
  if (!is.R6Class(subclass))
    stop("subclass is not a R6ClassGenerator.", call. = F)
  parent <- subclass$get_inherit()
  cls %in% c(subclass$classname, IsSubR6Class(parent))
}
