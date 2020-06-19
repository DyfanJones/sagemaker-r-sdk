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

