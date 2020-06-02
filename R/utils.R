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

# validation check of s3 uri
is.s3_uri <- function(x) {
  if(is.null(x)) return(FALSE)
  regex <- '^s3://[a-z0-9][a-z0-9\\.-]+[a-z0-9](/(.*)?)?$'
  grepl(regex, x)
}

# split s3 uri
split_s3_uri <- function(uri) {
  stopifnot(is.s3_uri(uri))
  path <- gsub('^s3://', '', uri)
  list(
    bucket = gsub('/.*$', '', path),
    key = gsub('^[a-z0-9][a-z0-9\\.-]+[a-z0-9]/', '', path)
  )
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
