# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/deprecations.py

#' @import logger

V2_URL = "https://sagemaker.readthedocs.io/en/stable/v2.html"

.warn <- function(msg){
  full_msg = sprintf("%s to align with python sagemaker>=2.\nSee: %s for details.", msg, V2_URL)
  warning(full_msg, call. = F)
  log_warn(full_msg)
}

# Raise a warning for a no-op in sagemaker>=2
# Args:
#   phrase: the prefix phrase of the warning message.
removed_warning <- function(phrase){
  .warn(sprintf("%s is a no-op", phrase))
}

# Raise a warning for a rename in sagemaker>=2
# Args:
#   phrase: the prefix phrase of the warning message.
renamed_warning <- function(phrase){
  .warn(sprintf("%s has been renamed", phrase))
}

# Checks if the deprecated argument is in kwargs
# Raises warning, if present.
# Args:
#   old_name: name of deprecated argument
# new_name: name of the new argument
# value: value associated with new name, if supplied
# kwargs: keyword arguments dict
# Returns:
#   value of the keyword argument, if present
renamed_kwargs <- function(old_name,
                           new_name,
                           value,
                           kwargs){
  if(old_name %in% names(renamed_kwargs)){
    value = kwargs[[old_name]] %||% value
    eval.parent(substitute({kwargs[[new_name]] = value}))
    renamed_warning(old_name)}
  return(value)
}

# Checks if the deprecated argument is populated.
# Raises warning, if not None.
# Args:
#   name: name of deprecated argument
# arg: the argument to check
remove_arg <- function(name,
                       arg = NULL){
  if(!is.null(arg)){
    removed_warning(name)
  }
}

# Checks if the deprecated argument is in kwargs
# Raises warning, if present.
# Args:
#   name: name of deprecated argument
# kwargs: keyword arguments dict
removed_kwargs <- function(name,
                           kwargs){
  if (name %in% names(kwargs))
    removed_warning(name)
}
