# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/deprecations.py

#' @import logger

V2_URL <- "https://sagemaker.readthedocs.io/en/stable/v2.html"

# Generic warning raiser referencing V2
# Args:
#   phrase: The phrase to include in the warning.
.warn <- function(msg){
  full_msg = sprintf("%s in python sagemaker>=2.\nSee: %s for details.", msg, V2_URL)
  log_warn(full_msg)
}

# Checks if the deprecated argument is in kwargs
# Raises warning, if present.
# Args:
#   name: name of deprecated argument
# kwargs: keyword arguments dict
removed_kwargs <- function(name, kwargs){
  msg = "%s is a no-op"
  if (name %in% names(kwargs))
    .warn(sprintf(msg, name))
}


