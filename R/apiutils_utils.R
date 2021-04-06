# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/apiutils/_utils.py

#' @include utils.R
#' @include session.R

#' @import R6

ApiUtils = R6Class("ApiUtils",
  public = list(
    # Generate a random string of length 4.
    suffix = function(){
      alpha = paste(sample(letters, 4), collapse = "")
      return(paste(format(Sys.time(), "%Y-%m-%d-%H%M%S"), alpha, sep = "-"))
    },

    # Generate a new name with the specified prefix.
    name = function(prefix){
      return(paste(prefix, self$suffix(), sep = "-"))
    },

    # Create a default session.
    default_session = function(){
      return(Session$new())
    }
  )
)
