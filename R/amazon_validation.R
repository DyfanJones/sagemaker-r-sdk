# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/validation.py

#' @import R6

Validation = R6Class("Validation",
  public = list(
    gt = function(minimum){
      return(function(value){value > minimum})
    },

    ge = function(minimum){
      return(function(value){value >= minimum})
    },

    lt = function(maximum){
      return(function(value){value < maximum})
    },

    le = function(maximum){
      return(function(value){value <= maximum})
    },

    isin = function(expected){
      return(function(value){value %in% expected})
    },

    istype = function(expected){
      return(function(value){inherits(valuem, expected)})
    }
  )
)
