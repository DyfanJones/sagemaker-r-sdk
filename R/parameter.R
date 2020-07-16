# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/parameter.py


#' @import R6

#' @title ParameterRange Class
#' @description Base class for representing parameter ranges. This is used to define what
#'              hyperparameters to tune for an Amazon SageMaker hyperparameter tuning job
#'              and to verify hyperparameters for Marketplace Algorithms.
ParameterRange = R6Class("ParameterRange",
  public = list(
    #' @field .all_types
    #' All types of child class
    .all_types = c("Continuous", "Categorical", "Integer"),

    #' @field min_value
    #' The minimum value for the range
    min_value = NULL,

    #' @field max_value
    #' The maximum value for the rang
    max_value = NULL,

    #' @field scaling_type
    #' The scale used for searching the range during tuning
    scaling_type = NULL,

    #' @description Initialize a parameter range.
    #' @param min_value (float or int): The minimum value for the range.
    #' @param max_value (float or int): The maximum value for the range.
    #' @param scaling_type (str): The scale used for searching the range during
    #'              tuning (default: 'Auto'). Valid values: 'Auto', 'Linear',
    #'              Logarithmic' and 'ReverseLogarithmic'.
    initialize = function(min_value,
                          max_value,
                          scaling_type=c('Auto', 'Linear', 'Logarithmic', 'ReverseLogarithmic')){
      self$min_value = min_value
      self$max_value = max_value
      self$scaling_type = match.arg(scaling_type)
    },

    #' @description Determine if a value is valid within this ParameterRange.
    #' @param value (float or int): The value to be verified.
    #' @return bool: True if valid, False otherwise.
    is_valid = function(value){
      return ((self$min_value <= value &  value<= self$max_value))
    },

    #' @description cast value to numeric
    #' @param value The value to be verified.
    cast_to_type = function(value){
      return (as.numeric(value))
    },

    #' @description Represent the parameter range as a dicionary suitable for a request
    #'              to create an Amazon SageMaker hyperparameter tuning job.
    #' @param name (str): The name of the hyperparameter.
    #' @return dict[str, str]: A dictionary that contains the name and values of
    #'              the hyperparameter.
    as_tuning_range = function(name){
      return (list("Name"= name,
                   "MinValue"= as.character(self$min_value),
                   "MaxValue" = as.character(self$max_value),
                   "ScalingType"= self$scaling_type))
    }
  )
)

#' @title ContinuousParameter Class
#' @description A class for representing hyperparameters that have a continuous range of possible values.
#' @export
ContinuousParameter = R6Class("ContinuousParameter",
  inherit = ParameterRange,
  public = list(
    #' @field .name
    #' Helps to categorise Class
    .name = "Continuous",

    # Kept initialize in for individual class for roxygen2 documentation
    #' @description Initialize a ContinuousParameter
    #' @param min_value (float): The minimum value for the range.
    #' @param max_value (float): The maximum value for the range.
    #' @param scaling_type (str): The scale used for searching the range during
    #'              tuning (default: 'Auto'). Valid values: 'Auto', 'Linear',
    #'              Logarithmic' and 'ReverseLogarithmic'.
    initialize = function(min_value,
                          max_value,
                          scaling_type= c('Auto', 'Linear', 'Logarithmic', 'ReverseLogarithmic')) {
      super$initialize(min_value,
                       max_value,
                       scaling_type)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      cat("<ContinuousParameter>")
      invisible(self)
    }
  )
)

#' @title CategoricalParameter Class
#' @description A class for representing hyperparameters that have a discrete list of
#'              possible values.
#' @export
CategoricalParameter = R6Class("CategoricalParameter",
  inherit = ParameterRange,
  public = list(
   #' @field .name
   #' Helps to categorise Class
   .name = "Categorical",

   #' @field values
   #' The possible values for the hyperparameter
   values = NULL,

   # Kept initialize in for individual class for roxygen2 documentation
   #' @description Initialize a ``CategoricalParameter``.
   #' @param values (list or object): The possible values for the hyperparameter.
   #'              This input will be converted into a list of strings.
   initialize = function(values){
     if (inherits(values, "list"))
       self$values = values
     else
       self$values = as.list(values)
   },

   #' @description Represent the parameter range as a dicionary suitable for a request
   #'              to create an Amazon SageMaker hyperparameter tuning job.
   #' @param name (str): The name of the hyperparameter.
   #' @return dict[str, list[str]]: A dictionary that contains the name and values
   #'              of the hyperparameter.
   as_tuning_range = function(name){
   return (list("Name"= name, "Values"= self$values))
   },

   #' @description Represent the parameter range as a dictionary suitable for a request
   #'              to create an Amazon SageMaker hyperparameter tuning job using one of the
   #'              deep learning frameworks.
   #'              The deep learning framework images require that hyperparameters be
   #'              serialized as JSON.
   #' @param name (str): The name of the hyperparameter.
   #' @return dict[str, list[str]]: A dictionary that contains the name and values of the
   #'              hyperparameter, where the values are serialized as JSON.
   as_json_range = function(name){
     return(list(Name = name, Values = list(shQuote(self$values))))
   },

   #' @description Determine if a value is valid within this CategoricalParameter
   #' @param value (object): Value of the hyperparameter
   #' @return boolean: TRUE` or `FALSE`
   is_valid = function(value){
     return(as.character(value) %in% self$values)
   },

   #' @description cast value to numeric
   #' @param value The value to be verified.
   cast_to_type = function(value){
     return (as.numeric(value))
   },

   #' @description
   #' Printer.
   #' @param ... (ignored).
   print = function(...){
     cat("<CategoricalParameter>")
     invisible(self)
   }
  )
)

#' @title IntegerParameter Class
#' @description A class for representing hyperparameters that have an integer range of possible values.
#' @export
IntegerParameter = R6Class("IntegerParameter",
  inherit = ParameterRange,
  public = list(
   #' @field .name
   #' Helps to categorise Class
   .name = "Integer",

   # Kept initialize in for individual class for roxygen2 documentation
   #' @description Initialize a IntegerParameter
   #' @param min_value (int): The minimum value for the range.
   #' @param max_value (int): The maximum value for the range.
   #' @param scaling_type (str): The scale used for searching the range during
   #'              tuning (default: 'Auto'). Valid values: 'Auto', 'Linear',
   #'              Logarithmic' and 'ReverseLogarithmic'.
   initialize = function(min_value,
                         max_value,
                         scaling_type = c('Auto', 'Linear', 'Logarithmic', 'ReverseLogarithmic')){
     super$initialize(min_value,
                      max_value,
                      scaling_type)
   },

   #' @description cast value to integer
   #' @param value The value to be verified.
   cast_to_type = function(value){
     return(as.integer(value))
   },

   #' @description
   #' Printer.
   #' @param ... (ignored).
   print = function(...){
     cat("<IntegerParameter>")
     invisible(self)
   }
  )
)

