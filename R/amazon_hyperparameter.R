# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/hyperparameter.py

#' @import R6

#' @title Hyperparameter Class
#' @description An algorithm hyperparameter with optional validation. Implemented as a
#'              python descriptor object.
Hyperparameter = R6Class("Hyperparameter",
  public = list(

    #' @field validation
    #' validation function
    validation = NULL,

    #' @field validation_message
    #' validation message
    validation_message = NULL,

    #' @field name
    #' name of hyperparameter validate
    name = NULL,

    #' @field data_type
    #' function to convert data type
    data_type = NULL,

    #' @param name (str): The name of this hyperparameter validate
    #' @param validate (callable[object]->[bool]): A validation function or list of validation
    #'                    functions.
    #'                    Each function validates an object and returns False if the object
    #'                    value is invalid for this hyperparameter.
    #' @param validation_message (str): A usage guide to display on validation
    #'                    failure.
    #' @param data_type : function to convert data types
    #' @param obj (R6Class): R6Class for descriptor class to modify
    initialize = function(name,
                          validate= function(x) TRUE,
                          validation_message="",
                          data_type=as.character,
                          obj){
      stopifnot(is.character(validation_message),
                is.function(data_type))

      self$validation = validate
      self$validation_message = validation_message
      self$name = name
      self$data_type = data_type
      self$obj = obj
    },

    #' @description Validate value
    #' @param value : values to be validated
    validate = function(value = NULL){
      if (is.null(value))  # We allow assignment from None, but Nones are not sent to training.
        return(NULL)

      for (valid in self$validation){
        if (!valid(value)){
          error_message = sprintf("Invalid hyperparameter value %s for %s", value, self$name)
          if (!is.null(self$validation_message))
            error_message = error_message + ". Expecting: " + self.validation_message
          stop(error_message, call. = F)
        }
      }
    },

    #' @description Return all non-None ``hyperparameter`` values on ``obj`` as a
    #'              ``dict[str,str].``
    #' @param obj : R object to be serialized
    serialize_all = function(obj){

      if (!(".hyperparameters" %in% names(obj)))
        return(list())

      return(Filter(Negate(is.null), obj[[".hyperparameters"]]))
    }
  ),
  private = list(
    # Placeholder: until get R6 equivalent
    .get = function(){
      if (!(".hyperparameters" %in% names(self$obj)) || !(self$name %in% self$obj[[".hyperparameters"]]))
        stop("Attribute Error", call. = F)
      return(self$obj[[".hyperparameters"]][[self$name]])
    },

    # Validate the supplied value and set this hyperparameter to value
    # Placeholder: until get R6 equivalent
    .set = function(value = NULL){
      value = if(is.null(value)) NULL else self$data_type(value)
      self$validate(value)
      if (!(".hyperparameters" %in% names(self$obj)))
        self$obj[[".hyperparameters"]] = list()
      self$obj[[".hyperparameters"]][[self$name]] = value
    },

    # Placeholder: until get R6 equivalent
    .delete = function(){
      self$obj[[".hyperparameters"]][[self$name]] <- NULL
    }
  ),
  active = list(
    # Use active binding to mimic Python Descriptor class

    #' @field descriptor
    #' active binding that mimic's python descriptor class
    descriptor = function(value){
      # get method
      if(missing(value))
        return(private$.get())

      # delete method
      if(is.null(value))
        return(private$.delete(value))

      # set method
      return(private$.set(value))
    }
  )
)

