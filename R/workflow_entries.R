# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/workflow/entities.py

#' @import R6
#' @import R6sagemaker.common

#' @title Base object for workflow entities.
#' @description Entities must implement the to_request method.
Entity = R6Class("Entity",
  public = list(

    #' @description Get the request structure for workflow service calls.
    to_request = function(){
      NotImplementedError$new()
    }
  )
)

# TODO: investigate EnumMeta and how it contributes to airflow workflow
# DefaultEnumMeta


#' @title Base object for expressions.
#' @description Expressions must implement the expr property.
Expression = R6Class("Expression",
  active = list(

    #' @field expr
    #' Get the expression structure for workflow service calls.
    expr = function(){
      NotImplementedError$new()
    }
  )
)
