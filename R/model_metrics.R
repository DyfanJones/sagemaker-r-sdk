# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/model_metrics.py

#' @import R6
#' @import utils

#' @title ModelMetrics Class
#' @description Accepts model metrics parameters for conversion to request dict.
#' @export
ModelMetrics = R6Class("ModelMetrics",
  public = list(

    #' @description Initialize a ``ModelMetrics`` instance and turn parameters into dict.
    #' @param model_statistics (MetricsSource):
    #' @param model_constraints (MetricsSource):
    #' @param model_data_constraints (MetricsSource):
    #' @param model_data_statistics (MetricsSource):
    #' @param bias (MetricsSource):
    #' @param explainability (MetricsSource):
    initialize = function(model_statistics=NULL,
                          model_constraints=NULL,
                          model_data_statistics=NULL,
                          model_data_constraints=NULL,
                          bias=NULL,
                          explainability=NULL){
      self$model_statistics = model_statistics
      self$model_constraints = model_constraints
      self$model_data_statistics = model_data_statistics
      self$model_data_constraints = model_data_constraints
      self$bias = bias
      self$explainability = explainability
    },

    #' @description Generates a request dictionary using the parameters provided to the class.
    to_request_list = function(){
      model_metrics_request = list()

      model_quality = list()
      model_quality$Statistics = self$model_statistics$to_request_list()
      model_quality$Constraints = self$model_constraints$to_request_list()
      if(!islistempty(model_quality))
        model_metrics_request$ModelQuality = model_quality

      model_data_quality = list()
      model_data_quality$Statistics = self$model_data_statistics$to_request_list()
      model_data_quality$Constraints = self.model_data_constraints$to_request_list()
      if(!islistempty(model_data_quality))
        model_metrics_request$ModelDataQuality = model_data_quality

      model_metrics_request$Bias = self$bias$to_request_list()
      model_metrics_request$Explainability = self$explainability$to_request_list()
      return(model_metrics_request)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      return(print_class(self))
    }
  ),
  lock_objects = F
)

#' @title MetricsSource Class
#' @description Accepts metrics source parameters for conversion to request dict.
#' @export
MetricsSource = R6Class("MetricsSource",
  public = list(

    #' @description Initialize a ``MetricsSource`` instance and turn parameters into dict.
    #' @param content_type (str):
    #' @param s3_uri (str):
    #' @param content_digest (str):
    initialize = function(content_type,
                          s3_uri,
                          content_digest=NULL){
      self$content_type = content_type
      self$s3_uri = s3_uri
      self$content_digest = content_digest
    },

    #' @description Generates a request dictionary using the parameters provided to the class.
    to_request_list = function(){
      metrics_source_request = list("ContentType"= self$content_type, "S3Uri"= self$s3_uri)
      metrics_source_request$ContentDigest = self$content_digest
      return(metrics_source_request)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      return(print_class(self))
    }
  ),
  lock_objects = F
)
