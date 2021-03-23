# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/session.py

#' @include s3.R
#' @include session.R

#' @import R6

#' @title DataCaptureConfig Class
#' @description Configuration object passed in when deploying models to Amazon SageMaker Endpoints.
#'              This object specifies configuration related to endpoint data capture for use with
#'              Amazon SageMaker Model Monitoring.
#' @export
DataCaptureConfig = R6Class("DataCaptureConfig",
  public = list(

    #' @field enable_capture
    #' Whether data capture should be enabled or not.
    enable_capture = TRUE,

    #' @field sampling_percentage
    #' The percentage of data to sample.
    sampling_percentage = NULL,

    #' @field destination_s3_uri
    #' Defaults to "s3://<default-session-bucket>/model-monitor/data-capture"
    destination_s3_uri = NULL,

    #' @field kms_key_id
    #' The kms key to use when writing to S3.
    kms_key_id = NULL,

    #' @field capture_options
    #' Denotes which data to capture between request and response.
    capture_options = NULL,

    #' @field csv_content_types
    #' Default=["text/csv"].
    csv_content_types=NULL,

    #' @field json_content_types
    #' Default=["application/json"].
    json_content_types=NULL,

    #' @field sagemaker_session
    #' A SageMaker Session object
    sagemaker_session=NULL,

    #' @field API_MAPPING
    #' Convert to API values or pass value directly through if unable to convert
    API_MAPPING = list("REQUEST"= "Input", "RESPONSE"= "Output"),

    #' @description Initialize a DataCaptureConfig object for capturing data from Amazon SageMaker Endpoints.
    #' @param enable_capture (bool): Required. Whether data capture should be enabled or not.
    #' @param sampling_percentage (int): Optional. Default=20. The percentage of data to sample.
    #'              Must be between 0 and 100.
    #' @param destination_s3_uri (str): Optional. Defaults to "s3://<default-session-bucket>/
    #'              model-monitor/data-capture".
    #' @param kms_key_id (str): Optional. Default=None. The kms key to use when writing to S3.
    #' @param capture_options ([str]): Optional. Must be a list containing any combination of the
    #'              following values: "REQUEST", "RESPONSE". Default=["REQUEST", "RESPONSE"]. Denotes
    #'              which data to capture between request and response.
    #' @param csv_content_types ([str]): Optional. Default=["text/csv"].
    #' @param json_content_types ([str]): Optional. Default=["application/json"].
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    initialize = function(enable_capture = TRUE,
                          sampling_percentage=20L,
                          destination_s3_uri=NULL,
                          kms_key_id=NULL,
                          capture_options=NULL,
                          csv_content_types=NULL,
                          json_content_types=NULL,
                          sagemaker_session=NULL){
      stopifnot(is.logical(enable_capture),
                is.integer(sampling_percentage),
                is.null(destination_s3_uri) || is.s3_uri(destination_s3_uri),
                is.null(kms_key_id) || is.character(kms_key_id),
                is.null(capture_options) || is.character(capture_options),
                is.null(csv_content_types) || is.character(csv_content_types),
                is.null(json_content_types) || is.character(json_content_types),
                is.null(sagemaker_session) || inherits(sagemaker_session, "Session"))

      self$enable_capture = enable_capture
      self$sampling_percentage = sampling_percentage
      self$destination_s3_uri = destination_s3_uri
      if (is.null(self$destination_s3_uri)){
        sagemaker_session = sagemaker_session %||% Session$new()
        self$destination_s3_uri = file.path("s3://", sagemaker_session$default_bucket(),
                                            private$.MODEL_MONITOR_S3_PATH, private$.DATA_CAPTURE_S3_PATH)}
      self$kms_key_id = kms_key_id
      self$capture_options = capture_options %||% c("REQUEST", "RESPONSE")
      self$csv_content_types = csv_content_types %||% "text/csv"
      self$json_content_types = json_content_types %||% "application/json"
    },

    #' @description Generates a request named list using the parameters provided to the class.
    to_request_list = function(){
      request_list = list(
        "EnableCapture"= self$enable_capture,
        "InitialSamplingPercentage"= self$sampling_percentage,
        "DestinationS3Uri"= self$destination_s3_uri,
        #  Convert to API values or pass value directly through if unable to convert.
        "CaptureOptions"= lapply(capture_options, function(x) list("CaptureMode"= self$API_MAPPING[[toupper(x)]]))
      )

      request_list["KmsKeyId"] = self$kms_key_id

      if (!is.null(self$csv_content_types))
        request_list[["CaptureContentTypeHeader"]][["CsvContentTypes"]] = list(self$csv_content_types)

      if (!is.null(self$json_content_types))
        request_list[["CaptureContentTypeHeader"]][["JsonContentTypes"]] = list(self$json_content_types)

      return(request_list)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      print_class(self)
    }
  ),
  private = list(
    .MODEL_MONITOR_S3_PATH = "model-monitor",
    .DATA_CAPTURE_S3_PATH = "data-capture"
  )
)
