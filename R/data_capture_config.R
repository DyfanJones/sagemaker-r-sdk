# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/session.py

#' @export
DataCaptureConfig = R6Class("DataCaptureConfig",
                            public = list(
                              enable_capture = TRUE,
                              sampling_percentage = NULL,
                              destination_s3_uri = NULL,
                              kms_key_id = NULL,
                              capture_options = NULL,
                              csv_content_types=NULL,
                              json_content_types=NULL,
                              sagemaker_session=NULL,
                              API_MAPPING = list("REQUEST"= "Input", "RESPONSE"= "Output"),
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
                              print = function(...){
                                cat("<DataCaptureConfig>")
                                invisible(self)
                                }
                              ),
                            private = list(
                              .MODEL_MONITOR_S3_PATH = "model-monitor",
                              .DATA_CAPTURE_S3_PATH = "data-capture"
                              ),
                            active = list(
                              to_request_list = function(){
                                request_list = list(
                                  "EnableCapture"= self$enable_capture,
                                  "InitialSamplingPercentage"= self$sampling_percentage,
                                  "DestinationS3Uri"= self$destination_s3_uri,
                                  "CaptureOptions"= lapply(capture_options, function(x) list("CaptureMode"= API_MAPPING[[toupper(x)]])) #  Convert to API values or pass value directly through if unable to convert.
                                )

                                request_list["KmsKeyId"] = self$kms_key_id

                                if (!is.null(self$csv_content_types))
                                  request_list[["CaptureContentTypeHeader"]][["CsvContentTypes"]] = list(self$csv_content_types)

                                if (!is.null(self$json_content_types))
                                  request_list[["CaptureContentTypeHeader"]][["JsonContentTypes"]] = list(self$json_content_types)

                                return(request_list)
                                }
                              )
                            )

