# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/f14d70a2dc92ad4b15e2260ee9e01f24a7e0bee4/src/sagemaker/model_monitor/monitoring_files.py

#' @include utils.R
#' @include session.R
#' @include s3.R

#' @import R6
#' @import lgr
#' @import jsonlite
#' @import uuid

NO_SUCH_KEY_CODE = "NoSuchKey"

#' @title ModelMonitoringFile Class
#' @description Represents a file with a body and an S3 uri.
ModelMonitoringFile = R6Class("ModelMonitoringFile",
  public = list(

    #' @description Initializes a file with a body and an S3 uri.
    #' @param body_dict (str): The body of the JSON file.
    #' @param file_s3_uri (str): The uri of the JSON file.
    #' @param kms_key (str): The kms key to be used to decrypt the file in S3.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    initialize = function(body_dict,
                          file_s3_uri,
                          kms_key,
                          sagemaker_session){
      self$body_dict = body_dict
      self$file_s3_uri = file_s3_uri
      self$kms_key = kms_key
      self$session = sagemaker_session
    },

    #' @description Save the current instance's body to s3 using the instance's s3 path.
    #'              The S3 path can be overridden by providing one. This also overrides the
    #'              default save location for this object.
    #' @param new_save_location_s3_uri (str): Optional. The S3 path to save the file to. If not
    #'              provided, the file is saved in place in S3. If provided, the file's S3 path is
    #'              permanently updated.
    #' @return str: The s3 location to which the file was saved.
    save = function(new_save_location_s3_uri = NULL){
      if (!is.null(new_save_location_s3_uri)){
        self$file_s3_uri = new_save_location_s3_uri}

        return(S3Uploader$new()$upload_string_as_file_body(
          body=toJSON(self$body_dict, auto_unbox = T),
          desired_s3_uri=self$file_s3_uri,
          kms_key=self$kms_key,
          session=self$session))
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      print_class(self)
    }
  ),
  lock_objects = F
)

#' @title Statistics Class
#' @description Represents the statistics JSON file used in Amazon SageMaker Model Monitoring.
#' @export
Statistics = R6Class("Statistics",
  inherit = ModelMonitoringFile,
  public = list(

   #' @description Initializes the Statistics object used in Amazon SageMaker Model Monitoring.
   #' @param body_dict (str): The body of the statistics JSON file.
   #' @param statistics_file_s3_uri (str): The uri of the statistics JSON file.
   #' @param kms_key (str): The kms key to be used to decrypt the file in S3.
   #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
   #'              object, used for SageMaker interactions (default: None). If not
   #'              specified, one is created using the default AWS configuration
   #'              chain.
   initialize = function(body_dict =NULL,
                         statistics_file_s3_uri = NULL,
                         kms_key=NULL,
                         sagemaker_session=NULL){
     super$initialize(body_dict=body_dict,
                      file_s3_uri=statistics_file_s3_uri,
                      kms_key=kms_key,
                      sagemaker_session=sagemaker_session)
   },

   #' @description Generates a Statistics object from an s3 uri.
   #' @param statistics_file_s3_uri (str): The uri of the statistics JSON file.
   #' @param kms_key (str): The kms key to be used to decrypt the file in S3.
   #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
   #'              object, used for SageMaker interactions (default: None). If not
   #'              specified, one is created using the default AWS configuration
   #'              chain.
   #' @return sagemaker.model_monitor.Statistics: The instance of Statistics generated from
   #'              the s3 uri.
   from_s3_uri = function(statistics_file_s3_uri,
                          kms_key=NULL,
                          sagemaker_session=NULL){
     tryCatch({body = S3Downloader$new()$read_file(s3_uri=statistics_file_s3_uri, sagemaker_session=sagemaker_session)},
              error = function(e) {
                LOGGER$error(paste0( "\nCould not retrieve statistics file at location '%s'. ",
                                  "To manually retrieve Statistics object from a given uri, ",
                                  "use 'my_model_monitor.statistics(my_s3_uri)' or ",
                                  "'Statistics.from_s3_uri(my_s3_uri)'") , statistics_file_s3_uri)
                stop(e$message)})
     body_dict = fromJSON(body)

    cls = self$clone()
    cls$body_dict = body_dict
    cls$statistics_file_s3_uri = statistics_file_s3_uri
    cls$kms_key = kms_key
    return(cls)
   },

   #' @description Generates a Statistics object from an s3 uri.
   #' @param statistics_file_string (str): The uri of the statistics JSON file.
   #' @param kms_key (str): The kms key to be used to encrypt the file in S3.
   #' @param file_name (str): The file name to use when uploading to S3.
   #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
   #'              object, used for SageMaker interactions (default: None). If not
   #'              specified, one is created using the default AWS configuration
   #'              chain.
   #' @return sagemaker.model_monitor.Statistics: The instance of Statistics generated from
   #'              the s3 uri.
   from_string = function(statistics_file_string,
                          kms_key=NULL,
                          file_name=NULL,
                          sagemaker_session=NULL){
     sagemaker_session = sagemaker_session %||% Session$new()
     file_name = file_name %||% "statistics.json"
     desired_s3_uri = file.path(
       "s3:/", sagemaker_session$default_bucket(), "monitoring", UUIDgenerate(), file_name)
     s3_uri = S3Uploader$new()$upload_string_as_file_body(
       body=statistics_file_string,
       desired_s3_uri=desired_s3_uri,
       kms_key=kms_key,
       session=sagemaker_session)

     return (self$from_s3_uri(statistics_file_s3_uri=s3_uri, kms_key=kms_key, sagemaker_session=sagemaker_session))
   },

   #' @description Initializes a Statistics object from a file path.
   #' @param statistics_file_path (str): The path to the statistics file.
   #' @param kms_key (str): The kms_key to use when encrypting the file in S3.
   #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
   #'              object, used for SageMaker interactions (default: None). If not
   #'              specified, one is created using the default AWS configuration
   #'              chain.
   #' @return sagemaker.model_monitor.Statistics: The instance of Statistics generated from
   #'              the local file path.
   from_file_path = function(statistics_file_path,
                             kms_key=NULL,
                             sagemaker_session=NULL){
     file_name = basename(statistics_file_path)

     file_body = paste(readLines(file_name),collapse = "\n")

     return(Statistics$new()$from_string(
              statistics_file_string=file_body,
              file_name=file_name,
              kms_key=kms_key,
              sagemaker_session=sagemaker_session))
   }
  ),
  lock_objects = F
)


#' @title Constraints Class
#' @description Represents the constraints JSON file used in Amazon SageMaker Model Monitoring.
#' @export
Constraints = R6Class("Constraints",
  inherit = ModelMonitoringFile,
  public = list(

    #' @description Initializes the Constraints object used in Amazon SageMaker Model Monitoring.
    #' @param body_dict (str): The body of the constraints JSON file.
    #' @param constraints_file_s3_uri (str): The uri of the constraints JSON file.
    #' @param kms_key (str): The kms key to be used to decrypt the file in S3.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    initialize = function(body_dict = NULL,
                          constraints_file_s3_uri = NULL,
                          kms_key = NULL,
                          sagemaker_session = NULL){
      super$initialize(body_dict=body_dict,
                       file_s3_uri=constraints_file_s3_uri,
                       kms_key=kms_key,
                       sagemaker_session=sagemaker_session)
    },

    #' @description Generates a Constraints object from an s3 uri.
    #' @param constraints_file_s3_uri (str): The uri of the constraints JSON file.
    #' @param kms_key (str): The kms key to be used to decrypt the file in S3.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    #' @return sagemaker.model_monitor.Constraints: The instance of Constraints generated from
    #'              the s3 uri.
    from_s3_uri = function(constraints_file_s3_uri,
                           kms_key=NULL,
                           sagemaker_session=NULL){
      tryCatch({body = S3Downloader$new()$read_file(s3_uri=constraints_file_s3_uri, sagemaker_session=sagemaker_session)},
               error = function(e) {
                 LOGGER$error(paste0( "\nCould not retrieve statistics file at location '%s'. ",
                                   "To manually retrieve Statistics object from a given uri, ",
                                   "use 'my_model_monitor.statistics(my_s3_uri)' or ",
                                   "'Statistics.from_s3_uri(my_s3_uri)'") , constraints_file_s3_uri)
                 stop(e)})
      body_dict = fromJSON(body)

      cls = self$clone()
      cls$body_dict = body_dict
      cls$statistics_file_s3_uri = constraints_file_s3_uri
      cls$kms_key = kms_key
      return(cls)
    },

    #' @description Generates a Constraints object from an s3 uri.
    #' @param constraints_file_string (str): The uri of the constraints JSON file.
    #' @param kms_key (str): The kms key to be used to encrypt the file in S3.
    #' @param file_name (str): The file name to use when uploading to S3.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    #' @return sagemaker.model_monitor.Constraints: The instance of Constraints generated from
    #'              the s3 uri.
    from_string = function(constraints_file_string,
                           kms_key=NULL,
                           file_name=NULL,
                           sagemaker_session=NULL){
      sagemaker_session = sagemaker_session %||% Session$new()
      file_name = file_name %||% "constraints.json"
      desired_s3_uri = file.path(
        "s3://", sagemaker_session$default_bucket(), "monitoring", UUIDgenerate(), file_name)
      s3_uri = S3Uploader$new()$upload_string_as_file_body(
        body=constraints_file_string,
        desired_s3_uri=desired_s3_uri,
        kms_key=kms_key,
        session=sagemaker_session)

      return (self$from_s3_uri(constraints_file_s3_uri=s3_uri, kms_key=kms_key, sagemaker_session=sagemaker_session))
    },

    #' @description Initializes a Constraints object from a file path.
    #' @param constraints_file_path (str): The path to the constraints file.
    #' @param kms_key (str): The kms_key to use when encrypting the file in S3.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    #' @return sagemaker.model_monitor.Constraints: The instance of Constraints generated from
    #'              the local file path.
    from_file_path = function(constraints_file_path,
                              kms_key=NULL,
                              sagemaker_session=NULL){
      file_name = os.path.basename(constraints_file_path)

      file_body = paste(readLines(file_name),collapse = "\n")

      return(self$from_string(
        constraints_file_string=file_body,
        file_name=file_name,
        kms_key=kms_key,
        sagemaker_session=sagemaker_session))
    },

    #' @description Sets the monitoring flags on this Constraints object.
    #'              If feature-name is provided, modify the feature-level override.
    #'              Else, modify the top-level monitoring flag.
    #' @param enable_monitoring (bool): Whether to enable monitoring or not.
    #' @param feature_name (str): Sets the feature-level monitoring flag if provided. Otherwise,
    #'              sets the file-level override.
    set_monitoring = function(enable_monitoring,
                              feature_name=NULL){
      flag = ifelse(enable_monitoring, "Enabled", "Disabled")
      if (is.null(feature_name)){
        self$body_dict$monitoring_config$evaluate_constraints = flag
      } else {
        for (feature in self$body_dict$features){
          if (feature$name == feature_name){
            string_constraints = feature$string_constraints
            if (islistempty(string_constraints$monitoring_config_overrides)){
              string_constraints$monitoring_config_overrides = list()}
            string_constraints$monitoring_config_overrides$evaluate_constraints = flag
          }
        }
      }
    }
  ),
  lock_objects = F
)

#' @title ConstraintViolations
#' @description Represents the constraint violations JSON file used in Amazon SageMaker Model Monitoring.
#' @export
ConstraintViolations = R6Class("ConstraintViolations",
  inherit = ModelMonitoringFile,
  public = list(

   #' @description Initializes the ConstraintViolations object used in Amazon SageMaker Model Monitoring.
   #' @param body_dict (str): The body of the constraint violations JSON file.
   #' @param constraint_violations_file_s3_uri (str): The uri of the constraint violations JSON file.
   #' @param kms_key (str): The kms key to be used to decrypt the file in S3.
   #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
   #'             object, used for SageMaker interactions (default: None). If not
   #'             specified, one is created using the default AWS configuration
   #'             chain.
   initialize = function(body_dict=NULL,
                         constraint_violations_file_s3_uri=NULL,
                         kms_key=NULL,
                         sagemaker_session=NULL){
     super$initialize(body_dict=body_dict,
                      file_s3_uri=constraint_violations_file_s3_uri,
                      kms_key=kms_key,
                      sagemaker_session=sagemaker_session)
   },

   #' @description Generates a ConstraintViolations object from an s3 uri.
   #' @param constraint_violations_file_s3_uri (str): The uri of the constraint violations JSON file.
   #' @param kms_key (str): The kms key to be used to decrypt the file in S3.
   #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
   #'              object, used for SageMaker interactions (default: None). If not
   #'              specified, one is created using the default AWS configuration
   #'              chain.
   #' @return sagemaker.model_monitor.ConstraintViolations: The instance of ConstraintViolations
   #'              generated from the s3 uri.
   from_s3_uri = function(constraint_violations_file_s3_uri,
                          kms_key=NULL,
                          sagemaker_session=NULL){
     tryCatch({body = S3Downloader$new()$read_file(s3_uri=constraint_violations_file_s3_uri, sagemaker_session=sagemaker_session)},
              error = function(e) {
                LOGGER$error(paste0("\nCould not retrieve statistics file at location '%s'. ",
                                  "To manually retrieve Statistics object from a given uri, ",
                                  "use 'my_model_monitor.statistics(my_s3_uri)' or ",
                                  "'Statistics.from_s3_uri(my_s3_uri)'") , constraints_file_s3_uri)
                stop(e)})
     body_dict = fromJSON(body)

     cls = self$clone()
     cls$body_dict = body_dict
     cls$statistics_file_s3_uri = constraints_file_s3_uri
     cls$kms_key = kms_key
     return(cls)
   },

   #' @description Generates a ConstraintViolations object from an s3 uri.
   #' @param constraint_violations_file_string (str): The uri of the constraint violations JSON file.
   #' @param kms_key (str): The kms key to be used to encrypt the file in S3.
   #' @param file_name (str): The file name to use when uploading to S3.
   #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
   #'              object, used for SageMaker interactions (default: None). If not
   #'              specified, one is created using the default AWS configuration
   #'              chain.
   #' @return sagemaker.model_monitor.ConstraintViolations: The instance of ConstraintViolations
   #'              generated from the s3 uri.
   from_string = function(constraint_violations_file_string,
                          kms_key=NULL,
                          file_name=NULL,
                          sagemaker_session=NULL){
     sagemaker_session = sagemaker_session %||% Session$new()
     file_name = file_name %||% "constraint_violations.json"
     desired_s3_uri = file.path(
       "s3://", sagemaker_session$default_bucket(), "monitoring", UUIDgenerate(), file_name)
     s3_uri = S3Uploader$new()$upload_string_as_file_body(
       body=constraints_file_string,
       desired_s3_uri=desired_s3_uri,
       kms_key=kms_key,
       session=sagemaker_session)

     return (self$from_s3_uri(constraints_file_s3_uri=s3_uri, kms_key=kms_key, sagemaker_session=sagemaker_session))
   },

   #' @description Initializes a ConstraintViolations object from a file path.
   #' @param constraint_violations_file_path (str): The path to the constraint violations file.
   #' @param kms_key (str): The kms_key to use when encrypting the file in S3.
   #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
   #'              object, used for SageMaker interactions (default: None). If not
   #'              specified, one is created using the default AWS configuration
   #'              chain.
   #' @return sagemaker.model_monitor.ConstraintViolations: The instance of ConstraintViolations
   #'              generated from the local file path.
   from_file_path = function(constraint_violations_file_path,
                             kms_key=NULL,
                             sagemaker_session=NULL){
     file_name = basename(constraint_violations_file_path)

     file_body = paste(readLines(file_name),collapse = "\n")

     return(self$from_string(
       constraint_violations_file_string=file_body,
       file_name=file_name,
       kms_key=kms_key,
       sagemaker_session=sagemaker_session))
   }
  ),
  lock_objects = F
)
