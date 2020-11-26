# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/job.py

#' @include utils.R
#' @include session.R
#' @include input.R

#' @import R6

#' @title .Job Class
#' @description Handle creating, starting and waiting for Amazon SageMaker jobs to
#'              finish.
#' @note This class shouldn't be directly instantiated.
#'              Subclasses must define a way to create, start and wait for an Amazon
#'              SageMaker job.
.Job = R6Class(".Job",
  public = list(

    #' @field sagemaker_session
    #' Sagemaker Session Class
    sagemaker_session = NULL,

    #' @field job_name
    #' name of job
    job_name = NULL,

    #' @description Base class initializer. Subclasses which override ``__init__`` should
    #'              invoke ``super()``
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    #' @param job_name (str): Prefix for training job name
    initialize = function(sagemaker_session = NULL,
                          job_name = NULL){
      self$sagemaker_session = sagemaker_session
      self$job_name = job_name
    },

    #' @description Create a new Amazon SageMaker job from the estimator.
    #' @param estimator (sagemaker.estimator.EstimatorBase): Estimator object
    #'              created by the user.
    #' @param inputs (str): Parameters used when called
    #'              :meth:`~sagemaker.estimator.EstimatorBase.fit`.
    #' @return sagemaker.job: Constructed object that captures all information
    #'              about the started job.
    start_new = function(estimator,
                         inputs){
      stop("I'm an abstract interface method", call. = F)
    },

    #' @description Wait for the Amazon SageMaker job to finish.
    wait = function(){stop("I'm an abstract interface method", call. = F)},

    #' @description Describe the job.
    describe = function(){stop("I'm an abstract interface method", call. = F)},

    #' @description Stop the job.
    stop = function(){stop("I'm an abstract interface method", call. = F)},

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      return(print_class(self))
    }
  ),
  private = list(
    .load_config = function(inputs,
                           estimator,
                           expand_role=TRUE,
                           validate_uri=TRUE){
      input_config = private$.format_inputs_to_input_config(inputs, validate_uri)
      role = if (expand_role) estimator$sagemaker_session$expand_role(estimator$role) else estimator$role


      output_config = private$.prepare_output_config(estimator$output_path, estimator$output_kms_key)

      resource_config = private$.prepare_resource_config(
        estimator$instance_count,
        estimator$instance_type,
        estimator$volume_size,
        estimator$volume_kms_key)

      stop_condition = private$.prepare_stop_condition(
        estimator$max_run,
        estimator$max_wait)

      vpc_config = estimator$get_vpc_config()

      model_channel = private$.prepare_channel(
        input_config,
        estimator$model_uri,
        estimator$model_channel_name,
        validate_uri,
        content_type="application/x-sagemaker-model",
        input_mode="File")

      if (!islistempty(model_channel)){
        input_config = if (islistempty(input_config)) list() else input_config
        input_config = c(input_config, model_channel)
      }

      if (estimator$enable_network_isolation()){
        code_channel = private$.prepare_channel(input_config, estimator$code_uri, estimator$code_channel_name, validate_uri)
        if (islistempty(code_channel)) {
          input_config = if (islistempty(input_config)) list() else input_config
          input_config = c(input_config, code_channel)}
      }

      return (list("input_config"= input_config,
                   "role"= role,
                   "output_config"= output_config,
                   "resource_config"= resource_config,
                   "stop_condition"= stop_condition,
                   "vpc_config"= vpc_config))
    },

    .format_inputs_to_input_config = function(inputs = NULL,
                                              validate_uri=TRUE){
      if (is.null(inputs))
        return(NULL)

      if (inherits(inputs, c("RecordSet", "FileSystemRecordSet")))
        inputs = inputs$data_channel()

      input_dict = list()
      if (inherits(inputs, "character")){
        input_dict[["training"]] = private$.format_string_uri_input(inputs, validate_uri)
      } else if (is.s3_uri(inputs)){
        input_dict[["training"]] = inputs
      } else if (inherits(inputs, "file_input")){
        input_dict[["training"]] = inputs
      } else if (inherits(inputs, "list")){
        for (v in 1:length(inputs)){
        input_dict[[names(inputs)[v]]] = private$.format_string_uri_input(inputs[[v]], validate_uri)
        }
      # TODO: need to check how viable this is in R!!!!!!
      # } else if (inherits(inputs, "list")){
      #   input_dict = private$.format_record_set_list_input(inputs)
      } else if (inherits(inputs, "FileSystemInput")) {
        input_dict[["training"]] = inputs
      } else{
        msg = "Cannot format input %s. Expecting one of str, dict, TrainingInput or FileSystemInput"
        stop(sprintf(msg,inputs), call. = F)
      }

      channels = lapply(1:length(input_dict), function(x) {private$.convert_input_to_channel(names(input_dict)[x], input_dict[[x]])})

      return(channels)
    },

    .convert_input_to_channel = function(channel_name,
                                         channel_s3_input){
      channel_config = channel_s3_input$config
      channel_config$ChannelName = channel_name
      return(channel_config)
    },

    .prepare_output_config = function(s3_path, kms_key_id){
      config = list("S3OutputPath"= s3_path)

      if (!is.null(kms_key_id) || length(kms_key_id) == 0)
        config[["KmsKeyId"]] = kms_key_id
      return(config)
    },

    .format_string_uri_input = function(uri_input,
                                        validate_uri=TRUE,
                                        content_type=NULL,
                                        input_mode=NULL,
                                        compression=NULL,
                                        target_attribute_name=NULL){
      if (inherits(uri_input, "character") && validate_uri && startsWith(uri_input,"s3://")){
        s3_input_result = TrainingInput$new(uri_input,
                                   content_type=content_type,
                                   input_mode=input_mode,
                                   compression=compression,
                                   target_attribute_name=target_attribute_name)
        return(s3_input_result)
      }
      if (inherits(uri_input, "character") && validate_uri && startsWith(uri_input,"file://")){
        stop("local session is currently not supported") # TODO: create local session functionality.
        return (file_input(uri_input))}
      if (inherits(uri_input, "character") && validate_uri)
        stop(sprintf('URI input %s must be a valid S3 or FILE URI: must start with "s3://" or "file://"',uri_input),
             call. = F)
      if (inherits(uri_input, "character")){
          s3_input_result = TrainingInput$new(uri_input,
                                    content_type=content_type,
                                    input_mode=input_mode,
                                    compression=compression,
                                    target_attribute_name=target_attribute_name)
        return(s3_input_result)}
      if (inherits(uri_input, c("TrainingInput", "file_input", "FileSystemInput")))
          return(uri_input)


      stop(sprintf("Cannot format input %s. Expecting one of str, TrainingInput, file_input or FileSystemInput", uri_input),
           call. = F)
    },

    .prepare_channel = function(input_config,
                                channel_uri=NULL,
                                channel_name=NULL,
                                validate_uri=TRUE,
                                content_type=NULL,
                                input_mode=NULL){
      if (is.null(channel_uri))
        return(NULL)
      if (is.null(channel_name))
        stop(sprintf("Expected a channel name if a channel URI %s is specified",channel_uri))

      if (!is.null(input_config) || length(input_config) > 0){
        for (existing_channel in input_config){
          if (existing_channel[["ChannelName"]] == channel_name)
            stop(sprintf("Duplicate channel %s not allowed.",channel_name))
        }
      }

      channel_input = private$.format_string_uri_input(channel_uri, validate_uri, content_type, input_mode)
      channel = private$.convert_input_to_channel(channel_name, channel_input)

      return(channel)
    },

    .format_record_set_list_input = function(inputs){
      # Deferred import due to circular dependency
      #from sagemaker.amazon.amazon_estimator import FileSystemRecordSet, RecordSet

      input_dict = list()
      for (record in inputs){
        if (!inherits(record, c("RecordSet", "FileSystemRecordSet")))
          stop("List compatible only with RecordSets or FileSystemRecordSets.", call. = F)

        if (record$channel %in% names(input_dict))
          stop("Duplicate channels not allowed.", call. = F)
        if (inherits(record, "RecordSet"))
          input_dict[[record$channel]] = record$records_s3_input()
        if (inherits(record, "FileSystemRecordSet"))
          input_dict[[record$channel]] = record$file_system_input
      }
      return (input_dict)
    },

    .prepare_resource_config = function(instance_count,
                                        instance_type,
                                        volume_size,
                                        volume_kms_key = NULL){
      resource_config = list("InstanceCount"= instance_count,
                             "InstanceType"= instance_type,
                             "VolumeSizeInGB"= volume_size)

      if (!is.null(volume_kms_key) || length(volume_kms_key) == 0)
        resource_config[["VolumeKmsKeyId"]] = volume_kms_key

        return(resource_config)
    },

    .prepare_stop_condition = function(max_run,
                                       max_wait){
      if (!is.null(max_wait))
        return(list("MaxRuntimeInSeconds"= max_run, "MaxWaitTimeInSeconds"= max_wait))
      return(list("MaxRuntimeInSeconds"= max_run))
    }
  ),
  active = list(
    #' @field name
    #' Returns job name
    name = function(){
      return(self$job_name)
    }
  ),
  lock_objects = F
)
