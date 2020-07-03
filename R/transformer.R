# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/deed079c8f51555b497bc723bd5902ce2a0edf53/src/sagemaker/transformer.py

#' @include session.R
#' @include utils.R

#' @import R6

#' @title A class for handling creating and interacting with Amazon SageMaker
#'        transform jobs
#' @export
Transformer = R6Class("Transformer",
                      public = list(

                        #' @description Initialize a ``Transformer``.
                        #' @param model_name (str): Name of the SageMaker model being used for the
                        #'              transform job.
                        #' @param instance_count (int): Number of EC2 instances to use.
                        #' @param instance_type (str): Type of EC2 instance to use, for example,
                        #'              'ml.c4.xlarge'.
                        #' @param strategy (str): The strategy used to decide how to batch records in
                        #'              a single request (default: None). Valid values: 'MultiRecord'
                        #'              and 'SingleRecord'.
                        #' @param assemble_with (str): How the output is assembled (default: None).
                        #'              Valid values: 'Line' or 'None'.
                        #' @param output_path (str): S3 location for saving the transform result. If
                        #'              not specified, results are stored to a default bucket.
                        #' @param output_kms_key (str): Optional. KMS key ID for encrypting the
                        #'              transform output (default: None).
                        #' @param accept (str): The accept header passed by the client to
                        #'              the inference endpoint. If it is supported by the endpoint,
                        #'              it will be the format of the batch transform output.
                        #' @param max_concurrent_transforms (int): The maximum number of HTTP requests
                        #'              to be made to each individual transform container at one time.
                        #' @param max_payload (int): Maximum size of the payload in a single HTTP
                        #'              request to the container in MB.
                        #' @param tags (list[dict]): List of tags for labeling a transform job
                        #'              (default: None). For more, see the SageMaker API documentation for
                        #'              `Tag <https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html>`_.
                        #' @param env (dict): Environment variables to be set for use during the
                        #'              transform job (default: None).
                        #' @param base_transform_job_name (str): Prefix for the transform job when the
                        #'              :meth:`~sagemaker.transformer.Transformer.transform` method
                        #'              launches. If not specified, a default prefix will be generated
                        #'              based on the training image name that was used to train the
                        #'              model associated with the transform job.
                        #' @param sagemaker_session (sagemaker.session.Session): Session object which
                        #'              manages interactions with Amazon SageMaker APIs and any other
                        #'              AWS services needed. If not specified, the estimator creates one
                        #'              using the default AWS configuration chain.
                        #' @param volume_kms_key (str): Optional. KMS key ID for encrypting the volume
                        #'              attached to the ML compute instance (default: None).
                        initialize = function(model_name,
                                              instance_count,
                                              instance_type,
                                              strategy=NULL,
                                              assemble_with=NULL,
                                              output_path=NULL,
                                              output_kms_key=NULL,
                                              accept=NULL,
                                              max_concurrent_transforms=NULL,
                                              max_payload=NULL,
                                              tags=NULL,
                                              env=NULL,
                                              base_transform_job_name=NULL,
                                              sagemaker_session=NULL,
                                              volume_kms_key=NULL){

                          self$model_name = model_name
                          self$strategy = strategy
                          self$env = env

                          self$output_path = output_path
                          self$output_kms_key = output_kms_key
                          self$accept = accept
                          self$assemble_with = assemble_with

                          self$instance_count = instance_count
                          self$instance_type = instance_type
                          self$volume_kms_key = volume_kms_key

                          self$max_concurrent_transforms = max_concurrent_transforms
                          self$max_payload = max_payload
                          self$tags = tags

                          self$base_transform_job_name = base_transform_job_name
                          self$.current_job_name = NULL
                          self$latest_transform_job = NULL
                          self$.reset_output_path = FALSE

                          self$sagemaker_session = sagemaker_session %||% Session$new()
                        },

                        #' @description Start a new transform job.
                        #' @param data (str): Input data location in S3.
                        #' @param data_type (str): What the S3 location defines (default: 'S3Prefix').
                        #'              Valid values:
                        #'              \itemize{
                        #'                \item{\strong{'S3Prefix'} - the S3 URI defines a key name prefix. All objects with this prefix
                        #'              will be used as inputs for the transform job.}
                        #'                \item{\strong{'ManifestFile'} - the S3 URI points to a single manifest file listing each S3
                        #'              object to use as an input for the transform job.}}
                        #' @param content_type (str): MIME type of the input data (default: None).
                        #' @param compression_type (str): Compression type of the input data, if
                        #'              compressed (default: None). Valid values: 'Gzip', None.
                        #' @param split_type (str): The record delimiter for the input object
                        #'              (default: 'None'). Valid values: 'None', 'Line', 'RecordIO', and
                        #'              'TFRecord'.
                        #' @param job_name (str): job name (default: None). If not specified, one will
                        #'              be generated.
                        #' @param input_filter (str): A JSONPath to select a portion of the input to
                        #'              pass to the algorithm container for inference. If you omit the
                        #'              field, it gets the value '$', representing the entire input.
                        #'              For CSV data, each row is taken as a JSON array,
                        #'              so only index-based JSONPaths can be applied, e.g. $[0], $[1:].
                        #'              CSV data should follow the `RFC format <https://tools.ietf.org/html/rfc4180>`_.
                        #'              See `Supported JSONPath Operators
                        #'              <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html#data-processing-operators>`_
                        #'              for a table of supported JSONPath operators.
                        #'              For more information, see the SageMaker API documentation for
                        #'              `CreateTransformJob
                        #'              <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html>`_.
                        #'              Some examples: "$[1:]", "$.features" (default: None).
                        #' @param output_filter (str): A JSONPath to select a portion of the
                        #'              joined/original output to return as the output.
                        #'              For more information, see the SageMaker API documentation for
                        #'              `CreateTransformJob
                        #'              <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html>`_.
                        #'              Some examples: "$[1:]", "$.prediction" (default: None).
                        #' @param join_source (str): The source of data to be joined to the transform
                        #'              output. It can be set to 'Input' meaning the entire input record
                        #'              will be joined to the inference result. You can use OutputFilter
                        #'              to select the useful portion before uploading to S3. (default:
                        #'              None). Valid values: Input, None.
                        #' @param experiment_config (dict[str, str]): Experiment management configuration.
                        #'              Dictionary contains three optional keys,
                        #'              'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                        #'              (default: ``None``).
                        #' @param wait (bool): Whether the call should wait until the job completes
                        #'              (default: False).
                        #' @param logs (bool): Whether to show the logs produced by the job.
                        #'              Only meaningful when wait is True (default: False).
                        #' @return NULL invisible
                        transform = function(data,
                                             data_type="S3Prefix",
                                             content_type=NULL,
                                             compression_type=NULL,
                                             split_type=NULL,
                                             job_name=NULL,
                                             input_filter=NULL,
                                             output_filter=NULL,
                                             join_source=NULL,
                                             experiment_config=NULL,
                                             wait=FALSE,
                                             logs=FALSE){

                          # TODO: sagemaker local mode
                          # local_mode = self$sagemaker_session$local_mode
                          if (#is.null(local_mode) &&
                              !startsWith(data, "s3://"))
                            stop(sprintf("Invalid S3 URI: %s",data), call. = F)

                          if (!is.null(job_name)) {
                            self$.current_job_name = job_name
                          } else {
                            base_name = self$base_transform_job_name

                            if (is.null(base_name)){
                              base_name = private$.retrieve_base_name()

                              self$.current_job_name = name_from_base(base_name)}
                          }

                          if (is.null(self$output_path) %||% self$.reset_output_path){
                            self$output_path = sprintf("s3://%s/%s",
                                                       self$sagemaker_session$default_bucket(),
                                                       self$.current_job_name)
                            self$.reset_output_path = TRUE}


                          self$latest_transform_job = private$.start_new(data,
                                                                         data_type,
                                                                         content_type,
                                                                         compression_type,
                                                                         split_type,
                                                                         input_filter,
                                                                         output_filter,
                                                                         join_source,
                                                                         experiment_config)

                          if(wait) self$wait(self$latest_transform_job, logs=logs)
                          return(invisible(NULL))
                        },

                        #' @description Delete the corresponding SageMaker model for this Transformer.
                        delete_model = function(){
                          self$sagemaker_session$delete_model(self$model_name)
                        },

                        #' @description Wait for latest running batch transform job
                        #' @param logs return logs
                        wait = function(logs = TRUE){
                          private$.ensure_last_transform_job()
                          if (logs)
                            self$sagemaker_session$logs_for_transform_job(self$latest_transform_job, wait=TRUE)
                          else
                            self$sagemaker_session$wait_for_transform_job(self$latest_transform_job)
                        },

                        #' @description Stop latest running batch transform job.
                        #' @param wait wait for transform job
                        stop_transform_job = function(wait=TRUE){
                          self$.ensure_last_transform_job()
                          self$sagemaker_session$stop_transform_job(name=self$latest_transform)
                          self$stop(self$latest_transform)
                          if(wait)
                            self$wait()
                        },

                        #' @description Attach an existing transform job to a new Transformer instance
                        #' @param transform_job_name (str): Name for the transform job to be attached.
                        #' @param sagemaker_session (sagemaker.session.Session): Session object which
                        #'              manages interactions with Amazon SageMaker APIs and any other
                        #'              AWS services needed. If not specified, one will be created using
                        #'              the default AWS configuration chain.
                        #' @return Transformer (class): The Transformer instance with the
                        #'              specified transform job attached.
                        attach = function(transform_job_name,
                                          sagemaker_session= NULL){
                          sagemaker_session = sagemaker_session %||% Session$new()

                          job_details = sagemaker_session$sagemaker$describe_transform_job(
                                                    TransformJobName=transform_job_name)

                          init_params = private$.prepare_init_params_from_job_description(job_details)

                          # clone current class
                          transfomer = self$clone()

                          # update transformer class variables
                          transformer$model_name = init_params$model_name
                          transformer$strategy = init_params$strategy
                          transformer$output_path = init_params$output_path
                          transformer$output_kms_key = init_params$output_kms_key
                          transformer$accept = init_params$accept
                          transformer$assemble_with = init_params$assemble_with
                          transformer$instance_count = init_params$instance_count
                          transformer$instance_type = init_params$instance_type
                          transformer$volume_kms_key = init_params$volume_kms_key
                          transformer$max_concurrent_transforms = init_params$max_concurrent_transforms
                          transformer$max_payload = init_params$max_payload
                          transformer$base_transform_job_name = init_params$base_transform_job_name
                          transformer$latest_transform_job = init_params$base_transform_job_name
                          transformer$sagemaker_session = sagemaker_session
                          transformer$latest_transform_job = init_params$base_transform_job_name

                          return(transformer)
                        }
                      ),
                      private = list(
                        .retrieve_base_name = function(){
                          image_name = private$.retrieve_image_name()

                          if (!is.null(image_name))
                            return(base_name_from_image(image_name))

                          return(self$model_name)
                        },

                        .retrieve_image_name = function(){

                          tryCatch({model_desc = self$sagemaker_session$sagemaker$describe_model(
                                              ModelName=self$model_name)},
                                   error = function(e){
                                     stop(sprintf(
                                       "Failed to fetch model information for %s. ", self$model_name),
                                       "Please ensure that the model exists. ",
                                       "Local instance types require locally created models.")}
                                   )
                          primary_container = model_desc$PrimaryContainer
                          if (!is.null(primary_container) || length(primary_container) > 0)
                            return(primary_container$Image)

                          containers = model_desc$Containers
                          if (!is.null(containers) || length(containers) > 0)
                            return(containers[[1]]$Image)

                          return(NULL)
                        },

                        .ensure_last_transform_job = function(){
                          if (is.null(self$latest_transform_job))
                            stop("No transform job available", call. = F)
                        },

                        # Convert the transform job description to init params that can be
                        # handled by the class constructor
                        # Args:
                        #   job_details (dict): the returned job details from a
                        # describe_transform_job API call.
                        # Returns:
                        #   dict: The transformed init_params
                        .prepare_init_params_from_job_description = function(job_details){
                          init_params = list()

                          init_params[["model_name"]] = job_details$ModelName
                          init_params[["instance_count"]] = job_details$TransformResources$InstanceCount
                          init_params[["instance_type"]] = job_details$TransformResources$InstanceType
                          init_params[["volume_kms_key"]] = job_details$TransformResources$VolumeKmsKeyId
                          init_params[["strategy"]] = job_details$BatchStrategy
                          init_params[["assemble_with"]] = job_details$TransformOutput$AssembleWith
                          init_params[["output_path"]] = job_details$TransformOutput$S3OutputPath
                          init_params[["output_kms_key"]] = job_details$TransformOutput$KmsKeyId
                          init_params[["accept"]] = job_details$TransformOutput$Accept
                          init_params[["max_concurrent_transforms"]] = job_details$MaxConcurrentTransforms
                          init_params[["max_payload"]] = job_details$MaxPayloadInMB
                          init_params[["base_transform_job_name"]] = job_details$TransformJobName

                          return(init_params)
                        },

                        .start_new = function(data,
                                              data_type,
                                              content_type,
                                              compression_type,
                                              split_type,
                                              input_filter,
                                              output_filter,
                                              join_source,
                                              experiment_config){
                          config = private$.load_config(data, data_type, content_type, compression_type, split_type, transformer)

                          data_processing = private$.prepare_data_processing(input_filter, output_filter, join_source)

                          # start transform job
                          self$sagemaker_session$transform(
                            job_name=self$.current_job_name,
                            model_name=self$model_name,
                            strategy=self$strategy,
                            max_concurrent_transforms=self$max_concurrent_transforms,
                            max_payload=self$max_payload,
                            env=self$env,
                            input_config=config$input_config,
                            output_config=config$output_config,
                            resource_config=config$resource_config,
                            experiment_config=experiment_config,
                            tags=self$tags,
                            data_processing=data_processing)

                          # return current_job name
                          return(self$.current_job_name)
                        },

                         .load_config = function(data,
                                                 data_type,
                                                 content_type,
                                                 compression_type,
                                                 split_type,
                                                 transformer){
                           input_config = private$.format_inputs_to_input_config(
                             data, data_type, content_type, compression_type, split_type)

                           output_config = private$.prepare_output_config(self$output_path,
                                                                          self$output_kms_key,
                                                                          self$assemble_with,
                                                                          self$accept)

                           resource_config = private$.prepare_resource_config(
                             self$instance_count, self$instance_type, self$volume_kms_key
                           )

                           return (list("input_config"= input_config,
                                       "output_config"= output_config,
                                       "resource_config"= resource_config))
                         },
                        .format_inputs_to_input_config = function(data,
                                                                  data_type,
                                                                  content_type = NULL,
                                                                  compression_type = NULL,
                                                                  split_type = NULL){
                          config = list("DataSource" = list("S3DataSource" = list("S3DataType"= data_type, "S3Uri"= data)))

                          if (!is.null(content_type))
                            config["ContentType"] = content_type

                          if (!is.null(compression_type))
                            config["CompressionType"] = compression_type

                          if (!is.null(split_type))
                            config["SplitType"] = split_type

                          return(config)
                        },

                        .prepare_output_config = function(s3_path,
                                                          kms_key_id,
                                                          assemble_with = NULL,
                                                          accept = NULL){
                          config = list("S3OutputPath"= s3_path)
                          if (!is.null(kms_key_id))
                            config["KmsKeyId"] = kms_key_id


                          if (!is.null(assemble_with))
                            config["AssembleWith"] = assemble_with

                          if (!is.null(accept))
                            config["Accept"] = accept

                          return(config)
                        },

                        .prepare_resource_config = function(instance_count,
                                                            instance_type,
                                                            volume_kms_key = NULL){
                          config = list("InstanceCount"= instance_count, "InstanceType"= instance_type)

                          if (!is.null(volume_kms_key))
                            config["VolumeKmsKeyId"] = volume_kms_key

                            return(config)
                        },

                        .prepare_data_processing = function(input_filter = NULL,
                                                            output_filter = NULL,
                                                            join_source = NULL){
                          config = list()

                          if (!is.null(input_filter))
                            config["InputFilter"] = input_filter

                          if (!is.null(output_filter))
                            config["OutputFilter"] = output_filter

                          if (!is.null(join_source))
                            config["JoinSource"] = join_source

                          if (length(config))
                            return(NULL)

                          return(config)
                        }
                      ),
                      lock_objects = FALSE
                    )
