# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/spark/processing.py

#' @include image_uris.R
#' @include processing.R
#' @include s3.R
#' @include session.R
#' @include utils.R

#' @import R6
#' @import logger
#' @importFrom jsonlite toJSON fromJSON
#' @importFrom urltools url_parse
#' @importFrom httr GET
#' @importFrom sys exec_background

#' @title Handles Amazon SageMaker processing tasks for jobs using Spark.
#' @description Base class for either PySpark or SparkJars.
#' @export
.SparkProcessorBase = R6Class(".SparkProcessorBase",
   inherit = ScriptProcessor,
   public = list(

     #' @description Initialize a ``_SparkProcessorBase`` instance.
     #'              The _SparkProcessorBase handles Amazon SageMaker processing tasks for
     #'              jobs using SageMaker Spark.
     #' @param role (str): An AWS IAM role name or ARN. The Amazon SageMaker training jobs
     #'              and APIs that create Amazon SageMaker endpoints use this role
     #'              to access training data and model artifacts. After the endpoint
     #'              is created, the inference code might use the IAM role, if it
     #'              needs to access an AWS resource.
     #' @param instance_type (str): Type of EC2 instance to use for
     #'              processing, for example, 'ml.c4.xlarge'.
     #' @param instance_count (int): The number of instances to run
     #'              the Processing job with. Defaults to 1.
     #' @param framework_version (str): The version of SageMaker PySpark.
     #' @param py_version (str): The version of python.
     #' @param container_version (str): The version of spark container.
     #' @param image_uri (str): The container image to use for training.
     #' @param volume_size_in_gb (int): Size in GB of the EBS volume to
     #'              use for storing data during processing (default: 30).
     #' @param volume_kms_key (str): A KMS key for the processing
     #'              volume.
     #' @param output_kms_key (str): The KMS key id for all ProcessingOutputs.
     #' @param max_runtime_in_seconds (int): Timeout in seconds.
     #'              After this amount of time Amazon SageMaker terminates the job
     #'              regardless of its current status.
     #' @param base_job_name (str): Prefix for processing name. If not specified,
     #'              the processor generates a default job name, based on the
     #'              training image name and current timestamp.
     #' @param sagemaker_session (sagemaker.session.Session): Session object which
     #'              manages interactions with Amazon
     #'              SageMaker APIs and any other AWS services needed. If not specified,
     #'              the processor creates one using the default AWS configuration chain.
     #' @param env (dict): Environment variables to be passed to the processing job.
     #' @param tags ([dict]): List of tags to be passed to the processing job.
     #'              network_config (sagemaker.network.NetworkConfig): A NetworkConfig
     #'              object that configures network isolation, encryption of
     #'              inter-container traffic, security group IDs, and subnets.
     #' @param network_config (sagemaker.network.NetworkConfig): A NetworkConfig
     #'              object that configures network isolation, encryption of
     #'              inter-container traffic, security group IDs, and subnets.
     initialize = function(role,
                           instance_type,
                           instance_count,
                           framework_version=NULL,
                           py_version=NULL,
                           container_version=NULL,
                           image_uri=NULL,
                           volume_size_in_gb=30,
                           volume_kms_key=NULL,
                           output_kms_key=NULL,
                           max_runtime_in_seconds=NULL,
                           base_job_name=NULL,
                           sagemaker_session=NULL,
                           env=NULL,
                           tags=NULL,
                           network_config=NULL){
       self$history_server = NULL
       private$.spark_event_logs_s3_uri = NULL
       session = sagemaker_session %||% Session$new()
       region = session$paws_region_name

       self$image_uri = private$.retrieve_image_uri(
         image_uri, framework_version, py_version, container_version, region, instance_type
       )

       env = env %||% list()
       command = list(private$.default_command)

       super$initialize(
         role=role,
         image_uri=self$image_uri,
         instance_count=instance_count,
         instance_type=instance_type,
         command=command,
         volume_size_in_gb=volume_size_in_gb,
         volume_kms_key=volume_kms_key,
         output_kms_key=output_kms_key,
         max_runtime_in_seconds=max_runtime_in_seconds,
         base_job_name=base_job_name,
         sagemaker_session=session,
         env=env,
         tags=tags,
         network_config=network_config)
     },

     #' @description Runs a processing job.
     #' @param submit_app (str): .py or .jar file to submit to Spark as the primary application
     #' @param inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
     #'              the processing job. These must be provided as
     #'              :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
     #' @param outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
     #'              the processing job. These can be specified as either path strings or
     #'              :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
     #' @param arguments (list[str]): A list of string arguments to be passed to a
     #'              processing job (default: None).
     #' @param wait (bool): Whether the call should wait until the job completes (default: True).
     #' @param logs (bool): Whether to show the logs produced by the job.
     #'              Only meaningful when wait is True (default: True).
     #' @param job_name (str): Processing job name. If not specified, the processor generates
     #'              a default job name, based on the base job name and current timestamp.
     #' @param experiment_config (dict[str, str]): Experiment management configuration.
     #'              Dictionary contains three optional keys:
     #'              'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
     #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
     #'              user code file (default: None).
     run = function(submit_app,
                    inputs=NULL,
                    outputs=NULL,
                    arguments=NULL,
                    wait=TRUE,
                    logs=TRUE,
                    job_name=NULL,
                    experiment_config=NULL,
                    kms_key=NULL){
       self$.current_job_name = private$.generate_current_job_name(job_name=job_name)

       super$run(submit_app,
                 inputs,
                 outputs,
                 arguments,
                 wait,
                 logs,
                 job_name,
                 experiment_config,
                 kms_key)
     },

     #' @description Starts a Spark history server.
     #' @param spark_event_logs_s3_uri (str): S3 URI where Spark events are stored.
     start_history = function(spark_event_logs_s3_uri=NULL){
       # TODO: .ecr_login_if_needed function
       if (.ecr_login_if_needed(self$sagemaker_session$paws_session, self$image_uri)){
         log_info("Pulling spark history server image...")
         # TODO: .pull_image function
         .pull_image(self$image_uri)}

       history_server_env_variables = private$.prepare_history_server_env_variables(
         spark_event_logs_s3_uri
       )

       self$history_server = .HistoryServer$new(
         history_server_env_variables, self$image_uri, private$.get_network_config()
       )
       self$history_server$run()
       private$.check_history_server()
     },

     #' @description Terminates the Spark history server.
     terminate_history_server = function(){
       if (!islistempty(self$history_server)){
         log_info("History server is running, terminating history server")
         self$history_server$down()
         self$history_server = NULL}
     }
   ),
   private = list(
     .default_command = "smspark-submit",
     .conf_container_base_path = "/opt/ml/processing/input/",
     .conf_container_input_name = "conf",
     .conf_file_name = "configuration.json",
     .valid_configuration_keys = list("Classification", "Properties", "Configurations"),
     .valid_configuration_classifications = list(
       "core-site",
       "hadoop-env",
       "hadoop-log4j",
       "hive-env",
       "hive-log4j",
       "hive-exec-log4j",
       "hive-site",
       "spark-defaults",
       "spark-env",
       "spark-log4j",
       "spark-hive-site",
       "spark-metrics",
       "yarn-env",
       "yarn-site",
       "export"),
     .submit_jars_input_channel_name = "jars",
     .submit_files_input_channel_name = "files",
     .submit_py_files_input_channel_name = "py-files",
     .submit_deps_error_message = paste(
       "Please specify a list of one or more S3 URIs,",
       "local file paths, and/or local directory paths"),
     # history server vars
     .history_server_port = "15050",
     .history_server_url_suffix = sprintf("/proxy/%s", "15050"),
     .spark_event_log_default_local_path = "/opt/ml/processing/spark-events/",

     # Extends processing job args such as inputs.
     .extend_processing_args = function(inputs,
                                        outputs,
                                        ...){
       kwargs = list(...)
       if (!islistempty(kwargs$spark_event_logs_s3_uri)) {
         spark_event_logs_s3_uri = kwargs$spark_event_logs_s3_uri
         private$.validate_s3_uri(spark_event_logs_s3_uri)

         private$.spark_event_logs_s3_uri = spark_event_logs_s3_uri
         self$command = c(self$command,
                          list("--local-spark-event-logs-dir",
                          .SparkProcessorBase$private_fields$.spark_event_log_default_local_path))

         output = ProcessingOutput$new(
           source=.SparkProcessorBase$private_fields$.spark_event_log_default_local_path,
           destination=spark_event_logs_s3_uri,
           s3_upload_mode="Continuous")

         outputs = outputs %||% list()
         outputs = c(outputs, output)

         if (!islistempty(kwargs$configuration)){
           configuration = kwargs$configuration
           private$.validate_configuration(configuration)
           inputs = inputs %||% list()
           inputs = c(inputs, self$.stage_configuration(configuration))
         }
       }
       return(list("Inputs" = inputs, "Outputs" = outputs))
   },
   
   # Builds an image URI.
   .retrieve_image_uri = function(image_uri = NULL,
                                  framework_version = NULL,
                                  py_version = NULL,
                                  container_version = NULL,
                                  region = NULL,
                                  instance_type = NULL){
      if (!is.null(image_uri)){
         if (is.null(py_version) || is.null(container_version))
            stop(
               "Both or neither of py_version and container_version should be set",
               call. = F)
      
         if (!is.null(container_version))
            container_version = sprintf("v%s", container_version)
            
            return(ImageUris$new()$retrieve(
               "spark",
               region,
               version=framework_version,
               instance_type=instance_type,
               py_version=py_version,
               container_version=container_version)
            )
      }
      return(image_uri)
   },
   
   # Validates the user-provided Hadoop/Spark/Hive configuration.
   # This ensures that the list or dictionary the user provides will serialize to
   # JSON matching the schema of EMR's application configuration:
   #      https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html
   .validate_configuration = function(configuration){
      if(inherits(configuration, "list")){
         keys = names(configuration)
         if (!("Classification" %in% keys) || !("Properties" %in% keys))
            stop("Missing one or more required keys in configuration dictionary ",
                 sprintf("%s Please see %s for more information", configuration, emr_configure_apps_url),
                 call. = F)
         
         for (key in keys){
            if (!(key %in% private$.valid_configuration_keys))
               stop(sprintf("Invalid key: %s. Must be one of %s. ", key, toJSON(private$.valid_configuration_keys, auto_unbox = T)),
                    sprintf("Please see %sfor more information.", emr_configure_apps_url),
                    call. = F)
            if (key == "Classification")
               if (!(configuration[[key]] %in% private$.valid_configuration_classifications))
                  stop(sprintf("Invalid classification: %s. Must be one of %s", key,
                       toJSON(private$.valid_configuration_classifications, auto_unbox = T)),
                       call. = F)
         }
      }
      
      # if list is unnamed check components
      if (is.null(names(configuration))){
         for(item in configuration)
            private$.validate_configuration(item)
      }
   },
   
   # Serializes and uploads the user-provided EMR application configuration to S3.
   # This method prepares an input channel.
   # Args:
   #    configuration (Dict): the configuration dict for the EMR application configuration.
   .stage_configuration = function(configuration){
      serialized_configuration = toJSON(configuration, auto_unbox = T)
      s3_uri = sprintf("s3://%s/%s/input/%s/%s", 
                       self$sagemaker_session$default_bucket(), 
                       self$.current_job_name,
                       private$.conf_container_input_name,
                       private$.conf_file_name)
      
      S3Uploader$new()$upload_string_as_file_body(
         body=serialized_configuration,
         desired_s3_uri=s3_uri,
         sagemaker_session=self$sagemaker_session)
      
      conf_input = ProcessingInput$new(
         source=s3_uri,
         destination=sprintf("%s%s", private$.conf_container_base_path, private$.conf_container_input_name),
         input_name=private$.conf_container_input_name)
      return(conf_input)
   },
   
   # Prepares a list of paths to jars, py-files, or files dependencies.
   # This prepared list of paths is provided as `spark-submit` options.
   # The submit_deps list may include a combination of S3 URIs and local paths.
   # Any S3 URIs are appended to the `spark-submit` option value without modification.
   # Any local file paths are copied to a temp directory, uploaded to a default S3 URI,
   # and included as a ProcessingInput channel to provide as local files to the SageMaker
   # Spark container.
   # :param submit_deps (list[str]): List of one or more dependency paths to include.
   # :param input_channel_name (str): The `spark-submit` option name associated with
   # the input channel.
   # :return (Optional[ProcessingInput], str): Tuple of (left) optional ProcessingInput
   # for the input channel, and (right) comma-delimited value for
   # `spark-submit` option.
   .stage_submit_deps = function(submit_deps = NULL,
                                 input_channel_name = NULL){
      if (is.null(submit_deps))
         stop(sprintf("submit_deps value may not be empty. %s",private$.submit_deps_error_message),
              call. = F)
      if (is.null(input_channel_name))
         stop("input_channel_name value may not be empty.", call.= F)
      
      input_channel_s3_uri = sprintf("s3://%s/%s/input/%s", self$sagemaker_session$default_bucket(), self$.current_job_name, input_channel_name)
      
      use_input_channel = FALSE
      spark_opt_s3_uris = list()
      
      tmpdir = tempdir()
      
      for (dep_path in submit_deps){
         dep_url = url_parse(dep_path)
         # S3 URIs are included as-is in the spark-submit argument
         if (dep_url$scheme %in% c("s3", "s3a")){
            spark_opt_s3_uris = c(spark_opt_s3_uris, dep_path)
         # Local files are copied to temp directory to be uploaded to S3
         } else if (is.null(dep_url$scheme) || dep_url$scheme == "file") {
            if (!file_test("-f", dep_path)){
               stop(sprintf("submit_deps path %s is not a valid local file. %s", dep_path, private$.submit_deps_error_message),
                    call. = F)
            log_info("Copying dependency from local path %s to tmpdir %s", dep_path, tmpdir)
            file.copy(dep_path, tmpdir)
            }
         } else {
            stop(sprintf("submit_deps path %s references unsupported filesystem ", dep_path),
                 sprintf("scheme: %s %s", dep_url$scheme, private$.submit_deps_error_message)
            )
         }
      }
      
      # If any local files were found and copied, upload the temp directory to S3
      if (!islistempty(list.dirs(tmpdir)))
         log_info("Uploading dependencies from tmpdir %s to S3 %s", tmpdir, input_channel_s3_uri)
         S3Uploader$new()$upload(
            local_path=tmpdir,
            desired_s3_uri=input_channel_s3_uri,
            sagemaker_session=self$sagemaker_session)
         use_input_channel = TRUE
         
      # If any local files were uploaded, construct a ProcessingInput to provide
      # them to the Spark container  and form the spark-submit option from a
      # combination of S3 URIs and container's local input path
      if (use_input_channel) {
         input_channel = ProcessingInput$new(
            source=input_channel_s3_uri,
            destination=sprintf("%s%s", private$.conf_container_base_path, input_channel_name),
            input_name=input_channel_name)
         spark_opt = paste(c(spark_opt_s3_uris, input_channel.destination), collapse = ",")
      # If no local files were uploaded, form the spark-submit option from a list of S3 URIs
      } else {
         input_channel = NULL
         spark_opt = paste(spark_opt_s3_uris, collapse = ",")
      }
      
      return(list("InputChannel" = input_channel, "SparkOpt" = spark_opt))
   },
   
   # Runs container with different network config based on different env.
   .get_network_config = function(){
      if (private$.is_notebook_instance())
         return ("--network host")
      
      return(sprintf("-p 80:80 -p %s:%s", private$.history_server_port, private$.history_server_port))
   },
   
   # Gets all parameters required to run history server.
   .prepare_history_server_env_variables = function(spark_event_logs_s3_uri = NULL){
      # prepare env varibles
      history_server_env_variables = list()
      
      if (!is.null(spark_event_logs_s3_uri)){
         history_server_env_variables[[
            .HistoryServer$new()$arg_event_logs_s3_uri
            ]] = spark_event_logs_s3_uri
      # this variable will be previously set by run() method
      } else if (!is.null(private$.spark_event_logs_s3_uri)){
         history_server_env_variables[[
            .HistoryServer$new()$arg_event_logs_s3_uri
            ]] = private$.spark_event_logs_s3_uri
      } else {
         stop(
            "SPARK_EVENT_LOGS_S3_URI not present. You can specify spark_event_logs_s3_uri ",
            "either in run() or start_history_server()",
            call. = F)
      }
         
      history_server_env_variables = c(history_server_env_variables, private$.config_aws_credentials())
      region = self$sagemaker_session$paws_region_name
      history_server_env_variables[["AWS_REGION"]] = region
      
      if (private$.is_notebook_instance())
         history_server_env_variables[[
            .HistoryServer$new()$arg_remote_domain_name
            ]] = private$.get_notebook_instance_domain()
      
      return(history_server_env_variables)
   },
   
   # Determine whether it is a notebook instance.
   .is_notebook_instance = function(){
      return (file_test("-f","/opt/ml/metadata/resource-metadata.json"))
   },

   # Get the instance's domain.   
   .get_notebook_instance_domain = function(){
      region = self$sagemaker_session$paws_region_name
      data = fromJSON("/opt/ml/metadata/resource-metadata.json")
      notebook_name = data$ResourceName
      
      return(sprintf("https://%s.notebook.%s.sagemaker.aws", notebook_name, region))
   },
   
   # Print message indicating the status of history server.
   # Pings port 15050 to check whether the history server is up.
   # Times out after `ping_timeout`.
   # Args:
   #    ping_timeout (int): Timeout in seconds (defaults to 40).
   .check_history_server = function(ping_timeout = 40){
      # ping port 15050 to check history server is up
      timeout = Sys.time() + ping_timeout
      
      while(TRUE){
         if (private$.is_history_server_started()){
            if (private$.is_notebook_instance()){
               log_info(
                  "History server is up on %s%s",
                  private$.get_notebook_instance_domain(),
                  private$.history_server_url_suffix,
               )
            } else {
               log_info(
                  "History server is up on http://0.0.0.0%s", private$.history_server_url_suffix)
            }
            break
         }
         if (Sys.time() > timeout){
            log_error(
               "History server failed to start. Please run 'docker logs history_server' to see logs")
            break
         }
         Sys.sleep(1)
      }
   },
   
   # Check if history server is started.
   .is_history_server_started = function(){
      tryCatch({response = httr::GET(sprintf("http://localhost:%s", private$.history_server_port))
      return (response$status_code == 200)},
      error = function(){
         return(FALSE)}
      )
   },
   
   # TODO (note from sagemaker-v2): method only checks urlparse scheme, need to perform deep s3 validation
   # Validate whether the URI uses an S3 scheme.
   # In the future, this validation will perform deeper S3 validation.
   # Args:
   #    spark_output_s3_path (str): The URI of the Spark output S3 Path.
   .validate_s3_uri = function(spark_output_s3_path){
      if (url_parse(spark_output_s3_path)$scheme != "s3")
         stop(sprintf("Invalid s3 path: %s. Please enter something like ", spark_output_s3_path),
            "s3://bucket-name/folder-name",
            call. = F)
   },

   # Configure AWS credentials.
   .config_aws_credentials = function(){
      tryCatch({
         creds = self$sagemaker_session$paws_credentials
         access_key = creds$aws_access_key_id
         secret_key = creds$aws_secret_access_key
         token = creds$aws_session_token
         
         return(list("AWS_ACCESS_KEY_ID"= access_key,
            "AWS_SECRET_ACCESS_KEY"= secret_key,
            "AWS_SESSION_TOKEN" = token))},
         error = function(){
            return(list())
         })
   },
   
   # Handle script dependencies
   # The method extends inputs and command based on input files and file_type
   .handle_script_dependencies = function(inputs,
                                          submit_files = NULL,
                                          file_type){
      
      if (is.null(submit_files))
         return(inputs)
      
      input_channel_name_dict = list(
         private$.submit_jars_input_channel_name,
         private$.submit_py_files_input_channel_name,
         private$.submit_files_input_channel_name)
      
      spark_files  = private$.stage_submit_deps(
         submit_files, input_channel_name_dict[[file_type]]
      )
      
      inputs = inputs %||% list()
      
      if (!islistempty(spark_files$InputChannel))
         inputs = c(inputs, spark_files$InputChannel)
      
      if (!islistempty(spark_files$SparkOpt))
         self$command = c(self$command, c(sprintf("--%s", input_channel_name_dict[[file_type]]), spark_files$SparkOpt))
      
      return(inputs)
   }
   ),
   lock_objects = F
)

#' @title PySparkProcessor Class
#' @description Handles Amazon SageMaker processing tasks for jobs using PySpark.
#' @export
PySparkProcessor = R6Class("PySparkProcessor",
   inherit = .SparkProcessorBase,
   public = list(
      
      #' @description Initialize an ``PySparkProcessor`` instance.
      #'              The PySparkProcessor handles Amazon SageMaker processing tasks for jobs
      #'              using SageMaker PySpark.
      #' @param role (str): An AWS IAM role name or ARN. The Amazon SageMaker training jobs
      #'              and APIs that create Amazon SageMaker endpoints use this role
      #'              to access training data and model artifacts. After the endpoint
      #'              is created, the inference code might use the IAM role, if it
      #'              needs to access an AWS resource.
      #' @param instance_type (str): Type of EC2 instance to use for
      #'              processing, for example, 'ml.c4.xlarge'.
      #' @param instance_count (int): The number of instances to run
      #'              the Processing job with. Defaults to 1.
      #' @param framework_version (str): The version of SageMaker PySpark.
      #' @param py_version (str): The version of python.
      #' @param container_version (str): The version of spark container.
      #' @param image_uri (str): The container image to use for training.
      #' @param volume_size_in_gb (int): Size in GB of the EBS volume to
      #'              use for storing data during processing (default: 30).
      #' @param volume_kms_key (str): A KMS key for the processing
      #'              volume.
      #' @param output_kms_key (str): The KMS key id for all ProcessingOutputs.
      #' @param max_runtime_in_seconds (int): Timeout in seconds.
      #'              After this amount of time Amazon SageMaker terminates the job
      #'              regardless of its current status.
      #' @param base_job_name (str): Prefix for processing name. If not specified,
      #'              the processor generates a default job name, based on the
      #'              training image name and current timestamp.
      #' @param sagemaker_session (sagemaker.session.Session): Session object which
      #'              manages interactions with Amazon SageMaker APIs and any other
      #'              AWS services needed. If not specified, the processor creates one
      #'              using the default AWS configuration chain.
      #' @param env (dict): Environment variables to be passed to the processing job.
      #' @param tags ([dict]): List of tags to be passed to the processing job.
      #' @param network_config (sagemaker.network.NetworkConfig): A NetworkConfig
      #'              object that configures network isolation, encryption of
      #'              inter-container traffic, security group IDs, and subnets.
      initialize = function(role,
                            instance_type,
                            instance_count,
                            framework_version=NULL,
                            py_version=NULL,
                            container_version=NULL,
                            image_uri=NULL,
                            volume_size_in_gb=30,
                            volume_kms_key=NULL,
                            output_kms_key=NULL,
                            max_runtime_in_seconds=NULL,
                            base_job_name=NULL,
                            sagemaker_session=NULL,
                            env=NULL,
                            tags=NULL,
                            network_config=NULL){
         super$initialize(
            role=role,
            instance_count=instance_count,
            instance_type=instance_type,
            framework_version=framework_version,
            py_version=py_version,
            container_version=container_version,
            image_uri=image_uri,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config)
      },
      
      #' @description Runs a processing job.
      #' @param submit_app (str): Path (local or S3) to Python file to submit to Spark
      #'              as the primary application
      #' @param submit_py_files (list[str]): List of paths (local or S3) to provide for
      #'              `spark-submit --py-files` option
      #' @param submit_jars (list[str]): List of paths (local or S3) to provide for
      #'              `spark-submit --jars` option
      #' @param submit_files (list[str]): List of paths (local or S3) to provide for
      #'              `spark-submit --files` option
      #' @param inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
      #'              the processing job. These must be provided as
      #'              :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
      #' @param outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
      #'              the processing job. These can be specified as either path strings or
      #'              :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
      #' @param arguments (list[str]): A list of string arguments to be passed to a
      #'              processing job (default: None).
      #' @param wait (bool): Whether the call should wait until the job completes (default: True).
      #' @param logs (bool): Whether to show the logs produced by the job.
      #'              Only meaningful when wait is True (default: True).
      #' @param job_name (str): Processing job name. If not specified, the processor generates
      #'              a default job name, based on the base job name and current timestamp.
      #' @param experiment_config (dict[str, str]): Experiment management configuration.
      #'              Dictionary contains three optional keys:
      #'              'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
      #' @param configuration (list[dict] or dict): Configuration for Hadoop, Spark, or Hive.
      #'              List or dictionary of EMR-style classifications.
      #'              https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html
      #' @param spark_event_logs_s3_uri (str): S3 path where spark application events will
      #'              be published to.
      #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
      #'              user code file (default: None).
      run = function(submit_app,
                     submit_py_files=NULL,
                     submit_jars=NULL,
                     submit_files=NULL,
                     inputs=NULL,
                     outputs=NULL,
                     arguments=NULL,
                     wait=TRUE,
                     logs=TRUE,
                     job_name=NULL,
                     experiment_config=NULL,
                     configuration=NULL,
                     spark_event_logs_s3_uri=NULL,
                     kms_key=NULL){
         self$.current_job_name = private$.generate_current_job_name(job_name=job_name)
         self$command = list(.SparkProcessorBase$private_fields$.default_command)
         
         if (missing(submit_app))
            stop("submit_app is required", call. = F)
         
         extended_args = private$.extend_processing_args(
            inputs,
            outputs,
            submit_py_files=submit_py_files,
            submit_jars=submit_jars,
            submit_files=submit_files,
            configuration=configuration,
            spark_event_logs_s3_uri=spark_event_logs_s3_uri
         )
         
         super$run(
            submit_app=submit_app,
            inputs=extended_args$Inputs,
            outputs=extended_args$Outputs,
            arguments=arguments,
            wait=wait,
            logs=logs,
            job_name=self$.current_job_name,
            experiment_config=experiment_config,
         )
      }
   ),
   private = list(
      
      # Extends inputs and outputs.
      # Args:
      #    inputs: Processing inputs.
      # outputs: Processing outputs.
      # kwargs: Additional keyword arguments passed to `super()`.
      .extend_processing_args = function(inputs,
                                         outputs,
                                         ...){
         kwargs = list(...)
         extended_inputs = private$.handle_script_dependencies(
            inputs, kwargs$submit_py_files, FileType$new()$PYTHON
         )
         extended_inputs = private$.handle_script_dependencies(
            extended_inputs, kwargs$submit_jars, FileType$new()$JAR
         )
         extended_inputs = private$.handle_script_dependencies(
            extended_inputs, kwargs$submit_files, FileType$new()$FILE
         )
         
         return(super$.extend_processing_args(extended_inputs, outputs, ...))
      }
   ),
   lock_objects =  F
)

#' @title SparkJarProcessor Class
#' @description Handles Amazon SageMaker processing tasks for jobs using Spark with Java or Scala Jars.
#' @export
SparkJarProcessor = R6Class("SparkJarProcessor",
   inherit = .SparkProcessorBase,
   public = list(
      
      #' @description Initialize a ``SparkJarProcessor`` instance.
      #'              The SparkProcessor handles Amazon SageMaker processing tasks for jobs
      #'              using SageMaker Spark.
      #' @param role (str): An AWS IAM role name or ARN. The Amazon SageMaker training jobs
      #'              and APIs that create Amazon SageMaker endpoints use this role
      #'              to access training data and model artifacts. After the endpoint
      #'              is created, the inference code might use the IAM role, if it
      #'              needs to access an AWS resource.
      #' @param instance_type (str): Type of EC2 instance to use for
      #'              processing, for example, 'ml.c4.xlarge'.
      #' @param instance_count (int): The number of instances to run
      #'              the Processing job with. Defaults to 1.
      #' @param framework_version (str): The version of SageMaker PySpark.
      #' @param py_version (str): The version of python.
      #' @param container_version (str): The version of spark container.
      #' @param image_uri (str): The container image to use for training.
      #' @param volume_size_in_gb (int): Size in GB of the EBS volume to
      #'              use for storing data during processing (default: 30).
      #' @param volume_kms_key (str): A KMS key for the processing
      #'              volume.
      #' @param output_kms_key (str): The KMS key id for all ProcessingOutputs.
      #' @param max_runtime_in_seconds (int): Timeout in seconds.
      #'              After this amount of time Amazon SageMaker terminates the job
      #'              regardless of its current status.
      #' @param base_job_name (str): Prefix for processing name. If not specified,
      #'              the processor generates a default job name, based on the
      #'              training image name and current timestamp.
      #' @param sagemaker_session (sagemaker.session.Session): Session object which
      #'              manages interactions with Amazon SageMaker APIs and any other
      #'              AWS services needed. If not specified, the processor creates one
      #'              using the default AWS configuration chain.
      #' @param env (dict): Environment variables to be passed to the processing job.
      #' @param tags ([dict]): List of tags to be passed to the processing job.
      #' @param network_config (sagemaker.network.NetworkConfig): A NetworkConfig
      #'              object that configures network isolation, encryption of
      #'              inter-container traffic, security group IDs, and subnets.
      initialize = function(role,
                            instance_type,
                            instance_count,
                            framework_version=NULL,
                            py_version=NULL,
                            container_version=NULL,
                            image_uri=NULL,
                            volume_size_in_gb=30,
                            volume_kms_key=NULL,
                            output_kms_key=NULL,
                            max_runtime_in_seconds=NULL,
                            base_job_name=NULL,
                            sagemaker_session=NULL,
                            env=NULL,
                            tags=NULL,
                            network_config=NULL){
         super$intialize(
            role=role,
            instance_count=instance_count,
            instance_type=instance_type,
            framework_version=framework_version,
            py_version=py_version,
            container_version=container_version,
            image_uri=image_uri,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config)
      },
      
      #' @description Runs a processing job.
      #' @param submit_app (str): Path (local or S3) to Jar file to submit to Spark as
      #'              the primary application
      #' @param submit_class (str): Java class reference to submit to Spark as the primary
      #'              application
      #' @param submit_jars (list[str]): List of paths (local or S3) to provide for
      #'              `spark-submit --jars` option
      #' @param submit_files (list[str]): List of paths (local or S3) to provide for
      #'              `spark-submit --files` option
      #' @param inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
      #'              the processing job. These must be provided as
      #'              :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
      #' @param outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
      #'              the processing job. These can be specified as either path strings or
      #'              :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
      #' @param arguments (list[str]): A list of string arguments to be passed to a
      #'              processing job (default: None).
      #' @param wait (bool): Whether the call should wait until the job completes (default: True).
      #' @param logs (bool): Whether to show the logs produced by the job.
      #'              Only meaningful when wait is True (default: True).
      #' @param job_name (str): Processing job name. If not specified, the processor generates
      #'              a default job name, based on the base job name and current timestamp.
      #' @param experiment_config (dict[str, str]): Experiment management configuration.
      #'              Dictionary contais three optional keys:
      #'              'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
      #' @param configuration (list[dict] or dict): Configuration for Hadoop, Spark, or Hive.
      #'              List or dictionary of EMR-style classifications.
      #'              https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html
      #' @param spark_event_logs_s3_uri (str): S3 path where spark application events will
      #'              be published to.
      #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
      #'              user code file (default: None).
      run = function(submit_app,
                     submit_class=NULL,
                     submit_jars=NULL,
                     submit_files=NULL,
                     inputs=NULL,
                     outputs=NULL,
                     arguments=NULL,
                     wait=TRUE,
                     logs=TRUE,
                     job_name=NULL,
                     experiment_config=NULL,
                     configuration=NULL,
                     spark_event_logs_s3_uri=NULL,
                     kms_key=NULL){
         self$.current_job_name = private$.generate_current_job_name(job_name=job_name)
         self$command = list(.SparkProcessorBase$private_fields$.default_command)
         
         if (missing(submit_app))
            stop("submit_app is required", call. = F)
         
         extended_args = private$.extend_processing_args(
            inputs,
            outputs,
            submit_class=submit_class,
            submit_jars=submit_jars,
            submit_files=submit_files,
            configuration=configuration,
            spark_event_logs_s3_uri=spark_event_logs_s3_uri)
         
         super$run(
            submit_app=submit_app,
            inputs=extended_args$Inputs,
            outputs=extended_args$Outputs,
            arguments=arguments,
            wait=wait,
            logs=logs,
            job_name=self._current_job_name,
            experiment_config=experiment_config,
            kms_key=kms_key)
      }
   ),
   private = list(
      .extend_processing_args = function(inputs,
                                         outputs,
                                         ...){
         kwargs = list(...)
         if (!islistempty(kwargs$submit_class))
            self$command = c(self$command, c("--class", kwargs$submit_class))
         else
            stop("submit_class is required", call. = F)
         
         extended_inputs = private$.handle_script_dependencies(
            inputs, kwargs$submit_jars, FileType$new()$JAR
         )
         extended_inputs = private$.handle_script_dependencies(
            extended_inputs, kwargs$submit_files, FileType$new()$FILE
         )
         
         return(super$.extend_processing_args(extended_inputs, outputs, ...))
      }
   ),
   lock_objects = F
)

# History server class that is responsible for starting history server.
.HistoryServer = R6Class(".HistoryServer",
   public = list(
      initialize = function(cli_args,
                            image_uri,
                            network_config){
         self$cli_args = cli_args
         self$image_uri = image_uri
         self$network_config = network_config
         self$run_history_server_command = private$.get_run_history_server_cmd()
      },
      
      # Runs the history server.
      run = function(){
         self$down()
         log_info("Starting history server...")
         cmd <- split_str(self$run_history_server_command, " ")
         exec_background(cmd[1], cmd[-1], std_out = TRUE, std_err = TRUE)
      },
      
      # Stops and removes the container.
      down = function(){
         sys_jupyter("docker", c("stop", private$.container_name))
         sys_jupyter("docker", c("rm", private$.container_name))
         log_info("History server terminated")
      }
   ),
   private = list(
      .container_name = "history_server",
      .entry_point = "smspark-history-server",
      arg_event_logs_s3_uri = "event_logs_s3_uri",
      arg_remote_domain_name = "remote_domain_name",
      # .history_server_args_format_map name c(arg_event_logs_s3_uri, arg_remote_domain_name)
      .history_server_args_format_map = list(
         "event_logs_s3_uri"= "--event-logs-s3-uri %s ",
         "remote_domain_name"= "--remote-domain-name %s "),
      
      # Gets the history server command.
      .get_run_history_server_cmd = function(){
         env_options = ""
         ser_cli_args = ""
         for (i in seq_along(self$cli_args)) {
            key = names(self$cli_args)[i]
            value = self$cli_args[[i]]
            if (key %in% names(private$.history_server_args_format_map))
               ser_cli_args = paste0(ser_cli_args, sprintf(private$.history_server_args_format_map[[key]], value))
            else
               env_options = paste0(env_options, sprintf("--env %s=%s ", key, value))
         }
         
         cmd = paste(
            sprintf("docker run %s --name %s", env_options, private$.container_name),
            sprintf("%s --entrypoint %s %s", self$network_config, private$.entry_point, self$image_uri),
            sprintf("%s", ser_cli_args))
         return(cmd)
      }
   ),
   lock_objects = F
)

# Enum of file type
FileType = R6Class("FileType",
   public = list(
      JAR = 1,
      PYTHON = 2,
      FILE = 3
   )
)
