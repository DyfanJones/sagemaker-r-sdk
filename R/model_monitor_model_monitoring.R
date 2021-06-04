# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/model_monitor/model_monitoring.py

#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common
#' @import lgr
#' @importFrom urltools url_parse
#' @import jsonlite
#' @import uuid

DEFAULT_REPOSITORY_NAME <- "sagemaker-model-monitor-analyzer"

STATISTICS_JSON_DEFAULT_FILE_NAME <- "statistics.json"
CONSTRAINTS_JSON_DEFAULT_FILE_NAME <- "constraints.json"
CONSTRAINT_VIOLATIONS_JSON_DEFAULT_FILE_NAME <- "constraint_violations.json"

.CONTAINER_BASE_PATH <- "/opt/ml/processing"
.CONTAINER_INPUT_PATH <- "input"
.CONTAINER_ENDPOINT_INPUT_PATH <- "endpoint"
.BASELINE_DATASET_INPUT_NAME <- "baseline_dataset_input"
.RECORD_PREPROCESSOR_SCRIPT_INPUT_NAME <- "record_preprocessor_script_input"
.POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME <- "post_analytics_processor_script_input"
.CONTAINER_OUTPUT_PATH <- "output"
.DEFAULT_OUTPUT_NAME <- "monitoring_output"
.MODEL_MONITOR_S3_PATH <- "model-monitor"
.BASELINING_S3_PATH <- "baselining"
.MONITORING_S3_PATH <- "monitoring"
.RESULTS_S3_PATH <- "results"
.INPUT_S3_PATH <- "input"

.SUGGESTION_JOB_BASE_NAME <- "baseline-suggestion-job"
.MONITORING_SCHEDULE_BASE_NAME <- "monitoring-schedule"

.DATASET_SOURCE_PATH_ENV_NAME <- "dataset_source"
.DATASET_FORMAT_ENV_NAME <- "dataset_format"
.OUTPUT_PATH_ENV_NAME <- "output_path"
.RECORD_PREPROCESSOR_SCRIPT_ENV_NAME <- "record_preprocessor_script"
.POST_ANALYTICS_PROCESSOR_SCRIPT_ENV_NAME <- "post_analytics_processor_script"
.PUBLISH_CLOUDWATCH_METRICS_ENV_NAME <- "publish_cloudwatch_metrics"
.ANALYSIS_TYPE_ENV_NAME <- "analysis_type"
.PROBLEM_TYPE_ENV_NAME <- "problem_type"
.GROUND_TRUTH_ATTRIBUTE_ENV_NAME <- "ground_truth_attribute"
.INFERENCE_ATTRIBUTE_ENV_NAME <- "inference_attribute"
.PROBABILITY_ATTRIBUTE_ENV_NAME <- "probability_attribute"
.PROBABILITY_THRESHOLD_ATTRIBUTE_ENV_NAME <- "probability_threshold_attribute"

#' @title Sets up Amazon SageMaker Monitoring Schedules and baseline suggestions.
#' @description Use this class when you want to provide your own container image containing the code
#'              you'd like to run, in order to produce your own statistics and constraint validation files.
#'              For a more guided experience, consider using the DefaultModelMonitor class instead.
#' @export
ModelMonitor = R6Class("ModelMonitor",
  public = list(

   #' @description Initializes a ``Monitor`` instance. The Monitor handles baselining datasets and
   #'              creating Amazon SageMaker Monitoring Schedules to monitor SageMaker endpoints.
   #' @param role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
   #' @param image_uri (str): The uri of the image to use for the jobs started by
   #'              the Monitor.
   #' @param instance_count (int): The number of instances to run
   #'              the jobs with.
   #' @param instance_type (str): Type of EC2 instance to use for
   #'              the job, for example, 'ml.m5.xlarge'.
   #' @param entrypoint ([str]): The entrypoint for the job.
   #' @param volume_size_in_gb (int): Size in GB of the EBS volume
   #'              to use for storing data during processing (default: 30).
   #' @param volume_kms_key (str): A KMS key for the job's volume.
   #' @param output_kms_key (str): The KMS key id for the job's outputs.
   #' @param max_runtime_in_seconds (int): Timeout in seconds. After this amount of
   #'              time, Amazon SageMaker terminates the job regardless of its current status.
   #'              Default: 3600
   #' @param base_job_name (str): Prefix for the job name. If not specified,
   #'              a default name is generated based on the training image name and
   #'              current timestamp.
   #' @param sagemaker_session (sagemaker.session.Session): Session object which
   #'              manages interactions with Amazon SageMaker APIs and any other
   #'              AWS services needed. If not specified, one is created using
   #'              the default AWS configuration chain.
   #' @param env (dict): Environment variables to be passed to the job.
   #' @param tags ([dict]): List of tags to be passed to the job.
   #' @param network_config (sagemaker.network.NetworkConfig): A NetworkConfig
   #'              object that configures network isolation, encryption of
   #'              inter-container traffic, security group IDs, and subnets.
   initialize = function(role =NULL,
                         image_uri = NULL,
                         instance_count=1,
                         instance_type="ml.m5.xlarge",
                         entrypoint=NULL,
                         volume_size_in_gb=30,
                         volume_kms_key=NULL,
                         output_kms_key=NULL,
                         max_runtime_in_seconds=NULL,
                         base_job_name=NULL,
                         sagemaker_session=NULL,
                         env=NULL,
                         tags=NULL,
                         network_config=NULL){
     self$role = role
     self$image_uri = image_uri
     self$instance_count = instance_count
     self$instance_type = instance_type
     self$entrypoint = entrypoint
     self$volume_size_in_gb = volume_size_in_gb
     self$volume_kms_key = volume_kms_key
     self$output_kms_key = output_kms_key
     self$max_runtime_in_seconds = max_runtime_in_seconds
     self$base_job_name = base_job_name
     self$sagemaker_session = sagemaker_session %||% Session$new()
     self$env = env
     self$tags = tags
     self$network_config = network_config

     self$baselining_jobs = list()
     self$latest_baselining_job = NULL
     self$arguments = NULL
     self$latest_baselining_job_name = NULL
     self$monitoring_schedule_name = NULL
     self$job_definition_name = NULL
   },

   #' @description Run a processing job meant to baseline your dataset.
   #' @param baseline_inputs ([sagemaker.processing.ProcessingInput]): Input files for the processing
   #'              job. These must be provided as ProcessingInput objects.
   #' @param output (sagemaker.processing.ProcessingOutput): Destination of the constraint_violations
   #'              and statistics json files.
   #' @param arguments ([str]): A list of string arguments to be passed to a processing job.
   #' @param wait (bool): Whether the call should wait until the job completes (default: True).
   #' @param logs (bool): Whether to show the logs produced by the job.
   #'              Only meaningful when wait is True (default: True).
   #' @param job_name (str): Processing job name. If not specified, the processor generates
   #'              a default job name, based on the image name and current timestamp.
   run_baseline = function(baseline_inputs,
                           output,
                           arguments=NULL,
                           wait=TRUE,
                           logs=TRUE,
                           job_name=NULL){
     self$latest_baselining_job_name = private$.generate_baselining_job_name(job_name=job_name)
     self$arguments = arguments
     normalized_baseline_inputs = private$.normalize_baseline_inputs(
                  baseline_inputs=baseline_inputs)
     normalized_output = private$.normalize_processing_output(output=output)

     baselining_processor = Processor$new(
       role=self$role,
       image_uri=self$image_uri,
       instance_count=self$instance_count,
       instance_type=self$instance_type,
       entrypoint=self$entrypoint,
       volume_size_in_gb=self$volume_size_in_gb,
       volume_kms_key=self$volume_kms_key,
       output_kms_key=self$output_kms_key,
       max_runtime_in_seconds=self$max_runtime_in_seconds,
       base_job_name=self$base_job_name,
       sagemaker_session=self$sagemaker_session,
       env=self$env,
       tags=self$tags,
       network_config=self$network_config)

     baselining_processor$run(
       inputs=normalized_baseline_inputs,
       outputs=list(normalized_output),
       arguments=self$arguments,
       wait=wait,
       logs=logs,
       job_name=self$latest_baselining_job_name)

     self$latest_baselining_job = BaseliningJob$new()$from_processing_job(
       processing_job=baselining_processor$latest_job)

     self$baselining_jobs = c(self$baselining_jobs, self$latest_baselining_job)
   },

   #' @description Creates a monitoring schedule to monitor an Amazon SageMaker Endpoint.
   #'              If constraints and statistics are provided, or if they are able to be retrieved from a
   #'              previous baselining job associated with this monitor, those will be used.
   #'              If constraints and statistics cannot be automatically retrieved, baseline_inputs will be
   #'              required in order to kick off a baselining job.
   #' @param endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
   #'              This can either be the endpoint name or an EndpointInput.
   #' @param output (sagemaker.model_monitor.MonitoringOutput): The output of the monitoring
   #'              schedule.
   #' @param statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
   #'              constraints, these will be used for monitoring the endpoint. This can be a
   #'              sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
   #'              JSON file.
   #' @param constraints (sagemaker.model_monitor.Constraints or str): If provided alongside
   #'              statistics, these will be used for monitoring the endpoint. This can be a
   #'              sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
   #'              JSON file.
   #' @param monitor_schedule_name (str): Schedule name. If not specified, the processor generates
   #'              a default job name, based on the image name and current timestamp.
   #' @param schedule_cron_expression (str): The cron expression that dictates the frequency that
   #'              this job runs at. See sagemaker.model_monitor.CronExpressionGenerator for valid
   #'              expressions. Default: Daily.
   create_monitoring_schedule = function(endpoint_input,
                                         output,
                                         statistics=NULL,
                                         constraints=NULL,
                                         monitor_schedule_name=NULL,
                                         schedule_cron_expression=NULL){
     if(!islistempty(self$monitoring_schedule_name)){
       msg = paste("It seems that this object was already used to create an Amazon Model ",
                   "Monitoring Schedule. To create another, first delete the existing one ",
                   "using my_monitor.delete_monitoring_schedule().", sep = "\n")
       ValueError$new(msg)}

     self$monitoring_schedule_name = private$.generate_monitoring_schedule_name(
       schedule_name=monitor_schedule_name
     )

     normalized_endpoint_input = private$.normalize_endpoint_input(endpoint_input=endpoint_input)

     normalized_monitoring_output = private$.normalize_monitoring_output_fields(output=output)

     file_objects = private$.get_baseline_files(
       statistics=statistics, constraints=constraints, sagemaker_session=self$sagemaker_session)

     statistics_s3_uri = NULL
     if (!islistempty(file_objects$statistics))
       statistics_s3_uri = file_objects$statistics$file_s3_uri

     constraints_s3_uri = NULL
     if (!islistempty(file_objects$constraint))
       constraints_s3_uri = file_objects$constraint$file_s3_uri

     monitoring_output_config = list(
       "MonitoringOutputs"= list(normalized_monitoring_output$to_request_list()))

     if (!is.null(self$output_kms_key))
       monitoring_output_config$KmsKeyId = self$output_kms_key

     self$monitoring_schedule_name = (
        monitor_schedule_name %||%
        private$.generate_monitoring_schedule_name(schedule_name=monitor_schedule_name)
     )

     network_config_list = NULL
     if (!is.null(self$network_config)){
       network_config_list = self$network_config$to_request_list()
       private$.validate_network_config(network_config_list)}

     self$sagemaker_session$create_monitoring_schedule(
       monitoring_schedule_name=self$monitoring_schedule_name,
       schedule_expression=schedule_cron_expression,
       statistics_s3_uri=statistics_s3_uri,
       constraints_s3_uri=constraints_s3_uri,
       monitoring_inputs=list(normalized_endpoint_input$to_request_list()),
       monitoring_output_config=monitoring_output_config,
       instance_count=self$instance_count,
       instance_type=self$instance_type,
       volume_size_in_gb=self$volume_size_in_gb,
       volume_kms_key=self$volume_kms_key,
       image_uri=self$image_uri,
       entrypoint=self$entrypoint,
       arguments=self$arguments,
       record_preprocessor_source_uri=NULL,
       post_analytics_processor_source_uri=NULL,
       max_runtime_in_seconds=self$max_runtime_in_seconds,
       environment=self$env,
       network_config=network_config_list,
       role_arn=self$sagemaker_session$expand_role(self$role),
       tags=self$tags)
   },

   #' @description Updates the existing monitoring schedule.
   #' @param endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
   #'              This can either be the endpoint name or an EndpointInput.
   #' @param output (sagemaker.model_monitor.MonitoringOutput): The output of the monitoring
   #'              schedule.
   #' @param statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
   #'              constraints, these will be used for monitoring the endpoint. This can be a
   #'              sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
   #'              JSON file.
   #' @param constraints (sagemaker.model_monitor.Constraints or str): If provided alongside
   #'              statistics, these will be used for monitoring the endpoint. This can be a
   #'              sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
   #'              JSON file.
   #' @param schedule_cron_expression (str): The cron expression that dictates the frequency that
   #'              this job runs at.  See sagemaker.model_monitor.CronExpressionGenerator for valid
   #'              expressions.
   #' @param instance_count (int): The number of instances to run
   #'              the jobs with.
   #' @param instance_type (str): Type of EC2 instance to use for
   #'              the job, for example, 'ml.m5.xlarge'.
   #' @param entrypoint (str): The entrypoint for the job.
   #' @param volume_size_in_gb (int): Size in GB of the EBS volume
   #'              to use for storing data during processing (default: 30).
   #' @param volume_kms_key (str): A KMS key for the job's volume.
   #' @param output_kms_key (str): The KMS key id for the job's outputs.
   #' @param arguments ([str]): A list of string arguments to be passed to a processing job.
   #' @param max_runtime_in_seconds (int): Timeout in seconds. After this amount of
   #'              time, Amazon SageMaker terminates the job regardless of its current status.
   #'              Default: 3600
   #' @param env (dict): Environment variables to be passed to the job.
   #' @param network_config (sagemaker.network.NetworkConfig): A NetworkConfig
   #'              object that configures network isolation, encryption of
   #'              inter-container traffic, security group IDs, and subnets.
   #' @param role (str): An AWS IAM role name or ARN. The Amazon SageMaker jobs use this role.
   #' @param image_uri (str): The uri of the image to use for the jobs started by
   #'              the Monitor.
   update_monitoring_schedule = function(endpoint_input=NULL,
                                         output=NULL,
                                         statistics=NULL,
                                         constraints=NULL,
                                         schedule_cron_expression=NULL,
                                         instance_count=NULL,
                                         instance_type=NULL,
                                         entrypoint=NULL,
                                         volume_size_in_gb=NULL,
                                         volume_kms_key=NULL,
                                         output_kms_key=NULL,
                                         arguments=NULL,
                                         max_runtime_in_seconds=NULL,
                                         env=NULL,
                                         network_config=NULL,
                                         role=NULL,
                                         image_uri=NULL){
     monitoring_inputs = NULL
     if (!is.null(endpoint_input))
        monitoring_inputs = list(
           private$.normalize_endpoint_input(endpoint_input=endpoint_input)$to_request_list()
        )

     monitoring_output_config = NULL
     if (!is.null(output)){
        normalized_monitoring_output = private$.normalize_monitoring_output_fields(output=output)
        monitoring_output_config = list(
           "MonitoringOutputs"= list(normalized_monitoring_output$to_request_list())
        )
     }

     file_objects = private$.get_baseline_files(
       statistics=statistics, constraints=constraints, sagemaker_session=self$sagemaker_session)

     statistics_s3_uri = NULL
     if (!islistempty(file_objects$statistics))
       statistics_s3_uri = file_objects$statistics$file_s3_uri

     constraints_s3_uri = NULL
     if (!islistempty(file_objects$constraint))
       constraints_s3_uri = file_objects$constraint$file_s3_uri


     if (!is.null(instance_type))
       self$instance_type = instance_type

     if (!is.null(instance_count))
       self$instance_count = instance_count

     if (!is.null(entrypoint))
       self$entrypoint = entrypoint

     if (!is.null(volume_size_in_gb))
       self$volume_size_in_gb = volume_size_in_gb

     if (!is.null(volume_kms_key))
       self$volume_kms_key = volume_kms_key

     if (!is.null(output_kms_key)){
        self$output_kms_key = output_kms_key
        monitoring_output_config$KmsKeyId = self$output_kms_key}

     if (!is.null(arguments))
       self$arguments = arguments

     if (!is.null(max_runtime_in_seconds))
       self$max_runtime_in_seconds = max_runtime_in_seconds

     if (!islistempty(env))
       self$env = env

     if (!islistempty(network_config))
       self$network_config = network_config

     if (!is.null(role))
       self$role = role

     if (!is.null(image_uri))
       self$image_uri = image_uri

     network_config_list = NULL
     if (!is.null(self$network_config)){
       network_config_list = self$network_config$to_request_list()
       private$.validate_network_config(network_config_list)}

     self$sagemaker_session$update_monitoring_schedule(
       monitoring_schedule_name=self$monitoring_schedule_name,
       schedule_expression=schedule_cron_expression,
       statistics_s3_uri=statistics_s3_uri,
       constraints_s3_uri=constraints_s3_uri,
       monitoring_inputs=monitoring_inputs,
       monitoring_output_config=monitoring_output_config,
       instance_count=instance_count,
       instance_type=instance_type,
       volume_size_in_gb=volume_size_in_gb,
       volume_kms_key=volume_kms_key,
       image_uri=image_uri,
       entrypoint=entrypoint,
       arguments=arguments,
       max_runtime_in_seconds=max_runtime_in_seconds,
       environment=env,
       network_config=network_config_list,
       role_arn=self$sagemaker_session$expand_role(self$role))

     private$.wait_for_schedule_changes_to_apply()
   },

   #' @description Starts the monitoring schedule.
   start_monitoring_schedule = function(){
     self$sagemaker_session$start_monitoring_schedule(
       monitoring_schedule_name=self$monitoring_schedule_name)

     private$.wait_for_schedule_changes_to_apply()
   },

   #' @description Stops the monitoring schedule.
   stop_monitoring_schedule = function(){
     self$sagemaker_session$stop_monitoring_schedule(
       monitoring_schedule_name=self.monitoring_schedule_name)

     private$.wait_for_schedule_changes_to_apply()
   },

   #' @description Deletes the monitoring schedule.
   delete_monitoring_schedule = function(){
      self$sagemaker_session$delete_monitoring_schedule(
       monitoring_schedule_name=self$monitoring_schedule_name)

      if(!is.null(self$job_definition_name)){
         # Job definition is locked by schedule so need to wait for the schedule to be deleted
         tryCatch({
            private$.wait_for_schedule_changes_to_apply()
         },
         error = function(e){
            # OK the schedule is gone
            if(attributes(e)$error_response$`__type` == "ResourceNotFound")
               return(NULL)
            else stop(e)
         })
      }
      self$monitoring_schedule_name = NULL
   },

   #' @description Returns a Statistics object representing the statistics json file generated by the
   #'              latest baselining job.
   #' @param file_name (str): The name of the .json statistics file
   #' @return sagemaker.model_monitor.Statistics: The Statistics object representing the file that
   #'              was generated by the job.
   baseline_statistics = function(file_name=STATISTICS_JSON_DEFAULT_FILE_NAME){
     return(self$latest_baselining_job$baseline_statistics(
       file_name=file_name, kms_key=self$output_kms_key))
   },

   #' @description Returns a Statistics object representing the constraints json file generated by the
   #'              latest baselining job
   #' @param file_name (str): The name of the .json constraints file
   #' @param sagemaker.model_monitor.Constraints: The Constraints object representing the file that
   #'              was generated by the job.
   suggested_constraints = function(file_name=CONSTRAINTS_JSON_DEFAULT_FILE_NAME){
     return (self$latest_baselining_job$suggested_constraints(
       file_name=file_name, kms_key=self$output_kms_key))
   },

   #' @description Returns the sagemaker.model_monitor.Statistics generated by the latest monitoring
   #'              execution.
   #' @param file_name (str): The name of the statistics file to be retrieved. Only override if
   #'              generating a custom file name.
   #' @return sagemaker.model_monitoring.Statistics: The Statistics object representing the file
   #'              generated by the latest monitoring execution.
   latest_monitoring_statistics = function(file_name=STATISTICS_JSON_DEFAULT_FILE_NAME){
      executions = self$list_executions()
      if (length(executions) == 0){
        message(sprintf("No executions found for schedule. monitoring_schedule_name: %s",
                       self$monitoring_schedule_name))
        return(NULL)}
      latest_monitoring_execution = executions[[length(executions)]]
      return(latest_monitoring_execution$statistics(file_name=file_name))
   },

   #' @description Returns the sagemaker.model_monitor.ConstraintViolations generated by the latest
   #'              monitoring execution.
   #' @param file_name (str): The name of the constraint violdations file to be retrieved. Only
   #'              override if generating a custom file name.
   #' @return sagemaker.model_monitoring.ConstraintViolations: The ConstraintViolations object
   #'              representing the file generated by the latest monitoring execution.
   latest_monitoring_constraint_violations = function(file_name=CONSTRAINT_VIOLATIONS_JSON_DEFAULT_FILE_NAME){
     executions = self$list_executions()
     if (length(executions) == 0){
       message(sprintf("No executions found for schedule. monitoring_schedule_name: %s",
                       self$monitoring_schedule_name))
       return(NULL)}

     latest_monitoring_execution = executions[[length(executions)]]
     return(latest_monitoring_execution$constraint_violations(file_name=file_name))
   },

   #' @description Describe the latest baselining job kicked off by the suggest workflow.
   describe_latest_baselining_job = function(){
     if (is.null(self.latest_baselining_job)){
       ValueError$new("No suggestion jobs were kicked off.")}
     return(self$latest_baselining_job$describe())
   },

   #' @description Describes the schedule that this object represents.
   #' @return dict: A dictionary response with the monitoring schedule description.
   describe_schedule = function(){
      return(self$sagemaker_session$describe_monitoring_schedule(
         monitoring_schedule_name=self$monitoring_schedule_name)
      )
   },

   #' @description Get the list of the latest monitoring executions in descending order of "ScheduledTime".
   #'              Statistics or violations can be called following this example:
   #'              Example:
   #'              >>> my_executions = my_monitor.list_executions()
   #'              >>> second_to_last_execution_statistics = my_executions[-1].statistics()
   #'              >>> second_to_last_execution_violations = my_executions[-1].constraint_violations()
   #' @return [sagemaker.model_monitor.MonitoringExecution]: List of MonitoringExecutions in
   #'              descending order of "ScheduledTime".
   list_executions = function(){
     monitoring_executions_list = self$sagemaker_session$list_monitoring_executions(
       monitoring_schedule_name=self$monitoring_schedule_name)

     if (islistempty(monitoring_executions_list$MonitoringExecutionSummaries)){
        message(
           sprintf("No executions found for schedule. monitoring_schedule_name: %s",
                   self$monitoring_schedule_name))
        return(list())}

     processing_job_arns = lapply(
        monitoring_executions_list$MonitoringExecutionSummaries,
        function(execution_list) execution_list$ProcessingJobArn
     )
     processing_job_arns = processing_job_arns[!islistempty(processing_job_arns$ProcessingJobArn)]

     monitoring_executions = lapply(
        processing_job_arns,
        function(processing_job_arn) MonitoringExecution$new()$from_processing_arn(
           sagemaker_session=self$sagemaker_session,
           processing_job_arn=processing_job_arn)
        )

     return(rev(monitoring_executions))
   },

   #' @description Sets this object's schedule name to point to the Amazon Sagemaker Monitoring Schedule
   #'              name provided. This allows subsequent describe_schedule or list_executions calls to point
   #'              to the given schedule.
   #' @param monitor_schedule_name (str): The name of the schedule to attach to.
   #' @param sagemaker_session (sagemaker.session.Session): Session object which
   #'              manages interactions with Amazon SageMaker APIs and any other
   #'              AWS services needed. If not specified, one is created using
   #'              the default AWS configuration chain.
   attach = function(monitor_schedule_name, sagemaker_session=NULL){
     sagemaker_session = sagemaker_session %||% Session$new()
     schedule_desc = sagemaker_session$describe_monitoring_schedule(
       monitoring_schedule_name=monitor_schedule_name)

     initial_param = list()

     initial_param$role = schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$RoleArn
     initial_param$image_uri = schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$ImageUri
     initial_param$instance_count = schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringResources$ClusterConfig$InstanceCount
     initial_param$instance_type = schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringResources$ClusterConfig$InstanceType
     initial_param$entrypoint = schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$ContainerEntrypoint
     initial_param$volume_size_in_gb = schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringResources$ClusterConfig$VolumeSizeInGB
     initial_param$volume_kms_key = schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringResources$ClusterConfig$VolumeKmsKeyId
     initial_param$output_kms_key = schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringOutputConfig$KmsKeyId

     if (!islistempty(schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$StoppingCondition)){
        initial_param$max_runtime_in_seconds = schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$StoppingCondition$MaxRuntimeInSeconds}

     initial_param$env = schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$Environment
     network_config_list = schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$NetworkConfig
     vpc_config = schedule_desc$MonitoringScheduleConfig$MonitoringJobDefinition$NetworkConfig$VpcConfig

     security_group_ids = NULL
     subnets = NULL
     if (!islistempty(vpc_config)){
       security_group_ids = vpc_config$SecurityGroupIds
       subnets = vpc_config$Subnets}

     if (!islistempty(network_config_list)){
        initial_param$network_config = NetworkConfig$new(
         enable_network_isolation=network_config_list$EnableNetworkIsolation,
         security_group_ids=security_group_ids,
         subnets=subnets)}

     initial_param$tags = sagemaker_session$list_tags(resource_arn=schedule_desc$MonitoringScheduleArn)

     attached_monitor = self$clone()

     # initialize new class
     do.call(attached_monitor$initialize, initial_param)

     # modify clone class with new inputs
     attached_monitor$monitoring_schedule_name = monitor_schedule_name

     return (attached_monitor)
   },

   #' @description Type of the monitoring job.
   monitoring_type = function(){
      TypeError$new(sprintf("Subclass of %s shall define this property",class(self)[1]))
   },

   #' @description format class
   format = function(){
      format_class(self)
   }
  ),
  private = list(

   # Attach a model monitor object to an existing monitoring schedule.
   # Args:
   #   clazz: a subclass of this class
   # sagemaker_session (sagemaker.session.Session): Session object which
   # manages interactions with Amazon SageMaker APIs and any other
   # AWS services needed. If not specified, one is created using
   # the default AWS configuration chain.
   # schedule_desc (dict): output of describe monitoring schedule API.
   # job_desc (dict): output of describe job definition API.
   # Returns:
   #   Object of a subclass of this class.
   .attach = function(clazz,
                      sagemaker_session,
                      schedule_desc,
                      job_desc,
                      tags){
      monitoring_schedule_name = schedule_desc[["MonitoringScheduleName"]]
      job_definition_name = schedule_desc[["MonitoringScheduleConfig"]][[
         "MonitoringJobDefinitionName"
      ]]
      monitoring_type = schedule_desc[["MonitoringScheduleConfig"]][["MonitoringType"]]
      role = job_desc[["RoleArn"]]
      cluster_config = job_desc[["JobResources"]][["ClusterConfig"]]
      instance_count = cluster_config[["InstanceCount"]]
      instance_type = cluster_config[["InstanceType"]]
      volume_size_in_gb = cluster_config[["VolumeSizeInGB"]]
      volume_kms_key = cluster_config[["VolumeKmsKeyId"]]
      output_kms_key = job_desc[[sprintf("%sJobOutputConfig", monitoring_type)]][["KmsKeyId"]]
      network_config_dict = job_desc[["NetworkConfig"]] %||% list()

      max_runtime_in_seconds = NULL
      stopping_condition = job_desc[["StoppingCondition"]]
      if (!islistempty(stopping_condition))
         max_runtime_in_seconds = stopping_condition[["MaxRuntimeInSeconds"]]

      env = job_desc[[sprintf("%sAppSpecification",monitoring_type)]][["Environment"]]

      vpc_config = network_config_dict[["VpcConfig"]]

      security_group_ids = NULL
      if (!islistempty(vpc_config))
         security_group_ids = vpc_config[["SecurityGroupIds"]]

      subnets = NULL
      if (!islistempty(vpc_config))
         subnets = vpc_config[["Subnets"]]

      network_config = NULL
      if (!islistempty(network_config_dict))
         network_config = NetworkConfig$new(
            enable_network_isolation=network_config_dict[["EnableNetworkIsolation"]],
            security_group_ids=security_group_ids,
            subnets=subnets)
      attached_monitor = clazz$new(
         role=role,
         instance_count=instance_count,
         instance_type=instance_type,
         volume_size_in_gb=volume_size_in_gb,
         volume_kms_key=volume_kms_key,
         output_kms_key=output_kms_key,
         max_runtime_in_seconds=max_runtime_in_seconds,
         sagemaker_session=sagemaker_session,
         env=env,
         tags=tags,
         network_config=network_config)
      attached_monitor$monitoring_schedule_name = monitoring_schedule_name
      attached_monitor$job_definition_name = job_definition_name
      return(attached_monitor)
   },

   # Generate the job name before running a suggestion processing job.
   # Args:
   #   job_name (str): Name of the suggestion processing job to be created. If not
   # specified, one is generated using the base name given to the
   # constructor, if applicable.
   # Returns:
   #   str: The supplied or generated job name.
   .generate_baselining_job_name = function(job_name = NULL){
     if (!is.null(job_name))
       return(job_name)

     if (!is.null(self$base_job_name)){
       base_name = self$base_job_name
     } else {
       base_name = "baseline-suggestion-job"}

     return(name_from_base(base=base_name))
   },

   # Generate the monitoring schedule name.
   # Args:
   #   schedule_name (str): Name of the monitoring schedule to be created. If not
   # specified, one is generated using the base name given to the
   # constructor, if applicable.
   # Returns:
   #   str: The supplied or generated job name.
   .generate_monitoring_schedule_name = function(schedule_name=NULL){

     if (!is.null(schedule_name)){
       return(schedule_name)}

     if (!islistempty(self$base_job_name)){
       base_name = self$base_job_name
     } else {
       base_name = .MONITORING_SCHEDULE_BASE_NAME}

     return(name_from_base(base=base_name))
   },

   # Generate a list of environment variables from first-class parameters.
   # Args:
   #    output_path (str): Local path to the output.
   # enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
   # the baselining or monitoring jobs.
   # record_preprocessor_script_container_path (str): The path to the record preprocessor
   # script.
   # post_processor_script_container_path (str): The path to the post analytics processor
   # script.
   # dataset_format (dict): The format of the baseline_dataset.
   # dataset_source_container_path (str): The path to the dataset source.
   # inference_attribute (str): Index or JSONpath to locate predicted label(s).
   # Only used for ModelQualityMonitor, ModelBiasMonitor, and ModelExplainabilityMonitor
   # probability_attribute (str or int): Index or JSONpath to locate probabilities.
   # Only used for ModelQualityMonitor, ModelBiasMonitor and ModelExplainabilityMonitor
   # ground_truth_attribute (str): Index or JSONpath to locate actual label(s).
   # probability_threshold_attribute (float): threshold to convert probabilities to binaries
   # Only used for ModelQualityMonitor, ModelBiasMonitor and ModelExplainabilityMonitor
   # Returns:
   #    dict: Dictionary of environment keys and values.
   .generate_env_map = function(env,
                                output_path=NULL,
                                enable_cloudwatch_metrics=NULL,
                                record_preprocessor_script_container_path=NULL,
                                post_processor_script_container_path=NULL,
                                dataset_format=NULL,
                                dataset_source_container_path=NULL,
                                analysis_type=NULL,
                                problem_type=NULL,
                                inference_attribute=NULL,
                                probability_attribute=NULL,
                                ground_truth_attribute=NULL,
                                probability_threshold_attribute=NULL){
      cloudwatch_env_map = function(x) ifelse(x, "Enabled", "Disabled")

      env = env %||% list()

      env[[.OUTPUT_PATH_ENV_NAME]] = output_path

      if(!is.null(enable_cloudwatch_metrics))
         env[[.PUBLISH_CLOUDWATCH_METRICS_ENV_NAME]] = cloudwatch_env_map(
            enable_cloudwatch_metrics)

      if(!is.null(dataset_format))
         env[[.DATASET_FORMAT_ENV_NAME]] = jsonlite::toJSON(dataset_format, auto_unbox = T)

      env[[.RECORD_PREPROCESSOR_SCRIPT_ENV_NAME]] = record_preprocessor_script_container_path
      env[[.POST_ANALYTICS_PROCESSOR_SCRIPT_ENV_NAME]] = post_processor_script_container_path
      env[[.DATASET_SOURCE_PATH_ENV_NAME]] = dataset_source_container_path
      env[[.ANALYSIS_TYPE_ENV_NAME]] = analysis_type
      env[[.PROBLEM_TYPE_ENV_NAME]] = problem_type
      env[[.INFERENCE_ATTRIBUTE_ENV_NAME]] = inference_attribute
      env[[.PROBABILITY_ATTRIBUTE_ENV_NAME]] = probability_attribute
      env[[.GROUND_TRUTH_ATTRIBUTE_ENV_NAME]] = ground_truth_attribute
      env[[.PROBABILITY_THRESHOLD_ATTRIBUTE_ENV_NAME]] = probability_threshold_attribute

      return(env)
   },

   # Populates baseline values if possible.
   # Args:
   #   statistics (sagemaker.model_monitor.Statistics or str): The statistics object or str.
   # If none, this method will attempt to retrieve a previously baselined constraints
   # object.
   # constraints (sagemaker.model_monitor.Constraints or str): The constraints object or str.
   # If none, this method will attempt to retrieve a previously baselined constraints
   # object.
   # sagemaker_session (sagemaker.session.Session): Session object which manages interactions
   # with Amazon SageMaker APIs and any other AWS services needed. If not specified, one
   # is created using the default AWS configuration chain.
   # Returns:
   #   sagemaker.model_monitor.Statistics, sagemaker.model_monitor.Constraints: The Statistics
   # and Constraints objects that were provided or created by the latest
   # baselining job. If none were found, returns None.
   .get_baseline_files = function(statistics,
                                  constraints,
                                  sagemaker_session=NULL){

     if (!is.null(statistics) && inherits(statistics, "character")){
       statistics = Statistics$new()$from_s3_uri(
         statistics_file_s3_uri=statistics, sagemaker_session=sagemaker_session)}

     if (!is.null(constraints) && inherits(constraints, "character")){
       constraints = Constraints$new()$from_s3_uri(
         constraints_file_s3_uri=constraints, sagemaker_session=sagemaker_session)}

     return(list(statistics = statistics, constraints = constraints))
   },

   # Ensure that the input is an EndpointInput object.
   # Args:
   #   endpoint_input ([str or sagemaker.processing.EndpointInput]): An endpoint input
   # to be normalized. Can be either a string or a EndpointInput object.
   # Returns:
   #   sagemaker.processing.EndpointInput: The normalized EndpointInput object.
   .normalize_endpoint_input = function(endpoint_input){
     # If the input is a string, turn it into an EndpointInput object.
     if (inherit(endpoint_input, "character"))
       endpoint_input = EndpointInput$new(
         endpoint_name=endpoint_input,
         destination=file.path(
           .CONTAINER_BASE_PATH, .CONTAINER_INPUT_PATH, .CONTAINER_ENDPOINT_INPUT_PATH)
       )

     return(endpoint_input)
   },

   # Ensure that all the ProcessingInput objects have names and S3 uris.
   # Args:
   #   baseline_inputs ([sagemaker.processing.ProcessingInput]): A list of ProcessingInput
   # objects to be normalized.
   # Returns:
   #   [sagemaker.processing.ProcessingInput]: The list of normalized
   # ProcessingInput objects.
   .normalize_baseline_inputs = function(baseline_inputs=NULL){
     normalized_inputs = list()
     if (!is.null(baseline_inputs)){
       # Iterate through the provided list of inputs.
       for (count in seq_along(baseline_inputs)){
          file_input = baseline_inputs[[count]]
          if (!inherits(file_input, "ProcessingInput")){
             TypeError$new(
                "Your inputs must be provided as ProcessingInput objects.")}
          # Generate a name for the ProcessingInput if it doesn't have one.
          if (is.null(file_input$input_name)){
             file_input$input_name = sprintf("input-%s",count)}
          # If the source is a local path, upload it to S3
          # and save the S3 uri in the ProcessingInput source.
          parse_result = parse_url(file_input$source)
          if (parse_result$scheme != "s3"){
            s3_uri = file.path(
              "s3://",
               self$sagemaker_session$default_bucket(),
               self$latest_baselining_job_name,
               file_input$input_name)

           S3Uploader$new()$upload(
             local_path=file_input.source,
             desired_s3_uri=s3_uri,
             session=self.sagemaker_session)

           file_input$source = s3_uri
           normalized_inputs = c(normalized_inputs, file_input)}
          }
       }
      return(normalized_inputs)
   },

   # Ensure that the output is a ProcessingOutput object.
   # Args:
   #   output_s3_uri (str): The output S3 uri to deposit the baseline files in.
   # Returns:
   #   sagemaker.processing.ProcessingOutput: The normalized ProcessingOutput object.
   .normalize_baseline_output = function(output_s3_uri=NULL) {
      s3_uri = output_s3_uri %||% file.path(
         "s3://",
         self$sagemaker_session$default_bucket(),
         .MODEL_MONITOR_S3_PATH,
         .BASELINING_S3_PATH,
         self$latest_baselining_job_name,
         .RESULTS_S3_PATH)
      return(ProcessingOutput$new(
         source=file.path(.CONTAINER_BASE_PATH, .CONTAINER_OUTPUT_PATH),
         destination=s3_uri,
         output_name=.DEFAULT_OUTPUT_NAME))
   },

   # Ensure that the output is a ProcessingOutput object.
   # Args:
   #   output (str or sagemaker.processing.ProcessingOutput): An output to be normalized.
   # Returns:
   #   sagemaker.processing.ProcessingOutput: The normalized ProcessingOutput object.
   .normalize_processing_output = function(output = NULL){
     # If the output is a string, turn it into a ProcessingOutput object.
     if (inherits(output, "character")){
       s3_uri = file.path(
         "s3://",
         self$sagemaker_session$default_bucket(),
         self$latest_baselining_job_name,
         "output")
       output = ProcessingOutput$new(
         source=output, destination=s3_uri, output_name=.DEFAULT_OUTPUT_NAME)
     }
     return(output)
   },

   # Ensure that the output is a MonitoringOutput object.
   # Args:
   #    monitoring_schedule_name (str): Monitoring schedule name
   # output_s3_uri (str): The output S3 uri to deposit the monitoring evaluation files in.
   # Returns:
   #    sagemaker.model_monitor.MonitoringOutput: The normalized MonitoringOutput object.
   .normalize_monitoring_output = function(monitoring_schedule_name,
                                           output_s3_uri=NULL){
      s3_uri = output_s3_uri %||% file.path(
         "s3://",
         self$sagemaker_session$default_bucket(),
         .MODEL_MONITOR_S3_PATH,
         .MONITORING_S3_PATH,
         monitoring_schedule_name,
         .RESULTS_S3_PATH)
      output = MonitoringOutput$new(
         source=file.path(.CONTAINER_BASE_PATH, .CONTAINER_OUTPUT_PATH),
         destination=s3_uri)
      return(output)
   },

   # Ensure that output has the correct fields.
   # Args:
   #   output (sagemaker.processing.MonitoringOutput): An output to be normalized.
   # Returns:
   #   sagemaker.processing.MonitoringOutput: The normalized MonitoringOutput
   .normalize_monitoring_output_fields = function(output = NULL){
     # If the output is a string, turn it into a ProcessingOutput object.
     if (is.null(output$destination)){
       output$destination = file.path(
         "s3://",
         self$sagemaker_session$default_bucket(),
         self$monitoring_schedule_name,
         "output")}

       return(output)
   },

   # If path is local, uploads to S3 and returns S3 uri. Otherwise returns S3 uri as-is.
   # Args:
   #   path (str): Path to file. This can be a local path or an S3 path.
   # Returns:
   #   str: S3 uri to file.
   .s3_uri_from_local_path = function(path){
     parse_result = url_parse(path)
     if (parse_result$scheme != "s3"){
       s3_uri = file.path(
         "s3://",
         self$sagemaker_session$default_bucket(),
         .MODEL_MONITOR_S3_PATH,
         .MONITORING_S3_PATH,
         self$monitoring_schedule_name,
         .INPUT_S3_PATH,
         UUIDgenerate())
       S3Uploader$new()$upload(
         local_path=path, desired_s3_uri=s3_uri, session=self$sagemaker_session)
       path = file.path(s3_uri, basename(path))
     }
     return (path)
   },

   # Waits for the schedule associated with this monitor to no longer be in the 'Pending'
   # state.
   .wait_for_schedule_changes_to_apply = function(){
      f <- retries(max_retry_count=36, # 36*5 = 3min
                   seconds_to_sleep=5,
                   exception_message_prefix = "Waiting for schedule to leave 'Pending' status")
      while(f()){
         schedule_desc = self$describe_schedule()
         if(schedule_desc[["MonitoringScheduleStatus"]] != "Pending")
            break
      }
   },

   # Validates that EnableInterContainerTrafficEncryption is not set in the provided
   # NetworkConfig request dictionary.
   # Args:
   #   network_config_dict (dict): NetworkConfig request dictionary.
   # Contains parameters from :class:`~sagemaker.network.NetworkConfig` object
   # that configures network isolation, encryption of
   # inter-container traffic, security group IDs, and subnets.
   .validate_network_config = function(network_config_list){
     if ("EnableInterContainerTrafficEncryption" %in% names(network_config_list)){
       msg = paste("EnableInterContainerTrafficEncryption is not supported in Model Monitor. ",
                   "Please ensure that encrypt_inter_container_traffic=None ",
                   "when creating your NetworkConfig object. ",
             sprintf("Current encrypt_inter_container_traffic value: %s",
                     self$network_config$encrypt_inter_container_traffic), sep = "\n")

       LOGGER$info(msg)
       ValueError$new(msg)
     }
   },

   # Creates a monitoring schedule.
   # Args:
   #    monitor_schedule_name (str): Monitoring schedule name.
   # job_definition_name (str): Job definition name.
   # schedule_cron_expression (str): The cron expression that dictates the frequency that
   # this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
   # expressions. Default: Daily.
   .create_monitoring_schedule_from_job_definition = function(monitor_schedule_name,
                                                              job_definition_name,
                                                              schedule_cron_expression=NULL){
      message = sprintf("Creating Monitoring Schedule with name: %s", monitor_schedule_name)
      LOGGER$info(message)

      monitoring_schedule_config = list(
         "MonitoringJobDefinitionName"=job_definition_name,
         "MonitoringType"=self$monitoring_type())
      if (!is.null(schedule_cron_expression))
         monitoring_schedule_config[["ScheduleConfig"]] = list(
            "ScheduleExpression"=schedule_cron_expression)
      self$sagemaker_session$sagemaker$create_monitoring_schedule(
         MonitoringScheduleName=monitor_schedule_name,
         MonitoringScheduleConfig=monitoring_schedule_config,
         Tags=self$tags %||% list())
   },

   # Generates a ProcessingInput object from a source. Source can be a local path or an S3
   # uri.
   # Args:
   #   source (str): The source of the data. This can be a local path or an S3 uri.
   # destination (str): The desired container path for the data to be downloaded to.
   # name (str): The name of the ProcessingInput.
   # Returns:
   #   sagemaker.processing.ProcessingInput: The ProcessingInput object.
   .upload_and_convert_to_processing_input = function(source = NULL,
                                                      destination = NULL,
                                                      name = NULL){
      if (is.null(source))
         return(NULL)

      parse_result = url_parse(source)

      if (parse_result$scheme != "s3"){
         s3_uri = file.path(
            "s3://",
            self$sagemaker_session$default_bucket(),
            .MODEL_MONITOR_S3_PATH,
            .BASELINING_S3_PATH,
            self$latest_baselining_job_name,
            .INPUT_S3_PATH,
            name)
         S3Uploader$new()$upload(
            local_path=source, desired_s3_uri=s3_uri, session=self$sagemaker_session)
         source = s3_uri
      }
      return(ProcessingInput$new(source=source, destination=destination, input_name=name))
   },

   # Updates existing monitoring schedule with new job definition and/or schedule expression.
   # Args:
   #    job_definition_name (str): Job definition name.
   # schedule_cron_expression (str or None): The cron expression that dictates the frequency
   # that this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
   # expressions.
   .update_monitoring_schedule = function(job_definition_name,
                                          schedule_cron_expression=NULL){
      if (is.null(self.job_definition_name) || is.null(self.monitoring_schedule_name)){
         message = "Nothing to update, please create a schedule first."
         LOGGER$error(message)
         ValueError$new(message)
      }
      monitoring_schedule_config = list(
         "MonitoringJobDefinitionName"= job_definition_name,
         "MonitoringType"=self$monitoring_type())
      if (!is.null(schedule_cron_expression)){
         monitoring_schedule_config[["ScheduleConfig"]] = list(
            "ScheduleExpression"=schedule_cron_expression)
         }
      self$sagemaker_session$sagemaker$update_monitoring_schedule(
         MonitoringScheduleName=self$monitoring_schedule_name,
         MonitoringScheduleConfig=monitoring_schedule_config)
      private$.wait_for_schedule_changes_to_apply()
   },
   .frame_work="model-monitor"
  ),
  lock_objects = F
)

#' @title DefaultModelMonitor Class
#' @description Sets up Amazon SageMaker Monitoring Schedules and baseline suggestions. Use this class when
#'              you want to utilize Amazon SageMaker Monitoring's plug-and-play solution that only requires
#'              your dataset and optional pre/postprocessing scripts.
#'              For a more customized experience, consider using the ModelMonitor class instead.
#' @export
DefaultModelMonitor = R6Class("DefaultModelMonitor",
  inherit = ModelMonitor,
  public = list(

    #' @field JOB_DEFINITION_BASE_NAME
    #' Model definition base name
    JOB_DEFINITION_BASE_NAME = "data-quality-job-definition",

    #' @description Initializes a ``Monitor`` instance. The Monitor handles baselining datasets and
    #'              creating Amazon SageMaker Monitoring Schedules to monitor SageMaker endpoints.
    #' @param role (str): An AWS IAM role name or ARN. The Amazon SageMaker jobs use this role.
    #' @param instance_count (int): The number of instances to run the jobs with.
    #' @param instance_type (str): Type of EC2 instance to use for the job, for example,
    #'              'ml.m5.xlarge'.
    #' @param volume_size_in_gb (int): Size in GB of the EBS volume
    #'              to use for storing data during processing (default: 30).
    #' @param volume_kms_key (str): A KMS key for the processing volume.
    #' @param output_kms_key (str): The KMS key id for the job's outputs.
    #' @param max_runtime_in_seconds (int): Timeout in seconds. After this amount of
    #'              time, Amazon SageMaker terminates the job regardless of its current status.
    #'              Default: 3600
    #' @param base_job_name (str): Prefix for the job name. If not specified,
    #'              a default name is generated based on the training image name and
    #'              current timestamp.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, one is created using
    #'              the default AWS configuration chain.
    #' @param env (dict): Environment variables to be passed to the job.
    #' @param tags ([dict]): List of tags to be passed to the job.
    #' @param network_config (sagemaker.network.NetworkConfig): A NetworkConfig
    #'              object that configures network isolation, encryption of
    #'              inter-container traffic, security group IDs, and subnets.
    initialize = function(role,
                          instance_count=1,
                          instance_type="ml.m5.xlarge",
                          volume_size_in_gb=30,
                          volume_kms_key=NULL,
                          output_kms_key=NULL,
                          max_runtime_in_seconds=NULL,
                          base_job_name=NULL,
                          sagemaker_session=NULL,
                          env=NULL,
                          tags=NULL,
                          network_config=NULL){
      session = sagemaker_session %||% Session$new()
      super$intialize(
        role=role,
        image_uri=private$.get_default_image_uri(session$paws_region_name),
        instance_count=instance_count,
        instance_type=instance_type,
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

    #' @description Type of the monitoring job.
    monitoring_type = function(){
       return("DataQuality")
    },

    #' @description Suggest baselines for use with Amazon SageMaker Model Monitoring Schedules.
    #' @param baseline_dataset (str): The path to the baseline_dataset file. This can be a local path
    #'              or an S3 uri.
    #' @param dataset_format (dict): The format of the baseline_dataset.
    #' @param record_preprocessor_script (str): The path to the record preprocessor script. This can
    #'              be a local path or an S3 uri.
    #' @param post_analytics_processor_script (str): The path to the record post-analytics processor
    #'              script. This can be a local path or an S3 uri.
    #' @param output_s3_uri (str): Desired S3 destination Destination of the constraint_violations
    #'              and statistics json files.
    #'              Default: "s3://<default_session_bucket>/<job_name>/output"
    #' @param wait (bool): Whether the call should wait until the job completes (default: True).
    #' @param logs (bool): Whether to show the logs produced by the job.
    #'              Only meaningful when wait is True (default: True).
    #' @param job_name (str): Processing job name. If not specified, the processor generates
    #'              a default job name, based on the image name and current timestamp.
    #' @return sagemaker.processing.ProcessingJob: The ProcessingJob object representing the
    #'              baselining job.
    suggest_baseline = function(baseline_dataset,
                                dataset_format,
                                record_preprocessor_script=NULL,
                                post_analytics_processor_script=NULL,
                                output_s3_uri=NULL,
                                wait=TRUE,
                                logs=TRUE,
                                job_name=NULL){

      self$latest_baselining_job_name = private$.generate_baselining_job_name(job_name=job_name)

      normalized_baseline_dataset_input = private$.upload_and_convert_to_processing_input(
        source=baseline_dataset,
        destination=file.path(
          .CONTAINER_BASE_PATH, .CONTAINER_INPUT_PATH, .BASELINE_DATASET_INPUT_NAME),
        name=.BASELINE_DATASET_INPUT_NAME)

      # Unlike other input, dataset must be a directory for the Monitoring image.
      baseline_dataset_container_path = normalized_baseline_dataset_input.destination

      normalized_record_preprocessor_script_input = private$.upload_and_convert_to_processing_input(
        source=record_preprocessor_script,
        destination=file.path(
          .CONTAINER_BASE_PATH, .CONTAINER_INPUT_PATH, .RECORD_PREPROCESSOR_SCRIPT_INPUT_NAME),
        name=.RECORD_PREPROCESSOR_SCRIPT_INPUT_NAME)

      record_preprocessor_script_container_path = NULL
      if (!islistempty(normalized_record_preprocessor_script_input)){
        record_preprocessor_script_container_path = file.path(
          normalized_record_preprocessor_script_input$destination,
          basename(record_preprocessor_script))}

      normalized_post_processor_script_input = private$.upload_and_convert_to_processing_input(
        source=post_analytics_processor_script,
        destination=file.path(
          .CONTAINER_BASE_PATH,
          .CONTAINER_INPUT_PATH,
          .POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME),
        name=.POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME)

      post_processor_script_container_path = NULL
      if (!islistempty(normalized_post_processor_script_input)){
        post_processor_script_container_path = file.path(
          normalized_post_processor_script_input$destination,
          asename(post_analytics_processor_script))}

      normalized_baseline_output = private$.normalize_baseline_output(output_s3_uri=output_s3_uri)

      normalized_env = private$.generate_env_map(
        env=self$env,
        dataset_format=dataset_format,
        output_path=normalized_baseline_output.source,
        enable_cloudwatch_metrics=FALSE,  # Only supported for monitoring schedules
        dataset_source_container_path=baseline_dataset_container_path,
        record_preprocessor_script_container_path=record_preprocessor_script_container_path,
        post_processor_script_container_path=post_processor_script_container_path)

      baselining_processor = Processor$new(
        role=self$role,
        image_uri=self$image_uri,
        instance_count=self$instance_count,
        instance_type=self$instance_type,
        entrypoint=self$entrypoint,
        volume_size_in_gb=self$volume_size_in_gb,
        volume_kms_key=self$volume_kms_key,
        output_kms_key=self$output_kms_key,
        max_runtime_in_seconds=self$max_runtime_in_seconds,
        base_job_name=self$base_job_name,
        sagemaker_session=self$sagemaker_session,
        env=normalized_env,
        tags=self$tags,
        network_config=self$network_config)

      baseline_job_inputs_with_nones = list(
        normalized_baseline_dataset_input,
        normalized_record_preprocessor_script_input,
        normalized_post_processor_script_input)

      baseline_job_inputs = baseline_job_inputs_with_nones[!sapply(baseline_job_inputs_with_nones, islistempty)]

      baselining_processor$run(
        inputs=baseline_job_inputs,
        outputs=list(normalized_baseline_output),
        arguments=self$arguments,
        wait=wait,
        logs=logs,
        job_name=self$latest_baselining_job_name)


      self$latest_baselining_job = BaseliningJob$new()$from_processing_job(
        processing_job=baselining_processor$latest_job)
      self$baselining_jobs = c(self$baselining_jobs, self$latest_baselining_job)
      return (baselining_processor$latest_job)
    },

    #' @description Creates a monitoring schedule to monitor an Amazon SageMaker Endpoint.
    #'              If constraints and statistics are provided, or if they are able to be retrieved from a
    #'              previous baselining job associated with this monitor, those will be used.
    #'              If constraints and statistics cannot be automatically retrieved, baseline_inputs will be
    #'              required in order to kick off a baselining job.
    #' @param endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
    #'              This can either be the endpoint name or an EndpointInput.
    #' @param record_preprocessor_script (str): The path to the record preprocessor script. This can
    #'              be a local path or an S3 uri.
    #' @param post_analytics_processor_script (str): The path to the record post-analytics processor
    #'              script. This can be a local path or an S3 uri.
    #' @param output_s3_uri (str): Desired S3 destination of the constraint_violations and
    #'              statistics json files.
    #'              Default: "s3://<default_session_bucket>/<job_name>/output"
    #' @param constraints (sagemaker.model_monitor.Constraints or str): If provided alongside
    #'              statistics, these will be used for monitoring the endpoint. This can be a
    #'              sagemaker.model_monitor.Constraints object or an s3_uri pointing to a constraints
    #'              JSON file.
    #' @param statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
    #'              constraints, these will be used for monitoring the endpoint. This can be a
    #'              sagemaker.model_monitor.Constraints object or an s3_uri pointing to a constraints
    #'              JSON file.
    #' @param monitor_schedule_name (str): Schedule name. If not specified, the processor generates
    #'              a default job name, based on the image name and current timestamp.
    #' @param schedule_cron_expression (str): The cron expression that dictates the frequency that
    #'              this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
    #'              expressions. Default: Daily.
    #' @param enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
    #'              the baselining or monitoring jobs.
    create_monitoring_schedule = function(endpoint_input,
                                          record_preprocessor_script=NULL,
                                          post_analytics_processor_script=NULL,
                                          output_s3_uri=NULL,
                                          constraints=NULL,
                                          statistics=NULL,
                                          monitor_schedule_name=NULL,
                                          schedule_cron_expression=NULL,
                                          enable_cloudwatch_metrics=TRUE){

      if (!is.null(self$job_definition_name) || !is.null(self$monitoring_schedule_name)){
        msg = paste(
          "It seems that this object was already used to create an Amazon Model",
          "Monitoring Schedule. To create another, first delete the existing one",
          "using my_monitor.delete_monitoring_schedule().")
        LOGGER$error(msg)
        ValueError$new(msg)
      }

      # create job definition
      monitoring_schedule_name = private$.generate_monitoring_schedule_name(
        schedule_name=monitor_schedule_name)
      new_job_definition_name = name_from_base(self$JOB_DEFINITION_BASE_NAME)
      request_dict = private$.build_create_data_quality_job_definition_request(
         monitoring_schedule_name=monitor_schedule_name,
         job_definition_name=new_job_definition_name,
         image_uri=self$image_uri,
         latest_baselining_job_name=self$latest_baselining_job_name,
         endpoint_input=endpoint_input,
         record_preprocessor_script=record_preprocessor_script,
         post_analytics_processor_script=post_analytics_processor_script,
         output_s3_uri=private$.normalize_monitoring_output(
            monitor_schedule_name, output_s3_uri)$destination,
         constraints=constraints,
         statistics=statistics,
         enable_cloudwatch_metrics=enable_cloudwatch_metrics,
         role=self$role,
         instance_count=self$instance_count,
         instance_type=self$instance_type,
         volume_size_in_gb=self$volume_size_in_gb,
         volume_kms_key=self$volume_kms_key,
         output_kms_key=self$output_kms_key,
         max_runtime_in_seconds=self$max_runtime_in_seconds,
         env=self$env,
         tags=self$tags,
         network_config=self$network_config)

      do.call(
         self$sagemaker_session$sagemaker$create_data_quality_job_definition,
         request_dict)

      # create schedule
      tryCatch({
         private$.create_monitoring_schedule_from_job_definition(
            monitor_schedule_name=monitor_schedule_name,
            job_definition_name=new_job_definition_name,
            schedule_cron_expression=schedule_cron_expression)
         self$job_definition_name = new_job_definition_name
         self$monitoring_schedule_name = monitor_schedule_name
         },
         error = function(e){
            LOGGER$error("Failed to create_monitoring schedule.")

            tryCatch(
               self$sagemaker_session$sagemaker$delete_data_quality_job_definition(
                  JobDefinitionName=new_job_definition_name),
               error = function(ee){
                  message = sprintf("Failed to delete job definition %s.",new_job_definition_name)
                  LOGGER$error(message)
                  stop(ee)
               })
         })
    },

    #' @description Updates the existing monitoring schedule.
    #' @param endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
    #'              This can either be the endpoint name or an EndpointInput.
    #' @param record_preprocessor_script (str): The path to the record preprocessor script. This can
    #'              be a local path or an S3 uri.
    #' @param post_analytics_processor_script (str): The path to the record post-analytics processor
    #'              script. This can be a local path or an S3 uri.
    #' @param output_s3_uri (str): Desired S3 destination of the constraint_violations and
    #'              statistics json files.
    #' @param statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
    #'              constraints, these will be used for monitoring the endpoint. This can be a
    #'              sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
    #'              JSON file.
    #' @param constraints (sagemaker.model_monitor.Constraints or str): If provided alongside
    #'              statistics, these will be used for monitoring the endpoint. This can be a
    #'              sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
    #'              JSON file.
    #' @param schedule_cron_expression (str): The cron expression that dictates the frequency that
    #'              this job runs at. See sagemaker.model_monitor.CronExpressionGenerator for valid
    #'              expressions.
    #' @param instance_count (int): The number of instances to run
    #'              the jobs with.
    #' @param instance_type (str): Type of EC2 instance to use for
    #'              the job, for example, 'ml.m5.xlarge'.
    #' @param volume_size_in_gb (int): Size in GB of the EBS volume
    #'              to use for storing data during processing (default: 30).
    #' @param volume_kms_key (str): A KMS key for the job's volume.
    #' @param output_kms_key (str): The KMS key id for the job's outputs.
    #' @param max_runtime_in_seconds (int): Timeout in seconds. After this amount of
    #'              time, Amazon SageMaker terminates the job regardless of its current status.
    #'              Default: 3600
    #' @param env (dict): Environment variables to be passed to the job.
    #' @param network_config (sagemaker.network.NetworkConfig): A NetworkConfig
    #'              object that configures network isolation, encryption of
    #'              inter-container traffic, security group IDs, and subnets.
    #' @param enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
    #'              the baselining or monitoring jobs.
    #' @param role (str): An AWS IAM role name or ARN. The Amazon SageMaker jobs use this role.
    update_monitoring_schedule = function(endpoint_input=NULL,
                                          record_preprocessor_script=NULL,
                                          post_analytics_processor_script=NULL,
                                          output_s3_uri=NULL,
                                          statistics=NULL,
                                          constraints=NULL,
                                          schedule_cron_expression=NULL,
                                          instance_count=NULL,
                                          instance_type=NULL,
                                          volume_size_in_gb=NULL,
                                          volume_kms_key=NULL,
                                          output_kms_key=NULL,
                                          max_runtime_in_seconds=NULL,
                                          env=NULL,
                                          network_config=NULL,
                                          enable_cloudwatch_metrics=NULL,
                                          role=NULL){
       # check if this schedule is in v2 format and update as per v2 format if it is
       if (!is.null(self$job_definition_name)){
          private$.update_data_quality_monitoring_schedule(
             endpoint_input=endpoint_input,
             record_preprocessor_script=record_preprocessor_script,
             post_analytics_processor_script=post_analytics_processor_script,
             output_s3_uri=output_s3_uri,
             statistics=statistics,
             constraints=constraints,
             schedule_cron_expression=schedule_cron_expression,
             instance_count=instance_count,
             instance_type=instance_type,
             volume_size_in_gb=volume_size_in_gb,
             volume_kms_key=volume_kms_key,
             output_kms_key=output_kms_key,
             max_runtime_in_seconds=max_runtime_in_seconds,
             env=env,
             network_config=network_config,
             enable_cloudwatch_metrics=enable_cloudwatch_metrics,
             role=role)
          return(NULL)}

      monitoring_inputs = NULL
      if (!is.null(endpoint_input)){
        monitoring_inputs = list(private$.normalize_endpoint_input(endpoint_input)$to_request_list())}

      record_preprocessor_script_s3_uri = NULL
      if (!is.null(record_preprocessor_script)){
        record_preprocessor_script_s3_uri = private$.s3_uri_from_local_path(
          path=record_preprocessor_script)}

      post_analytics_processor_script_s3_uri = NULL
      if (!is.null(post_analytics_processor_script)){
        post_analytics_processor_script_s3_uri = private$.s3_uri_from_local_path(
          path=post_analytics_processor_script)}

      monitoring_output_config = NULL
      output_path = NULL
      if (!is.null(output_s3_uri)){
        normalized_monitoring_output = private$.normalize_monitoring_output(
          output_s3_uri=output_s3_uri)
        monitoring_output_config = list(
          "MonitoringOutputs"= list(normalized_monitoring_output$to_request_list()))
        output_path = normalized_monitoring_output.source}

      if (!is.null(env))
        self$env = env

      normalized_env = private$.generate_env_map(
        env=env, output_path=output_path, enable_cloudwatch_metrics=enable_cloudwatch_metrics)

      file_objects = private$.get_baseline_files(
        statistics=statistics, constraints=constraints, sagemaker_session=self$sagemaker_session)

      constraints_s3_uri = NULL
      if (!is.null(file_objects$constraints)){
        constraints_s3_uri = file_objects$constraints$file_s3_uri}

      statistics_s3_uri = NULL
      if (!is.null(file_objects$statistics)){
        statistics_s3_uri = file_objects$statistics$file_s3_uri}

      if (!is.null(instance_type)){
        self$instance_type = instance_type}

      if (!is.null(instance_count)){
        self$instance_count = instance_count}

      if (!is.null(volume_size_in_gb)){
        self$volume_size_in_gb = volume_size_in_gb}

      if (!is.null(volume_kms_key)){
        self$volume_kms_key = volume_kms_key}

      if (!is.null(output_kms_key)){
        self$output_kms_key = output_kms_key
        monitoring_output_config$KmsKeyId = self$output_kms_key}

      if (!is.null(max_runtime_in_seconds)){
        self$max_runtime_in_seconds = max_runtime_in_seconds}

      if (!is.null(network_config)){
        self$network_config = network_config}

      network_config_list = NULL
      if (!is.null(self$network_config)){
        network_config_list = self$network_config$to_request_list()
        super$.validate_network_config(network_config_list)}

      if (!is.null(role)){
        self$role = role}

      self$sagemaker_session$update_monitoring_schedule(
        monitoring_schedule_name=self$monitoring_schedule_name,
        schedule_expression=schedule_cron_expression,
        constraints_s3_uri=constraints_s3_uri,
        statistics_s3_uri=statistics_s3_uri,
        monitoring_inputs=monitoring_inputs,
        monitoring_output_config=monitoring_output_config,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size_in_gb=volume_size_in_gb,
        volume_kms_key=volume_kms_key,
        record_preprocessor_source_uri=record_preprocessor_script_s3_uri,
        post_analytics_processor_source_uri=post_analytics_processor_script_s3_uri,
        max_runtime_in_seconds=max_runtime_in_seconds,
        environment=normalized_env,
        network_config=network_config_dict,
        role_arn=self$sagemaker_session$expand_role(self$role))

      private$.wait_for_schedule_changes_to_apply()
    },

    #' @description Deletes the monitoring schedule and its job definition.
    delete_monitoring_schedule = function(){
       super$delete_monitoring_schedule()
       if(!is.null(self$job_definition_name)){
          # Delete job definition.
          message = sprintf("Deleting Data Quality Job Definition with name: %s",
             self$job_definition_name
          )
          LOGGER$info(message)
          self$sagemaker_session$sagemaker$delete_data_quality_job_definition(
             JobDefinitionName=self$job_definition_name
          )
          self$job_definition_name = NULL
       }
    },

    #' @description `run_baseline()` is only allowed for ModelMonitor objects. Please use suggest_baseline
    #'               for DefaultModelMonitor objects, instead.
    run_baseline = function(){
      NotImplementedError$new(
         "`run_baseline()`` is only allowed for ModelMonitor objects.",
         " Please use suggest_baseline for DefaultModelMonitor objects, instead.")
    },

    #' @description Sets this object's schedule name to point to the Amazon Sagemaker Monitoring Schedule
    #'              name provided. This allows subsequent describe_schedule or list_executions calls to point
    #'              to the given schedule.
    #' @param monitor_schedule_name (str): The name of the schedule to attach to.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, one is created using
    #'              the default AWS configuration chain.
    attach = function(monitor_schedule_name,
                      sagemaker_session=NULL){
      sagemaker_session = sagemaker_session %||% Session$new()
      schedule_desc = sagemaker_session$describe_monitoring_schedule(
        monitoring_schedule_name=monitor_schedule_name)

      job_definition_name = schedule_desc[["MonitoringScheduleConfig"]][[
         "MonitoringJobDefinitionName"]]

      if(!islistempty(job_definition_name)){
         monitoring_type = schedule_desc[["MonitoringScheduleConfig"]][["MonitoringType"]]
         if(monitoring_type != self$monitor_type()){
            TypeError$new(
               sprintf("%s can only attach to Data quality monitoring schedule.",
                       class(self)[1])
            )
         }
         job_desc = sagemaker_session.sagemaker_client.describe_data_quality_job_definition(
            JobDefinitionName=job_definition_name
         )
         tags = sagemaker_session$list_tags(resource_arn=schedule_desc[["MonitoringScheduleArn"]])
         return (ModelMonitor$private_methods$.attach(
            clazz=self,
            sagemaker_session=sagemaker_session,
            schedule_desc=schedule_desc,
            job_desc=job_desc,
            tags=tags)
         )
      }

      job_definition = schedule_desc[["MonitoringScheduleConfig"]][["MonitoringJobDefinition"]]
      role = job_definition[["RoleArn"]]
      cluster_config = job_definition[["MonitoringResources"]][["ClusterConfig"]]
      instance_count = cluster_config[["InstanceCount"]]
      instance_type = cluster_config[["InstanceType"]]
      volume_size_in_gb = cluster_config[["VolumeSizeInGB"]]
      volume_kms_key = cluster_config[["VolumeKmsKeyId"]]
      output_kms_key = job_definition[["MonitoringOutputConfig"]][["KmsKeyId"]]
      max_runtime_in_seconds = job_definition[["StoppingCondition"]][["MaxRuntimeInSeconds"]]
      env = job_definition[["Environment"]]

      network_config_list = job_definition[["NetworkConfig"]] %||% list()
      network_config = NULL
      if (!islistempty(network_config_list)){
         vpc_config = network_config_list[["VpcConfig"]] %||% list()
         security_group_ids = vpc_config[["SecurityGroupIds"]]
         subnets = vpc_config[["Subnets"]]

         network_config = NetworkConfig$new(
            enable_network_isolation=network_config_list$EnableNetworkIsolation,
            security_group_ids=security_group_ids,
            subnets=subnets)
      }

      tags = sagemaker_session$list_tags(resource_arn=schedule_desc$MonitoringScheduleArn)

      attached_monitor = self$clone()
      attached_monitor$role = role
      attached_monitor$instance_count=instance_count
      attached_monitor$instance_type=instance_type
      attached_monitor$volume_size_in_gb=volume_size_in_gb
      attached_monitor$volume_kms_key=volume_kms_key
      attached_monitor$output_kms_key=output_kms_key
      attached_monitor$max_runtime_in_seconds=max_runtime_in_seconds
      attached_monitor$sagemaker_session=sagemaker_session
      attached_monitor$env=env
      attached_monitor$tags=tags
      attached_monitor$network_config=network_config
      attached_monitor$monitoring_schedule_name=monitor_schedule_name

      return(attached_monitor)
    },

    #' @description Returns the sagemaker.model_monitor.Statistics generated by the latest monitoring
    #'              execution.
    #' @return sagemaker.model_monitoring.Statistics: The Statistics object representing the file
    #'              generated by the latest monitoring execution.
    latest_monitoring_statistics = function(){
      executions = self$list_executions()
      if (islistempty(executions)){
        message(sprintf("No executions found for schedule. monitoring_schedule_name: %s",
                 self$monitoring_schedule_name))
        return(NULL)}

      latest_monitoring_execution = executions[[length(executions)]]

      tryCatch(
         latest_monitoring_execution$statistics(),
         error = function(e){
            status = latest_monitoring_execution$describe()$ProcessingJobStatus
            message(paste0("Unable to retrieve statistics as job is in status '%s'. Latest statistics only ",
                            "available for completed executions."),status)}
         )
    },

    #' @description Returns the sagemaker.model_monitor.ConstraintViolations generated by the latest
    #'              monitoring execution.
    #' @return sagemaker.model_monitoring.ConstraintViolations: The ConstraintViolations object
    #'              representing the file generated by the latest monitoring execution.
    latest_monitoring_constraint_violations = function(){
      executions = self$list_executions()
      if (islistempty(executions)){
        message("No executions found for schedule. monitoring_schedule_name: %s",
                 self$monitoring_schedule_name)
        return(NULL)}

      latest_monitoring_execution = executions[[length(executions)]]

      tryCatch(
         latest_monitoring_execution$constraint_violations(),
         error = function(e){
            status = latest_monitoring_execution$describe()$ProcessingJobStatus
            message(sprintf(
               paste("Unable to retrieve constraint violations as job is in status '%s'. Latest",
                     "violations only available for completed executions."),status))
            }
         )
    }
  ),
  private = list(

     # Updates the existing monitoring schedule.
     # Args:
     #    endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
     # This can either be the endpoint name or an EndpointInput.
     # record_preprocessor_script (str): The path to the record preprocessor script. This can
     # be a local path or an S3 uri.
     # post_analytics_processor_script (str): The path to the record post-analytics processor
     # script. This can be a local path or an S3 uri.
     # output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
     # Default: "s3://<default_session_bucket>/<job_name>/output"
     # constraints (sagemaker.model_monitor.Constraints or str): If provided it will be used
     # for monitoring the endpoint. It can be a Constraints object or an S3 uri pointing
     # to a constraints JSON file.
     # statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
     # constraints, these will be used for monitoring the endpoint. This can be a
     # sagemaker.model_monitor.Statistics object or an S3 uri pointing to a statistics
     # JSON file.
     # schedule_cron_expression (str): The cron expression that dictates the frequency that
     # this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
     # expressions. Default: Daily.
     # enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
     # the baselining or monitoring jobs.
     # role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
     # instance_count (int): The number of instances to run
     # the jobs with.
     # instance_type (str): Type of EC2 instance to use for
     # the job, for example, 'ml.m5.xlarge'.
     # volume_size_in_gb (int): Size in GB of the EBS volume
     # to use for storing data during processing (default: 30).
     # volume_kms_key (str): A KMS key for the job's volume.
     #        output_kms_key (str): The KMS key id for the job's outputs.
     # max_runtime_in_seconds (int): Timeout in seconds. After this amount of
     # time, Amazon SageMaker terminates the job regardless of its current status.
     # Default: 3600
     # env (dict): Environment variables to be passed to the job.
     # network_config (sagemaker.network.NetworkConfig): A NetworkConfig
     # object that configures network isolation, encryption of
     # inter-container traffic, security group IDs, and subnets.
    .update_data_quality_monitoring_schedule = function(endpoint_input=NULL,
                                                        record_preprocessor_script=NULL,
                                                        post_analytics_processor_script=NULL,
                                                        output_s3_uri=NULL,
                                                        constraints=NULL,
                                                        statistics=NULL,
                                                        schedule_cron_expression=NULL,
                                                        enable_cloudwatch_metrics=NULL,
                                                        role=NULL,
                                                        instance_count=NULL,
                                                        instance_type=NULL,
                                                        volume_size_in_gb=NULL,
                                                        volume_kms_key=NULL,
                                                        output_kms_key=NULL,
                                                        max_runtime_in_seconds=NULL,
                                                        env=NULL,
                                                        network_config=NULL){
       valid_args = Filter(Negate(is.null), as.list(environment()))

       # Nothing to update
       if (length(valid_args) <= 0)
          return

       # Only need to update schedule expression
       if (length(valid_args) == 1 && !is.null(schedule_cron_expression)){
          private$.update_monitoring_schedule(self$job_definition_name, schedule_cron_expression)
          return(NULL)
       }

       # Need to update schedule with a new job definition
       job_desc = self$sagemaker_session$sagemaker$describe_data_quality_job_definition(
          JobDefinitionName=self$job_definition_name)

       new_job_definition_name = name_from_base(self$JOB_DEFINITION_BASE_NAME)
       request_dict = private$.build_create_data_quality_job_definition_request(
          monitoring_schedule_name=self$monitoring_schedule_name,
          job_definition_name=new_job_definition_name,
          image_uri=self$image_uri,
          existing_job_desc=job_desc,
          endpoint_input=endpoint_input,
          record_preprocessor_script=record_preprocessor_script,
          post_analytics_processor_script=post_analytics_processor_script,
          output_s3_uri=output_s3_uri,
          statistics=statistics,
          constraints=constraints,
          enable_cloudwatch_metrics=enable_cloudwatch_metrics,
          role=role,
          instance_count=instance_count,
          instance_type=instance_type,
          volume_size_in_gb=volume_size_in_gb,
          volume_kms_key=volume_kms_key,
          output_kms_key=output_kms_key,
          max_runtime_in_seconds=max_runtime_in_seconds,
          env=env,
          tags=self$tags,
          network_config=network_config)

       do.call(self$sagemaker_session$sagemaker$create_data_quality_job_definition, request_dict)
       tryCatch({
          private$.update_monitoring_schedule(new_job_definition_name, schedule_cron_expression)
          self$job_definition_name = new_job_definition_name
          if (!is.null(role))
             self$role = role
          if (!is.null(instance_count))
             self$instance_count = instance_count
          if (!is.null(instance_type))
             self$instance_type = instance_type
          if (!is.null(volume_size_in_gb))
             self$volume_size_in_gb = volume_size_in_gb
          if (!is.null(volume_kms_key))
             self$volume_kms_key = volume_kms_key
          if (!is.null(output_kms_key))
             self$output_kms_key = output_kms_key
          if (!is.null(max_runtime_in_seconds))
             self$max_runtime_in_seconds = max_runtime_in_seconds
          if (!is.null(env))
             self$env = env
          if (!is.null(network_config))
             self$network_config = network_config
          },
       error=function(e){
          LOGGER$error("Failed to update monitoring schedule.")
          # noinspection PyBroadException
          tryCatch(
             self$sagemaker_session$sagemaker$delete_data_quality_job_definition(
                JobDefinitionName=new_job_definition_name),
             error = function(ee){
                message = sprintf("Failed to delete job definition %s.", new_job_definition_name)
                LOGGER$error(message)
                stop(ee)
            })
         })
    },

    # Ensure that the output is a MonitoringOutput object.
    # Args:
    #   output_s3_uri (str): The output S3 uri to deposit the monitoring evaluation files in.
    # Returns:
    #   sagemaker.model_monitor.MonitoringOutput: The normalized MonitoringOutput object.
    .normalize_monitoring_output = function(output_s3_uri=NULL){
      s3_uri = output_s3_uri %||% file.path(
        "s3://",
        self$sagemaker_session$default_bucket(),
        .MODEL_MONITOR_S3_PATH,
        .MONITORING_S3_PATH,
        self$monitoring_schedule_name,
        .RESULTS_S3_PATH)
      output = MonitoringOutput$new(
        source=file.path(.CONTAINER_BASE_PATH, .CONTAINER_OUTPUT_PATH), destination=s3_uri)

      return(output)
    },

    # Generate a list of environment variables from first-class parameters.
    # Args:
    #   dataset_format (dict): The format of the baseline_dataset.
    # output_path (str): Local path to the output.
    # record_preprocessor_script_container_path (str): The path to the record preprocessor
    #   script.
    # post_processor_script_container_path (str): The path to the post analytics processor
    #   script.
    # dataset_source_container_path (str): The path to the dataset source.
    # Returns:
    #   dict: Dictionary of environment keys and values.
    .generate_env_map = function(env,
                                 output_path=NULL,
                                 enable_cloudwatch_metrics=NULL,
                                 record_preprocessor_script_container_path=NULL,
                                 post_processor_script_container_path=NULL,
                                 dataset_format=NULL,
                                 dataset_source_container_path=NULL){
      env = env %||% list()

      env[[.OUTPUT_PATH_ENV_NAME]] = output_path

      if (!is.null(enable_cloudwatch_metrics))
        env[[.PUBLISH_CLOUDWATCH_METRICS_ENV_NAME]] = ifelse(enable_cloudwatch_metrics, "Enabled", "Disabled")

      env[[.DATASET_FORMAT_ENV_NAME]] = list(dataset_format)
      env[[.RECORD_PREPROCESSOR_SCRIPT_ENV_NAME]] = record_preprocessor_script_container_path
      env[[.POST_ANALYTICS_PROCESSOR_SCRIPT_ENV_NAME]] = post_processor_script_container_path
      env[[.DATASET_SOURCE_PATH_ENV_NAME]] = dataset_source_container_path

      return(env)
    },

    # Returns the Default Model Monitoring image uri based on the region.
    # Args:
    #   region (str): The AWS region.
    # Returns:
    #   str: The Default Model Monitoring image uri based on the region.
    .get_default_image_uri = function(region){
      return (ImageUris$new()$retrieve(framework=private$.frame_work, region=region))
    },

    # Build the request for job definition creation API
    # Args:
    #    monitoring_schedule_name (str): Monitoring schedule name.
    # job_definition_name (str): Job definition name.
    # If not specified then a default one will be generated.
    # image_uri (str): The uri of the image to use for the jobs started by the Monitor.
    # latest_baselining_job_name (str): name of the last baselining job.
    # existing_job_desc (dict): description of existing job definition. It will be updated by
    # values that were passed in, and then used to create the new job definition.
    # endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
    # This can either be the endpoint name or an EndpointInput.
    # output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
    # Default: "s3://<default_session_bucket>/<job_name>/output"
    # constraints (sagemaker.model_monitor.Constraints or str): If provided it will be used
    # for monitoring the endpoint. It can be a Constraints object or an S3 uri pointing
    # to a constraints JSON file.
    # enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
    # the baselining or monitoring jobs.
    # role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
    # instance_count (int): The number of instances to run
    # the jobs with.
    # instance_type (str): Type of EC2 instance to use for
    # the job, for example, 'ml.m5.xlarge'.
    # volume_size_in_gb (int): Size in GB of the EBS volume
    # to use for storing data during processing (default: 30).
    # volume_kms_key (str): A KMS key for the job's volume.
    # output_kms_key (str): KMS key id for output.
    # max_runtime_in_seconds (int): Timeout in seconds. After this amount of
    #     time, Amazon SageMaker terminates the job regardless of its current status.
    #     Default: 3600
    # env (dict): Environment variables to be passed to the job.
    # tags ([dict]): List of tags to be passed to the job.
    # network_config (sagemaker.network.NetworkConfig): A NetworkConfig
    #     object that configures network isolation, encryption of
    #     inter-container traffic, security group IDs, and subnets.
    # Returns:
    # dict: request parameters to create job definition.
    .build_create_data_quality_job_definition_request = function(
      monitoring_schedule_name,
      job_definition_name,
      image_uri,
      latest_baselining_job_name=NULL,
      existing_job_desc=NULL,
      endpoint_input=NULL,
      record_preprocessor_script=NULL,
      post_analytics_processor_script=NULL,
      output_s3_uri=NULL,
      statistics=NULL,
      constraints=NULL,
      enable_cloudwatch_metrics=NULL,
      role=NULL,
      instance_count=NULL,
      instance_type=NULL,
      volume_size_in_gb=NULL,
      volume_kms_key=NULL,
      output_kms_key=NULL,
      max_runtime_in_seconds=NULL,
      env=NULL,
      tags=NULL,
      network_config=NULL){
       if (!is.null(existing_job_desc)){
          app_specification = existing_job_desc[[
             sprintf("%sAppSpecification",self$monitoring_type())
          ]]
          baseline_config = existing_job_desc[[
             sprintf("%sBaselineConfig", self$monitoring_type())]] %||% list()
          job_input = existing_job_desc[[sprintf("%sJobInput",self$monitoring_type())]]
          job_output = existing_job_desc[[sprintf("%sJobOutputConfig",self$monitoring_type())]]
          cluster_config = existing_job_desc[["JobResources"]][["ClusterConfig"]]
          if (is.null(role))
             role = existing_job_desc[["RoleArn"]]
          existing_network_config = existing_job_desc[["NetworkConfig"]]
          stop_condition = existing_job_desc[["StoppingCondition"]] %||% list()
       } else {
          app_specification = list()
          baseline_config = list()
          job_input = list()
          job_output = list()
          cluster_config = list()
          existing_network_config = NULL
          stop_condition = list()
      }

      # app specification
      record_preprocessor_script_s3_uri = NULL
      if (!is.null(record_preprocessor_script))
         record_preprocessor_script_s3_uri = private$.s3_uri_from_local_path(
            path=record_preprocessor_script)

      post_analytics_processor_script_s3_uri = NULL
      if (!is.null(post_analytics_processor_script))
         post_analytics_processor_script_s3_uri = private$.s3_uri_from_local_path(
            path=post_analytics_processor_script)

      app_specification[["ImageUri"]] = image_uri
      if (!islistempty(post_analytics_processor_script_s3_uri))
         app_specification[[
          "PostAnalyticsProcessorSourceUri"
       ]] = post_analytics_processor_script_s3_uri
      if (!islistempty(record_preprocessor_script_s3_uri))
         app_specification[["RecordPreprocessorSourceUri"]] = record_preprocessor_script_s3_uri

      normalized_env = private$.generate_env_map(
         env=env,
         enable_cloudwatch_metrics=enable_cloudwatch_metrics)
      if (!islistempty(normalized_env))
         app_specification[["Environment"]] = normalized_env

      # baseline config
      # noinspection PyTypeChecker
      ll = privatet$.get_baseline_files(
         statistics=statistics,
         constraints=constraints,
         sagemaker_session=self.sagemaker_session)
      if (!is.null(ll$constraints_object)){
         constraints_s3_uri = constraints_object$file_s3_uri
         baseline_config[["ConstraintsResource"]] = list(S3Uri=constraints_s3_uri)}
      if (!is.null(statistics_object)){
         statistics_s3_uri = statistics_object$file_s3_uri
         baseline_config[["StatisticsResource"]] = list(S3Uri=statistics_s3_uri)}
      # ConstraintsResource and BaseliningJobName can co-exist in BYOC case
      if (!is.null(latest_baselining_job_name))
         baseline_config[["BaseliningJobName"]] = latest_baselining_job_name

      # job input
      if (!is.null(endpoint_input)){
         normalized_endpoint_input = private$.normalize_endpoint_input(
            endpoint_input=endpoint_input)
         job_input = normalized_endpoint_input$to_request_list()
      }

      # job output
      if (!is.null(output_s3_uri)){
         normalized_monitoring_output = private$.normalize_monitoring_output(
            monitoring_schedule_name, output_s3_uri)
         job_output[["MonitoringOutputs"]] = list(normalized_monitoring_output$to_request_list())
      }
      if (!is.null(output_kms_key))
         job_output[["KmsKeyId"]] = output_kms_key

      # cluster config
      if (!is.null(instance_count))
         cluster_config[["InstanceCount"]] = instance_count
      if (!is.null(instance_type))
         cluster_config[["InstanceType"]] = instance_type
      if (!is.null(volume_size_in_gb))
         cluster_config[["VolumeSizeInGB"]] = volume_size_in_gb
      if (!is.null(volume_kms_key))
         cluster_config[["VolumeKmsKeyId"]] = volume_kms_key

      # stop condition
      if (!is.null(max_runtime_in_seconds))
         stop_condition[["MaxRuntimeInSeconds"]] = max_runtime_in_seconds

      request_dict = list(
         job_definition_name, app_specification, job_input,
         job_output, list("ClusterConfig"=cluster_config),
         self$sagemaker_session$expand_role(role))
      names(request_dict) = c(
         "JobDefinitionName",
         sprintf("%sAppSpecification",self$monitoring_type()),
         sprintf("%sJobInput",self$monitoring_type()),
         sprintf("%sJobOutputConfig",self$monitoring_type()),
         "JobResources",
         "RoleArn")

      if (!islistempty(baseline_config))
         request_dict[[sprintf("%sBaselineConfig",self$monitoring_type())]] = baseline_config

      if (!is.null(network_config)) {
         network_config_dict = network_config$.to_request_dict()
         private$.validate_network_config(network_config_dict)
         request_dict[["NetworkConfig"]] = network_config_dict
      } else if (!is.null(existing_network_config))
         request_dict[["NetworkConfig"]] = existing_network_config

      if (!islistempty(stop_condition))
         request_dict[["StoppingCondition"]] = stop_condition

      if (!is.null(tags))
         request_dict[["Tags"]] = tags

      return(request_dict)
   }
  ),
  lock_objects = F
)

#' @title Amazon SageMaker model monitor to monitor quality metrics for an endpoint.
#' @description Please see the `initialize` method of its base class for how to instantiate it.
#' @export
ModelQualityMonitor = R6Class("ModelQualityMonitor",
   inherit = ModelMonitor,
   public = list(

      #' @field JOB_DEFINITION_BASE_NAME
      #' Model definition base name
      JOB_DEFINITION_BASE_NAME = "model-quality-job-definition",

      #' @description Initializes a monitor instance.
      #'              The monitor handles baselining datasets and creating Amazon SageMaker
      #'              Monitoring Schedules to monitor SageMaker endpoints.
      #' @param role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
      #' @param instance_count (int): The number of instances to run
      #'              the jobs with.
      #' @param instance_type (str): Type of EC2 instance to use for
      #'              the job, for example, 'ml.m5.xlarge'.
      #' @param volume_size_in_gb (int): Size in GB of the EBS volume
      #'              to use for storing data during processing (default: 30).
      #' @param volume_kms_key (str): A KMS key for the job's volume.
      #' @param output_kms_key (str): The KMS key id for the job's outputs.
      #' @param max_runtime_in_seconds (int): Timeout in seconds. After this amount of
      #'              time, Amazon SageMaker terminates the job regardless of its current status.
      #'              Default: 3600
      #' @param base_job_name (str): Prefix for the job name. If not specified,
      #'              a default name is generated based on the training image name and
      #'              current timestamp.
      #' @param sagemaker_session (sagemaker.session.Session): Session object which
      #'              manages interactions with Amazon SageMaker APIs and any other
      #'              AWS services needed. If not specified, one is created using
      #'              the default AWS configuration chain.
      #' @param env (dict): Environment variables to be passed to the job.
      #' @param tags ([dict]): List of tags to be passed to the job.
      #' @param network_config (sagemaker.network.NetworkConfig): A NetworkConfig
      #'              object that configures network isolation, encryption of
      #'              inter-container traffic, security group IDs, and subnets.
      initialize = function(role,
                            instance_count=1,
                            instance_type="ml.m5.xlarge",
                            volume_size_in_gb=30,
                            volume_kms_key=NULL,
                            output_kms_key=NULL,
                            max_runtime_in_seconds=NULL,
                            base_job_name=NULL,
                            sagemaker_session=NULL,
                            env=NULL,
                            tags=NULL,
                            network_config=NULL){
         session = sagemaker_session %||% Session$new()
         super$initialize(
            role=role,
            image_uri=private$.get_default_image_uri(session$region_name),
            instance_count=instance_count,
            instance_type=instance_type,
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

      #' @description Type of the monitoring job.
      monitoring_type = function(){
         return("ModelQuality")
      },

      #' @description Suggest baselines for use with Amazon SageMaker Model Monitoring Schedules.
      #' @param baseline_dataset (str): The path to the baseline_dataset file. This can be a local
      #'              path or an S3 uri.
      #' @param dataset_format (dict): The format of the baseline_dataset.
      #' @param problem_type (str): The type of problem of this model quality monitoring. Valid
      #'              values are "Regression", "BinaryClassification", "MulticlassClassification".
      #' @param inference_attribute (str): Index or JSONpath to locate predicted label(s).
      #' @param probability_attribute (str or int): Index or JSONpath to locate probabilities.
      #' @param ground_truth_attribute (str): Index or JSONpath to locate actual label(s).
      #' @param probability_threshold_attribute (float): threshold to convert probabilities to binaries
      #'              Only used for ModelQualityMonitor, ModelBiasMonitor and ModelExplainabilityMonitor
      #' @param post_analytics_processor_script (str): The path to the record post-analytics processor
      #'              script. This can be a local path or an S3 uri.
      #' @param output_s3_uri (str): Desired S3 destination Destination of the constraint_violations
      #'              and statistics json files.
      #'              Default: "s3://<default_session_bucket>/<job_name>/output"
      #' @param wait (bool): Whether the call should wait until the job completes (default: False).
      #' @param logs (bool): Whether to show the logs produced by the job.
      #'              Only meaningful when wait is True (default: False).
      #' @param job_name (str): Processing job name. If not specified, the processor generates
      #'              a default job name, based on the image name and current timestamp.
      #' @return sagemaker.processing.ProcessingJob: The ProcessingJob object representing the
      #'              baselining job.
      suggest_baseline = function(baseline_dataset,
                                  dataset_format,
                                  problem_type,
                                  inference_attribute=NULL,
                                  probability_attribute=NULL,
                                  ground_truth_attribute=NULL,
                                  probability_threshold_attribute=NULL,
                                  post_analytics_processor_script=NULL,
                                  output_s3_uri=NULL,
                                  wait=FALSE,
                                  logs=FALSE,
                                  job_name=NULL){
         self$latest_baselining_job_name = private$.generate_baselining_job_name(job_name=job_name)

         normalized_baseline_dataset_input = private$.upload_and_convert_to_processing_input(
            source=baseline_dataset,
            destination=file.path(
               .CONTAINER_BASE_PATH, .CONTAINER_INPUT_PATH, .BASELINE_DATASET_INPUT_NAME),
            name=.BASELINE_DATASET_INPUT_NAME)

         # Unlike other input, dataset must be a directory for the Monitoring image.
         baseline_dataset_container_path = normalized_baseline_dataset_input$destination

         normalized_post_processor_script_input = private$.upload_and_convert_to_processing_input(
            source=post_analytics_processor_script,
            destination=file.path(
                  .CONTAINER_BASE_PATH,
                  .CONTAINER_INPUT_PATH,
                  .POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME),
            name=.POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME)

         post_processor_script_container_path = NULL
         if (!is.null(normalized_post_processor_script_input))
            post_processor_script_container_path = file.path(
                  normalized_post_processor_script_input$destination,
                  basename(post_analytics_processor_script))

         normalized_baseline_output = private$.normalize_baseline_output(output_s3_uri=output_s3_uri)

         normalized_env = private$.generate_env_map(
            env=self$env,
            dataset_format=dataset_format,
            output_path=normalized_baseline_output.source,
            enable_cloudwatch_metrics=FALSE,  # Only supported for monitoring schedules
            dataset_source_container_path=baseline_dataset_container_path,
            post_processor_script_container_path=post_processor_script_container_path,
            analysis_type="MODEL_QUALITY",
            problem_type=problem_type,
            inference_attribute=inference_attribute,
            probability_attribute=probability_attribute,
            ground_truth_attribute=ground_truth_attribute,
            probability_threshold_attribute=probability_threshold_attribute)

         baselining_processor = Processor$new(
            role=self$role,
            image_uri=self$image_uri,
            instance_count=self$instance_count,
            instance_type=self$instance_type,
            entrypoint=self$entrypoint,
            volume_size_in_gb=self$volume_size_in_gb,
            volume_kms_key=self$volume_kms_key,
            output_kms_key=self$output_kms_key,
            max_runtime_in_seconds=self$max_runtime_in_seconds,
            base_job_name=self$base_job_name,
            sagemaker_session=self$sagemaker_session,
            env=normalized_env,
            tags=self$tags,
            network_config=self$network_config)

         baseline_job_inputs_with_nones = list(
            normalized_baseline_dataset_input,
            normalized_post_processor_script_input)

         baseline_job_inputs = Filter(
            Negate(is.null), baseline_job_inputs_with_nones)

         baselining_processor$run(
            inputs=baseline_job_inputs,
            outputs=list(normalized_baseline_output),
            arguments=self$arguments,
            wait=wait,
            logs=logs,
            job_name=self$latest_baselining_job_name)

         self$latest_baselining_job = BaseliningJob$new()$from_processing_job(
            processing_job=baselining_processor$latest_job)

         self$baselining_jobs = c(self$baselining_jobs, self$latest_baselining_job)
         return (baselining_processor$latest_job)
      },

      #' @description Creates a monitoring schedule.
      #' @param endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to
      #'              monitor. This can either be the endpoint name or an EndpointInput.
      #' @param ground_truth_input (str): S3 URI to ground truth dataset.
      #' @param problem_type (str): The type of problem of this model quality monitoring. Valid
      #'              values are "Regression", "BinaryClassification", "MulticlassClassification".
      #' @param record_preprocessor_script (str): The path to the record preprocessor script. This can
      #'              be a local path or an S3 uri.
      #' @param post_analytics_processor_script (str): The path to the record post-analytics processor
      #'              script. This can be a local path or an S3 uri.
      #' @param output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
      #'              Default: "s3://<default_session_bucket>/<job_name>/output"
      #' @param constraints (sagemaker.model_monitor.Constraints or str): If provided it will be used
      #'              for monitoring the endpoint. It can be a Constraints object or an S3 uri pointing
      #'              to a constraints JSON file.
      #' @param monitor_schedule_name (str): Schedule name. If not specified, the processor generates
      #'              a default job name, based on the image name and current timestamp.
      #' @param schedule_cron_expression (str): The cron expression that dictates the frequency that
      #'              this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
      #'              expressions. Default: Daily.
      #' @param enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
      #'              the baselining or monitoring jobs.
      create_monitorying_schedule = function(endpoint_input,
                                             ground_truth_input,
                                             problem_type,
                                             record_preprocessor_script=NULL,
                                             post_analytics_processor_script=NULL,
                                             output_s3_uri=NULL,
                                             constraints=NULL,
                                             monitor_schedule_name=NULL,
                                             schedule_cron_expression=NULL,
                                             enable_cloudwatch_metrics=TRUE){
         if(!is.null(self$job_definition_name) || !is.null(self$monitoring_schedule_name)){
            message = paste(
               "It seems that this object was already used to create an Amazon Model",
               "Monitoring Schedule. To create another, first delete the existing one",
               "using my_monitor.delete_monitoring_schedule().")
            LOGGER$error(message)
            ValueError$new(message)
         }
         # create job definition
         monitor_schedule_name = private$.generate_monitoring_schedule_name(
            schedule_name=monitor_schedule_name)

         new_job_definition_name = name_from_base(self$JOB_DEFINITION_BASE_NAME)
         request_dict = private$.build_create_model_quality_job_definition_request(
            monitoring_schedule_name=monitor_schedule_name,
            job_definition_name=new_job_definition_name,
            image_uri=self$image_uri,
            latest_baselining_job_name=self$latest_baselining_job_name,
            endpoint_input=endpoint_input,
            ground_truth_input=ground_truth_input,
            problem_type=problem_type,
            record_preprocessor_script=record_preprocessor_script,
            post_analytics_processor_script=post_analytics_processor_script,
            output_s3_uri=private$.normalize_monitoring_output(
               monitor_schedule_name, output_s3_uri)$destination,
            constraints=constraints,
            enable_cloudwatch_metrics=enable_cloudwatch_metrics,
            role=self$role,
            instance_count=self$instance_count,
            instance_type=self$instance_type,
            volume_size_in_gb=self$volume_size_in_gb,
            volume_kms_key=self$volume_kms_key,
            output_kms_key=self$output_kms_key,
            max_runtime_in_seconds=self$max_runtime_in_seconds,
            env=self$env,
            tags=self$tags,
            network_config=self$network_config)

         do.call(
            self$sagemaker_session$sagemaker$create_model_quality_job_definition,
            request_dict)

         # create schedule
         tryCatch({
            private$.create_monitoring_schedule_from_job_definition(
               monitor_schedule_name=monitor_schedule_name,
               job_definition_name=new_job_definition_name,
               schedule_cron_expression=schedule_cron_expression)
            self$job_definition_name = new_job_definition_name
            self$monitoring_schedule_name = monitor_schedule_name},
            error = function(e){
               LOGGER$error("Failed to create monitoring schedule.")
               tryCatch(
                  self$sagemaker_session$sagemaker$delete_model_quality_job_definition(
                     JobDefinitionName=new_job_definition_name),
                  error = function(ee){
                     message = sprintf(
                        "Failed to delete job definition %s",
                        new_job_definition_name)
                     LOGGER$error(message)
                     stop(ee)
            })
         })
      },

      #' @description Updates the existing monitoring schedule.
      #'              If more options than schedule_cron_expression are to be updated, a new job definition will
      #'              be created to hold them. The old job definition will not be deleted.
      #' @param endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint
      #'              to monitor. This can either be the endpoint name or an EndpointInput.
      #' @param ground_truth_input (str): S3 URI to ground truth dataset.
      #' @param problem_type (str): The type of problem of this model quality monitoring. Valid values
      #'              are "Regression", "BinaryClassification", "MulticlassClassification".
      #' @param record_preprocessor_script (str): The path to the record preprocessor script. This can
      #'              be a local path or an S3 uri.
      #' @param post_analytics_processor_script (str): The path to the record post-analytics processor
      #'              script. This can be a local path or an S3 uri.
      #' @param output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
      #'              Default: "s3://<default_session_bucket>/<job_name>/output"
      #' @param constraints (sagemaker.model_monitor.Constraints or str): If provided it will be used
      #'              for monitoring the endpoint. It can be a Constraints object or an S3 uri pointing
      #'              to a constraints JSON file.
      #' @param schedule_cron_expression (str): The cron expression that dictates the frequency that
      #'              this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
      #'              expressions. Default: Daily.
      #' @param enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
      #'              the baselining or monitoring jobs.
      #' @param role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
      #' @param instance_count (int): The number of instances to run
      #'              the jobs with.
      #' @param instance_type (str): Type of EC2 instance to use for
      #'              the job, for example, 'ml.m5.xlarge'.
      #' @param volume_size_in_gb (int): Size in GB of the EBS volume
      #'              to use for storing data during processing (default: 30).
      #' @param volume_kms_key (str): A KMS key for the job's volume.
      #' @param output_kms_key (str): The KMS key id for the job's outputs.
      #' @param max_runtime_in_seconds (int): Timeout in seconds. After this amount of
      #'              time, Amazon SageMaker terminates the job regardless of its current status.
      #'              Default: 3600
      #' @param env (dict): Environment variables to be passed to the job.
      #' @param network_config (sagemaker.network.NetworkConfig): A NetworkConfig
      #'              object that configures network isolation, encryption of
      #'              inter-container traffic, security group IDs, and subnets.
      update_monitoring_schedule = function(
         endpoint_input=NULL,
         ground_truth_input=NULL,
         problem_type=NULL,
         record_preprocessor_script=NULL,
         post_analytics_processor_script=NULL,
         output_s3_uri=NULL,
         constraints=NULL,
         schedule_cron_expression=NULL,
         enable_cloudwatch_metrics=NULL,
         role=NULL,
         instance_count=NULL,
         instance_type=NULL,
         volume_size_in_gb=NULL,
         volume_kms_key=NULL,
         output_kms_key=NULL,
         max_runtime_in_seconds=NULL,
         env=NULL,
         network_config=NULL){
         valid_args = Filter(Negate(is.null), as.list(environment()))

         # Nothing to update
         if (length(valid_args) <= 0)
            return(NULL)

         # Only need to update schedule expression
         if (length(valid_args) == 1 && !is.null(schedule_cron_expression)){
            private$.update_monitoring_schedule(self$job_definition_name, schedule_cron_expression)
            return(NULL)
         }
         # Need to update schedule with a new job definition
         job_desc = self$sagemaker_session$sagemaker$describe_model_quality_job_definition(
            JobDefinitionName=self$job_definition_name)

         new_job_definition_name = name_from_base(self$JOB_DEFINITION_BASE_NAME)
         request_dict = private$.build_create_model_quality_job_definition_request(
            monitoring_schedule_name=self$monitoring_schedule_name,
            job_definition_name=new_job_definition_name,
            image_uri=self$image_uri,
            existing_job_desc=job_desc,
            endpoint_input=endpoint_input,
            ground_truth_input=ground_truth_input,
            problem_type=problem_type,
            record_preprocessor_script=record_preprocessor_script,
            post_analytics_processor_script=post_analytics_processor_script,
            output_s3_uri=output_s3_uri,
            constraints=constraints,
            enable_cloudwatch_metrics=enable_cloudwatch_metrics,
            role=role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            env=env,
            tags=self$tags,
            network_config=network_config)

         do.call(
            self$sagemaker_session$sagemaker$create_model_quality_job_definition,
            request_dict)
         tryCatch({
            private$.update_monitoring_schedule(new_job_definition_name, schedule_cron_expression)
            private$.job_definition_name = new_job_definition_name
            if (!is.null(role))
               self$role = role
            if (!is.null(instance_count))
               self$instance_count = instance_count
            if (!is.null(instance_type))
               self$instance_type = instance_type
            if (!is.null(volume_size_in_gb))
               self$volume_size_in_gb = volume_size_in_gb
            if (!is.null(volume_kms_key))
               self$volume_kms_key = volume_kms_key
            if (!is.null(output_kms_key))
               self$output_kms_key = output_kms_key
            if (!is.null(max_runtime_in_seconds))
               self$max_runtime_in_seconds = max_runtime_in_seconds
            if (!is.null(env))
               self$env = env
            if (!is.null(network_config))
               self$network_config = network_config
         },
         error = function(e){
            LOGGER$error("Failed to update monitoring schedule.")
            # noinspection PyBroadException
            tryCatch(
               self$sagemaker_session$sagemaker$delete_model_quality_job_definition(
                  JobDefinitionName=new_job_definition_name),
               error = function(ee){
                  message = sprintf("Failed to delete job definition %s.", new_job_definition_name)
                  LOGGER$error(message)
                  stop(ee)
               })
         })
      },

      #' @description Deletes the monitoring schedule and its job definition."
      delete_monitoring_schedule = function(){
         super$delete_monitoring_schedule()
         # Delete job definition.
         message = sprintf("Deleting Model Quality Job Definition with name: %s",
            self$job_definition_name)
         LOGGER$info(message)
         self$sagemaker_session$sagemaker$delete_model_quality_job_definition(
            JobDefinitionName=self$job_definition_name)
         self$job_definition_name = NULL
      },

      #' @description Sets this object's schedule name to the name provided.
      #'              This allows subsequent describe_schedule or list_executions calls to point
      #'              to the given schedule.
      #' @param monitor_schedule_name (str): The name of the schedule to attach to.
      #' @param sagemaker_session (sagemaker.session.Session): Session object which
      #'              manages interactions with Amazon SageMaker APIs and any other
      #'              AWS services needed. If not specified, one is created using
      #'              the default AWS configuration chain.
      attach = function(monitor_schedule_name,
                        sagemaker_session=NULL){
         sagemaker_session = sagemaker_session %||% Session$new()
         schedule_desc = sagemaker_session$describe_monitoring_schedule(
            monitoring_schedule_name=monitor_schedule_name)
         monitoring_type = schedule_desc[["MonitoringScheduleConfig"]][["MonitoringType"]]
         if (monitoring_type != self.monitoring_type())
            TypeError$new(sprintf(
               "%s can only attach to ModelQuality monitoring schedule.", class(self)[1]))

         job_definition_name = schedule_desc[["MonitoringScheduleConfig"]][[
            "MonitoringJobDefinitionName"
         ]]
         job_desc = sagemaker_session$sagemaker$describe_model_quality_job_definition(
            JobDefinitionName=job_definition_name)
         tags = sagemaker_session$list_tags(resource_arn=schedule_desc["MonitoringScheduleArn"])
         return(ModelMonitor$private_methods$.attach(
            clazz=cls,
            sagemaker_session=sagemaker_session,
            schedule_desc=schedule_desc,
            job_desc=job_desc,
            tags=tags)
         )
      }
   ),
   private=list(

      # Build the request for job definition creation API
      # Args:
      #    monitoring_schedule_name (str): Monitoring schedule name.
      # job_definition_name (str): Job definition name.
      # If not specified then a default one will be generated.
      # image_uri (str): The uri of the image to use for the jobs started by the Monitor.
      # latest_baselining_job_name (str): name of the last baselining job.
      # existing_job_desc (dict): description of existing job definition. It will be updated by
      # values that were passed in, and then used to create the new job definition.
      # endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
      # This can either be the endpoint name or an EndpointInput.
      # ground_truth_input (str): S3 URI to ground truth dataset.
      # problem_type (str): The type of problem of this model quality monitoring. Valid
      # values are "Regression", "BinaryClassification", "MulticlassClassification".
      # output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
      # Default: "s3://<default_session_bucket>/<job_name>/output"
      # constraints (sagemaker.model_monitor.Constraints or str): If provided it will be used
      # for monitoring the endpoint. It can be a Constraints object or an S3 uri pointing
      # to a constraints JSON file.
      # enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
      # the baselining or monitoring jobs.
      # role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
      # instance_count (int): The number of instances to run
      # the jobs with.
      # instance_type (str): Type of EC2 instance to use for
      # the job, for example, 'ml.m5.xlarge'.
      # volume_size_in_gb (int): Size in GB of the EBS volume
      # to use for storing data during processing (default: 30).
      # volume_kms_key (str): A KMS key for the job's volume.
      # output_kms_key (str): KMS key id for output.
      # max_runtime_in_seconds (int): Timeout in seconds. After this amount of
      #     time, Amazon SageMaker terminates the job regardless of its current status.
      #     Default: 3600
      # env (dict): Environment variables to be passed to the job.
      # tags ([dict]): List of tags to be passed to the job.
      # network_config (sagemaker.network.NetworkConfig): A NetworkConfig
      #     object that configures network isolation, encryption of
      #     inter-container traffic, security group IDs, and subnets.
      # Returns:
      # dict: request parameters to create job definition.
      .build_create_model_quality_job_definition_request = function(
         monitoring_schedule_name,
         job_definition_name,
         image_uri,
         latest_baselining_job_name=NULL,
         existing_job_desc=NULL,
         endpoint_input=NULL,
         ground_truth_input=NULL,
         problem_type=NULL,
         record_preprocessor_script=NULL,
         post_analytics_processor_script=NULL,
         output_s3_uri=NULL,
         constraints=NULL,
         enable_cloudwatch_metrics=NULL,
         role=NULL,
         instance_count=NULL,
         instance_type=NULL,
         volume_size_in_gb=NULL,
         volume_kms_key=NULL,
         output_kms_key=NULL,
         max_runtime_in_seconds=NULL,
         env=NULL,
         tags=NULL,
         network_config=NULL){
         if (!is.null(existing_job_desc)) {
            app_specification = existing_job_desc[[
               sprintf("%sAppSpecification",self$monitoring_type())
            ]]
            baseline_config = existing_job_desc[[
               sprintf("%sBaselineConfig",self$monitoring_type())]] %||% list()
            job_input = existing_job_desc[[sprintf("%sJobInput",self$monitoring_type())]]
            job_output = existing_job_desc[[sprintf("%sJobOutputConfig",self$monitoring_type())]]
            cluster_config = existing_job_desc[["JobResources"]][["ClusterConfig"]]
            if (is.null(role))
               role = existing_job_desc[["RoleArn"]]
            existing_network_config = existing_job_desc[["NetworkConfig"]]
            stop_condition = existing_job_desc[["StoppingCondition"]] %||% list()
         } else {
            app_specification = list()
            baseline_config = list()
            job_input = list()
            job_output = list()
            cluster_config = list()
            existing_network_config = NULL
            stop_condition = list()
         }

         # app specification
         app_specification[["ImageUri"]] = image_uri
         if (!is.null(problem_type))
            app_specification[["ProblemType"]] = problem_type
         record_preprocessor_script_s3_uri = NULL
         if (!is.null(record_preprocessor_script))
            record_preprocessor_script_s3_uri = private$.s3_uri_from_local_path(
               path=record_preprocessor_script)

         post_analytics_processor_script_s3_uri = NULL
         if (!is.null(post_analytics_processor_script))
            post_analytics_processor_script_s3_uri = private$.s3_uri_from_local_path(
               path=post_analytics_processor_script)

         if (!is.null(post_analytics_processor_script_s3_uri))
            app_specification[[
               "PostAnalyticsProcessorSourceUri"
            ]] = post_analytics_processor_script_s3_uri
         if (!islistempty(record_preprocessor_script_s3_uri))
            app_specification[["RecordPreprocessorSourceUri"]] = record_preprocessor_script_s3_uri

         normalized_env = private$.generate_env_map(
            env=env, enable_cloudwatch_metrics=enable_cloudwatch_metrics)
         if (!islistempty(normalized_env))
            app_specification[["Environment"]] = normalized_env

         # baseline config
         if (!islistempty(constraints)){
            # noinspection PyTypeChecker
            ll = private$.get_baseline_files(
               statistics=NULL, constraints=constraints, sagemaker_session=self$sagemaker_session)
            constraints_s3_uri = NULL
            if (!is.null(ll$constraints_object))
               constraints_s3_uri = ll$constraints_object$file_s3_uri
            baseline_config[["ConstraintsResource"]] = list("S3Uri"=constraints_s3_uri)
         }
         if (!islistempty(latest_baselining_job_name))
            baseline_config[["BaseliningJobName"]] = latest_baselining_job_name

         # job input
         if (!is.null(endpoint_input))
            normalized_endpoint_input = private$.normalize_endpoint_input(
               endpoint_input=endpoint_input)
         job_input = normalized_endpoint_input$to_request_list()
         if (!is.null(ground_truth_input))
            job_input[["GroundTruthS3Input"]] = list("S3Uri"=ground_truth_input)

         # job output
         if (!is.null(output_s3_uri)){
            normalized_monitoring_output = private$.normalize_monitoring_output(
               monitoring_schedule_name, output_s3_uri)
            job_output[["MonitoringOutputs"]] = list(normalized_monitoring_output$to_request_list())}
         if (!is.null(output_kms_key))
            job_output[["KmsKeyId"]] = output_kms_key

         # cluster config
         if (!is.null(instance_count))
            cluster_config[["InstanceCount"]] = instance_count
         if (!is.null(instance_type))
            cluster_config[["InstanceType"]] = instance_type
         if (!is.null(volume_size_in_gb))
            cluster_config[["VolumeSizeInGB"]] = volume_size_in_gb
         if (!is.null(volume_kms_key))
            cluster_config[["VolumeKmsKeyId"]] = volume_kms_key

         # stop condition
         if (!is.null(max_runtime_in_seconds))
            stop_condition[["MaxRuntimeInSeconds"]] = max_runtime_in_seconds

         request_dict = list(
            job_definition_name,
            app_specification,
            job_input,
            job_output,
            list("ClusterConfig"=cluster_config),
            self$sagemaker_session$expand_role(role))
         names(request_dict) = c(
            "JobDefinitionName",
            sprintf("%sAppSpecification",self$monitoring_type()),
            sprintf("%sJobInput",self$monitoring_type()),
            sprintf("%sJobOutputConfig",self$monitoring_type()),
            "JobResources",
            "RoleArn")
         if (!islistempty(baseline_config))
            request_dict[[sprintf("%sBaselineConfig",self$monitoring_type())]] = baseline_config

         if (!is.null(network_config)) {
            network_config_dict = network_config$to_request_list()
            private$.validate_network_config(network_config_dict)
            request_dict[["NetworkConfig"]] = network_config_dict
         } else if (!is.null(existing_network_config))
            request_dict[["NetworkConfig"]] = existing_network_config

         if (!islistempty(stop_condition))
            request_dict[["StoppingCondition"]] = stop_condition

         if (!is.null(tags))
            request_dict[["Tags"]] = tags

         return(request_dict)
      },

      # Returns the Default Model Monitoring image uri based on the region.
      # Args:
      #    region (str): The AWS region.
      # Returns:
      #    str: The Default Model Monitoring image uri based on the region.
      .get_default_image_uri=function(region){
         return(ImageUris$new()$etrieve(framework=private$.framework_name, region=region))
      }
   ),
   lock_objects = F
)

#' @title Baselining Job Class
#' @description Provides functionality to retrieve baseline-specific files output from baselining job.
#' @export
BaseliningJob = R6Class("BaseliningJob",
  inherit = ProcessingJob,
  public = list(

    #' @description Initializes a Baselining job that tracks a baselining job kicked off by the suggest
    #'              workflow.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, one is created using
    #'              the default AWS configuration chain.
    #' @param job_name (str): Name of the Amazon SageMaker Model Monitoring Baselining Job.
    #' @param inputs ([sagemaker.processing.ProcessingInput]): A list of ProcessingInput objects.
    #' @param outputs ([sagemaker.processing.ProcessingOutput]): A list of ProcessingOutput objects.
    #' @param output_kms_key (str): The output kms key associated with the job. Defaults to None
    #'              if not provided.
    initialize = function(sagemaker_session = NULL,
                          job_name = NULL,
                          inputs = NULL,
                          outputs = NULL,
                          output_kms_key=NULL){
      self$inputs = inputs
      self$outputs = outputs
      super$initialize(
        sagemaker_session=sagemaker_session,
        job_name=job_name,
        inputs=inputs,
        outputs=outputs,
        output_kms_key=output_kms_key)
    },

    #' @description Initializes a Baselining job from a processing job.
    #' @param processing_job (sagemaker.processing.ProcessingJob): The ProcessingJob used for
    #'              baselining instance.
    #' @return sagemaker.processing.BaseliningJob: The instance of ProcessingJob created
    #'              using the current job name.
    from_processing_job = function(processing_job){
      cls = self$clone()
      cls$processing_job$sagemaker_session
      cls$processing_job$job_name
      cls$processing_job$inputs
      cls$processing_job$outputs
      cls$processing_job$output_kms_key

      return(cls)
    },

    #' @description Returns a sagemaker.model_monitor.Statistics object representing the statistics
    #'              JSON file generated by this baselining job.
    #' @param file_name (str): The name of the json-formatted statistics file
    #' @param kms_key (str): The kms key to use when retrieving the file.
    #' @return sagemaker.model_monitor.Statistics: The Statistics object representing the file that
    #'              was generated by the job.
    baseline_statistics = function(file_name=STATISTICS_JSON_DEFAULT_FILE_NAME,
                                   kms_key=NULL){
      tryCatch({
         baselining_job_output_s3_path = self$outputs[[1]]$destination
         Statistics$new()$from_s3_uri(
            statistics_file_s3_uri=file.path(baselining_job_output_s3_path, file_name),
            kms_key=kms_key,
            sagemaker_session=self$sagemaker_session)},
         error = function(e){
            error_code = attributes(e)$error_response$`__type`
            if(error_code == "NoSuchKey") {
               status = self$sagemaker_session$describe_processing_job(job_name=self$job_name)$ProcessingJobStatus
               if(status != "Completed"){
                  UnexpectedStatusError$new(
                     "The underlying job is not in 'Completed' state. You may only ",
                     "retrieve files for a job that has completed successfully.",
                     allowed_statuses="Completed",
                     actual_status=status)}
            } else {stop(e)}
      })
    },

    #' @description Returns a sagemaker.model_monitor.Constraints object representing the constraints
    #'              JSON file generated by this baselining job.
    #' @param file_name (str): The name of the json-formatted constraints file
    #' @param kms_key (str): The kms key to use when retrieving the file.
    #' @return sagemaker.model_monitor.Constraints: The Constraints object representing the file that
    #'              was generated by the job.
    suggested_constraints = function(file_name=CONSTRAINTS_JSON_DEFAULT_FILE_NAME,
                                     kms_key=NULL){
      tryCatch({
         baselining_job_output_s3_path = self$outputs[[1]]$destination
         Constraints$new()$from_s3_uri(
            statistics_file_s3_uri=file.path(baselining_job_output_s3_path, file_name),
            kms_key=kms_key,
            sagemaker_session=self$sagemaker_session)
         },
         error = function(e){
            error_code = attributes(e)$error_response$`__type`
            if(error_code == "NoSuchKey") {
               status = self$sagemaker_session$describe_processing_job(job_name=self$job_name)$ProcessingJobStatus
               if(status != "Completed"){
                  UnexpectedStatusError$new(
                     "The underlying job is not in 'Completed' state. You may only ",
                     "retrieve files for a job that has completed successfully.",
                     allowed_statuses="Completed",
                     actual_status=status)}
            } else {stop(e)}
      })
    }
  ),
  lock_objects = F
)

#' @title MonitoringExecution Class
#' @description Provides functionality to retrieve monitoring-specific files output from monitoring
#'              executions
#' @export
MonitoringExecution = R6Class("MonitoringExecution",
   inherit = ProcessingJob,
   public = list(
    #' @description Initializes a MonitoringExecution job that tracks a monitoring execution kicked off by
    #'              an Amazon SageMaker Model Monitoring Schedule.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, one is created using
    #'              the default AWS configuration chain.
    #' @param job_name (str): The name of the monitoring execution job.
    #' @param inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): A list of
    #'              :class:`~sagemaker.processing.ProcessingInput` objects.
    #' @param output (sagemaker.Processing.ProcessingOutput): The output associated with the
    #'              monitoring execution.
    #' @param output_kms_key (str): The output kms key associated with the job. Defaults to None
    #'              if not provided.
    initialize = function(sagemaker_session = NULL,
                          job_name = NULL,
                          inputs = NULL,
                          output = NULL,
                          output_kms_key=NULL){
      self$output = output
      super$initialize(
        sagemaker_session=sagemaker_session,
        job_name=job_name,
        inputs=inputs,
        outputs=list(output),
        output_kms_key=output_kms_key)
    },

    #' @description Initializes a Baselining job from a processing arn.
    #' @param processing_job_arn (str): ARN of the processing job to create a MonitoringExecution
    #'              out of.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, one is created using
    #'              the default AWS configuration chain.
    #' @return sagemaker.processing.BaseliningJob: The instance of ProcessingJob created
    #'              using the current job name.
    from_processing_arn = function(sagemaker_session,
                                   processing_job_arn){
      processing_job_name = split_str(processing_job_arn,":")[6]
      # This is necessary while the API only vends an arn.
      processing_job_name = substring(processing_job_name, nchar("processing-job/"), nchar(processing_job_name))
      job_desc = sagemaker_session$describe_processing_job(job_name=processing_job_name)

      output_config = job_desc[["ProcessingOutputConfig"]][["Outputs"]][[1]]

      cls = self$clone()
      cls$sagemaker_session=sagemaker_session
      cls$job_name=processing_job_name
      cls$inputs = lapply(job_desc[["ProcessingInputs"]], function(processing_input){
        ProcessingInput$new(source=processing_input[["S3Input"]][["S3Uri"]],
                            destination=processing_input[["S3Input"]][["LocalPath"]],
                            input_name=processing_input[["InputName"]],
                            s3_data_type=processing_input[["S3Input"]][["S3DataType"]],
                            s3_input_mode=processing_input[["S3Input"]][["S3InputMode"]],
                            s3_data_distribution_type=processing_input[["S3Input"]][["S3DataDistributionType"]],
                            s3_compression_type=processing_input[["S3Input"]][["S3CompressionType"]])})
      cls$output = ProcessingOutput$new(
         source=output_config[["S3Output"]][["LocalPath"]],
         destination=output_config[["S3Output"]][["S3Uri"]],
         output_name=output_config[["OutputName"]])
      cls$output_kms_key=job_desc[["ProcessingOutputConfig"]][["KmsKeyId"]]

      return(cls)
    },

    #' @description Returns a sagemaker.model_monitor.Statistics object representing the statistics
    #'              JSON file generated by this monitoring execution.
    #' @param file_name (str): The name of the json-formatted statistics file
    #' @param kms_key (str): The kms key to use when retrieving the file.
    #' @return sagemaker.model_monitor.Statistics: The Statistics object representing the file that
    #'              was generated by the execution.
    statistics = function(file_name=STATISTICS_JSON_DEFAULT_FILE_NAME,
                          kms_key=NULL){
      tryCatch({
         baselining_job_output_s3_path = self$outputs[[1]]$destination
         return(Statistics$new()$from_s3_uri(
            statistics_file_s3_uri=file.path(baselining_job_output_s3_path, file_name),
            kms_key=kms_key,
            sagemaker_session=self$sagemaker_session))},
      error = function(e){
         error_code = attributes(e)$error_response$`__type`
         if(error_code == "NoSuchKey") {
            status = self$sagemaker_session$describe_processing_job(job_name=self$job_name)$ProcessingJobStatus
            if(status != "Completed"){
               UnexpectedStatusError$new(
                  "The underlying job is not in 'Completed' state. You may only ",
                  "retrieve files for a job that has completed successfully.",
                  allowed_statuses="Completed",
                  actual_status=status)}
        } else {stop(e)}
      })
    },

    #' @description Returns a sagemaker.model_monitor.ConstraintViolations object representing the
    #'              constraint violations JSON file generated by this monitoring execution.
    #' @param file_name (str): The name of the json-formatted constraint violations file.
    #' @param kms_key (str): The kms key to use when retrieving the file.
    #' @return sagemaker.model_monitor.ConstraintViolations: The ConstraintViolations object
    #'              representing the file that was generated by the monitoring execution.
    constraint_violations = function(file_name=CONSTRAINT_VIOLATIONS_JSON_DEFAULT_FILE_NAME,
                                     kms_key=NULL){
      tryCatch({
         baselining_job_output_s3_path = self$outputs[[1]]$destination
         ConstraintViolations$new()$from_s3_uri(
            statistics_file_s3_uri=file.path(baselining_job_output_s3_path, file_name),
            kms_key=kms_key,
            sagemaker_session=self$sagemaker_session)},
      error = function(e){
        error_code = attributes(e)$error_response$`__type`
        if(error_code == "NoSuchKey") {
          status = self$sagemaker_session$describe_processing_job(job_name=self$job_name)$ProcessingJobStatus
          if(status != "Completed"){
             UnexpectedStatusError$new(
                "The underlying job is not in 'Completed' state. You may only ",
                "retrieve files for a job that has completed successfully.",
                allowed_statuses="Completed",
                actual_status=status)}
        } else {stop(e)}
     })
    }
   ),
   lock_objects = F
)

#' @title Accepts parameters that specify an endpoint input for monitoring execution.
#' @description It also provides a method to turn those parameters into a dictionary.
#' @export
EndpointInput = R6Class("EndpointInput",
  public = list(

    #' @description Initialize an ``EndpointInput`` instance. EndpointInput accepts parameters
    #'              that specify an endpoint input for a monitoring job and provides a method
    #'              to turn those parameters into a dictionary.
    #' @param endpoint_name (str): The name of the endpoint.
    #' @param destination (str): The destination of the input.
    #' @param s3_input_mode (str): The S3 input mode. Can be one of: "File", "Pipe. Default: "File".
    #' @param s3_data_distribution_type (str): The S3 Data Distribution Type. Can be one of:
    #'              "FullyReplicated", "ShardedByS3Key"
    #' @param start_time_offset (str): Monitoring start time offset, e.g. "-PT1H"
    #' @param end_time_offset (str): Monitoring end time offset, e.g. "-PT0H".
    #' @param features_attribute (str): JSONpath to locate features in JSONlines dataset.
    #'              Only used for ModelBiasMonitor and ModelExplainabilityMonitor
    #' @param inference_attribute (str): Index or JSONpath to locate predicted label(s).
    #'              Only used for ModelQualityMonitor, ModelBiasMonitor, and ModelExplainabilityMonitor
    #' @param probability_attribute (str or int): Index or JSONpath to locate probabilities.
    #'              Only used for ModelQualityMonitor, ModelBiasMonitor and ModelExplainabilityMonitor
    #' @param probability_threshold_attribute (float): threshold to convert probabilities to binaries
    #'              Only used for ModelQualityMonitor, ModelBiasMonitor and ModelExplainabilityMonitor
    initialize = function(endpoint_name,
                          destination,
                          s3_input_mode=c("File", "Pipe"),
                          s3_data_distribution_type=c("FullyReplicated", "ShardedByS3Key"),
                          start_time_offset=NULL,
                          end_time_offset=NULL,
                          features_attribute=NULL,
                          inference_attribute=NULL,
                          probability_attribute=NULL,
                          probability_threshold_attribute=NULL){
      self$endpoint_name = endpoint_name
      self$destination = destination
      self$s3_input_mode = match.arg(s3_input_mode)
      self$s3_data_distribution_type = match.arg(s3_data_distribution_type)
      self$start_time_offset = start_time_offset
      self$end_time_offset = end_time_offset
      self$features_attribute = features_attribute
      self$inference_attribute = inference_attribute
      self$probability_attribute = probability_attribute
      self$probability_threshold_attribute = probability_threshold_attribute
    },

    #' @description Generates a request dictionary using the parameters provided to the class.
    to_request_list = function(){
       endpoint_input = list(
          "EndpointName"= self$endpoint_name,
          "LocalPath"= self$destination,
          "S3InputMode"= self$s3_input_mode,
          "S3DataDistributionType"= self$s3_data_distribution_type)
      endpoint_input[["StartTimeOffset"]] = self$start_time_offset
      endpoint_input[["EndTimeOffset"]] = self$end_time_offset
      endpoint_input[["FeaturesAttribute"]] = self$features_attribute
      endpoint_input[["InferenceAttribute"]] = self$inference_attribute
      endpoint_input[["ProbabilityAttribute"]] = self$probability_attribute
      endpoint_input[["ProbabilityThresholdAttribute"]] = self$probability_threshold_attribute

      endpoint_input_request = list("EndpointInput"= endpoint_input)
      return (endpoint_input_request)
    },

    #' @description format class
    format = function(){
       format_class(self)
    }
  ),
  lock_objects = F
)

#' @title Accepts parameters that specify an S3 output for a monitoring job.
#' @description It also provides a method to turn those parameters into a dictionary.
#' @export
EndpointOutput = R6Class("EndpointOutput",
  public = list(

   #' @description Initialize a ``MonitoringOutput`` instance. MonitoringOutput accepts parameters that
   #'              specify an S3 output for a monitoring job and provides a method to turn
   #'              those parameters into a dictionary.
   #' @param source (str): The source for the output.
   #' @param destination (str): The destination of the output. Optional.
   #'              Default: s3://<default-session-bucket/schedule_name/output
   #' @param s3_upload_mode (str): The S3 upload mode.
   initialize = function(source,
                         destination=NULL,
                         s3_upload_mode="Continuous"){
     self$source = source
     self$destination = destination
     self$s3_upload_mode = s3_upload_mode
   },

   #' @description Generates a request dictionary using the parameters provided to the class.
   #' @return dict: The request dictionary.
   to_request_list = function(){
     s3_output_request = list(
       "S3Output" = list(
         "S3Uri"= self$destination,
         "LocalPath"= self$source,
         "S3UploadMode"= self$s3_upload_mode))

     return(s3_output_request)
   },

   #' @description format class
   format = function(){
      format_class(self)
   }
  ),
  lock_objects = F
)
