# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/model_monitor/clarify_model_monitoring.py

#' @include session.R
#' @include s3.R
#' @include r_utils.R
#' @include clarify.R
#' @include model_monitor_model_monitoring.R

#' @import R6
#' @import R6sagemaker.common
#' @import lgr
#' @import uuid
#' @import jsonlite

#' @title Base class of Amazon SageMaker Explainability API model monitors.
#' @description This class is an ``abstract base class``, please instantiate its subclasses
#'              if you want to monitor bias metrics or feature attribution of an endpoint.
#' @export
ClarifyModelMonitor = R6Class("ClarifyModelMonitor",
  inherit = ModelMonitor,
  public = list(

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

      if (class(self)[1] == class(self)[length(class(self))-2]) # To get abstract base class.
        TypeError$new(
          sprintf("%s is abstract, please instantiate its subclasses instead.", class(self)[1]))

      session = sagemaker_session %||% Session$new()
      clarify_image_uri = ImageUris$new()$retrieve("clarify", session$paws_region_name)

      super$new(
        role=role,
        image_uri=clarify_image_uri,
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
      self$latest_baselining_job_config = NULL
    },

    #' @description Not implemented.
    #'              .run_baseline()' is only allowed for ModelMonitor objects.
    #'              Please use `suggest_baseline` instead.
    #' @param ... : Unused argument
    run_baseline = function(...){
      NotImplementedError$new(
        "'run_baseline()' is only allowed for ModelMonitor objects. ",
        "Please use suggest_baseline instead.")
    },

    #' @description Not implemented.
    #'              The class doesn't support statistics.
    #'@param ... : Unused argument
    latest_monitorying_statistics = function(...){
      NotImplementedError$new(
        sprintf("%s doesn't support statistics.", class(self)[1]))
    },

    #' @description Get the list of the latest monitoring executions in descending order of "ScheduledTime".
    #' @return [sagemaker.model_monitor.ClarifyMonitoringExecution]: List of
    #'              ClarifyMonitoringExecution in descending order of "ScheduledTime".
    list_executions = function(){
      executions = super$list_executions()
      return (lapply(executions, function(execution){
        ClarifyMonitoringExecution$new(
          sagemaker_session=execution$sagemaker_session,
          job_name=execution$job_name,
          inputs=execution$inputs,
          output=execution$output,
          output_kms_key=execution$output_kms_key)})
      )
    }
  ),
  private = list(

    # Create and return a SageMakerClarifyProcessor object which will run the baselining job.
    # Returns:
    #   sagemaker.clarify.SageMakerClarifyProcessor object.
    .create_baselining_processor = function(){

      baselining_processor = SageMakerClarifyProcessor$new(
        role=self$role,
        instance_count=self$instance_count,
        instance_type=self$instance_type,
        volume_size_in_gb=self$volume_size_in_gb,
        volume_kms_key=self$volume_kms_key,
        output_kms_key=self$output_kms_key,
        max_runtime_in_seconds=self$max_runtime_in_seconds,
        sagemaker_session=self$sagemaker_session,
        env=self$env,
        tags=self$tags,
        network_config=self$network_config)
      baselining_processor$image_uri = self$image_uri
      return(baselining_processor)
    },

    # Upload analysis config to s3://<output path>/<job name>/analysis_config.json
    # Args:
    #   analysis_config (dict): analysis config of a Clarify model monitor.
    # output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
    # Default: "s3://<default_session_bucket>/<job_name>/output"
    # job_definition_name (str): Job definition name.
    # If not specified then a default one will be generated.
    # Returns:
    #   str: The S3 uri of the uploaded file(s).
    .upload_analysis_config = function(){
      s3_uri = file.path(
        output_s3_uri,
        job_definition_name,
        UUIDgenerate(),
        "analysis_config.json")

      LOGGER$info("Uploading analysis config to %s.", s3_uri)
      return (S3Uploader$new()$upload_string_as_file_body(
        toJSON(analysis_config, auto_unbox = T),
        desired_s3_uri=s3_uri,
        sagemaker_session=self$sagemaker_session)
      )
    },

    # Build the request for job definition creation API
    # Args:
    #   monitoring_schedule_name (str): Monitoring schedule name.
    # job_definition_name (str): Job definition name.
    # If not specified then a default one will be generated.
    # image_uri (str): The uri of the image to use for the jobs started by the Monitor.
    # latest_baselining_job_name (str): name of the last baselining job.
    # latest_baselining_job_config (ClarifyBaseliningConfig): analysis config from
    # last baselining job.
    # existing_job_desc (dict): description of existing job definition. It will be updated by
    # values that were passed in, and then used to create the new job definition.
    # endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
    # This can either be the endpoint name or an EndpointInput.
    # ground_truth_input (str): S3 URI to ground truth dataset.
    # analysis_config (str or BiasAnalysisConfig or ExplainabilityAnalysisConfig): URI to the
    # analysis_config.json for the bias job. If it is None then configuration of latest
    # baselining job config will be reused. If no baselining job then fail the call.
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
    # time, Amazon SageMaker terminates the job regardless of its current status.
    # Default: 3600
    # env (dict): Environment variables to be passed to the job.
    # tags ([dict]): List of tags to be passed to the job.
    # network_config (sagemaker.network.NetworkConfig): A NetworkConfig
    # object that configures network isolation, encryption of
    # inter-container traffic, security group IDs, and subnets.
    # Returns:
    # dict: request parameters to create job definition.
    .build_create_job_definition_request = function(
      monitoring_schedule_name,
      job_definition_name,
      image_uri,
      latest_baselining_job_name=NULL,
      latest_baselining_job_config=NULL,
      existing_job_desc=NULL,
      endpoint_input=NULL,
      ground_truth_input=NULL,
      analysis_config=NULL,
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
      if (!islistempty(existing_job_desc)){
        app_specification = existing_job_desc[[
          sprintf("%sAppSpecification", self$monitoring_type())
          ]]
        baseline_config = existing_job_desc[[
          sprintf("%sBaselineConfig", self$monitoring_type())
          ]] %||% list()
        job_input = existing_job_desc[[sprintf("%sJobInput", self$monitoring_type())]]
        job_output = existing_job_desc[[sprintf("%sJobOutputConfig", self$monitoring_type())]]
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

      # job output
      if (!is.null(output_s3_uri)){
        normalized_monitoring_output = private$.normalize_monitoring_output(
          monitoring_schedule_name, output_s3_uri)
        job_output[["MonitoringOutputs"]] = list(normalized_monitoring_output$to_request_list())
      }
      job_output[["KmsKeyId"]] = output_kms_key

      # app specification
      if (is.null(analysis_config)){
        if (!islistempty(latest_baselining_job_config)){
          analysis_config = latest_baselining_job_config$analysis_config
        } else if (!islistempty(app_specification)) {
          analysis_config = app_specification[["ConfigUri"]]
        } else {
          stop("analysis_config is mandatory.", call. = F)
        }
      }

      # backfill analysis_config
      if (is.character(analysis_config)){
        analysis_config_uri = analysis_config
      } else {
        analysis_config_uri = private$.upload_analysis_config(
          analysis_config$to_list(), output_s3_uri, job_definition_name)
      }
      app_specification[["ConfigUri"]] = analysis_config_uri
      app_specification[["ImageUri"]] = image_uri
      normalized_env = private$.generate_env_map(
        env=env, enable_cloudwatch_metrics=enable_cloudwatch_metrics)
      if (!islistempty(normalized_env))
        app_specification[["Environment"]] = normalized_env

      # baseline config
      if (!is.null(constraints)){
        # noinspection PyTypeChecker
        ll = private$.get_baseline_files(
          statistics=NULL, constraints=constraints, sagemaker_session=self$sagemaker_session)
        names(ll) = c("rm", "constraints_object")
        constraints_s3_uri = NULL
        if (!islistempty(lL$constraints_object))
          constraints_s3_uri = ll$constraints_object$file_s3_uri
        baseline_config[["ConstraintsResource"]] = list("S3Uri"=constraints_s3_uri)
      } else if (latest_baselining_job_name)
        baseline_config[["BaseliningJobName"]] = latest_baselining_job_name

      # job input
      if (!is.null(endpoint_input)){
        normalized_endpoint_input = private$.normalize_endpoint_input(
          endpoint_input=endpoint_input)
        # backfill attributes to endpoint input
        if (!islistempty(latest_baselining_job_config)){
          if (islistempty(normalized_endpoint_input$features_attribute)){
            normalized_endpoint_input.features_attribute = (
              latest_baselining_job_config$features_attribute)
          }
          if (islistempty(normalized_endpoint_input.inference_attribute)){
            normalized_endpoint_input$inference_attribute = (
              latest_baselining_job_config$inference_attribute)
          }
          if (islistempty(normalized_endpoint_input$probability_attribute)){
            normalized_endpoint_input$probability_attribute = (
              latest_baselining_job_config$probability_attribute)
          }
          if (islistempty(normalized_endpoint_input.probability_threshold_attribute)){
            normalized_endpoint_input$probability_threshold_attribute = (
              latest_baselining_job_config$probability_threshold_attribute)
          }
        }
        job_input = normalized_endpoint_input$to_request_list()
      }
      if (!is.null(ground_truth_input))
        job_input[["GroundTruthS3Input"]] = list("S3Uri"=ground_truth_input)

      # cluster config
      cluster_config[["InstanceCount"]] = instance_count
      cluster_config[["InstanceType"]] = instance_type
      cluster_config[["VolumeSizeInGB"]] = volume_size_in_gb
      cluster_config[["VolumeKmsKeyId"]] = volume_kms_key

      # stop condition
      stop_condition[["MaxRuntimeInSeconds"]] = max_runtime_in_seconds

      request_dict = list(
        job_definition_name, app_specification, job_input, job_output,
        list("ClusterConfig"=cluster_config), self$sagemaker_session$expand_role(role))
      names(request_dict) = c(
        "JobDefinitionName",
        sprintf("%sAppSpecification", self$monitoring_type()),
        sprintf("%sJobInput", self$monitoring_type()),
        sprintf("%sJobOutputConfig", self$monitoring_type()),
        "JobResources",
        "RoleArn")

      if (!islistempty(baseline_config))
        request_dict[[sprintf("%sBaselineConfig", self$monitoring_type())]] = baseline_config

      if (!islistempty(network_config)) {
        network_config_dict = network_config$to_request_list()
        private$.validate_network_config(network_config_dict)
        request_dict[["NetworkConfig"]] = network_config_dict
      } else if (!islistempty(existing_network_config))
        request_dict[["NetworkConfig"]] = existing_network_config

      if (!islistempty(stop_condition))
        request_dict[["StoppingCondition"]] = stop_condition

      if (!islistempty(tags))
        request_dict[["Tags"]] = tags

      return(request_dict)
    }
  ),
  lock_objects = F
)

#' @title Amazon SageMaker model monitor to monitor bias metrics of an endpoint.
#' @description Please see the `initialize` method of its base class for how to instantiate it.
#' @export
ModelBiasMonitor = R6Class("ModelBiasMonitor",
  inherit = ClarifyModelMonitor,
  public = list(

    #' @field JOB_DEFINITION_BASE_NAME
    #' Model definition base name
    JOB_DEFINITION_BASE_NAME = "model-bias-job-definition",

    #' @description Type of the monitoring job.
    monitoring_type = function(){
      return("ModelBias")
    },

    #' @description Suggests baselines for use with Amazon SageMaker Model Monitoring Schedules.
    #' @param data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
    #' @param bias_config (:class:`~sagemaker.clarify.BiasConfig`): Config of sensitive groups.
    #' @param model_config (:class:`~sagemaker.clarify.ModelConfig`): Config of the model and its
    #'              endpoint to be created.
    #' @param model_predicted_label_config (:class:`~sagemaker.clarify.ModelPredictedLabelConfig`):
    #'              Config of how to extract the predicted label from the model output.
    #' @param wait (bool): Whether the call should wait until the job completes (default: False).
    #' @param logs (bool): Whether to show the logs produced by the job.
    #'              Only meaningful when wait is True (default: False).
    #' @param job_name (str): Processing job name. If not specified, the processor generates
    #'              a default job name, based on the image name and current timestamp.
    #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
    #'              user code file (default: None).
    #' @return sagemaker.processing.ProcessingJob: The ProcessingJob object representing the
    #'              baselining job.
    suggest_baseline = function(data_config,
                                bias_config,
                                model_config,
                                model_predicted_label_config=NULL,
                                wait=FALSE,
                                logs=FALSE,
                                job_name=NULL,
                                kms_key=NULL){
      baselining_processor = private$.create_baselining_processor()
      baselining_job_name = private$.generate_baselining_job_name(job_name=job_name)
      baselining_processor$run_bias(
        data_config=data_config,
        bias_config=bias_config,
        model_config=model_config,
        model_predicted_label_config=model_predicted_label_config,
        wait=wait,
        logs=logs,
        job_name=baselining_job_name,
        kms_key=kms_key)

      latest_baselining_job_config = ClarifyBaseliningConfig$new(
        analysis_config=BiasAnalysisConfig$new(
          bias_config=bias_config, headers=data_config$headers, label=data_config$label),
        features_attribute=data_config$features)

      if (!islistempty(model_predicted_label_config)){
        latest_baselining_job_config$inference_attribute = model_predicted_label_config$label
        latest_baselining_job_config$probability_attribute = (
          model_predicted_label_config$probability)
        latest_baselining_job_config$probability_threshold_attribute = (
          model_predicted_label_config$probability_threshold)
      }
      self$latest_baselining_job_config = latest_baselining_job_config
      self$latest_baselining_job_name = baselining_job_name
      self$latest_baselining_job = ClarifyBaseliningJob$new(
        processing_job=baselining_processor$latest_job)

      self$baselining_jobs = c(self$baselining_jobs, self$latest_baselining_job)
      return(baselining_processor$latest_job)
    },

    #' @description Creates a monitoring schedule.
    #' @param endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
    #'              This can either be the endpoint name or an EndpointInput.
    #' @param ground_truth_input (str): S3 URI to ground truth dataset.
    #' @param analysis_config (str or BiasAnalysisConfig): URI to analysis_config for the bias job.
    #'              If it is None then configuration of the latest baselining job will be reused, but
    #'              if no baselining job then fail the call.
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
    create_monitoring_schedule = function(endpoint_input,
                                          ground_truth_input,
                                          analysis_config=NULL,
                                          output_s3_uri=NULL,
                                          constraints=NULL,
                                          monitor_schedule_name=NULL,
                                          schedule_cron_expression=NULL,
                                          enable_cloudwatch_metrics=TRUE){
      if (!is.null(self$job_definition_name) || !is.null(self$monitoring_schedule_name)){
        message = c("It seems that this object was already used to create an Amazon Model ",
                   "Monitoring Schedule. To create another, first delete the existing one ",
                   "using my_monitor.delete_monitoring_schedule().")
        LOGGER$error(message)
        ValueError$new(message)
      }

      # create job definition
      monitor_schedule_name = private$.generate_monitoring_schedule_name(
        schedule_name=monitor_schedule_name)
      new_job_definition_name = name_from_base(self$JOB_DEFINITION_BASE_NAME)
      request_dict = private$.build_create_job_definition_request(
        monitoring_schedule_name=monitor_schedule_name,
        job_definition_name=new_job_definition_name,
        image_uri=self$image_uri,
        latest_baselining_job_name=self$latest_baselining_job_name,
        latest_baselining_job_config=self$latest_baselining_job_config,
        endpoint_input=endpoint_input,
        ground_truth_input=ground_truth_input,
        analysis_config=analysis_config,
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

      self$sagemaker_session$sagemaker$create_model_bias_job_definition(
        JobDefinitionName = request_dict$JobDefinitionName,
        ModelBiasBaselineConfig = request_dict$ModelBiasBaselineConfig,
        ModelBiasAppSpecification = request_dict$ModelBiasAppSpecification,
        ModelBiasJobInput = request_dict$ModelBiasJobInput,
        ModelBiasJobOutputConfig = request_dict$ModelBiasJobOutputConfig,
        JobResources = request_dict$JobResources,
        NetworkConfig = request_dict$NetworkConfig,
        RoleArn = request_dict$RoleArn,
        StoppingCondition = request_dict$StoppingCondition,
        Tags = request_dict$Tags)

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
          # noinspection PyBroadException
          tryCatch({
            self$sagemaker_session$sagemaker_client$delete_model_bias_job_definition(
              JobDefinitionName=new_job_definition_name)},
            error = function(e){
              message = sprintf("Failed to delete job definition %s.", new_job_definition_name)
              LOGGER$error(message)
              stop(e)
            })
        })
    },

    #' @description Updates the existing monitoring schedule.
    #'              If more options than schedule_cron_expression are to be updated, a new job definition will
    #'              be created to hold them. The old job definition will not be deleted.
    #' @param endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
    #'              This can either be the endpoint name or an EndpointInput.
    #' @param ground_truth_input (str): S3 URI to ground truth dataset.
    #' @param analysis_config (str or BiasAnalysisConfig): URI to analysis_config for the bias job.
    #'              If it is None then configuration of the latest baselining job will be reused, but
    #'              if no baselining job then fail the call.
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
    update_monitoring_schedule = function(endpoint_input=NULL,
                                          ground_truth_input=NULL,
                                          analysis_config=NULL,
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
      valid_args <- Filter(Negate(is.null), as.list(environment()))

      # Nothing to update
      if (length(valid_args) <= 0)
        return(NULL)

      # Only need to update schedule expression
      if (length(valid_args) == 1 && !is.null(schedule_cron_expression)){
        private$.update_monitoring_schedule(self$job_definition_name, schedule_cron_expression)
        return(NULL)
      }

      # Need to update schedule with a new job definition
      job_desc = self$sagemaker_session$sagemaker$describe_model_bias_job_definition(
        JobDefinitionName=self$job_definition_name)
      new_job_definition_name = name_from_base(self$JOB_DEFINITION_BASE_NAME)
      request_dict = private$.build_create_job_definition_request(
        monitoring_schedule_name=self$monitoring_schedule_name,
        job_definition_name=new_job_definition_name,
        image_uri=self$image_uri,
        existing_job_desc=job_desc,
        endpoint_input=endpoint_input,
        ground_truth_input=ground_truth_input,
        analysis_config=analysis_config,
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
      self$sagemaker_session$sagemaker$create_model_bias_job_definition(
        JobDefinitionName=request_dict$JobDefinitionName,
        ModelBiasBaselineConfig=request_dict$ModelBiasBaselineConfig,
        ModelBiasAppSpecification=request_dict$ModelBiasAppSpecification,
        ModelBiasJobInput=request_dict$ModelBiasJobInput,
        ModelBiasJobOutputConfig=request_dict$ModelBiasJobOutputConfig,
        JobResources=request_dict$JobResources,
        NetworkConfig=request_dict$NetworkConfig,
        RoleArn=request_dict$RoleArn,
        StoppingCondition=request_dict$StoppingCondition,
        Tags=request_dict$Tags)

      tryCatch({
        private$.update_monitoring_schedule(new_job_definition_name, schedule_cron_expression)
        self$job_definition_name = new_job_definition_name
        if (is.null(role))
          self$role = role
        self$instance_count = instance_count
        self$instance_type = instance_type
        self$volume_size_in_gb = volume_size_in_gb
        self$volume_kms_key = volume_kms_key
        self$output_kms_key = output_kms_key
        self$max_runtime_in_seconds = max_runtime_in_seconds
        self$env = env
        self$network_config = network_config
        },
      error = function(e){
        LOGGER$error("Failed to update monitoring schedule.")

        tryCatch({
          self$sagemaker_session$sagemaker$delete_model_bias_job_definition(
            JobDefinitionName=new_job_definition_name)
          },
          error = function(ee){
            message = sprintf("Failed to delete job definition %s.", new_job_definition_name)
            LOGGER$error(message)
            stop(ee)
          })
      })
    },

    #' @description Deletes the monitoring schedule and its job definition.
    delete_monitoring_schedule = function(){
      super$delete_monitoring_schedule()
      # Delete job definition.
      message = sprintf("Deleting Model Bias Job Definition with name: %s",
                        self$job_definition_name)
      LOGGER$info(message)
      self$sagemaker_session$sagemaker$delete_model_bias_job_definition(
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
      if (monitoring_type != self$monitoring_type())
        stop(sprintf("%s can only attach to ModelBias schedule.", class(self)[1]), call. = F)
      job_definition_name = schedule_desc[["MonitoringScheduleConfig"]][[
        "MonitoringJobDefinitionName"]]
      job_desc = sagemaker_session$sagemaker$describe_model_bias_job_definition(
        JobDefinitionName=job_definition_name)
      tags = sagemaker_session$list_tags(resource_arn=schedule_desc[["MonitoringScheduleArn"]])

      cls = self$clone()

      return(ClarifyModelMonitor$private_methods$.attach(
        clazz=cls,
        sagemaker_session=sagemaker_session,
        schedule_desc=schedule_desc,
        job_desc=job_desc,
        tags=tags)
      )
    }

  ),
  lock_objects = F
)

#' @title BiasAnalysisConfig class
#' @description Analysis configuration for ModelBiasMonitor.
#' @export
BiasAnalysisConfig = R6Class("BiasAnalysisConfig",
  public = list(

    #' @field analysis_config
    #' Analysis config dictionary
    analysis_config = NULL,

    #' @description Creates an analysis config dictionary.
    #' @param bias_config (sagemaker.clarify.BiasConfig): Config object related to bias
    #'              configurations.
    #' @param headers (list[str]): A list of column names in the input dataset.
    #' @param label (str): Target attribute for the model required by bias metrics.
    #'              Specified as column name or index for CSV dataset, or as JSONPath for JSONLines.
    initialize = function(bias_config,
                          headers=NULL,
                          label=NULL){
      self$analysis_config = bias_config$get_config()
      self$analysis_config[["headers"]] = headers
      self$analysis_config[["label"]] = label
    },

    #' @description Generates a request dictionary using the parameters provided to the class.
    to_list = function(){
      return(self$analysis_config)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      print_class(self)
    }
  )
)

#' @title Amazon SageMaker model monitor to monitor feature attribution of an endpoint.
#' @description Please see the `initiliaze` method of its base class for how to instantiate it.
#' @export
ModelExplainabilityMonitor = R6Class("ModelExplainabilityMonitor",
  inherit = ClarifyModelMonitor,
  public = list(

    #' @field JOB_DEFINITION_BASE_NAME
    #' Model definition base name
    JOB_DEFINITION_BASE_NAME = "model-explainability-job-definition",

    #' @description Type of the monitoring job.
    monitoring_type = function(){
      return("ModelExplainability")
    },

    #' @description Suggest baselines for use with Amazon SageMaker Model Monitoring Schedules.
    #' @param data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
    #' @param explainability_config (:class:`~sagemaker.clarify.ExplainabilityConfig`): Config of the
    #'              specific explainability method. Currently, only SHAP is supported.
    #' @param model_config (:class:`~sagemaker.clarify.ModelConfig`): Config of the model and its
    #'              endpoint to be created.
    #' @param model_scores :  Index or JSONPath location in the model output for the predicted scores
    #'              to be explained. This is not required if the model output is a single score.
    #' @param wait (bool): Whether the call should wait until the job completes (default: False).
    #' @param logs (bool): Whether to show the logs produced by the job.
    #'              Only meaningful when wait is True (default: False).
    #' @param job_name (str): Processing job name. If not specified, the processor generates
    #'              a default job name, based on the image name and current timestamp.
    #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
    #'              user code file (default: None).
    #' @return sagemaker.processing.ProcessingJob: The ProcessingJob object representing the
    #'              baselining job.
    suggest_baseline = function(data_config,
                                explainability_config,
                                model_config,
                                model_scores=NULL,
                                wait=FALSE,
                                logs=FALSE,
                                job_name=NULL,
                                kms_key=NULL){
      baselining_processor = private$.create_baselining_processor()
      baselining_job_name = private$.generate_baselining_job_name(job_name=job_name)
      baselining_processor$run_explainability(
        data_config=data_config,
        model_config=model_config,
        explainability_config=explainability_config,
        model_scores=model_scores,
        wait=wait,
        logs=logs,
        job_name=baselining_job_name,
        kms_key=kms_key)

      # Explainability analysis doesn't need label
      headers = data_config$headers
      if (!islistempty(headers) && data_config$label %in% headers)
        headers[[data_config$label]] = NULL
      self$latest_baselining_job_config = ClarifyBaseliningConfig$new(
        analysis_config=ExplainabilityAnalysisConfig$new(
          explainability_config=explainability_config,
          model_config=model_config,
          headers=headers),
        features_attribute=data_config$features,
        inference_attribute=model_scores)
      self$latest_baselining_job_name = baselining_job_name
      self$latest_baselining_job = ClarifyBaseliningJob$new(
        processing_job=baselining_processor$latest_job)

      self$baselining_jobs = c(self$baselining_jobs, self$latest_baselining_job)
      return(baselining_processor$latest_job)
    },

    #' @description Creates a monitoring schedule.
    #' @param endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
    #'              This can either be the endpoint name or an EndpointInput.
    #' @param analysis_config (str or ExplainabilityAnalysisConfig): URI to the analysis_config for
    #'              the explainability job. If it is None then configuration of the latest baselining
    #'              job will be reused, but if no baselining job then fail the call.
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
    create_monitoring_schedule = function(endpoint_input,
                                          analysis_config=NULL,
                                          output_s3_uri=NULL,
                                          constraints=NULL,
                                          monitor_schedule_name=NULL,
                                          schedule_cron_expression=NULL,
                                          enable_cloudwatch_metrics=TRUE){
      if (!is.null(self$job_definition_name) || !is.null(self$monitoring_schedule_name)){
        message = paste(
          "It seems that this object was already used to create an Amazon Model",
          "Monitoring Schedule. To create another, first delete the existing one",
          "using my_monitor.delete_monitoring_schedule()."
        )
        LOGGER$error(message)
        ValueError$new(message)
      }

      # create job definition
      monitor_schedule_name = private$.generate_monitoring_schedule_name(
        schedule_name=monitor_schedule_name
      )
      new_job_definition_name = name_from_base(self$JOB_DEFINITION_BASE_NAME)
      request_dict = private$.build_create_job_definition_request(
        monitoring_schedule_name=monitor_schedule_name,
        job_definition_name=new_job_definition_name,
        image_uri=self$image_uri,
        latest_baselining_job_name=self$latest_baselining_job_name,
        latest_baselining_job_config=self$latest_baselining_job_config,
        endpoint_input=endpoint_input,
        analysis_config=analysis_config,
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
        self$sagemaker_session$sagemaker$create_model_explainability_job_definition,
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
        LOGGER$error("Failed to create monitoring schedule.")
        # noinspection PyBroadException
        tryCatch({
          self$sagemaker_session$sagemaker$delete_model_explainability_job_definition(
            JobDefinitionName=new_job_definition_name
          )
        },
        error = function(ee){
          message = sprintf("Failed to delete job definition %s.",new_job_definition_name)
          LOGGER$error(message)
          stop(ee)
        })
      })
    },

    #' @description Updates the existing monitoring schedule.
    #'              If more options than schedule_cron_expression are to be updated, a new job definition will
    #'              be created to hold them. The old job definition will not be deleted.
    #' @param endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
    #'              This can either be the endpoint name or an EndpointInput.
    #' @param analysis_config (str or BiasAnalysisConfig): URI to analysis_config for the bias job.
    #'              If it is None then configuration of the latest baselining job will be reused, but
    #'              if no baselining job then fail the call.
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
    update_monitoring_schedule = function(endpoint_input=NULL,
                                          analysis_config=NULL,
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
        ValueError$new("Nothing to update.")

      # Only need to update schedule expression
      if (length(valid_args) == 1 && !is.null(schedule_cron_expression)){
        private$.update_monitoring_schedule(self$job_definition_name, schedule_cron_expression)
        return(NULL)
      }

      # Need to update schedule with a new job definition
      job_desc = (
        self$sagemaker_session$sagemaker$describe_model_explainability_job_definition(
          JobDefinitionName=self$job_definition_name)
      )
      new_job_definition_name = name_from_base(self$JOB_DEFINITION_BASE_NAME)
      request_dict = private$.build_create_job_definition_request(
        monitoring_schedule_name=self$monitoring_schedule_name,
        job_definition_name=new_job_definition_name,
        image_uri=self$image_uri,
        existing_job_desc=job_desc,
        endpoint_input=endpoint_input,
        analysis_config=analysis_config,
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
        self$sagemaker_session$sagemaker$create_model_explainability_job_definition,
        request_dict)
      tryCatch({
        private$.update_monitoring_schedule(new_job_definition_name, schedule_cron_expression)
        self$job_definition_name = new_job_definition_name
        if (!is.null(role))
          self$role = role
        if (!is.null(instance_count))
          self$instance_count = instance_count
        if (!is.null(instance_type))
          self$instance_type = instance_type
        if (!is.nul(volume_size_in_gb))
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
        tyCatch({
          self$sagemaker_session$sagemaker$delete_model_explainability_job_definition(
            JobDefinitionName=new_job_definition_name)
        },
        error = function(ee){
          message = sprintf("Failed to delete job definition %s.", new_job_definition_name)
          LOGGER$error(message)
          stop(ee)
        })
      })
    },

    #' @description Deletes the monitoring schedule and its job definition.
    delete_monitoring_schedule = function(){
      super$delete_monitoring_schedule()
      # Delete job definition.
      message = sprintf("Deleting Model Explainability Job Definition with name: %s",
        self$job_definition_name)
      LOGGER$info(message)
      self$sagemaker_session$sagemaker$delete_model_explainability_job_definition(
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
      if (monitoring_type != self$monitoring_type())
        TypeError$new(
          sprintf("%s can only attach to ModelExplainability schedule.",class(self)[1]))
      job_definition_name = schedule_desc[["MonitoringScheduleConfig"]][[
        "MonitoringJobDefinitionName"
      ]]
      job_desc = sagemaker_session$sagemaker$describe_model_explainability_job_definition(
        JobDefinitionName=job_definition_name
      )
      tags = sagemaker_session$list_tags(resource_arn=schedule_desc[["MonitoringScheduleArn"]])

      cls = self$clone()

      return(ClarifyModelMonitor$private_methods$.attach(
        clazz=cls,
        sagemaker_session=sagemaker_session,
        schedule_desc=schedule_desc,
        job_desc=job_desc,
        tags=tags)
      )
    }
  ),
  lock_objects = F
)

#' @title ExplainabilityAnalysisConfig class
#' @description Analysis configuration for ModelExplainabilityMonitor.
#' @export
ExplainabilityAnalysisConfig = R6Class("ExplainabilityAnalysisConfig",
  public =list(

    #' @field analysis_config
    #' Analysis config dictionary
    analysis_config = NULL,

    #' @description Creates an analysis config dictionary.
    #' @param explainability_config (sagemaker.clarify.ExplainabilityConfig): Config object related
    #'              to explainability configurations.
    #' @param model_config (sagemaker.clarify.ModelConfig): Config object related to bias
    #'              configurations.
    #' @param headers (list[str]): A list of feature names (without label) of model/endpint input.
    initialize = function(explainability_config,
                          model_config,
                          headers=NULL){
      self$analysis_config = list(
        "methods"= explainability_config$get_explainability_config(),
        "predictor"= model_config$get_predictor_config()
      )
      if (!is.null(headers))
        self$analysis_config[["headers"]] = headers
    },

    #' @description Generates a request dictionary using the parameters provided to the class.
    to_list = function(){
      return(self$analysis_config)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      print_class(self)
    }
  )
)

#' @title ClarifyBaseliningConfig class
#' @description Data class to hold some essential analysis configuration of ClarifyBaseliningJob
#' @export
ClarifyBaseliningConfig = R6Class("ClarifyBaseliningConfig",
  public = list(

    #' @field analysis_config
    #' analysis config from configurations of the baselining job.
    analysis_config = NULL,

    #' @field features_attribute
    #'        JSONpath to locate features in predictor request payload.
    #'        Only required when predictor content type is JSONlines.
    features_attribute=NULL,

    #' @field inference_attribute
    #' Index, header or JSONpath to locate predicted label in predictor response payload
    inference_attribute=NULL,

    #' @field probability_attribute
    #'        Index or JSONpath location in the model output for
    #'        probabilities or scores to be used for explainability.
    probability_attribute=NULL,

    #' @field probability_threshold_attribute
    #'        Value to indicate the threshold to select
    #'        the binary label in the case of binary classification. Default is 0.5.
    probability_threshold_attribute=NULL,

    #' @description Initialization.
    #' @param analysis_config (BiasAnalysisConfig or ExplainabilityAnalysisConfig): analysis config
    #'              from configurations of the baselining job.
    #' @param features_attribute (str): JSONpath to locate features in predictor request payload.
    #'              Only required when predictor content type is JSONlines.
    #' @param inference_attribute (str): Index, header or JSONpath to locate predicted label in
    #'              predictor response payload.
    #' @param probability_attribute (str): Index or JSONpath location in the model output for
    #'              probabilities or scores to be used for explainability.
    #' @param probability_threshold_attribute (float): Value to indicate the threshold to select
    #'              the binary label in the case of binary classification. Default is 0.5.
    initialize = function(analysis_config,
                          features_attribute=NULL,
                          inference_attribute=NULL,
                          probability_attribute=NULL,
                          probability_threshold_attribute=NULL){
      self$analysis_config = analysis_config
      self$features_attribute = features_attribute
      self$inference_attribute = inference_attribute
      self$probability_attribute = probability_attribute
      self$probability_threshold_attribute = probability_threshold_attribute
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      print_class(self)
    }
  )
)

#' @title ClarifyBaseliningJob class
#' @description Provides functionality to retrieve baseline-specific output from Clarify baselining job.
#' @export
ClarifyBaseliningJob = R6Class("ClarifyBaseliningJob",
  inherit = BaseliningJob,
  public = list(

    #' @description Initializes a ClarifyBaseliningJob that tracks a baselining job by suggest_baseline()
    #' @param processing_job (sagemaker.processing.ProcessingJob): The ProcessingJob used for
    #'              baselining instance.
    initialize = function(processing_job){
      super$initialize(
        sagemaker_session=processing_job$sagemaker_session,
        job_name=processing_job$job_name,
        inputs=processing_job$inputs,
        outputs=processing_job$outputs,
        output_kms_key=processing_job.output_kms_key)
    },

    #' @description Not implemented.
    #'              The class doesn't support statistics.
    baseline_statistics = function(){
      NotImplementedError$new(sprintf("%s doesn't support statistics.", class(self)[1]))
    },

    #' @description Returns a sagemaker.model_monitor.
    #'              Constraints object representing the constraints JSON file generated by this baselining job.
    #' @param file_name (str): Keep this parameter to align with method signature in super class,
    #'              but it will be ignored.
    #' @param kms_key (str): The kms key to use when retrieving the file.
    #' @return sagemaker.model_monitor.Constraints: The Constraints object representing the file that
    #'              was generated by the job.
    suggested_constraints = function(file_name=NULL,
                                     kms_key=NULL){
      return(super$suggested_constraints(
        "analysis.json", kms_key)
      )
    }
  )
)

#' @title ClarifyMonitoringExecution class
#' @description Provides functionality to retrieve monitoring-specific files output from executions.
#' @export
ClarifyMonitoringExecution = R6Class("ClarifyMonitoringExecution",
  inherit = MonitoringExecution,
  public = list(

    #' @description Initializes an object that tracks a monitoring execution by a Clarify model monitor
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
    initialize = function(sagemaker_session,
                          job_name,
                          inputs,
                          output,
                          output_kms_key=NULL){
      super$initialize(
        sagemaker_session=sagemaker_session,
        job_name=job_name,
        inputs=inputs,
        output=output,
        output_kms_key=output_kms_key)
    },

    #' @description Not implemented.
    #'              The class doesn't support statistics.
    statistics = function(){
      NotImplementedError$new(
        sprintf("%s doesn't support statistics.", class(self)[1])
      )
    }
  )
)
