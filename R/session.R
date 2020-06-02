#' Sagemaker Session Class
#'
#' @description
#' Manage interactions with the Amazon SageMaker APIs and any other AWS services needed.
#' This class provides convenient methods for manipulating entities and resources that Amazon
#' SageMaker uses, such as training jobs, endpoints, and input datasets in S3.
#' AWS service calls are delegated to an underlying Boto3 session, which by default
#' is initialized using the AWS configuration chain. When you make an Amazon SageMaker API call
#' that accesses an S3 bucket location and one is not specified, the ``Session`` creates a default
#' bucket based on a naming convention which includes the current AWS account ID.
#' @import paws
#' @import jsonlite
#' @import R6
#' @import logger
#' @import utils
#' @export
Session = R6Class("Session",
                  public = list(
                    initialize = function(paws_credentials = NULL,
                                          bucket = NULL) {
                      self$paws_credentials <- .paws_cred(paws_credentials)
                      self$bucket <- NULL
                      self$config <- NULL
                      # get sagemaker object from paws
                      self$sagemaker = paws::sagemaker(config = self$paws_credentials$credentials)
                    },
                    # get execution arn role
                    get_caller_identity_arn = function(){

                      if(file.exists(NOTEBOOK_METADATA_FILE)){
                        instance_name = read_json(NOTEBOOK_METADATA_FILE)$ResourceName

                        tryCatch({instance_desc = self$sagemaker$describe_notebook_instance(NotebookInstanceName=instance_name)},
                                 error=function(e) stop(sprintf("Couldn't call 'describe_notebook_instance' to get the Role \nARN of the instance %s.",instance_name),
                                                        call. = F))
                        return(instance_desc$RoleArn)
                      }

                      assumed_role <- paws::sts(config = self$paws_credentials$credentials)$get_caller_identity()$Arn
                      if (grepl("AmazonSageMaker-ExecutionRole", assumed_role)){
                        role <- gsub("^(.+)sts::(\\d+):assumed-role/(.+?)/.*$", "\\1iam::\\2:role/service-role/\\3", assumed_role)
                        return(role)}

                      role <- gsub("^(.+)sts::(\\d+):assumed-role/(.+?)/.*$", "\\1iam::\\2:role/\\3", assumed_role)

                      # Call IAM to get the role's path
                      role_name = gsub(".*/","", role)

                      tryCatch({role = paws::iam(config = self$paws_credentials$credentials)$get_role(RoleName = role_name)$Role$Arn},
                               error = function(e) stop("Couldn't call 'get_role' to get Role ARN from role name ", role_name ," to get Role path.", call. = F))

                      return(role)
                    },

                    upload_data = function(path, bucket = NULL, key_prefix = "data", ...){

                      log_warn(paste("'upload_data' method will be deprecated in favor of 'S3Uploader' class",
                                     "(https://sagemaker.readthedocs.io/en/stable/s3.html#sagemaker.s3.S3Uploader)",
                                     "in SageMaker Python SDK v2."))

                      key_suffix = NULL

                      # get all files in directory
                      if(file.exists(path) && file_test("-f",path)) local_path <- path
                      else local_path <-list.files(path, full.names = T)[!(list.files(path, full.names = T) %in% list.dirs(path, full.names = T))]

                      s3_key = paste(key_prefix, basename(local_path), sep = "/")

                      # if bucke parameter hasn't been selected use class parameter
                      bucket = bucket %||% self$default_bucket()

                      # Get s3 object from paws
                      s3 <- paws::s3(config = self$paws_credentials$credentials)

                      # Upload file to s3
                      for (i in 1:length(local_path)){
                        obj <- readBin(local_path[i], "raw", n = file.size(local_path[i]))
                        s3$put_object(Body = obj, Bucket = bucket, Key = s3_key[i], ...)}
                    },

                    # Download file or directory from S3
                    download_data = function(path="", bucket, key_prefix = NULL, ...){
                      # path: directory to download data to


                      # Get s3 object from paws
                      s3 = paws::s3(config = self$paws_credentials$credentials)

                      next_token = NULL
                      keys = character()
                      # Loop through the contents of the bucket, 1,000 objects at a time. Gathering all keys into
                      # a "keys" list.
                      while(!identical(next_token, character(0))){
                        response = s3$list_objects_v2(Bucket = bucket, Prefix = key_prefix, ContinuationToken = next_token)
                        keys = c(keys, sapply(response$Contents, function(x) x$Key))
                        next_token = response$ContinuationToken
                      }

                      # convert key_prefix if NULL
                      if (is.null(key_prefix)) key_prefix = ""

                      tail_s3_uri_path = basename(keys)
                      # list of directories to be created
                      output = dirname(gsub(key_prefix,"", keys))

                      # get only files to be downloaded
                      files = ifelse(output == "/", FALSE, ifelse(output == ".", FALSE, TRUE))
                      list_dir = gsub("^/", "", output[files])

                      if (path =="") path = getwd()
                      destination_path = sapply(1:length(list_dir), function(i)if(list_dir[i] == "") file.path(path, tail_s3_uri_path[files][i])
                                                else file.path(path, list_dir[i], tail_s3_uri_path[files][i]))

                      # create directory
                      sapply(list_dir, dir.create, showWarnings = F)

                      for (i in 1:length(keys[files])){
                        obj = s3$get_object(Bucket = bucket, Key = keys[i], ...)
                        write_bin(obj$Body, destination_path[i])
                      }
                    },

                    list_s3_files = function(bucket, key_prefix){
                      # Get s3 object from paws
                      s3 = paws::s3(config = self$paws_credentials$credentials)
                      next_token = NULL
                      keys = character()
                      # Loop through the contents of the bucket, 1,000 objects at a time. Gathering all keys into
                      # a "keys" list.
                      while(!identical(next_token, character(0))){
                        response = s3$list_objects_v2(Bucket = bucket, Prefix = key_prefix, ContinuationToken = next_token)
                        keys = c(keys, sapply(response$Contents, function(x) x$Key))
                        next_token = response$ContinuationToken
                      }
                      return(keys)
                    },


                    # if default bucket has not been set create a S3 bucket
                    default_bucket = function(){
                      if (!is.null(self$bucket)) return(self$bucket)

                      region = self$paws_region_name

                      if(is.null(self$bucket)) {
                        account = paws::sts(config = self$paws_credentials$credentials)$get_caller_identity()$Account
                        self$bucket = sprintf("sagemaker-%s-%s", region, account)
                      }

                      private$.create_s3_bucket_if_it_does_not_exist(bucket_name = self$bucket, region = region)

                      return(self$bucket)
                    },

                    train = function(input_mode,
                                     input_config,
                                     role,
                                     job_name,
                                     output_config = NULL,
                                     resource_config = NULL,
                                     vpc_config = NULL,
                                     hyperparameters = NULL,
                                     stop_condition = NULL,
                                     tags = NULL,
                                     metric_definitions = NULL,
                                     enable_network_isolation=FALSE,
                                     image=NULL,
                                     algorithm_arn=NULL,
                                     encrypt_inter_container_traffic=FALSE,
                                     train_use_spot_instances=FALSE,
                                     checkpoint_s3_uri=NULL,
                                     checkpoint_local_path=NULL,
                                     experiment_config=NULL,
                                     debugger_rule_configs=NULL,
                                     debugger_hook_config=NULL,
                                     tensorboard_output_config=NULL,
                                     enable_sagemaker_metrics=NULL){

                      AlgorithmSpecification = list(TrainingInputMode = input_mode)
                      OutputDataConfig = output_config
                      TrainingJobName = job_name
                      StoppingCondition = stop_condition
                      ResourceConfig = resource_config
                      RoleArn = role

                      if(!is.null(image) && !is.null(algorithm_arn)) {
                        stop("image and algorithm_arn are mutually exclusive.",
                             sprintf("Both were provided: image: %s algorithm_arn: %s",image, algorithm_arn), call. = F)}

                      if(is.null(image) && is.null(algorithm_arn)){
                        stop("either image or algorithm_arn is required. None was provided.", call. = F)}

                      InputDataConfig = input_config

                      AlgorithmSpecification["TrainingImage"] = image
                      AlgorithmSpecification["AlgorithmName"] = algorithm_arn
                      AlgorithmSpecification["MetricDefinitions"] = metric_definitions
                      AlgorithmSpecification["EnableSageMakerMetricsTimeSeries"] = enable_sagemaker_metrics

                      HyperParameters = hyperparameters
                      Tags = tags
                      VpcConfig = vpc_config
                      ExperimentConfig = experiment_config
                      EnableNetworkIsolation = enable_network_isolation

                      EnableInterContainerTrafficEncryption = encrypt_inter_container_traffic
                      EnableManagedSpotTraining = train_use_spot_instances

                      CheckpointConfig = NULL

                      if (!is.null(checkpoint_s3_uri) || !is.null(checkpoint_local_path)) {
                        checkpoint_config = list()
                        checkpoint_config["S3Uri"] = checkpoint_s3_uri
                        checkpoint_config["LocalPath"] = checkpoint_local_path
                        CheckpointConfig = list(checkpoint_config)
                      }

                      DebugRuleConfigurations = debugger_rule_configs
                      DebugHookConfig = debugger_hook_config

                      TensorBoardOutputConfig = tensorboard_output_config

                      log_info("Creating training-job with name: ", job_name)
                      log_debug("train request: ", train_request)

                      self$sagemaker$create_training_job(TrainingJobName = TrainingJobName,
                                             HyperParameters = HyperParameters,
                                             AlgorithmSpecification = AlgorithmSpecification,
                                             RoleArn = RoleArn,
                                             InputDataConfig = InputDataConfig,
                                             OutputDataConfig = OutputDataConfig,
                                             ResourceConfig = ResourceConfig,
                                             VpcConfig = VpcConfig,
                                             StoppingCondition = StoppingCondition,
                                             Tags = Tags,
                                             EnableNetworkIsolation = EnableNetworkIsolation,
                                             EnableInterContainerTrafficEncryption = EnableInterContainerTrafficEncryption,
                                             EnableManagedSpotTraining = EnableManagedSpotTraining,
                                             CheckpointConfig = CheckpointConfig,
                                             DebugHookConfig = DebugHookConfig,
                                             DebugRuleConfigurations = DebugRuleConfigurations,
                                             TensorBoardOutputConfig = TensorBoardOutputConfig,
                                             ExperimentConfig = ExperimentConfig)
                    },

                    process = function(inputs = NULL,
                                       output_config = NULL,
                                       job_name = NULL,
                                       resources = NULL,
                                       stopping_condition = NULL,
                                       app_specification = NULL,
                                       environment = NULL,
                                       network_config = NULL,
                                       role_arn,
                                       tags = NULL,
                                       experiment_config=NULL){


                      ProcessingJobName = job_name
                      ProcessingResources = resources
                      AppSpecification = app_specification
                      RoleArn = role_arn

                      ProcessingInputs = inputs

                      if(!is.null(output_config$Outputs)) ProcessingOutputConfig = output_config

                      Environment = environment
                      NetworkConfig = network_config
                      StoppingCondition = stopping_condition
                      Tags = tags
                      ExperimentConfig = experiment_config

                      log_info("Creating processing-job with name ", job_name)
                      log_debug("process request: ", process_request)

                      self$sagemaker$create_processing_job(ProcessingInputs = ProcessingInputs,
                                               ProcessingOutputConfig = ProcessingOutputConfig,
                                               ProcessingJobName = ProcessingJobName,
                                               ProcessingResources = ProcessingResources,
                                               StoppingCondition = StoppingCondition,
                                               AppSpecification  = AppSpecification,
                                               Environment = Environment,
                                               NetworkConfig = NetworkConfig,
                                               RoleArn = RoleArn,
                                               Tags = Tags,
                                               ExperimentConfig = ExperimentConfig)
                    },

                    create_monitoring_schedule = function(monitoring_schedule_name,
                                                          schedule_expression = NULL,
                                                          statistics_s3_uri = NULL,
                                                          constraints_s3_uri = NULL,
                                                          monitoring_inputs = NULL,
                                                          monitoring_output_config = NULL,
                                                          instance_count = 1,
                                                          instance_type = c("ml.t3.medium","ml.t3.large","ml.t3.xlarge","ml.t3.2xlarge","ml.m4.xlarge","ml.m4.2xlarge","ml.m4.4xlarge","ml.m4.10xlarge","ml.m4.16xlarge","ml.c4.xlarge","ml.c4.2xlarge","ml.c4.4xlarge","ml.c4.8xlarge","ml.p2.xlarge","ml.p2.8xlarge","ml.p2.16xlarge","ml.p3.2xlarge","ml.p3.8xlarge","ml.p3.16xlarge","ml.c5.xlarge","ml.c5.2xlarge","ml.c5.4xlarge","ml.c5.9xlarge","ml.c5.18xlarge","ml.m5.large","ml.m5.xlarge","ml.m5.2xlarge","ml.m5.4xlarge","ml.m5.12xlarge","ml.m5.24xlarge","ml.r5.large","ml.r5.xlarge","ml.r5.2xlarge","ml.r5.4xlarge","ml.r5.8xlarge","ml.r5.12xlarge","ml.r5.16xlarge","ml.r5.24xlarge"),
                                                          volume_size_in_gb = NULL,
                                                          volume_kms_key = NULL,
                                                          image_uri = NULL,
                                                          entrypoint = NULL,
                                                          arguments = NULL,
                                                          record_preprocessor_source_uri = NULL,
                                                          post_analytics_processor_source_uri = NULL,
                                                          max_runtime_in_seconds = NULL,
                                                          environment = NULL,
                                                          network_config = NULL,
                                                          role_arn= NULL,
                                                          tags = NULL){

                      instance_type = match.arg(instance_type)


                      MonitoringScheduleName = monitoring_schedule_name
                      MonitoringScheduleConfig = list(MonitoringJobDefinition =
                                                        list(MonitoringInputs = monitoring_inputs,
                                                             RoleArn = role_arn,
                                                             MonitoringAppSpecification = list(ImageUri = image_uri)))

                      MonitoringResources = list(ClusterConfig = list(
                        InstanceCount = instance_count,
                        InstanceType = instance_type,
                        VolumeSizeInGB = volume_size_in_gb))

                      MonitoringScheduleConfig[["MonitoringJobDefinition"]]["MonitoringResources"] = list(MonitoringResources)

                      if(!is.null(schedule_expression)) MonitoringScheduleConfig[["ScheduleConfig"]] = list(ScheduleExpression = schedule_expression)

                      MonitoringScheduleConfig[["MonitoringJobDefinition"]]["MonitoringOutputConfig"] = monitoring_output_config

                      BaselineConfig = NULL
                      if (!is.null(statistics_s3_uri) || !is.null(constraints_s3_uri)){
                        BaselineConfig = list()
                        if(!is.null(statistics_s3_uri)) BaselineConfig[["StatisticsResource"]] = list(S3Uri = statistics_s3_uri)
                        if(!is.null(constraints_s3_uri)) BaselineConfig[["ConstraintsResource"]] = list(S3Uri = constraints_s3_uri)
                      }

                      MonitoringScheduleConfig[["MonitoringJobDefinition"]][["BaselineConfig"]] = BaselineConfig
                      MonitoringScheduleConfig[["MonitoringJobDefinition"]][["MonitoringAppSpecification"]]["RecordPreprocessorSourceUri"] = record_preprocessor_source_uri
                      MonitoringScheduleConfig[["MonitoringJobDefinition"]][["MonitoringAppSpecification"]]["PostAnalyticsProcessorSourceUri"] = post_analytics_processor_source_uri
                      MonitoringScheduleConfig[["MonitoringJobDefinition"]][["MonitoringAppSpecification"]][["ContainerEntrypoint"]] = entrypoint
                      MonitoringScheduleConfig[["MonitoringJobDefinition"]][["MonitoringAppSpecification"]][["ContainerArguments"]] = arguments
                      MonitoringScheduleConfig[["MonitoringJobDefinition"]][["MonitoringResources"]][["ClusterConfig"]]["VolumeKmsKeyId"] = volume_kms_key

                      if(!is.null(max_runtime_in_seconds)) MonitoringScheduleConfig[["MonitoringScheduleConfig"]][[
                                                                                      "MonitoringJobDefinition"]][["StoppingCondition"]] = list(MaxRuntimeInSeconds = max_runtime_in_seconds)

                      MonitoringScheduleConfig[["MonitoringJobDefinition"]][["Environment"]] = environment
                      MonitoringScheduleConfig[["MonitoringJobDefinition"]][["NetworkConfig"]] = network_config

                      Tags = tags

                      log_info("Creating monitoring schedule name ", monitoring_schedule_name)
                      log_debug("monitoring_schedule_request= %s", monitoring_schedule_request)

                      self$sagemaker$create_monitoring_schedule(MonitoringScheduleName = MonitoringScheduleName,
                                                    MonitoringScheduleConfig = MonitoringScheduleConfig,
                                                    Tags= Tags)
                    },

                    # TODO: translate function https://github.com/aws/sagemaker-python-sdk/blob/762b509f711daf4d0d7b759626f614fcf618b74e/src/sagemaker/session.py#L835
                    update_monitoring_schedule = function(monitoring_schedule_name,
                                                          schedule_expression=NULL,
                                                          statistics_s3_uri=NULL,
                                                          constraints_s3_uri=NULL,
                                                          monitoring_inputs=NULL,
                                                          monitoring_output_config=NULL,
                                                          instance_count=NULL,
                                                          instance_type=NULL,
                                                          volume_size_in_gb=NULL,
                                                          volume_kms_key=NULL,
                                                          image_uri=NULL,
                                                          entrypoint=NULL,
                                                          arguments=NULL,
                                                          record_preprocessor_source_uri=NULL,
                                                          post_analytics_processor_source_uri=NULL,
                                                          max_runtime_in_seconds=NULL,
                                                          environment=NULL,
                                                          network_config=NULL,
                                                          role_arn=NULL){
                      existing_desc = self$sagemaker$describe_monitoring_schedule(MonitoringScheduleName=monitoring_schedule_name)

                      existing_schedule_config = NULL
                      if (!is.null(existing_desc$MonitoringScheduleConfig)
                          && !is.null(existing_desc$MonitoringScheduleConfig$ScheduleConfig)
                          && !is.null(existing_desc$MonitoringScheduleConfig$ScheduleConfig$ScheduleExpression)){
                        existing_schedule_config = existing_desc$MonitoringScheduleConfig$ScheduleConfig$ScheduleExpression
                      }
                    },

                    start_monitoring_schedule = function(monitoring_schedule_name){
                      message(sprintf("Starting Monitoring Schedule with name: %s",monitoring_schedule_name))

                      self$sagemaker$start_monitoring_schedule(MonitoringScheduleName=monitoring_schedule_name)
                    },

                    stop_monitoring_schedule = function(monitoring_schedule_name){
                      message(sprintf("Stopping Monitoring Schedule with name: %s",monitoring_schedule_name))
                      self$sagemaker$stop_monitoring_schedule(MonitoringScheduleName=monitoring_schedule_name)
                    },

                    delete_monitoring_schedule = function(monitoring_schedule_name){
                      message(sprintf("Deleting Monitoring Schedule with name: %s",monitoring_schedule_name))
                      self$sagemaker$delete_monitoring_schedule(MonitoringScheduleName=monitoring_schedule_name)
                    },

                    describe_monitoring_schedule = function(monitoring_schedule_name){
                      self$sagemaker$describe_monitoring_schedule(MonitoringScheduleName=monitoring_schedule_name)
                    },

                    list_monitoring_executions = function(monitoring_schedule_name,
                                                          sort_by="ScheduledTime",
                                                          sort_order="Descending",
                                                          max_results=100){
                      self$sagemaker$list_monitoring_executions(
                        MonitoringScheduleName=monitoring_schedule_name,
                        SortBy=sort_by,
                        SortOrder=sort_order,
                        MaxResults=max_results)
                    },

                    list_monitoring_schedules = function(endpoint_name=NULL,
                                                         sort_by="CreationTime",
                                                         sort_order="Descending",
                                                         max_results=100){

                      if (!is.null(endpoint_name)){
                        response = self$sagemaker$list_monitoring_schedules(
                          EndpointName=endpoint_name,
                          SortBy=sort_by,
                          SortOrder=sort_order,
                          MaxResults=max_results,
                        )
                      } else{
                          response = self.sagemaker_client.list_monitoring_schedules(
                            SortBy=sort_by,
                            SortOrder=sort_order,
                            MaxResults=max_results)}

                      return(response)
                    },

                    was_processing_job_successful = function(job_name){
                      job_desc = self$sagemaker$describe_processing_job(ProcessingJobName=job_name)
                      return(job_desc$ProcessingJobStatus == "Completed")
                    },

                    describe_processing_job = function(job_name){
                      return(self$sagemaker$describe_processing_job(ProcessingJobName=job_name))
                    },

                    stop_processing_job = function(job_name){
                      return(self$sagemaker$stop_processing_job(ProcessingJobName=job_name))
                    },

                    stop_training_job = function(job_name){
                      return(self$sagemaker$stop_training_job(TrainingJobName=job_name))
                    },

                    describe_training_job = function(job_name){
                      return(self$sagemaker$describe_training_job(TrainingJobName=job_name))
                    },

                    print = function(...){
                      cat("<sagemaker$Session>")
                      invisible(self)
                    }
                  ),
                  private = list(
                    .create_s3_bucket_if_it_does_not_exist = function(bucket_name, region){
                      # Creates an S3 Bucket if it does not exist.
                      # Also swallows a few common exceptions that indicate that the bucket already exists or
                      # that it is being created.
                      # Args:
                      #   bucket_name (str): Name of the S3 bucket to be created.
                      #   region (str): The region in which to create the bucket.

                      s3 <- paws::s3(config = self$paws_credentials$credentials)

                      resp <- tryCatch(s3$head_bucket(Bucket = bucket_name), error = function(e) e)

                      # check if bucket exists: HTTP 404 bucket not found
                      if(inherits(resp, "http_404")){
                        s3$create_bucket(Bucket = bucket_name, CreateBucketConfiguration= list(LocationConstraint = region))
                       log_info(sprintf("Created S3 bucket: %s", bucket_name))}

                    }
                  ),
                  active = list(
                    # return aws region
                    paws_region_name = function() {self$paws_credentials$credentials$region}
                  ),
                  lock_objects = F
)

get_execution_role <- function(sagemaker_session = NULL){
  sagemaker_session <- if(!inherits(sagemaker_session, "Session")) Session$new() else sagemaker_session

  arn <- sagemaker_session$get_caller_identity_arn()

  if (grepl(":role/", arn)) {
    return(arn)
  } else {
   message <- sprintf("The current AWS identity is not a role: %s, therefore it cannot be used as a \nSageMaker execution role", arn)
   stop(message, call.= F)
  }
}

NOTEBOOK_METADATA_FILE = "/opt/ml/metadata/resource-metadata.json"
