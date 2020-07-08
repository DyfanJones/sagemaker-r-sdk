# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/session.py

#' @include utils.R
#' @include logs.R
#' @include set_credentials.R
#' @include vpc_utils.R

#' @import paws
#' @import jsonlite
#' @import R6
#' @import logger
#' @import utils

#' @title Sagemaker Session Class
#'
#' @name Session
#' @description
#' Manage interactions with the Amazon SageMaker APIs and any other AWS services needed.
#' This class provides convenient methods for manipulating entities and resources that Amazon
#' SageMaker uses, such as training jobs, endpoints, and input datasets in S3.
#' AWS service calls are delegated to an underlying paws session, which by default
#' is initialized using the AWS configuration chain. When you make an Amazon SageMaker API call
#' that accesses an S3 bucket location and one is not specified, the ``Session`` creates a default
#' bucket based on a naming convention which includes the current AWS account ID.
#'
#' @export
Session = R6Class("Session",
  public = list(

    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #'              Initialize a SageMaker \code{Session}.
    #'
    #' @param paws_credentials (PawsCredentials): The underlying AWS credentails passed to paws SDK.
    #' @param bucket (str): The default Amazon S3 bucket to be used by this session.
    #'              This will be created the next time an Amazon S3 bucket is needed (by calling
    #'              :func:\code{default_bucket}).
    #'              If not provided, a default bucket will be created based on the following format:
    #'              "sagemaker-{region}-{aws-account-id}". Example: "sagemaker-my-custom-bucket".
    initialize = function(paws_credentials = NULL,
                          bucket = NULL) {
      self$paws_credentials <- if(inherits(paws_credentials, "PawsCredentials")) paws_credentials else PawsCredentials$new()
      self$bucket <- bucket
      self$config <- NULL
      # get sagemaker object from paws
      self$sagemaker = paws::sagemaker(config = self$paws_credentials$credentials)

      self$local_mode = FALSE
    },

    #' @description Upload local file or directory to S3.If a single file is specified for upload, the resulting S3 object key is
    #'              ``{key_prefix}/{filename}`` (filename does not include the local path, if any specified).
    #'              If a directory is specified for upload, the API uploads all content, recursively,
    #'              preserving relative structure of subdirectories. The resulting object key names are:
    #'              ``{key_prefix}/{relative_subdirectory_path}/filename``.
    #' @param path (str): Path (absolute or relative) of local file or directory to upload.
    #' @param bucket (str): Name of the S3 Bucket to upload to (default: None). If not specified, the
    #'              default bucket of the ``Session`` is used (if default bucket does not exist, the
    #'              ``Session`` creates it).
    #' @param key_prefix (str): Optional S3 object key name prefix (default: 'data'). S3 uses the
    #'              prefix to create a directory structure for the bucket content that it display in
    #'              the S3 console.
    #' @param ... (any): Optional extra arguments that may be passed to the upload operation.
    #'              Similar to ExtraArgs parameter in S3 upload_file function. Please refer to the
    #'              ExtraArgs parameter documentation here:
    #'              https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html#the-extraargs-parameter
    #' @return str: The S3 URI of the uploaded file(s). If a file is specified in the path argument,
    #'              the URI format is: ``s3://{bucket name}/{key_prefix}/{original_file_name}``.
    #'              If a directory is specified in the path argument, the URI format is ``s3://{bucket name}/{key_prefix}``.
    upload_data = function(path, bucket = NULL, key_prefix = "data", ...){

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

    #' @description Upload a string as a file body.
    #' @param body (str): String representing the body of the file.
    #' @param bucket (str): Name of the S3 Bucket to upload to (default: None). If not specified, the
    #'              default bucket of the ``Session`` is used (if default bucket does not exist, the
    #'              ``Session`` creates it).
    #' @param key (str): S3 object key. This is the s3 path to the file.
    #' @param kms_key (str): The KMS key to use for encrypting the file.
    #' @return str: The S3 URI of the uploaded file.
    #'              The URI format is: ``s3://{bucket name}/{key}``.
    upload_string_as_file_body = function(body,
                                          bucket,
                                          key,
                                          kms_key=NULL){
      # Get s3 object from paws
      s3 <- paws::s3(config = self$paws_credentials$credentials)

      if (!is.null(kms_key))
        s3$put_object(Bucket = bucket, Body=charToRaw(body), SSEKMSKeyId=kms_key, ServerSideEncryption="aws:kms")
      else
        s3$put_object(Bucket = bucket, Body=charToRaw(body))

      s3_uri = sprintf("s3://%s/%s",bucket, key)
      return (s3_uri)
    },

    #' @description Download file or directory from S3.
    #' @param path (str): Local path where the file or directory should be downloaded to.
    #' @param bucket (str): Name of the S3 Bucket to download from.
    #' @param key_prefix (str): Optional S3 object key name prefix.
    #' @param ... (any): Optional extra arguments that may be passed to the
    #'              download operation. Please refer to the ExtraArgs parameter in the boto3
    #'              documentation here:
    #'              https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-example-download-file.html
    #' @return
    #' NULL invisibly
    download_data = function(path="", bucket, key_prefix = NULL, ...){

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
      return(invisible(NULL))
    },

    #' @description Read a single file from S3.
    #' @param bucket (str): Name of the S3 Bucket to download from.
    #' @param key_prefix (str): S3 object key name prefix.
    #' @return str: The body of the s3 file as a string.
    read_s3_file = function(bucket,
                            key_prefix){

      # Get s3 object from paws
      s3 = paws::s3(config = self$paws_credentials$credentials)

      # Explicitly passing a None kms_key to boto3 throws a validation error.
      s3_object = s3$get_object(Bucket=bucket, Key=key_prefix)

      return(rawToChar(s3_object$Body))
    },

    #' @description Lists the S3 files given an S3 bucket and key.
    #' @param bucket (str): Name of the S3 Bucket to download from.
    #' @param key_prefix (str): S3 object key name prefix.
    #' @return (str): The list of files at the S3 path.
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

    #' @description Return the name of the default bucket to use in relevant Amazon SageMaker interactions.
    #' @return (str): The name of the default bucket, which is of the form:
    #'  ``sagemaker-{region}-{AWS account ID}``.
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

    #' @description Create an Amazon SageMaker training job. Train the learner on a set of observations of the provided `task`.
    #'              Mutates the learner by reference, i.e. stores the model alongside other information in field `$state`.
    #'
    #' @param input_mode (str): The input mode that the algorithm supports. Valid modes:
    #'              \itemize{
    #'                \item{\strong{'File':} Amazon SageMaker copies the training dataset from the S3 location to
    #'                        a directory in the Docker container.}
    #'                \item{\strong{'Pipe':} Amazon SageMaker streams data directly from S3 to the container via a
    #'                        Unix-named pipe.}}
    #' @param input_config (list): A list of Channel objects. Each channel is a named input source.
    #'              Please refer to the format details described:
    #'              https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
    #'              jobs and APIs that create Amazon SageMaker endpoints use this role to access
    #'              training data and model artifacts. You must grant sufficient permissions to this
    #'              role.
    #' @param job_name (str): Name of the training job being created.
    #' @param output_config (dict): The S3 URI where you want to store the training results and
    #'              optional KMS key ID.
    #' @param resource_config (dict): Contains values for ResourceConfig:
    #'              \itemize{
    #'                \item{\strong{instance_count (int):} Number of EC2 instances to use for training.
    #'                              The key in resource_config is 'InstanceCount'.}
    #'                \item{\strong{instance_type (str):} Type of EC2 instance to use for training, for example,
    #'                              'ml.c4.xlarge'. The key in resource_config is 'InstanceType'.}}
    #' @param vpc_config (dict): Contains values for VpcConfig:
    #'              \itemize{
    #'                \item{\strong{subnets (list[str]):} List of subnet ids.
    #'                              The key in vpc_config is 'Subnets'.}
    #'                \item{\strong{security_group_ids (list[str]):} List of security group ids.
    #'                              The key in vpc_config is 'SecurityGroupIds'.}}
    #' @param hyperparameters (dict): Hyperparameters for model training. The hyperparameters are
    #'              made accessible as a dict[str, str] to the training code on SageMaker. For
    #'              convenience, this accepts other types for keys and values, but ``str()`` will be
    #'              called to convert them before training.
    #' @param stop_condition (dict): Defines when training shall finish. Contains entries that can
    #'              be understood by the service like ``MaxRuntimeInSeconds``.
    #' @param tags (list[dict]): List of tags for labeling a training job. For more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @param metric_definitions (list[dict]): A list of dictionaries that defines the metric(s)
    #'              used to evaluate the training jobs. Each dictionary contains two keys: 'Name' for
    #'              the name of the metric, and 'Regex' for the regular expression used to extract the
    #'              metric from the logs.
    #' @param enable_network_isolation (bool): Whether to request for the training job to run with
    #'              network isolation or not.
    #' @param image (str): Docker image containing training code.
    #' @param algorithm_arn (str): Algorithm Arn from Marketplace.
    #' @param encrypt_inter_container_traffic (bool): Specifies whether traffic between training
    #'              containers is encrypted for the training job (default: ``False``).
    #' @param train_use_spot_instances (bool): whether to use spot instances for training.
    #' @param checkpoint_s3_uri (str): The S3 URI in which to persist checkpoints
    #'              that the algorithm persists (if any) during training. (default: ``None``).
    #' @param checkpoint_local_path (str): The local path that the algorithm
    #'              writes its checkpoints to. SageMaker will persist all files
    #'              under this path to `checkpoint_s3_uri` continually during
    #'              training. On job startup the reverse happens - data from the
    #'              s3 location is downloaded to this path before the algorithm is
    #'              started. If the path is unset then SageMaker assumes the
    #'              checkpoints will be provided under `/opt/ml/checkpoints/`.
    #'              (Default: \code{NULL}).
    #' @param experiment_config (dict): Experiment management configuration. Dictionary contains
    #'              three optional keys, 'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
    #'              (Default: \code{NULL})
    #' @param debugger_rule_configs Configuration information for debugging rules
    #' @param debugger_hook_config Configuration information for debugging rules
    #' @param tensorboard_output_config Xonfiguration information for tensorboard output
    #' @param enable_sagemaker_metrics (bool): enable SageMaker Metrics Time
    #'              Series. For more information see:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_AlgorithmSpecification.html#SageMaker-Type-AlgorithmSpecification-EnableSageMakerMetricsTimeSeries
    #'              (Default: \code{NULL}).
    #'
    #' @return
    #' str: ARN of the training job, if it is created.
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
      train_request = list(
        AlgorithmSpecification = list(TrainingInputMode = input_mode),
        OutputDataConfig = output_config,
        TrainingJobName = job_name,
        StoppingCondition = stop_condition,
        ResourceConfig = resource_config,
        RoleArn = role)

      if(!is.null(image) && !is.null(algorithm_arn)) {
        stop("image and algorithm_arn are mutually exclusive.",
             sprintf("Both were provided: image: %s algorithm_arn: %s",image, algorithm_arn), call. = F)}

      if(is.null(image) && is.null(algorithm_arn)){
        stop("either image or algorithm_arn is required. None was provided.", call. = F)}

      train_request$InputDataConfig = input_config

      train_request$AlgorithmSpecification$TrainingImage = image
      train_request$AlgorithmSpecification$AlgorithmName = algorithm_arn
      train_request$AlgorithmSpecification$MetricDefinitions = metric_definitions
      train_request$AlgorithmSpecification$EnableSageMakerMetricsTimeSeries = enable_sagemaker_metrics

      train_request$HyperParameters = hyperparameters
      train_request$Tags = tags
      train_request$VpcConfig = vpc_config
      train_request$ExperimentConfig = experiment_config
      train_request$EnableNetworkIsolation = enable_network_isolation

      train_request$EnableInterContainerTrafficEncryption = encrypt_inter_container_traffic
      train_request$EnableManagedSpotTraining = train_use_spot_instances

      train_request$CheckpointConfig = NULL

      if (!is.null(checkpoint_s3_uri) || !is.null(checkpoint_local_path)) {
        checkpoint_config = list()
        checkpoint_config["S3Uri"] = checkpoint_s3_uri
        checkpoint_config["LocalPath"] = checkpoint_local_path
        train_request$CheckpointConfig = list(checkpoint_config)
      }

      train_request$DebugRuleConfigurations = debugger_rule_configs
      train_request$DebugHookConfig = debugger_hook_config

      train_request$TensorBoardOutputConfig = tensorboard_output_config

      log_info("Creating training-job with name: %s", job_name)
      log_debug("train request: %s", toJSON(train_request, pretty = T, auto_unbox = T))

      self$sagemaker$create_training_job(TrainingJobName = train_request$TrainingJobName,
                             HyperParameters = train_request$HyperParameters,
                             AlgorithmSpecification = train_request$AlgorithmSpecification,
                             RoleArn = train_request$RoleArn,
                             InputDataConfig = train_request$InputDataConfig,
                             OutputDataConfig = train_request$OutputDataConfig,
                             ResourceConfig = train_request$ResourceConfig,
                             VpcConfig = train_request$VpcConfig,
                             StoppingCondition = train_request$StoppingCondition,
                             Tags = train_request$Tags,
                             EnableNetworkIsolation = train_request$EnableNetworkIsolation,
                             EnableInterContainerTrafficEncryption = train_request$EnableInterContainerTrafficEncryption,
                             EnableManagedSpotTraining = train_request$EnableManagedSpotTraining,
                             CheckpointConfig = train_request$CheckpointConfig,
                             DebugHookConfig = train_request$DebugHookConfig,
                             DebugRuleConfigurations = train_request$DebugRuleConfigurations,
                             TensorBoardOutputConfig = train_request$TensorBoardOutputConfig,
                             ExperimentConfig = train_request$ExperimentConfig)
    },

    #' @description Create an Amazon SageMaker processing job.
    #' @param inputs ([dict]): List of up to 10 ProcessingInput dictionaries.
    #' @param output_config (dict): A config dictionary, which contains a list of up
    #'               to 10 ProcessingOutput dictionaries, as well as an optional KMS key ID.
    #' @param job_name (str): The name of the processing job. The name must be unique
    #'               within an AWS Region in an AWS account. Names should have minimum
    #'               length of 1 and maximum length of 63 characters.
    #' @param resources (dict): Encapsulates the resources, including ML instances
    #'               and storage, to use for the processing job.
    #' @param stopping_condition (dict[str,int]): Specifies a limit to how long
    #'               the processing job can run, in seconds.
    #' @param app_specification (dict[str,str]): Configures the processing job to
    #'               run the given image. Details are in the processing container
    #'               specification.
    #' @param environment (dict): Environment variables to start the processing
    #'               container with.
    #' @param network_config (dict): Specifies networking options, such as network
    #'               traffic encryption between processing containers, whether to allow
    #'               inbound and outbound network calls to and from processing containers,
    #'               and VPC subnets and security groups to use for VPC-enabled processing
    #'               jobs.
    #' @param role_arn (str): The Amazon Resource Name (ARN) of an IAM role that
    #'               Amazon SageMaker can assume to perform tasks on your behalf.
    #' @param tags ([dict[str,str]]): A list of dictionaries containing key-value
    #'               pairs.
    #' @param experiment_config (dict): Experiment management configuration. Dictionary contains
    #'               three optional keys, 'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
    #'               (Default: \code{NULL})
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


      process_request = list(
        ProcessingJobName = job_name,
        ProcessingResources = resources,
        AppSpecification = app_specification,
        RoleArn = role_arn)

      process_request$ProcessingInputs = inputs

      if(!is.null(output_config$Outputs)) process_request$ProcessingOutputConfig = output_config

      process_request$Environment = environment
      process_request$NetworkConfig = network_config
      process_request$StoppingCondition = stopping_condition
      process_request$Tags = tags
      process_request$ExperimentConfig = experiment_config

      log_info("Creating processing-job with name %s", job_name)
      log_debug("process request: %s", toJSON(process_request, pretty = T, auto_unbox = T))

      self$sagemaker$create_processing_job(ProcessingInputs = process_request$ProcessingInputs,
                               ProcessingOutputConfig = process_request$ProcessingOutputConfig,
                               ProcessingJobName = process_request$ProcessingJobName,
                               ProcessingResources = process_request$ProcessingResources,
                               StoppingCondition = process_request$StoppingCondition,
                               AppSpecification  = process_request$AppSpecification,
                               Environment = process_request$Environment,
                               NetworkConfig = process_request$NetworkConfig,
                               RoleArn = process_request$RoleArn,
                               Tags = process_request$Tags,
                               ExperimentConfig = process_request$ExperimentConfig)
    },

    #' @description Create an Amazon SageMaker monitoring schedule.
    #' @param monitoring_schedule_name (str): The name of the monitoring schedule. The name must be
    #'              unique within an AWS Region in an AWS account. Names should have a minimum length
    #'              of 1 and a maximum length of 63 characters.
    #' @param schedule_expression (str): The cron expression that dictates the monitoring execution
    #'              schedule.
    #' @param statistics_s3_uri (str): The S3 uri of the statistics file to use.
    #' @param constraints_s3_uri (str): The S3 uri of the constraints file to use.
    #' @param monitoring_inputs ([dict]): List of MonitoringInput dictionaries.
    #' @param monitoring_output_config (dict): A config dictionary, which contains a list of
    #'              MonitoringOutput dictionaries, as well as an optional KMS key ID.
    #' @param instance_count (int): The number of instances to run.
    #' @param instance_type (str): The type of instance to run.
    #' @param volume_size_in_gb (int): Size of the volume in GB.
    #' @param volume_kms_key (str): KMS key to use when encrypting the volume.
    #' @param image_uri (str): The image uri to use for monitoring executions.
    #' @param entrypoint (str): The entrypoint to the monitoring execution image.
    #' @param arguments (str): The arguments to pass to the monitoring execution image.
    #' @param record_preprocessor_source_uri (str or None): The S3 uri that points to the script that
    #'              pre-processes the dataset (only applicable to first-party images).
    #' @param post_analytics_processor_source_uri (str or None): The S3 uri that points to the script
    #'              that post-processes the dataset (only applicable to first-party images).
    #' @param max_runtime_in_seconds (int): Specifies a limit to how long
    #'              the processing job can run, in seconds.
    #' @param environment (dict): Environment variables to start the monitoring execution
    #'              container with.
    #' @param network_config (dict): Specifies networking options, such as network
    #'              traffic encryption between processing containers, whether to allow
    #'              inbound and outbound network calls to and from processing containers,
    #'              and VPC subnets and security groups to use for VPC-enabled processing
    #'              jobs.
    #' @param role_arn (str): The Amazon Resource Name (ARN) of an IAM role that
    #'              Amazon SageMaker can assume to perform tasks on your behalf.
    #' @param tags ([dict[str,str]]): A list of dictionaries containing key-value
    #'              pairs.
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

      monitoring_schedule_request = list(
        MonitoringScheduleName = monitoring_schedule_name,
        MonitoringScheduleConfig = list(MonitoringJobDefinition =
                                        list(MonitoringInputs = monitoring_inputs,
                                             RoleArn = role_arn,
                                             MonitoringAppSpecification = list(ImageUri = image_uri))))

      MonitoringResources = list(ClusterConfig = list(
        InstanceCount = instance_count,
        InstanceType = instance_type,
        VolumeSizeInGB = volume_size_in_gb))

      monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringResources = list(MonitoringResources)

      if(!is.null(schedule_expression)) monitoring_schedule_request$MonitoringScheduleConfig$ScheduleConfig = list(ScheduleExpression = schedule_expression)

      monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringOutputConfig = monitoring_output_config

      BaselineConfig = NULL
      if (!is.null(statistics_s3_uri) || !is.null(constraints_s3_uri)){
        BaselineConfig = list()
        if(!is.null(statistics_s3_uri)) BaselineConfig$StatisticsResource = list(S3Uri = statistics_s3_uri)
        if(!is.null(constraints_s3_uri)) BaselineConfig$ConstraintsResource = list(S3Uri = constraints_s3_uri)
      }

      monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$BaselineConfig = BaselineConfig
      monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$RecordPreprocessorSourceUri = record_preprocessor_source_uri
      monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$PostAnalyticsProcessorSourceUri = post_analytics_processor_source_uri
      monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$ContainerEntrypoint = entrypoint
      monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$ContainerArguments = arguments
      monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringResources$ClusterConfig$VolumeKmsKeyId = volume_kms_key

      if(!is.null(max_runtime_in_seconds))
        monitoring_schedule_request$MonitoringScheduleConfig$MonitoringScheduleConfig$MonitoringJobDefinition$StoppingCondition = list(MaxRuntimeInSeconds = max_runtime_in_seconds)

      monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$Environment = environment
      monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$NetworkConfig = network_config

      monitoring_schedule_request$Tags = tags

      log_info("Creating monitoring schedule name %s", monitoring_schedule_name)
      log_debug("monitoring_schedule_request= %s", toJSON(monitoring_schedule_request, pretty = T, auto_unbox = T))

      self$sagemaker$create_monitoring_schedule(MonitoringScheduleName = monitoring_schedule_request$MonitoringScheduleName,
                                    MonitoringScheduleConfig = monitoring_schedule_request$MonitoringScheduleConfig,
                                    Tags= monitoring_schedule_request$Tags)
    },

    #' @description Update an Amazon SageMaker monitoring schedule.
    #' @param monitoring_schedule_name (str): The name of the monitoring schedule. The name must be
    #'              unique within an AWS Region in an AWS account. Names should have a minimum length
    #'              of 1 and a maximum length of 63 characters.
    #' @param schedule_expression (str): The cron expression that dictates the monitoring execution
    #'              schedule.
    #' @param statistics_s3_uri (str): The S3 uri of the statistics file to use.
    #' @param constraints_s3_uri (str): The S3 uri of the constraints file to use.
    #' @param monitoring_inputs ([dict]): List of MonitoringInput dictionaries.
    #' @param monitoring_output_config (dict): A config dictionary, which contains a list of
    #'              MonitoringOutput dictionaries, as well as an optional KMS key ID.
    #' @param instance_count (int): The number of instances to run.
    #' @param instance_type (str): The type of instance to run.
    #' @param volume_size_in_gb (int): Size of the volume in GB.
    #' @param volume_kms_key (str): KMS key to use when encrypting the volume.
    #' @param image_uri (str): The image uri to use for monitoring executions.
    #' @param entrypoint (str): The entrypoint to the monitoring execution image.
    #' @param arguments (str): The arguments to pass to the monitoring execution image.
    #' @param record_preprocessor_source_uri (str or None): The S3 uri that points to the script that
    #' @param pre-processes the dataset (only applicable to first-party images).
    #' @param post_analytics_processor_source_uri (str or None): The S3 uri that points to the script
    #'              that post-processes the dataset (only applicable to first-party images).
    #' @param max_runtime_in_seconds (int): Specifies a limit to how long
    #'              the processing job can run, in seconds.
    #' @param environment (dict): Environment variables to start the monitoring execution
    #'              container with.
    #' @param network_config (dict): Specifies networking options, such as network
    #'             traffic encryption between processing containers, whether to allow
    #'             inbound and outbound network calls to and from processing containers,
    #'             and VPC subnets and security groups to use for VPC-enabled processing
    #'             jobs.
    #' @param role_arn (str): The Amazon Resource Name (ARN) of an IAM role that
    #'             Amazon SageMaker can assume to perform tasks on your behalf.
    #' @param tags ([dict[str,str]]): A list of dictionaries containing key-value
    #'             pairs.
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

      request_schedule_expression = schedule_expression %||% existing_schedule_config
      request_monitoring_inputs = monitoring_inputs %||% existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringInputs
      request_instance_count = instance_count %||% existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringResources$ClusterConfig$InstanceCount
      request_instance_type = instance_type %||% existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringResources$ClusterConfig$InstanceType
      request_volume_size_in_gb = volume_size_in_gb %||% existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringResources$ClusterConfig$VolumeSizeInGB
      request_image_uri = image_uri %||% existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$ImageUri
      request_role_arn = role_arn %||% existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$RoleArn

      monitoring_schedule_request = list(
        MonitoringScheduleName = monitoring_schedule_name,
        MonitoringScheduleConfig = list(MonitoringJobDefinition = list(
          MonitoringInputs = request_monitoring_inputs,
          MonitoringResources = list(
            ClusterConfig = list(
              InstanceCount = request_instance_count,
              InstanceType = request_instance_type,
              VolumeSizeInGB = request_volume_size_in_gb)),
          MonitoringAppSpecification = list(ImageUri = request_image_uri),
          RoleArn = request_role_arn)))

      if(!is.null(existing_schedule_config))
        monitoring_schedule_request$MonitoringScheduleConfig$ScheduleConfig$ScheduleExpression = list(ScheduleExpression = request_schedule_expression)

      existing_monitoring_output_config = existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringOutputConfig

      monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringOutputConfig = monitoring_output_config %||% existing_monitoring_output_config

      existing_statistics_s3_uri = NULL
      existing_constraints_s3_uri = NULL

      if(!is.null(existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$BaselineConfig)){

        if (!is.null(existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$BaselineConfig$StatisticsResource)){
          existing_statistics_s3_uri = existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$BaselineConfig$StatisticsResource$S3Uri}

        if (!is.null(existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$BaselineConfig$ConstraintsResource)){
          existing_constraints_s3_uri = existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$BaselineConfig$ConstraintsResource$S3Uri
        }


        if (!is.null(statistics_s3_uri)
            || !is.null(constraints_s3_uri)
            || !is.null(existing_statistics_s3_uri)
            || !is.null(existing_constraints_s3_uri)){
          monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$BaselineConfig = list()}

        if (!is.null(statistics_s3_uri) || !is.null(existing_statistics_s3_uri)){
          monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$BaselineConfig$StatisticsResource = list(S3Uri= statistics_s3_uri %||% existing_statistics_s3_uri)}

        if (!is.null(constraints_s3_uri) || !is.null(existing_constraints_s3_uri)){
          monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$BaselineConfig$ConstraintsResource = list(S3Uri= constraints_s3_uri %||% existing_constraints_s3_uri)}

        existing_record_preprocessor_source_uri = existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$RecordPreprocessorSourceUri

        monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$RecordPreprocessorSourceUri = record_preprocessor_source_uri %||% existing_record_preprocessor_source_uri

        existing_post_analytics_processor_source_uri = existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$PostAnalyticsProcessorSourceUri

        monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$PostAnalyticsProcessorSourceUri = post_analytics_processor_source_uri %||% existing_post_analytics_processor_source_uri

        existing_entrypoint = existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$ContainerEntrypoint

        monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$ContainerEntrypoint = entrypoint %||% existing_entrypoint

        existing_arguments = existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$ContainerArguments

        monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringAppSpecification$ContainerArguments = arguments %||% existing_arguments

        existing_volume_kms_key = existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringResources$ClusterConfig$VolumeKmsKeyId

        monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$MonitoringResources$ClusterConfig$VolumeKmsKeyId = volume_kms_key %||% existing_volume_kms_key

        existing_max_runtime_in_seconds = NULL
        if (!is.null(existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$StoppingCondition)){
          existing_max_runtime_in_seconds = existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$StoppingCondition$MaxRuntimeInSeconds}

        if (!is.null(max_runtime_in_seconds) || !is.null(existing_max_runtime_in_seconds)){
          monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$StoppingCondition = list(MaxRuntimeInSeconds= max_runtime_in_seconds %||% existing_max_runtime_in_seconds)}

          existing_environment = existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$Environment

          monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$Environment = environment %||% existing_environment

          existing_network_config = existing_desc$MonitoringScheduleConfig$MonitoringJobDefinition$NetworkConfig

          monitoring_schedule_request$MonitoringScheduleConfig$MonitoringJobDefinition$NetworkConfig = network_config %||% existing_network_config
      }

      log_info("Updating monitoring schedule with name: %s", monitoring_schedule_name)
      log_debug("monitoring_schedule_request= %s", toJSON(monitoring_schedule_request, pretty = T, auto_unbox = T))

      self$sagemaker$update_monitoring_schedule(monitoring_schedule_request$MonitoringScheduleName,
                                                monitoring_schedule_request$MonitoringScheduleConfig)
    },

    #' @description Starts a monitoring schedule.
    #' @param monitoring_schedule_name (str): The name of the Amazon SageMaker Monitoring
    #'              Schedule to start.
    start_monitoring_schedule = function(monitoring_schedule_name){
      message(sprintf("Starting Monitoring Schedule with name: %s",monitoring_schedule_name))

      self$sagemaker$start_monitoring_schedule(MonitoringScheduleName=monitoring_schedule_name)
    },

    #' @description Stops a monitoring schedule.
    #' @param monitoring_schedule_name (str): The name of the Amazon SageMaker Monitoring
    #'              Schedule to stop.
    stop_monitoring_schedule = function(monitoring_schedule_name){
      message(sprintf("Stopping Monitoring Schedule with name: %s",monitoring_schedule_name))
      self$sagemaker$stop_monitoring_schedule(MonitoringScheduleName=monitoring_schedule_name)
    },

    #' @description Deletes a monitoring schedule.
    #' @param monitoring_schedule_name (str): The name of the Amazon SageMaker Monitoring
    #'              Schedule to delete.
    delete_monitoring_schedule = function(monitoring_schedule_name){
      message(sprintf("Deleting Monitoring Schedule with name: %s",monitoring_schedule_name))
      self$sagemaker$delete_monitoring_schedule(MonitoringScheduleName=monitoring_schedule_name)
    },

    #' @description  Calls the DescribeMonitoringSchedule API for the given monitoring schedule name
    #'               and returns the response.
    #' @param monitoring_schedule_name (str): The name of the processing job to describe.
    #' @return dict: A dictionary response with the processing job description.
    describe_monitoring_schedule = function(monitoring_schedule_name){
      self$sagemaker$describe_monitoring_schedule(MonitoringScheduleName=monitoring_schedule_name)
    },

    #' @description Lists the monitoring executions associated with the given monitoring_schedule_name.
    #' @param monitoring_schedule_name (str): The monitoring_schedule_name for which to retrieve the
    #'              monitoring executions.
    #' @param sort_by (str): The field to sort by. Can be one of: "CreationTime", "ScheduledTime",
    #'              "Status". Default: "ScheduledTime".
    #' @param sort_order (str): The sort order. Can be one of: "Ascending", "Descending".
    #'               Default: "Descending".
    #' @param max_results (int): The maximum number of results to return. Must be between 1 and 100.
    #' @return dict: Dictionary of monitoring schedule executions.
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

    #' @description Lists the monitoring executions associated with the given monitoring_schedule_name.
    #' @param endpoint_name (str): The name of the endpoint to filter on. If not provided, does not
    #'              filter on it. Default: None.
    #' @param sort_by (str): The field to sort by. Can be one of: "Name", "CreationTime", "Status".
    #'              Default: "CreationTime".
    #' @param sort_order (str): The sort order. Can be one of: "Ascending", "Descending".
    #'              Default: "Descending".
    #' @param max_results (int): The maximum number of results to return. Must be between 1 and 100.
    #' @return dict: Dictionary of monitoring schedule executions.
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
          response = self$sagemaker$list_monitoring_schedules(
            SortBy=sort_by,
            SortOrder=sort_order,
            MaxResults=max_results)}

      return(response)
    },

    #' @description Calls the DescribeProcessingJob API for the given job name
    #'              and returns the True if the job was successful. False otherwise.
    #' @param job_name (str): The name of the processing job to describe.
    #' @return bool: Whether the processing job was successful.
    was_processing_job_successful = function(job_name){
      job_desc = self$sagemaker$describe_processing_job(ProcessingJobName=job_name)
      return(job_desc$ProcessingJobStatus == "Completed")
    },

    #' @description Calls the DescribeProcessingJob API for the given job name
    #'              and returns the response.
    #' @param job_name (str): The name of the processing job to describe.
    #' @return dict: A dictionary response with the processing job description.
    describe_processing_job = function(job_name){
      return(self$sagemaker$describe_processing_job(ProcessingJobName=job_name))
    },

    #' @description Calls the StopProcessingJob API for the given job name.
    #' @param job_name (str): The name of the processing job to stop.
    stop_processing_job = function(job_name){
      return(self$sagemaker$stop_processing_job(ProcessingJobName=job_name))
    },

    #' @description Calls the StopTrainingJob API for the given job name.
    #' @param job_name (str): The name of the training job to stop.
    stop_training_job = function(job_name){
      return(self$sagemaker$stop_training_job(TrainingJobName=job_name))
    },

    #' @description Calls the DescribeTrainingJob API for the given job name
    #'              and returns the response.
    #' @param job_name (str): The name of the training job to describe.
    #' @return dict: A dictionary response with the training job description.
    describe_training_job = function(job_name){
      return(self$sagemaker$describe_training_job(TrainingJobName=job_name))
    },

    #' @description Create an Amazon SageMaker AutoML job.
    #' @param input_config (list[dict]): A list of Channel objects. Each channel contains "DataSource"
    #'              and "TargetAttributeName", "CompressionType" is an optional field.
    #' @param output_config (dict): The S3 URI where you want to store the training results and
    #'              optional KMS key ID.
    #' @param auto_ml_job_config (dict): A dict of AutoMLJob config, containing "StoppingCondition",
    #'              "SecurityConfig", optionally contains "VolumeKmsKeyId".
    #' @param role (str): The Amazon Resource Name (ARN) of an IAM role that
    #'               Amazon SageMaker can assume to perform tasks on your behalf.
    #' @param job_name (str): A string that can be used to identify an AutoMLJob. Each AutoMLJob
    #'               should have a unique job name.
    #' @param problem_type (str): The type of problem of this AutoMLJob. Valid values are
    #'               "Regression", "BinaryClassification", "MultiClassClassification". If None,
    #'               SageMaker AutoMLJob will infer the problem type automatically.
    #' @param job_objective (dict): AutoMLJob objective, contains "AutoMLJobObjectiveType" (optional),
    #'               "MetricName" and "Value".
    #' @param generate_candidate_definitions_only (bool): Indicates whether to only generate candidate
    #'                definitions. If True, AutoML.list_candidates() cannot be called. Default: False.
    #' @param tags ([dict[str,str]]): A list of dictionaries containing key-value
    #'                pairs.
    #' @return NULL invisible
    auto_ml = function(input_config,
                       output_config,
                       auto_ml_job_config,
                       role,
                       job_name,
                       problem_type=NULL,
                       job_objective=NULL,
                       generate_candidate_definitions_only=FALSE,
                       tags=NULL){
      self$sagemaker$create_auto_ml_job(AutoMLJobName = job_name,
                                        InputDataConfig = input_config,
                                        OutputDataConfig = output_config,
                                        ProblemType = problem_type,
                                        AutoMLJobObjective = job_objective,
                                        AutoMLJobConfig = auto_ml_job_config,
                                        RoleArn = role,
                                        GenerateCandidateDefinitionsOnly = generate_candidate_definitions_only,
                                        Tags = tags)
      return(invisible(NULL))
    },

    #' @description Calls the DescribeAutoMLJob API for the given job name
    #'              and returns the response.
    #' @param job_name (str): The name of the AutoML job to describe.
    #' @return dict: A dictionary response with the AutoML Job description.
    describe_auto_ml_job = function(job_name){
      return(self$sagemaker$describe_auto_ml_job(AutoMLJobName=job_name))
    },

    #' @description Returns the list of candidates of an AutoML job for a given name.
    #' @param job_name (str): The name of the AutoML job. If None, will use object's
    #'              latest_auto_ml_job name.
    #' @param status_equals (str): Filter the result with candidate status, values could be
    #'              "Completed", "InProgress", "Failed", "Stopped", "Stopping"
    #' @param candidate_name (str): The name of a specified candidate to list.
    #'              Default to NULL
    #' @param candidate_arn (str): The Arn of a specified candidate to list.
    #'              Default to NULL.
    #' @param sort_order (str): The order that the candidates will be listed in result.
    #'              Default to NULL.
    #' @param sort_by (str): The value that the candidates will be sorted by.
    #'              Default to NULL.
    #' @param max_results (int): The number of candidates will be listed in results,
    #'              between 1 to 100. Default to None. If None, will return all the candidates.
    #' @return list: A list of dictionaries with candidates information
    list_candidates = function(job_name,
                               status_equals=NULL,
                               candidate_name=NULL,
                               candidate_arn=NULL,
                               sort_order=NULL,
                               sort_by=NULL,
                               max_results=NULL){

      return(self$sagemaker$list_candidates_for_auto_ml_job(AutoMLJobName = job_name,
                                                            StatusEquals = status_equals,
                                                            CandidateNameEquals = candidate_name,
                                                            SortOrder = sort_order,
                                                            SortBy = sort_by,
                                                            MaxResults = max_results))
    },

    #' @description Wait for an Amazon SageMaker AutoML job to complete.
    #' @param job (str): Name of the auto ml job to wait for.
    #' @param poll (int): Polling interval in seconds (default: 5).
    #' @return (dict): Return value from the ``DescribeAutoMLJob`` API.
    wait_for_auto_ml_job = function(job, poll=5){
      desc = private$.wait_until(private$.auto_ml_job_status(job), poll)
      private$.check_job_status(job, desc, "AutoMLJobStatus")
      return(desc)
    },

    #' @description Display the logs for a given AutoML job, optionally tailing them until the
    #'              job is complete. If the output is a tty or a Jupyter cell, it will be color-coded
    #'              based on which instance the log entry is from.
    #' @param job_name (str): Name of the Auto ML job to display the logs for.
    #' @param wait (bool): Whether to keep looking for new log entries until the job completes
    #'              (Default: FALSE).
    #' @param poll (int): The interval in seconds between polling for new log entries and job
    #'              completion (Default: 10).
    logs_for_auto_ml = function(job_name,
                                wait=False,
                                poll=10){

      description = self$sagemaker$describe_training_job(TrainingJobName=job_name)
      cloudwatchlogs = paws::cloudwatchlogs(config = self$paws_credentials$credentials)

      init_log = .log_init(description, "Processing")

      state = .get_initial_job_state(description, "AutoMLJobStatus", wait)

      last_describe_job_call = Sys.time()
      while(TRUE){
        .flush_log_streams(init_log$stream_names,
                           init_log$instance_count,
                           cloudwatchlogs,
                           init_log$log_group,
                           job_name,
                           sm_env$positions)

        if(state == LogState$COMPLETE) {break}

        Sys.sleep(poll)

        if(state == LogState$JOB_COMPLETE) {
          writeLines("\n")
          state = LogState$COMPLETE
        } else if(Sys.time() - last_describe_job_call >= 30){
          description = self.sagemaker_client.describe_auto_ml_job(AutoMLJobName=job_name)
          last_describe_job_call = Sys.time()

          status = description$ProcessingJobStatus

          if (status %in% c("Completed", "Failed", "Stopped")) state = LogState$JOB_COMPLETE
        }
      }

      if (wait) {
        private$.check_job_status(job_name, description, "AutoMLJobStatus")}
    },

    #' @description Create an Amazon SageMaker Neo compilation job.
    #' @param input_model_config (dict): the trained model and the Amazon S3 location where it is
    #'              stored.
    #' @param output_model_config (dict): Identifies the Amazon S3 location where you want Amazon
    #'              SageMaker Neo to save the results of compilation job
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker Neo
    #'              compilation jobs use this role to access model artifacts. You must grant
    #'              sufficient permissions to this role.
    #' @param job_name (str): Name of the compilation job being created.
    #' @param stop_condition (dict): Defines when compilation job shall finish. Contains entries
    #'              that can be understood by the service like ``MaxRuntimeInSeconds``.
    #' @param tags (list[dict]): List of tags for labeling a compile model job. For more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @return str: ARN of the compile model job, if it is created.
    compile_model = function(input_model_config,
                             output_model_config,
                             role,
                             job_name,
                             stop_condition,
                             tags){

        log_info("Creating compilation-job with name: %s", job_name)
        self$sagemaker$create_compilation_job(CompilationJobName = job_name,
                                              RoleArn = role,
                                              InputConfig = input_model_config,
                                              OutputConfig = output_model_config,
                                              StoppingCondition = stop_condition)
    },

    #' @description Create an Amazon SageMaker hyperparameter tuning job
    #' @param job_name (str): Name of the tuning job being created.
    #' @param strategy (str): Strategy to be used for hyperparameter estimations.
    #' @param objective_type (str): The type of the objective metric for evaluating training jobs.
    #'              This value can be either 'Minimize' or 'Maximize'.
    #' @param objective_metric_name (str): Name of the metric for evaluating training jobs.
    #' @param max_jobs (int): Maximum total number of training jobs to start for the hyperparameter
    #'              tuning job.
    #' @param max_parallel_jobs (int): Maximum number of parallel training jobs to start.
    #' @param parameter_ranges (dict): Dictionary of parameter ranges. These parameter ranges can be
    #'              one of three types: Continuous, Integer, or Categorical.
    #' @param static_hyperparameters (dict): Hyperparameters for model training. These
    #'              hyperparameters remain unchanged across all of the training jobs for the
    #'              hyperparameter tuning job. The hyperparameters are made accessible as a dictionary
    #'              for the training code on SageMaker.
    #' @param image (str): Docker image containing training code.
    #' @param algorithm_arn (str): Resource ARN for training algorithm created on or subscribed from
    #'              AWS Marketplace (Default: \code{NULL}).
    #' @param input_mode (str): The input mode that the algorithm supports. Valid modes:
    #'              \itemize{
    #'                \item{\strong{'File'} - Amazon SageMaker copies the training dataset from the S3 location to
    #'                      a directory in the Docker container.}
    #'                \item{\strong{'Pipe'} - Amazon SageMaker streams data directly from S3 to the container via a
    #'                      Unix-named pipe.}}
    #' @param metric_definitions (list[dict]): A list of dictionaries that defines the metric(s)
    #'              used to evaluate the training jobs. Each dictionary contains two keys: 'Name' for
    #'              the name of the metric, and 'Regex' for the regular expression used to extract the
    #'              metric from the logs. This should be defined only for jobs that don't use an
    #'              Amazon algorithm.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker
    #'              training jobs and APIs that create Amazon SageMaker endpoints use this role to
    #'              access training data and model artifacts. You must grant sufficient permissions
    #'              to this role.
    #' @param input_config (list): A list of Channel objects. Each channel is a named input source.
    #'              Please refer to the format details described:
    #'              https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job
    #' @param output_config (dict): The S3 URI where you want to store the training results and
    #'              optional KMS key ID.
    #' @param resource_config (dict): Contains values for ResourceConfig:
    #'              \itemize{
    #'                \item{\strong{instance_count (int):} Number of EC2 instances to use for training.
    #'                              The key in resource_config is 'InstanceCount'.}
    #'                \item{\strong{instance_type (str):} Type of EC2 instance to use for training, for example,
    #'                              'ml.c4.xlarge'. The key in resource_config is 'InstanceType'.}}
    #' @param stop_condition (dict): When training should finish, e.g. ``MaxRuntimeInSeconds``.
    #' @param tags (list[dict]): List of tags for labeling the tuning job. For more, see
    #'             https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @param warm_start_config (dict): Configuration defining the type of warm start and
    #'             other required configurations.
    #' @param early_stopping_type (str): Specifies whether early stopping is enabled for the job.
    #'             Can be either 'Auto' or 'Off'. If set to 'Off', early stopping will not be
    #'             attempted. If set to 'Auto', early stopping of some training jobs may happen, but
    #'             is not guaranteed to.
    #' @param enable_network_isolation (bool): Specifies whether to isolate the training container
    #'             (Default: \code{FALSE}).
    #' @param encrypt_inter_container_traffic (bool): Specifies whether traffic between training
    #'             containers is encrypted for the training jobs started for this hyperparameter
    #'             tuning job (Default: \code{FALSE}).
    #' @param vpc_config (dict): Contains values for VpcConfig (default: None):
    #'              \itemize{
    #'                \item{\strong{subnets (list[str]):} List of subnet ids.
    #'                              The key in vpc_config is 'Subnets'.}
    #'                \item{\strong{security_group_ids (list[str]):} List of security group ids.
    #'                              The key in vpc_config is 'SecurityGroupIds'.}}
    #' @param train_use_spot_instances (bool): whether to use spot instances for training.
    #' @param checkpoint_s3_uri (str): The S3 URI in which to persist checkpoints
    #'             that the algorithm persists (if any) during training. (Default: \code{FALSE}).
    #' @param checkpoint_local_path (str): The local path that the algorithm
    #'             writes its checkpoints to. SageMaker will persist all files
    #'             under this path to `checkpoint_s3_uri` continually during
    #'             training. On job startup the reverse happens - data from the
    #'             s3 location is downloaded to this path before the algorithm is
    #'             started. If the path is unset then SageMaker assumes the
    #'             checkpoints will be provided under `/opt/ml/checkpoints/`.
    #'             (Default: \code{NULL}).
    tune = function(job_name,
                    strategy = c("Bayesian", "Random"),
                    objective_type,
                    objective_metric_name,
                    max_jobs,
                    max_parallel_jobs,
                    parameter_ranges,
                    static_hyperparameters,
                    input_mode,
                    metric_definitions,
                    role,
                    input_config,
                    output_config,
                    resource_config,
                    stop_condition,
                    tags,
                    warm_start_config,
                    enable_network_isolation=FALSE,
                    image=NULL,
                    algorithm_arn=NULL,
                    early_stopping_type="Off",
                    encrypt_inter_container_traffic=FALSE,
                    vpc_config=NULL,
                    train_use_spot_instances=FALSE,
                    checkpoint_s3_uri=NULL,
                    checkpoint_local_path=NULL){

      strategy = match.arg(strategy)

      tune_request$HyperParameterTuningJobName = job_name
      tune_request$HyperParameterTuningJobConfig = private$.map_tuning_config(
        strategy=strategy,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        objective_type=objective_type,
        objective_metric_name=objective_metric_name,
        parameter_ranges=parameter_ranges,
        early_stopping_type=early_stopping_type)
      tune_request$TrainingJobDefinition = private$.map_training_config(
          static_hyperparameters=static_hyperparameters,
          role=role,
          input_mode=input_mode,
          image=image,
          algorithm_arn=algorithm_arn,
          metric_definitions=metric_definitions,
          input_config=input_config,
          output_config=output_config,
          resource_config=resource_config,
          vpc_config=vpc_config,
          stop_condition=stop_condition,
          enable_network_isolation=enable_network_isolation,
          encrypt_inter_container_traffic=encrypt_inter_container_traffic,
          train_use_spot_instances=train_use_spot_instances,
          checkpoint_s3_uri=checkpoint_s3_uri,
          checkpoint_local_path=checkpoint_local_path
        )

      tune_request$WarmStartConfig = warm_start_config
      tune_request$Tags = tags

      log_info("Creating hyperparameter tuning job with name: %s", job_name)
      log_debug("tune request: %s", toJSON(tune_request, pretty = T, auto_unbox = T))
      self$sagemaker$create_hyper_parameter_tuning_job(HyperParameterTuningJobName = tune_request$HyperParameterTuningJobName,
                                                       HyperParameterTuningJobConfig = tune_request$HyperParameterTuningJobConfig,
                                                       TrainingJobDefinition = tune_request$TrainingJobDefinition,
                                                       WarmStartConfig = tune_request$WarmStartConfig,
                                                       Tags = tune_request$Tags)
      },

    #' @description Create an Amazon SageMaker hyperparameter tuning job. This method supports creating
    #'              tuning jobs with single or multiple training algorithms (estimators), while the ``tune()``
    #'              method above only supports creating tuning jobs with single training algorithm.
    #' @param job_name (str): Name of the tuning job being created.
    #' @param tuning_config (dict): Configuration to launch the tuning job.
    #' @param training_config (dict): Configuration to launch training jobs under the tuning job
    #'              using a single algorithm.
    #' @param training_config_list (list[dict]): A list of configurations to launch training jobs
    #'              under the tuning job using one or multiple algorithms. Either training_config
    #'              or training_config_list should be provided, but not both.
    #' @param warm_start_config (dict): Configuration defining the type of warm start and
    #'              other required configurations.
    #' @param tags (list[dict]): List of tags for labeling the tuning job. For more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    create_tuning_job = function(job_name,
                                 tuning_config,
                                 training_config=NULL,
                                 training_config_list=NULL,
                                 warm_start_config=NULL,
                                 tags=NULL){
      if (is.null(training_config) && is.null(training_config_list)){
        stop("Either training_config or training_config_list should be provided.", call. = F)}
      if (!is.null(training_config) && !is.null(training_config_list)){
        stop("Only one of training_config and training_config_list should be provided.", call. = F)}

      tune_request = list(HyperParameterTuningJobName = job_name,
                          HyperParameterTuningJobConfig = do.call(private$.map_tuning_config, tuning_config))

      if(!is.null(training_config))
        tune_request$TrainingJobDefinition = do.call(private$.map_training_config, training_config)

      if (!is.null(training_config_list))
        tune_request$TrainingJobDefinitions= lapply(training_config_list, function(training_cfg) do.call(private$.map_training_config, training_cfg))

      tune_request$Tags = tags
      tune_request$WarmStartConfig = warm_start_config

      log_info("Creating hyperparameter tuning job with name: %s", job_name)
      log_debug("tune request: %s", toJSON(tune_request, pretty = T, auto_unbox = T))

      self$sagemaker$create_hyper_parameter_tuning_job(HyperParameterTuningJobName = tune_request$HyperParameterTuningJobName,
                                                       HyperParameterTuningJobConfig = tune_request$HyperParameterTuningJobConfig,
                                                       TrainingJobDefinition = tune_request$TrainingJobDefinition,
                                                       TrainingJobDefinitions = tune_request$TrainingJobDefinitions,
                                                       WarmStartConfig = tune_request$WarmStartConfig,
                                                       Tags = tune_request$Tags)
    },

    #' @description Stop the Amazon SageMaker hyperparameter tuning job with the specified name.
    #' @param name (str): Name of the Amazon SageMaker hyperparameter tuning job.
    stop_tuning_job = function(name){
      log_info("Stopping tuning job: %s", name)
      tryCatch(self$sagemaker$stop_hyper_parameter_tuning_job(HyperParameterTuningJobName=name),
               error = function(e) {
                 error_code = attributes(e)$error_response$`__type`
                 if(error_code == "ValidationException") {log_info("Tuning job: %s is alread stopped or not running.", name)
                 } else {log_error("Error occurred while attempting to stop tuning job: %s. Please try again.", name)}
                 stop(e$message, call. = F)
               })
    },

    #' @description Create an Amazon SageMaker transform job.
    #' @param job_name (str): Name of the transform job being created.
    #' @param model_name (str): Name of the SageMaker model being used for the transform job.
    #' @param strategy (str): The strategy used to decide how to batch records in a single request.
    #'              Possible values are 'MultiRecord' and 'SingleRecord'.
    #' @param max_concurrent_transforms (int): The maximum number of HTTP requests to be made to
    #'              each individual transform container at one time.
    #' @param max_payload (int): Maximum size of the payload in a single HTTP request to the
    #'              container in MB.
    #' @param env (dict): Environment variables to be set for use during the transform job.
    #' @param input_config (dict): A dictionary describing the input data (and its location) for the
    #'              job.
    #' @param output_config (dict): A dictionary describing the output location for the job.
    #' @param resource_config (dict): A dictionary describing the resources to complete the job.
    #' @param experiment_config (dict): A dictionary describing the experiment configuration for the
    #'              job. Dictionary contains three optional keys,
    #'              'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
    #' @param tags (list[dict]): List of tags for labeling a transform job.
    #' @param data_processing (dict): A dictionary describing config for combining the input data and
    #'              transformed data. For more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    transform = function(job_name = NULL,
                         model_name = NULL,
                         strategy = NULL,
                         max_concurrent_transforms = NULL,
                         max_payload = NULL,
                         env = NULL,
                         input_config = NULL,
                         output_config = NULL,
                         resource_config = NULL,
                         experiment_config = NULL,
                         tags = NULL,
                         data_processing = NULL){
      request_list = list(job_name, model_name, max_concurrent_transforms, max_payload,strategy,
                          env, input_config, output_config, resource_config, data_processing,
                          tags, experiment_config)
      log_info("Creating transform job with name: %s", job_name)
      log_debug("Transform request: %s", toJSON(request_list, pretty = T, auto_unbox = T))
      self$sagemaker$create_transform_job(TransformJobName = job_name,
                                          ModelName = model_name,
                                          MaxConcurrentTransforms = max_concurrent_transforms,
                                          MaxPayloadInMB = max_payload,
                                          BatchStrategy = strategy,
                                          Environment = env,
                                          TransformInput = input_config,
                                          TransformOutput = output_config,
                                          TransformResources = resource_config,
                                          DataProcessing = data_processing,
                                          Tags = tags,
                                          ExperimentConfig = experiment_config)
    },

    #' @description Create an Amazon SageMaker ``Model``.
    #'              Specify the S3 location of the model artifacts and Docker image containing
    #'              the inference code. Amazon SageMaker uses this information to deploy the
    #'              model in Amazon SageMaker. This method can also be used to create a Model for an Inference
    #'              Pipeline if you pass the list of container definitions through the containers parameter.
    #' @param name (str): Name of the Amazon SageMaker ``Model`` to create.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
    #'              jobs and APIs that create Amazon SageMaker endpoints use this role to access
    #'              training data and model artifacts. You must grant sufficient permissions to this
    #'              role.
    #' @param container_defs (list[dict[str, str]] or [dict[str, str]]): A single container
    #'              definition or a list of container definitions which will be invoked sequentially
    #'              while performing the prediction. If the list contains only one container, then
    #'              it'll be passed to SageMaker Hosting as the ``PrimaryContainer`` and otherwise,
    #'              it'll be passed as ``Containers``.You can also specify the  return value of
    #'              ``sagemaker.get_container_def()`` or ``sagemaker.pipeline_container_def()``,
    #'              which will used to create more advanced container configurations, including model
    #'              containers which need artifacts from S3.
    #' @param vpc_config (dict[str, list[str]]): The VpcConfig set on the model (default: None)
    #'              \itemize{
    #'                \item{\strong{'Subnets' (list[str]):} List of subnet ids.}
    #'                \item{\strong{'SecurityGroupIds' (list[str]):} List of security group ids.}}
    #' @param enable_network_isolation (bool): Wether the model requires network isolation or not.
    #' @param primary_container (str or dict[str, str]): Docker image which defines the inference
    #'              code. You can also specify the return value of ``sagemaker.container_def()``,
    #'              which is used to create more advanced container configurations, including model
    #'              containers which need artifacts from S3. This field is deprecated, please use
    #'              container_defs instead.
    #' @param tags (list[list[str, str]]): Optional. The list of tags to add to the model.
    #'              Example: \code{tags = list(list('Key'= 'tagname', 'Value'= 'tagvalue'))}
    #'              For more information about tags, see
    #'              https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags
    #' @return str: Name of the Amazon SageMaker ``Model`` created.
    create_model = function(name,
                            role,
                            container_defs = NULL,
                            vpc_config = NULL,
                            enable_network_isolation = FALSE,
                            primary_container = NULL,
                            tags = NULL){

      if(!is.null(container_defs) && !is.null(primary_container))
        stop("Both container_defs and primary_container can not be passed as input", call.= F)

      create_model_request = list(ModelName= name,
                                  ExecutionRoleArn = role)

      if (!is.null(primary_container)){

        msg = paste0("primary_container is going to be deprecated in a future release. Please use ",
                     "container_defs instead.")
        warning(msg)
        container_defs = primary_container
      }

      role = self$expand_role(role)

      if(inherits(container_def, "list"))
        create_model_request$Containers = container_def
      else
        create_model_request$PrimaryContainer = container_def

      create_model_request$Tags = tags
      create_model_request$VpcConfig = vpc_config
      create_model_request$EnableNetworkIsolation = enable_network_isolation

      log_info("Creating model with name: %s", name)
      log_debug("CreateModel request: %s", toJSON(create_model_request, pretty = T, auto_unbox = T))

      tryCatch({self$sagemaker$create_model(ModelName = create_model_request$ModelName,
                                            PrimaryContainer = create_model_request$PrimaryContainer,
                                            Containers = create_model_request$Containers,
                                            ExecutionRoleArn = create_model_request$ExecutionRoleArn,
                                            Tags = create_model_request$Tags,
                                            VpcConfig = create_model_request$VpcConfig,
                                            EnableNetworkIsolation = create_model_request$EnableNetworkIsolation)},
               error=function(e){
                 error_code = attributes(e)$error_response$`__type`
                 if (error_code == "ValidationException"
                   && grepl("Cannot create already existing model", e$message)){
                   log_warn("Using already existing model: %s", name)
                   } else {stop(e$message, call. = F)}
                 }
               )
      return(name)
    },


    #' @description Create an Amazon SageMaker ``Model`` from a SageMaker Training Job.
    #' @param training_job_name (str): The Amazon SageMaker Training Job name.
    #' @param name (str): The name of the SageMaker ``Model`` to create (default: None).
    #'              If not specified, the training job name is used.
    #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, specified either
    #'              by an IAM role name or role ARN. If None, the ``RoleArn`` from the SageMaker
    #'              Training Job will be used.
    #' @param primary_container_image (str): The Docker image reference (default: None). If None, it
    #'              defaults to the Training Image in ``training_job_name``.
    #' @param model_data_url (str): S3 location of the model data (default: None). If None, defaults
    #'              to the ``ModelS3Artifacts`` of ``training_job_name``.
    #' @param env (dict[string,string]): Model environment variables (default: {}).
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the
    #'              model. Default: use VpcConfig from training job.
    #'              \itemize{
    #'                \item{\strong{'Subnets' (list[str])} List of subnet ids.}
    #'                \item{\strong{'SecurityGroupIds' (list[str])} List of security group ids.}}
    #' @param tags (list[list[str, str]]): Optional. The list of tags to add to the model.
    #'              For more, see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @return str: The name of the created ``Model``.
    create_model_from_job = function(training_job_name,
                                     name=NULL,
                                     role=NULL,
                                     primary_container_image=NULL,
                                     model_data_url=NULL,
                                     env=NULL,
                                     vpc_config_override= "VPC_CONFIG_DEFAULT",
                                     tags=NULL){

      training_job = self$sagemaker$describe_training_job(TrainingJobName=training_job_name)

      name = name %||% training_job_name
      role = role %||% training_job$RoleArn
      primary_container = list(ContainerHostname = primary_container_image %||% training_job$AlgorithmSpecification$TrainingImage,
                               ModelDataUrl=model_data_url %||% training_job$ModelArtifacts$S3ModelArtifacts,
                               Environment = env %||% list())

      vpc_config = private$.vpc_config_from_training_job(training_job, vpc_config_override)

      return (self$create_model(name, role, primary_container,
                                vpc_config = vpc_config, tags=tags))
    },

    #' @description Create a SageMaker Model Package from the results of training with an Algorithm Package
    #' @param name (str): ModelPackage name
    #' @param description (str): Model Package description
    #' @param algorithm_arn (str): arn or name of the algorithm used for training.
    #' @param model_data (str): s3 URI to the model artifacts produced by training
    create_model_package_from_algorithm = function(name,
                                                   description = NULL,
                                                   algorithm_arn = NULL,
                                                   model_data = NULL){

      SourceAlgorithmSpecification = list(SourceAlgorithms = list(list(ModelDataUrl = model_data, AlgorithmName = algorithm_arn)))

      log_info("Creating model package with name: %s", name)

      tryCatch(
        self$sagemaker$create_model_package(ModelPackageName = name,
                                                   ModelPackageDescription = description,
                                                   SourceAlgorithmSpecification= SourceAlgorithmSpecification),
        error = function(e) {
          error_code = attributes(e)$error_response$`__type`
          if (error_code == "ValidationException"
              && grepl("ModelPackage already exists", e$message)) {
            log_warn("Using already existing model package: %s", name)
          } else {stop(e$message, call. = F)}
        })
    },

    #' @description Wait for an Amazon SageMaker endpoint deployment to complete.
    #' @param model_package_name (str): Name of the ``Endpoint`` to wait for.
    #' @param poll (int): Polling interval in seconds (default: 5).
    #' @return dict: Return value from the ``DescribeEndpoint`` API.
    wait_for_model_package = function(model_package_name,
                                      poll = 5){
      desc = private$.wait_until(private$.create_model_package_status(model_package_name), poll)

      status = desc$ModelPackageStatus

      if (status != "Completed"){
        reason = desc$FailureReason
        message = sprintf("Error creating model package %s: %s Reason: %s", model_package_name, status, reason)
        stop(message, call. = F)
      }
      return(desc)
    },

    #' @description Create an Amazon SageMaker endpoint configuration.
    #'              The endpoint configuration identifies the Amazon SageMaker model (created using the
    #'              ``CreateModel`` API) and the hardware configuration on which to deploy the model. Provide
    #'              this endpoint configuration to the ``CreateEndpoint`` API, which then launches the
    #'              hardware and deploys the model.
    #' @param name (str): Name of the Amazon SageMaker endpoint configuration to create.
    #' @param model_name (str): Name of the Amazon SageMaker ``Model``.
    #' @param initial_instance_count (int): Minimum number of EC2 instances to launch. The actual
    #'              number of active instances for an endpoint at any given time varies due to
    #'              autoscaling.
    #' @param instance_type (str): Type of EC2 instance to launch, for example, 'ml.c4.xlarge'.
    #' @param accelerator_type (str): Type of Elastic Inference accelerator to attach to the
    #'              instance. For example, 'ml.eia1.medium'.
    #'              For more information: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
    #' @param tags (list[list[str, str]]): Optional. The list of tags to add to the endpoint config.
    #' @param kms_key (str): The KMS key that is used to encrypt the data on the storage volume
    #'              attached to the instance hosting the endpoint.
    #' @param data_capture_config_dict (dict): Specifies configuration related to Endpoint data
    #'              capture for use with Amazon SageMaker Model Monitoring. Default: None.
    #'              Example: \code{tags = list(list('Key'= 'tagname', 'Value'= 'tagvalue'))}
    #'              For more information about tags, see
    #'              https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags
    #' @return str: Name of the endpoint point configuration created.
    create_endpoint_config = function(name,
                                      model_name,
                                      initial_instance_count,
                                      instance_type,
                                      accelerator_type=NULL,
                                      tags=NULL,
                                      kms_key=NULL,
                                      data_capture_config_dict=NULL){
      log_info("Creating endpoint-config with name %s", name)

      ProductionVariants = list(production_variant(
        model_name,
        instance_type,
        initial_instance_count,
        accelerator_type=accelerator_type))

      self$sagemaker$create_endpoint_config(EndpointConfigName = name,
                                            ProductionVariants = ProductionVariants,
                                            DataCaptureConfig = data_capture_config_dict,
                                            Tags = tags,
                                            KmsKeyId = kms_key)
      return(name)
    },

    #' @description Create an Amazon SageMaker endpoint configuration from an existing one. Updating any
    #'              values that were passed in.
    #'              The endpoint configuration identifies the Amazon SageMaker model (created using the
    #'              ``CreateModel`` API) and the hardware configuration on which to deploy the model. Provide
    #'              this endpoint configuration to the ``CreateEndpoint`` API, which then launches the
    #'              hardware and deploys the model.
    #' @param existing_config_name (str): Name of the existing Amazon SageMaker endpoint
    #'              configuration.
    #' @param new_config_name (str): Name of the Amazon SageMaker endpoint configuration to create.
    #' @param new_tags (List[list[str, str]]): Optional. The list of tags to add to the endpoint
    #'              config. If not specified, the tags of the existing endpoint configuration are used.
    #'              If any of the existing tags are reserved AWS ones (i.e. begin with "aws"),
    #'              they are not carried over to the new endpoint configuration.
    #' @param new_kms_key (str): The KMS key that is used to encrypt the data on the storage volume
    #'              attached to the instance hosting the endpoint (default: None). If not specified,
    #'              the KMS key of the existing endpoint configuration is used.
    #' @param new_data_capture_config_dict (dict): Specifies configuration related to Endpoint data
    #'              capture for use with Amazon SageMaker Model Monitoring (default: None).
    #'              If not specified, the data capture configuration of the existing
    #'              endpoint configuration is used.
    #' @return str: Name of the endpoint point configuration created.
    create_endpoint_config_from_existing = function(existing_config_name,
                                                    new_config_name,
                                                    new_tags=NULL,
                                                    new_kms_key=NULL,
                                                    new_data_capture_config_dict=NULL){

      log_info("Creating endpoint-config with name ", new_config_name)

      existing_endpoint_config_desc = self$sagemaker$describe_endpoint_config(
        EndpointConfigName=existing_config_name
      )

      request_tags = new_tags %||% self$list_tags(existing_endpoint_config_desc$EndpointConfigArn)

      return(self$sagemaker$create_endpoint_config(EndpointConfigName = new_config_name,
                                                   ProductionVariants = existing_endpoint_config_desc$ProductionVariants,
                                                   DataCaptureConfig = new_data_capture_config_dict %||% existing_endpoint_config_desc$DataCaptureConfig,
                                                   Tags = request_tags,
                                                   KmsKeyId = new_kms_key %||% existing_endpoint_config_desc.get("KmsKeyId")))

    },

    #' @description Create an Amazon SageMaker ``Endpoint`` according to the endpoint configuration
    #'              specified in the request.
    #'              Once the ``Endpoint`` is created, client applications can send requests to obtain
    #'              inferences. The endpoint configuration is created using the ``CreateEndpointConfig`` API.
    #' @param endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` being created.
    #' @param config_name (str): Name of the Amazon SageMaker endpoint configuration to deploy.
    #' @param tags (list[list[str, str]]): Optional. The list of tags to add to the endpoint config.
    #' @param wait (bool): Whether to wait for the endpoint deployment to complete before returning
    #'              (Default: \code{TRUE}).
    #' @return str: Name of the Amazon SageMaker ``Endpoint`` created.
    create_endpoint = function(endpoint_name,
                               config_name,
                               tags=NULL,
                               wait=TRUE){

      log_info("Creating endpoint with name %s", endpoint_name)

      self$sagemaker$create_endpoint(EndpointName=endpoint_name,
                                     EndpointConfigName=config_name,
                                     Tags=tags)
      if (wait) self$wait_for_endpoint(endpoint_name)
      return(endpoint_name)
    },

    #' @description Update an Amazon SageMaker ``Endpoint`` according to the endpoint configuration
    #'              specified in the request
    #' @param endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` being created.
    #' @param endpoint_config_name (str): Name of the Amazon SageMaker endpoint configuration to deploy.
    #' @param wait (bool): Whether to wait for the endpoint deployment to complete before returning
    #'              (Default: \code{TRUE}).
    #' @return str: Name of the Amazon SageMaker ``Endpoint`` being updated.
    update_endpoint = function(endpoint_name,
                               endpoint_config_name,
                               wait=TRUE){
      if (!.deployment_entity_exists(self$sagemaker$describe_endpoint(EndpointName=endpoint_name)))
        stop(sprintf("Endpoint with name '%s' does not exist; please use an existing endpoint name",
                     endpoint_name), call. = F)

      self$sagemaker$update_endpoint(EndpointName=endpoint_name,
                                     EndpointConfigName=endpoint_config_name)

      if (wait) self$wait_for_endpoint(endpoint_name)
      return(endpoint_name)
    },

    #' @description Delete an Amazon SageMaker ``Endpoint``.
    #' @param endpoint_name (str): Name of the Amazon SageMaker ``Endpoint`` to delete.
    delete_endpoint = function(endpoint_name){
      log_info("Deleting endpoint with name: %s", endpoint_name)
      self$sagemaker$delete_endpoint(EndpointName=endpoint_name)
    },
    #' @description Delete an Amazon SageMaker endpoint configuration.
    #' @param endpoint_config_name (str): Name of the Amazon SageMaker endpoint configuration to
    #'              delete.
    delete_endpoint_config = function(endpoint_config_name){
      log_info("Deleting endpoint configuration with name: %s", endpoint_config_name)
      self$sagemaker$delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    },

    #' @description Delete an Amazon SageMaker Model.
    #' @param model_name (str): Name of the Amazon SageMaker model to delete.
    delete_model= function(model_name){
      log_info("Deleting model with name: %s", model_name)
      self$sagemaker$delete_model(ModelName=model_name)
    },

    #' @description List the tags given an Amazon Resource Name
    #' @param resource_arn (str): The Amazon Resource Name (ARN) for which to get the tags list.
    #' @param max_results (int): The maximum number of results to include in a single page.
    #'              This method takes care of that abstraction and returns a full list.
    list_tags = function(resource_arn,
                         max_results=50){
      tags_list = list()

      tryCatch({
        list_tags_response = self$sagemaker$list_tags(ResourceArn=resource_arn, MaxResults=max_results)
        tags_list = c(tags_list, list_tags_response$Tags)

        next_token = list_tags_response$nextToken

        while(!is.null(next_token) || length(next_token) != 0){
          list_tags_response = self$sagemaker$list_tags(ResourceArn=resource_arn,
                                                        MaxResults=max_results,
                                                        NextToken=next_token)
          tags_list = c(tags_list, list_tags_response$Tags)
          next_token = list_tags_response$nextToken
        }
        non_aws_tags = list()
        for(tag in tag_list){
          if (!grepl("aws:", tag$Key)){
            non_aws_tags = c(non_aws_tags, tag)}
        }
        return(non_aws_tags)
      },
      error = function(e){
        log_error("Error retrieving tags. resource_arn: %s",resource_arn)
        return(e)
      })
    },

    #' @description Wait for an Amazon SageMaker training job to complete.
    #' @param job (str): Name of the training job to wait for.
    #' @param poll (int): Polling interval in seconds (default: 5).
    #' @return (dict): Return value from the ``DescribeTrainingJob`` API.
    wait_for_job = function(job, poll = 5){

      # make sure no previous job descriptions are picked up
      private$.last_job_desc = NULL

      desc = private$.wait_until_training_done(private$.train_done(job), poll)

      private$.check_job_status(job, desc, "TrainingJobStatus")

      # clean up last job description
      private$.last_job_desc = NULL

      return(desc)
    },

    #' @description Wait for an Amazon SageMaker Processing job to complete.
    #' @param job (str): Name of the processing job to wait for.
    #' @param poll (int): Polling interval in seconds (Default: 5).
    #' @return (dict): Return value from the ``DescribeProcessingJob`` API.
    wait_for_processing_job = function(job, poll=5){
      desc = private$.wait_until(private$compilation_job_status(job), poll)
      private$.check_job_status(job, desc, "CompilationJobStatus")
      return(desc)
    },

    #' @description Wait for an Amazon SageMaker Neo compilation job to complete.
    #' @param job (str): Name of the compilation job to wait for.
    #' @param poll (int): Polling interval in seconds (Default: 5).
    #' @return (dict): Return value from the ``DescribeCompilationJob`` API.
    wait_for_compilation_job = function(job,
                                        poll=5){
      desc = private$.wait_until(private$.compilation_job_status(job), poll)
      private$.check_job_status(job, desc, "CompilationJobStatus")
      return(desc)
    },

    #' @description Wait for an Amazon SageMaker hyperparameter tuning job to complete.
    #' @param job (str): Name of the tuning job to wait for.
    #' @param poll (int): Polling interval in seconds (default: 5).
    #' @return (dict): Return value from the ``DescribeHyperParameterTuningJob`` API.
    wait_for_tuning_job = function(job, poll=5){
      desc = private$.wait_until(private$.tuning_job_status(job), poll)
      private$.check_job_status(job, desc, "HyperParameterTuningJobStatus")
      return(desc)
    },

    #' @description Calls the DescribeTransformJob API for the given job name
    #'              and returns the response.
    #' @param job_name (str): The name of the transform job to describe.
    #' @return dict: A dictionary response with the transform job description.
    describe_transform_job = function(job_name){
      return (self$sagemaker$describe_transform_job(TransformJobName=job_name))
    },

    #' @description Wait for an Amazon SageMaker transform job to complete.
    #' @param job (str): Name of the transform job to wait for.
    #' @param poll (int): Polling interval in seconds (default: 5).
    #' @return (dict): Return value from the ``DescribeTransformJob`` API.
    wait_for_transform_job = function(job, poll = 5){
      desc = private$.wait_until(private$.transform_job_status(job), poll)
      private$.check_job_status(job, desc, "TransformJobStatus")
      return(desc)
    },

    #' @description Stop the Amazon SageMaker hyperparameter tuning job with the specified name.
    #' @param name (str): Name of the Amazon SageMaker batch transform job.
    stop_transform_job = function(name){
      log_info("Stopping transform job: %s", name)
      tryCatch(self$sagemaker$stop_transform_job(TransformJobName=name),
               error = function(e){
                 error_code = attributes(e)$error_response$`__type`
                 # allow to pass if the job already stopped
                 if (error_code == "ValidationException"){
                   log_info("Transform job: %s is already stopped or not running.", name)
                 } else{
                   log_error("Error occurred while attempting to stop transform job: %s", name)
                   stop(e$message, call. = F)}
               })
    },

    #' @description Wait for an Amazon SageMaker endpoint deployment to complete.
    #' @param endpoint (str): Name of the ``Endpoint`` to wait for.
    #' @param poll (int): Polling interval in seconds (Default: 30).
    #' @return dict: Return value from the ``DescribeEndpoint`` API.
    wait_for_endpoint = function(endpoint, poll=30){
      desc = private$.wait_until(private$.deploy_done(endpoint), poll)
      status = desc$EndpointStatus

      if(status != "InService"){
        reason = desc$FailureReason
        message = sprintf("Error hosting endpoint %s: %s. Reason: %s.", endpoint, status, reason)
        stop(message, call. = F)
      }
      return(desc)
    },

    #' @description Create an ``Endpoint`` using the results of a successful training job.
    #'              Specify the job name, Docker image containing the inference code, and hardware
    #'              configuration to deploy the model. Internally the API, creates an Amazon SageMaker model
    #'              (that describes the model artifacts and the Docker image containing inference code),
    #'              endpoint configuration (describing the hardware to deploy for hosting the model), and
    #'              creates an ``Endpoint`` (launches the EC2 instances and deploys the model on them). In
    #'              response, the API returns the endpoint name to which you can send requests for inferences.
    #' @param job_name (str): Name of the training job to deploy the results of.
    #' @param initial_instance_count (int): Minimum number of EC2 instances to launch. The actual
    #'              number of active instances for an endpoint at any given time varies due to
    #'              autoscaling.
    #' @param instance_type (str): Type of EC2 instance to deploy to an endpoint for prediction,
    #'              for example, 'ml.c4.xlarge'.
    #' @param deployment_image (str): The Docker image which defines the inference code to be used
    #'              as the entry point for accepting prediction requests. If not specified, uses the
    #'              image used for the training job.
    #' @param name (str): Name of the ``Endpoint`` to create. If not specified, uses the training job
    #'              name.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
    #'              jobs and APIs that create Amazon SageMaker endpoints use this role to access
    #'              training data and model artifacts. You must grant sufficient permissions to this
    #'              role.
    #' @param wait (bool): Whether to wait for the endpoint deployment to complete before returning
    #'              (Default: True).
    #' @param model_environment_vars (dict[str, str]): Environment variables to set on the model
    #'              container (Default: NULL).
    #' @param vpc_config_override (dict[str, list[str]]): Overrides VpcConfig set on the model.
    #'              Default: use VpcConfig from training job.
    #'              \itemize{
    #'                \item{\strong{'Subnets' (list[str]):} List of subnet ids.}
    #'                \item{\strong{'SecurityGroupIds' (list[str]):} List of security group ids.}}
    #' @param accelerator_type (str): Type of Elastic Inference accelerator to attach to the
    #'              instance. For example, 'ml.eia1.medium'.
    #'              For more information: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
    #' @param data_capture_config (DataCaptureConfig): Specifies
    #'              configuration related to Endpoint data capture for use with
    #'              Amazon SageMaker Model Monitoring. Default: None.
    #' @return str: Name of the ``Endpoint`` that is created.
    endpoint_from_job = function(job_name,
                                 initial_instance_count,
                                 instance_type,
                                 deployment_image=NULL,
                                 name=NULL,
                                 role=NULL,
                                 wait=True,
                                 model_environment_vars=NULL,
                                 vpc_config_override="VPC_CONFIG_DEFAULT",
                                 accelerator_type=NULL,
                                 data_capture_config=NULL){

      job_desc = self$sagemaker$describe_training_job(TrainingJobName=job_name)
      output_url = job_desc$ModelArtifacts$S3ModelArtifacts

      deployment_image = deployment_image %||% job_desc$AlgorithmSpecification$TrainingImage
      role = role %||% job_desc$RoleArn
      name = name %||% job_name
      vpc_config_override = if(vpc_config_override == "VPC_CONFIG_DEFAULT") job_desc$VpcConfig else vpc_sannitize(vpc_config_override)

      return (self$endpoint_from_model_data(
        model_s3_location=output_url,
        deployment_image=deployment_image,
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        name=name,
        role=role,
        wait=wait,
        model_environment_vars=model_environment_vars,
        model_vpc_config=vpc_config_override,
        accelerator_type=accelerator_type,
        data_capture_config=data_capture_config))

    },

    #' @description Create and deploy to an ``Endpoint`` using existing model data stored in S3.
    #' @param model_s3_location (str): S3 URI of the model artifacts to use for the endpoint.
    #' @param deployment_image (str): The Docker image which defines the runtime code to be used as
    #'              the entry point for accepting prediction requests.
    #' @param initial_instance_count (int): Minimum number of EC2 instances to launch. The actual
    #'              number of active instances for an endpoint at any given time varies due to
    #'              autoscaling.
    #' @param instance_type (str): Type of EC2 instance to deploy to an endpoint for prediction,
    #'              e.g. 'ml.c4.xlarge'.
    #' @param name (str): Name of the ``Endpoint`` to create. If not specified, uses a name
    #'              generated by combining the image name with a timestamp.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
    #'              jobs and APIs that create Amazon SageMaker endpoints use this role to access
    #'              training data and model artifacts.
    #'              You must grant sufficient permissions to this role.
    #' @param wait (bool): Whether to wait for the endpoint deployment to complete before returning
    #'              (Default: True).
    #' @param model_environment_vars (dict[str, str]): Environment variables to set on the model
    #'              container (Default: NULL).
    #' @param model_vpc_config (dict[str, list[str]]): The VpcConfig set on the model (default: None)
    #'              \itemize{
    #'                \item{\strong{'Subnets' (list[str]):} List of subnet ids.}
    #'                \item{\strong{'SecurityGroupIds' (list[str]):} List of security group ids.}}
    #' @param accelerator_type (str): Type of Elastic Inference accelerator to attach to the instance.
    #'              For example, 'ml.eia1.medium'.
    #'              For more information: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
    #' @param data_capture_config (DataCaptureConfig): Specifies
    #'              configuration related to Endpoint data capture for use with
    #'              Amazon SageMaker Model Monitoring. Default: None.
    #' @return str: Name of the ``Endpoint`` that is created.
    endpoint_from_model_data = function(model_s3_location,
                                        deployment_image,
                                        initial_instance_count,
                                        instance_type,
                                        name=NULL,
                                        role=NULL,
                                        wait=True,
                                        model_environment_vars=NULL,
                                        model_vpc_config=NULL,
                                        accelerator_type=NULL,
                                        data_capture_config=NULL){
      model_environment_vars = model_environment_vars %||% list()
      name = name %||% name_from_image(deployment_image)
      model_vpc_config = vpc_sannitize(model_vpc_config)

      if (.deployment_entity_exists(self$sagemaker$describe_endpoint(EndpointName=name))){
        stop(sprintf('Endpoint with name "%s" already exists; please pick a different name.',name), call. = F)
      }

      if (!.deployment_entity_exists(self$sagemaker$describe_model(ModelName=name))){
        primary_container = container_def(image=deployment_image, model_data_url=model_s3_location, env=model_environment_vars)
      self$create_model(name=name, role=role, container_defs=primary_container, vpc_config=model_vpc_config)}

      data_capture_config_list = NULL
      if(!is.null(data_capture_config))
        data_capture_config_list = data_capture_config$new()$to_request_list

      if (!.deployment_entity_exists(self$sagemaker$describe_endpoint_config(EndpointConfigName=name))){
        self$create_endpoint_config(
          name=name,
          model_name=name,
          initial_instance_count=initial_instance_count,
          instance_type=instance_type,
          accelerator_type=accelerator_type,
          data_capture_config_dict=data_capture_config_list)
        }

      self$create_endpoint(endpoint_name=name, config_name=name, wait=wait)
      return(name)
    },

    #' @description Create an SageMaker ``Endpoint`` from a list of production variants.
    #' @param name (str): The name of the ``Endpoint`` to create.
    #' @param production_variants (list[dict[str, str]]): The list of production variants to deploy.
    #' @param tags (list[dict[str, str]]): A list of key-value pairs for tagging the endpoint
    #'              (Default: None).
    #' @param kms_key (str): The KMS key that is used to encrypt the data on the storage volume
    #'              attached to the instance hosting the endpoint.
    #' @param wait (bool): Whether to wait for the endpoint deployment to complete before returning
    #'              (Default: True).
    #' @param data_capture_config_list (list): Specifies configuration related to Endpoint data
    #'              capture for use with Amazon SageMaker Model Monitoring. Default: None.
    #' @return str: The name of the created ``Endpoint``.
    endpoint_from_production_variants = function(name,
                                                 production_variants = NULL,
                                                 tags=NULL,
                                                 kms_key=NULL,
                                                 wait=TRUE,
                                                 data_capture_config_list=NULL){
      if (!.deployment_entity_exists(self$sagemaker$describe_endpoint_config(EndpointConfigName=name))){
      self$sagemaker$create_endpoint_config(EndpointConfigName = name,
                                            ProductionVariants = production_variants,
                                            DataCaptureConfig = data_capture_config_list,
                                            Tags = tags,
                                            KmsKeyId = kms_key)}

      return (self$create_endpoint(endpoint_name=name, config_name=name, tags=tags, wait=wait))
    },

    #' @description Expand an IAM role name into an ARN.
    #'              If the role is already in the form of an ARN, then the role is simply returned. Otherwise
    #'              we retrieve the full ARN and return it.
    #' @param role (str): An AWS IAM role (either name or full ARN).
    #' @return str: The corresponding AWS IAM role ARN.
    expand_role = function(role){
      iam = paws::iam(config = self$paws_credentials$credentials)
      if(grepl("/", role)) return(role)
      return(iam$get_role(RoleName = role_name)$Role$Arn)
    },

    #' @description  Returns the ARN user or role whose credentials are used to call the API.
    #' @return str: The ARN user or role
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
               error = function(e) log_warn("Couldn't call 'get_role' to get Role ARN from role name %s to get Role path.", role_name))

      return(role)
    },

    #' @description Display the logs for a given training job, optionally tailing them until the
    #'              job is complete. If the output is a tty or a Jupyter cell, it will be color-coded
    #'              based on which instance the log entry is from.
    #' @param job_name (str): Name of the training job to display the logs for.
    #' @param wait (bool): Whether to keep looking for new log entries until the job completes
    #'              (Default: False).
    #' @param poll (int): The interval in seconds between polling for new log entries and job
    #'              completion (Default: 10).
    #' @param log_type (str): Type of logs to return from building sagemaker process
    logs_for_job = function(job_name,
                            wait=FALSE,
                            poll=10,
                            log_type="All"){

      description = self$sagemaker$describe_training_job(TrainingJobName=job_name)
      cloudwatchlogs = paws::cloudwatchlogs(config = self$paws_credentials$credentials)
      writeLines(secondary_training_status_message(description, NULL), sep = "")

      init_log = .log_init(description, "Training")

      state = .get_initial_job_state(description, "TrainingJobStatus", wait)


      last_describe_job_call = Sys.time()
      last_description = description
      last_debug_rule_statuses = NULL

      while(TRUE){
        .flush_log_streams(init_log$stream_names,
                           init_log$instance_count,
                           cloudwatchlogs,
                           init_log$log_group,
                           job_name,
                           sm_env$positions)
        if(state == LogState$COMPLETE) {break}

        Sys.sleep(poll)

        if(state == LogState$JOB_COMPLETE) {
          writeLines("\n")
          state = LogState$COMPLETE
        } else if(Sys.time() - last_describe_job_call >= 30){
          description = self$sagemaker$describe_training_job(TrainingJobName=job_name)
          last_describe_job_call = Sys.time()

          if(secondary_training_status_changed(description, last_description)){
            writeLines(secondary_training_status_message(description, last_description), sep = "")
            last_description = description
          }

          status = description$TrainingJobStatus

          if (status %in% c("Completed", "Failed", "Stopped")) state = LogState$JOB_COMPLETE

          debug_rule_statuses = description$DebugRuleEvaluationStatuses
          if(!islistempty(debug_rule_statuses)
             && .debug_rule_statuses_changed(debug_rule_statuses, last_debug_rule_statuses)
             && (log_type %in% c("All", "Rules"))){
            writeLines("\n")
            writeLines("********* Debugger Rule Status *********")
            writeLines("*")
            for (status in debug_rule_statuses){
              rule_log = sprintf("* %+18s: %-18s",
                status$RuleConfigurationName, status$RuleEvaluationStatus)
              writeLines(rule_log)
            }
            writeLines("*")
            writeLines(paste0(rep("*", 40), collapse = ""))
            last_debug_rule_statuses = debug_rule_statuses
          }

        }

      }

      if (wait) {
        private$.check_job_status(job_name, description, "TrainingJobStatus")

        spot_training = description$EnableManagedSpotTraining

        training_time = description$TrainingTimeInSeconds
        billable_time = description$BillableTimeInSeconds
        if (!is.null(training_time) || legnth(training_time) == 0)
          writeLines(sprintf("Training seconds: %s", training_time * init_log$instance_count))
        if (!is.null(billable_time) || legnth(billable_time) == 0)
          writeLines(sprintf("Billable seconds: %s", billable_time * init_log$instance_count))
        if (!is.null(spot_training) || legnth(spot_training) == 0){
          saving = (1 - as.numeric(billable_time) / training_time) * 100
          writeLines(sprintf("Managed Spot Training savings: %s", saving))}
      }

    },

    #' @description Display the logs for a given processing job, optionally tailing them until the
    #'              job is complete.
    #' @param job_name (str): Name of the training job to display the logs for.
    #' @param wait (bool): Whether to keep looking for new log entries until the job completes
    #'              (Default: False).
    #' @param poll (int): The interval in seconds between polling for new log entries and job
    #'              completion (Default: 10).
    logs_for_processing_job = function(job_name,
                                       wait=FALSE,
                                       poll=10){

      description = self$sagemaker$describe_training_job(TrainingJobName=job_name)
      cloudwatchlogs = paws::cloudwatchlogs(config = self$paws_credentials$credentials)

      init_log = .log_init(description, "Processing")

      state = .get_initial_job_state(description, "ProcessingJobStatus", wait)

      last_describe_job_call = Sys.time()
      while(TRUE){
        .flush_log_streams(init_log$stream_names,
                           init_log$instance_count,
                           cloudwatchlogs,
                           init_log$log_group,
                           job_name,
                           sm_env$positions)

        if(state == LogState$COMPLETE) {break}

        Sys.sleep(poll)

        if(state == LogState$JOB_COMPLETE) {
          writeLines("\n")
          state = LogState$COMPLETE
        } else if(Sys.time() - last_describe_job_call >= 30){
          description = self$sagemaker$describe_training_job(TrainingJobName=job_name)
          last_describe_job_call = Sys.time()

          status = description$ProcessingJobStatus

          if (status %in% c("Completed", "Failed", "Stopped")) state = LogState$JOB_COMPLETE
        }
      }

      if (wait) {
        private$.check_job_status(job_name, description, "ProcessingJobStatus")}
    },

    #' @description Display the logs for a given transform job, optionally tailing them until the
    #'              job is complete. If the output is a tty or a Jupyter cell, it will be color-coded
    #'              based on which instance the log entry is from.
    #' @param job_name (str): Name of the transform job to display the logs for.
    #' @param wait (bool): Whether to keep looking for new log entries until the job completes
    #'              (Default: FALSE).
    #' @param poll (int): The interval in seconds between polling for new log entries and job
    #'              completion (Default: 10).
    logs_for_transform_job = function(job_name,
                                      wait=FALSE,
                                      poll=10){

      description = self$sagemaker$describe_transform_job(TransformJobName=job_name)
      cloudwatchlogs = paws::cloudwatchlogs(config = self$paws_credentials$credentials)

      init_log = .log_init(description, "Transform")

      state = .get_initial_job_state(description, "TransformJobStatus", wait)

      last_describe_job_call = Sys.time()
      while(TRUE){
        .flush_log_streams(init_log$stream_names,
                           init_log$instance_count,
                           cloudwatchlogs,
                           init_log$log_group,
                           job_name,
                           sm_env$positions)

        if(state == LogState$COMPLETE) {break}

        Sys.sleep(poll)

        if(state == LogState$JOB_COMPLETE) {
          writeLines("\n")
          state = LogState$COMPLETE
        } else if(Sys.time() - last_describe_job_call >= 30){
          description = self$sagemaker$describe_training_job(TrainingJobName=job_name)
          last_describe_job_call = Sys.time()

          status = description$TransformJobStatus

          if (status %in% c("Completed", "Failed", "Stopped")) state = LogState$JOB_COMPLETE
        }
      }

      if (wait) {
        private$.check_job_status(job_name, description, "TransformJobStatus")}
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      cat("<Session>")
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
       log_info("Created S3 bucket: %s", bucket_name)}

    },

    .map_tuning_config = function(strategy,
                                  max_jobs,
                                  max_parallel_jobs,
                                  early_stopping_type="Off",
                                  objective_type=NULL,
                                  objective_metric_name=NULL,
                                  parameter_ranges=NULL){
      tuning_config = list(Strategy = strategy,
                           ResourceLimits = list(
                             MaxNumberOfTrainingJobs = max_jobs,
                             MaxParallelTrainingJobs = max_parallel_jobs),
                           TrainingJobEarlyStoppingType =  early_stopping_type)

      # ----- bring .map_tuning_objective into function ------
      tuning_objective = NULL

      if (!is.null(objective_type) || !is.null(objective_metric_name)) {
        tuning_objective = list()
        tuning_objective$Type = objective_type
        tuning_objective$MetricName = objective_metric_name}
      # ------------------------------------------------------

      tuning_config$HyperParameterTuningJobObjective = tuning_objective

      tuning_config$ParameterRanges = parameter_ranges

      return(tuning_config)
    },

    .map_training_config = function(static_hyperparameters,
                                    input_mode,
                                    role,
                                    output_config,
                                    resource_config,
                                    stop_condition,
                                    input_config=NULL,
                                    metric_definitions=NULL,
                                    image=NULL,
                                    algorithm_arn=NULL,
                                    vpc_config=NULL,
                                    enable_network_isolation=FALSE,
                                    encrypt_inter_container_traffic=FALSE,
                                    estimator_name=NULL,
                                    objective_type=NULL,
                                    objective_metric_name=NULL,
                                    parameter_ranges=NULL,
                                    train_use_spot_instances=FALSE,
                                    checkpoint_s3_uri=NULL,
                                    checkpoint_local_path=NULL){

      training_job_definition = list(
        StaticHyperParameters = static_hyperparameters,
        RoleArn = role,
        OutputDataConfig = output_config,
        ResourceConfig = resource_config,
        StoppingCondition = stop_condition)


      algorithm_spec = list(TrainingInputMode = input_mode)

      algorithm_spec[["MetricDefinitions"]] = metric_definitions


      if (!is.null(algorithm_arn)) {
        algorithm_spec["AlgorithmName"] = algorithm_arn
        } else {
        algorithm_spec["TrainingImage"] = image}

      training_job_definition[["AlgorithmSpecification"]] = algorithm_spec

      training_job_definition[["InputDataConfig"]] = input_config

      training_job_definition[["VpcConfig"]] = vpc_config

      if (enable_network_isolation) training_job_definition["EnableNetworkIsolation"] = TRUE

      if (encrypt_inter_container_traffic) training_job_definition["EnableInterContainerTrafficEncryption"] = TRUE

      if (train_use_spot_instances)  training_job_definition["EnableManagedSpotTraining"] = TRUE


      if (!is.null(checkpoint_s3_uri)){
        checkpoint_config = list()
        checkpoint_config = list(S3Uri = checkpoint_s3_uri)
        if (!is.null(checkpoint_local_path)){
          checkpoint_config["LocalPath"] = checkpoint_local_path}
        training_job_definition[["CheckpointConfig"]] = checkpoint_config}

      training_job_definition["DefinitionName"] = estimator_name

      tuning_objective = NULL

      if (!is.null(objective_type) || !is.null(objective_metric_name)) {
        tuning_objective = list()
        tuning_objective$Type = objective_type
        tuning_objective$MetricName = objective_metric_name}

      training_job_definition$TuningObjective = tuning_objective

      training_job_definition$HyperParameterRanges = parameter_ranges

      return(training_job_definition)
    },

    .vpc_config_from_training_job = function(training_job_desc,
                                             vpc_config_override="VPC_CONFIG_DEFAULT"){

      if (vpc_config_override == "VPC_CONFIG_DEFAULT"){return(training_job_desc$VpcConfig)}
      return(vpc_sannitize(vpc_config_override))

    },

    .wait_until = function(expr,
                          poll = 5){

      result = eval.parent(substitute(expr))
      while((is.null(result) || length(result) == 0)){
        Sys.sleep(poll)
        result = eval.parent(substitute(expr))
      }
      return(result)
    },

    .auto_ml_job_status = function(job_name){

      auto_ml_job_status_codes = list(
        "Completed"= "!",
        "InProgress"= ".",
        "Failed"= "*",
        "Stopped"= "s",
        "Stopping"= "_")

      in_progress_statuses = c("InProgress", "Stopping")

      desc = self$sagemaker$describe_auto_ml_job(AutoMLJobName=job_name)
      status = desc$AutoMLJobStatus

      msg = auto_ml_job_status_codes[[status]]
      if(is.null(msg)) msg = "?"

      writeLines(msg, sep="")
      flush(stdout())

      if (status %in% in_progress_statuses) return(NULL)

      writeLines("\n")
      return(desc)

    },

    .create_model_package_status = function(model_package_name){
      in_progress_statuses = c("InProgress", "Pending")

      desc = self$sagemaker$describe_model_package(ModelPackageName=model_package_name)
      status = desc$ModelPackageStatus

      writeLines(".", sep = "")
      flush(stdout())

      if (status %in% in_progress_statuses) return(NULL)

      writeLines("\n")
      return(desc)
    },

    .wait_until_training_done = function(expr,
                                         poll = 5){
      result = eval.parent(substitute(expr))
      while(!result$status){
        Sys.sleep(poll)
        result = eval.parent(substitute(expr))
      }
      return(result$job_desc)
    },

    .train_done = function(job_name){
      in_progress_statuses = c("InProgress", "Created")

      # Get last job description
      last_desc = private$.last_job_desc

      desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
      status = desc$TrainingJobStatus

      if(secondary_training_status_changed(desc, last_desc)){
        writeLines("\n")
        writeLines(secondary_training_status_message(desc, last_desc), sep = "")
      } else {
        writeLines(".", sep = "")
      }

      flush(stdout())
      # update last job description
      private$.last_job_desc = desc

      if(status %in% in_progress_statuses){
        return(list(job_desc = desc, status = FALSE))
      }

      writeLines("\n")
      return(list(job_desc = desc, status = TRUE))
    },

    .check_job_status = function(job,
                                 desc,
                                 status_key_name){
      status = desc[[status_key_name]]
      # convert status to camel case
      status = .STATUS_CODE_TABLE[[toupper(status)]]

      if(!(status %in% c("Completed", "Stopped"))){
        reason = desc$FailureReason
        job_type = gsub("JobStatus", " job", status_key_name)
        message = sprintf("Error for %s %s: %s. Reason: %s", job_type, job, status, reason)
        stop(message, call. = F)
      }
    },
    # last job desc to help check if job has finished or not
    .last_job_desc = NULL,

    .compilation_job_status = function(job_name){
      compile_status_codes = list(
        "Completed"= "!",
        "InProgress"= ".",
        "Failed"= "*",
        "Stopped"= "s",
        "Stopping"= "_")
      in_progress_statuses = c("InProgress", "Stopping", "Starting")

      desc = self$sagemaker$describe_compilation_job(CompilationJobName=job_name)
      status = desc$CompilationJobStatus
      status = .STATUS_CODE_TABLE[[toupper(status)]]

      msg = compile_status_codes[[status]]
      if(is.null(msg)) msg = "?"

      writeLines(msg, sep = "")
      flush(stdout())

      if (status %in% in_progress_statuses) return(NULL)

      return(desc)
    },

    .tuning_job_status = function(job_name){
      tuning_status_codes = list(
        "Completed"= "!",
        "InProgress"= ".",
        "Failed"= "*",
        "Stopped"= "s",
        "Stopping"= "_")

      in_progress_statuses = c("InProgress", "Stopping")

      desc = self$sagemaker$describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=job_name)

      status = desc$HyperParameterTuningJobStatus

      msg = tuning_status_codes[[status]]
      if(is.null(msg)) msg = "?"

      writeLines(msg, sep = "")
      flush(stdout())

      if (status %in% in_progress_statuses) return(NULL)

      writeLines("\n")
      return(desc)
    },

    .transform_job_status = function(job_name){
      transform_job_status_codes = list(
        "Completed"= "!",
        "InProgress"= ".",
        "Failed"= "*",
        "Stopped"= "s",
        "Stopping"= "_")

      in_progress_statuses = c("InProgress", "Stopping")

      desc = sagemaker_client.describe_transform_job(TransformJobName=job_name)
      status = desc$TransformJobStatus

      msg = transform_job_status_codes[[status]]
      if(is.null(msg)) msg = "?"

      writeLines(msg, sep = "")
      flush(stdout())

      if (status %in% in_progress_statuses) return(NULL)

      writeLines("\n")
      return(desc)
    },

    .deploy_done = function(endpoint_name){
      hosting_status_codes = list(
        "OutOfService"= "x",
        "Creating"= "-",
        "Updating"= "-",
        "InService"= "!",
        "RollingBack"= "<",
        "Deleting"= "o",
        "Failed"= "*")
      in_progress_statuses = c("Creating", "Updating")

      desc = self$sagemaker$describe_endpoint(EndpointName=endpoint_name)
      status = desc$EndpointStatus

      msg = hosting_status_codes[[status]]
      if(is.null(msg)) msg = "?"

      writeLines(msg, sep = "")
      flush(stdout())

      if (status %in% in_progress_statuses) return(NULL)

      return (desc)
    }
  ),
  active = list(
    #' @field paws_region_name
    #' Returns aws region associated with Session
    paws_region_name = function() {self$paws_credentials$credentials$region}
  ),
  lock_objects = F
)

#' @title Create a definition for executing a container as part of a SageMaker model.
#' @param image (str): Docker image to run for this container.
#' @param model_data_url (str): S3 URI of data required by this container,
#'              e.g. SageMaker training job model artifacts (default: None).
#' @param env (dict[str, str]): Environment variables to set inside the container (default: None).
#' @param container_mode (str): The model container mode. Valid modes:
#'              \itemize{
#'                \item{\strong{MultiModel:} Indicates that model container can support hosting multiple models}
#'                \item{\strong{SingleModel:} Indicates that model container can support hosting a single model
#'                              This is the default model container mode when container_mode = None}}
#' @return dict[str, str]: A complete container definition object usable with the CreateModel API if
#'              passed via `PrimaryContainers` field.
#' @export
container_def <- function(image,
                          model_data_url=NULL,
                          env=NULL,
                          container_mode=NULL){
  if(is.null(env)) env = list()
  c_def = list("Image" = image, "Environment"= env)

  c_def$ModelDataUrl = model_data_url
  c_def$Mode = container_mode
  return(list(c_def))
}

#' @title Create a definition for executing a pipeline of containers as part of a SageMaker model.
#' @param models (list[sagemaker.Model]): this will be a list of ``sagemaker.Model`` objects in the
#'              order the inference should be invoked.
#' @param instance_type (str): The EC2 instance type to deploy this Model to. For example,
#'              'ml.p2.xlarge' (default: None).
#' @return list[dict[str, str]]: list of container definition objects usable with with the
#'              CreateModel API for inference pipelines if passed via `Containers` field.
#' @export
pipeline_container_def <- function(models, instance_type=NULL){
  c_defs = list()  # should contain list of container definitions in the same order customer passed
  for (model in models){
    c_defs = c(c_defs, model$prepare_container_def(instance_type))}
  return(c_defs)
}

production_variant <- function(model_name,
                               instance_type = c("ml.t2.medium","ml.t2.large","ml.t2.xlarge","ml.t2.2xlarge","ml.m4.xlarge","ml.m4.2xlarge","ml.m4.4xlarge","ml.m4.10xlarge","ml.m4.16xlarge","ml.m5.large","ml.m5.xlarge","ml.m5.2xlarge","ml.m5.4xlarge","ml.m5.12xlarge","ml.m5.24xlarge","ml.m5d.large","ml.m5d.xlarge","ml.m5d.2xlarge","ml.m5d.4xlarge","ml.m5d.12xlarge","ml.m5d.24xlarge","ml.c4.large","ml.c4.xlarge","ml.c4.2xlarge","ml.c4.4xlarge","ml.c4.8xlarge","ml.p2.xlarge","ml.p2.8xlarge","ml.p2.16xlarge","ml.p3.2xlarge","ml.p3.8xlarge","ml.p3.16xlarge","ml.c5.large","ml.c5.xlarge","ml.c5.2xlarge","ml.c5.4xlarge","ml.c5.9xlarge","ml.c5.18xlarge","ml.c5d.large","ml.c5d.xlarge","ml.c5d.2xlarge","ml.c5d.4xlarge","ml.c5d.9xlarge","ml.c5d.18xlarge","ml.g4dn.xlarge","ml.g4dn.2xlarge","ml.g4dn.4xlarge","ml.g4dn.8xlarge","ml.g4dn.12xlarge","ml.g4dn.16xlarge","ml.r5.large","ml.r5.xlarge","ml.r5.2xlarge","ml.r5.4xlarge","ml.r5.12xlarge","ml.r5.24xlarge","ml.r5d.large","ml.r5d.xlarge","ml.r5d.2xlarge","ml.r5d.4xlarge","ml.r5d.12xlarge","ml.r5d.24xlarge","ml.inf1.xlarge","ml.inf1.2xlarge","ml.inf1.6xlarge","ml.inf1.24xlarge"),
                               initial_instance_count=1,
                               variant_name="AllTraffic",
                               initial_weight=1,
                               accelerator_type=NULL){

  instance_type = match.arg(instance_type)

  production_variant_configuration = list(
    ModelName = model_name,
    InstanceType = instance_type,
    InitialInstanceCount  = initial_instance_count,
    VariantName = variant_name,
    InitialVariantWeight =  initial_weight)

  production_variant_configuration["AcceleratorType"] = accelerator_type

  return(production_variant_configuration)

}

.deployment_entity_exists <- function(describe_fn){
  tryCatch(eval.parent(substitute(expr)),
           error = function(e){
             error_code = attributes(e)$error_response$`__type`
             if(error_code != "ValidationException"
                && grepl("Could not find", e$message)) {
             stop(e$message, call. = F)}
           })
  return (FALSE)
}

#' @title Return the role ARN whose credentials are used to call the API.
#' @param sagemaker_session(Session): Current sagemaker session
#' @return (str): The role ARN
#' @export
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

.get_initial_job_state <- function(description, status_key, wait){
  status = description[[status_key]]
  job_already_completed = status %in% c("Completed", "Failed", "Stopped")
  return(if(wait && !job_already_completed) LogState$TAILING else LogState$COMPLETE)
}

# Checks the rule evaluation statuses for SageMaker Debugger rules.
.debug_rule_statuses_changed <- function(current_statuses,
                                         last_statuses){
  if (islistempty(last_statuses)) return(TRUE)

  if (current_statuses$RuleConfigurationName == last_statuses$RuleConfigurationName
    && (current_statuses$RuleEvaluationStatus != last_statuses$RuleEvaluationStatus))
    return(TRUE)
  return(FALSE)
}


.log_init <- function(description, job){
  switch(job,
         "Training" = {instance_count = description$ResourceConfig$InstanceCount},
         "Transform" = {instance_count = description$TransformResources$InstanceCount},
         "Processing" = {instance_count = description$ProcessingResources$ClusterConfig$InstanceCount},
         "AutoML" = {instance_count = 0})
  stream_names = list()
  log_group = sprintf("/aws/sagemaker/%sJobs",job)

  # reset position pkg environmental variable
  sm_env$positions = NULL
  return(list("stream_names" = stream_names, "log_group" = log_group, "instance_count"= instance_count))
}

.flush_log_streams <- function(stream_names,
                              instance_count,
                              cloudwatchlogs,
                              log_group,
                              job_name,
                              positions = sm_env$positions){

  if (length(stream_names) < instance_count){
    tryCatch({streams = cloudwatchlogs$describe_log_streams(
      logGroupName=log_group,
      logStreamNamePrefix=paste0(job_name, "/"),
      orderBy="LogStreamName",
      limit=min(instance_count, 50))},
      error = function(e){
        # On the very first training job run on an account, there's no log group until
        # the container starts logging, so ignore any errors thrown about that
        error_code = attributes(e)$error_response$`__type`
        if (error_code != "ResourceNotFoundException")
          stop(e$message, call. = F)
        })

    stream_names = lapply(streams$logStreams, function(s) s$logStreamName)

    while (length(streams$nextToken) > 0){
      tryCatch(streams = cloudwatchlogs$describe_log_streams(
        logGroupName=log_group,
        logStreamNamePrefix=paste0(job_name, "/"),
        orderBy="LogStreamName",
        limit=min(instance_count, 50)),
        error = function(e){
          # On the very first training job run on an account, there's no log group until
          # the container starts logging, so ignore any errors thrown about that
          error_code = attributes(e)$error_response$`__type`
          if (error_code != "ResourceNotFoundException")
            stop(e$message, call. = F)
        })
      stream_names = c(stream_names, lapply(streams$logStreams, function(s) s$logStreamName))
    }
  }

  if (length(stream_names) > 0) {
    events = multi_stream_iter(cloudwatchlogs, log_group, stream_names, positions)
    for (e in seq_along(events)){
      logs = lapply(events[[e]], function(l) l$message)
      # break if nothing exists in list
      if(islistempty(logs)) break
      writeLines(sagemaker_colour_wrapper(logs))
      count = length(events[[e]])
      if(events[[e]][[count]]$timestamp == sm_env$positions[[e]]$timestamp){
        sm_env$positions[[e]]$timestamp = events[[e]][[count]]$timestamp
        sm_env$positions[[e]]$skip = count + 1
      } else {
        sm_env$positions[[e]]$timestamp = events[[e]][[count]]$timestamp
        sm_env$positions[[e]]$skip = 1
      }
    }
  } else{
    writeLines(".", sep = "")
    flush(stdout())
  }
}

LogState <- list(STARTING = 1,
                WAIT_IN_PROGRESS = 2,
                TAILING = 3,
                JOB_COMPLETE = 4,
                COMPLETE = 5)

.STATUS_CODE_TABLE <- list(
  "COMPLETED"= "Completed",
  "INPROGRESS"= "InProgress",
  "FAILED"= "Failed",
  "STOPPED"= "Stopped",
  "STOPPING"= "Stopping",
  "STARTING"= "Starting")

NOTEBOOK_METADATA_FILE <- "/opt/ml/metadata/resource-metadata.json"
