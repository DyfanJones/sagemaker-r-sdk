# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/estimator.py

#' @include utils.R
#' @include fw_utils.R
#' @include model.R
#' @include session.R
#' @include vpc_utils.R
#' @include analytics.R
#' @include image_uris.R
#' @include job.R

#' @import paws
#' @import jsonlite
#' @import R6
#' @import logger
#' @import utils
#' @importFrom urltools url_parse
#' @import uuid

#' @title Handle end-to-end Amazon SageMaker training and deployment tasks.
#' @description For introduction to model training and deployment, see
#'              http://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html
#'              Subclasses must define a way to determine what image to use for training,
#'              what hyperparameters to use, and how to create an appropriate predictor
#'              instance.
#' @export
EstimatorBase = R6Class("EstimatorBase",
  public = list(

    #' @description Initialize an ``EstimatorBase`` instance.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if it needs to access an AWS resource.
    #' @param instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.c4.xlarge'.
    #' @param volume_size (int): Size in GB of the EBS volume to use for
    #'              storing input data during training (default: 30). Must be large
    #'              enough to store training data if File Mode is used (which is the
    #'              default).
    #' @param volume_kms_key (str): Optional. KMS key ID for encrypting EBS
    #'              volume attached to the training instance (default: NULL).
    #' @param max_run (int): Timeout in seconds for training (default: 24 *
    #'              60 * 60). After this amount of time Amazon SageMaker terminates
    #'              the job regardless of its current status.
    #' @param input_mode (str): The input mode that the algorithm supports
    #'              (default: 'File'). Valid modes: 'File' - Amazon SageMaker copies
    #'              the training dataset from the S3 location to a local directory.
    #'              'Pipe' - Amazon SageMaker streams data directly from S3 to the
    #'              container via a Unix-named pipe. This argument can be overriden
    #'              on a per-channel basis using
    #'              ``TrainingInput.input_mode``.
    #' @param output_path (str): S3 location for saving the training result (model
    #'              artifacts and output files). If not specified, results are
    #'              stored to a default bucket. If the bucket with the specific name
    #'              does not exist, the estimator creates the bucket during the
    #'              :meth:`~sagemaker.estimator.EstimatorBase.fit` method execution.
    #'              file:// urls are used for local mode. For example: 'file://model/'
    #'              will save to the model folder in the current directory.
    #' @param output_kms_key (str): Optional. KMS key ID for encrypting the
    #'              training output (default: NULL).
    #' @param base_job_name (str): Prefix for training job name when the
    #'              :meth:`~sagemaker.estimator.EstimatorBase.fit` method launches.
    #'              If not specified, the estimator generates a default job name,
    #'              based on the training image name and current timestamp.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    #' @param tags (list[dict]): List of tags for labeling a training job. For
    #'              more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @param subnets (list[str]): List of subnet ids. If not specified training
    #'              job will be created without VPC config.
    #' @param security_group_ids (list[str]): List of security group ids. If not
    #'              specified training job will be created without VPC config.
    #' @param model_uri (str): URI where a pre-trained model is stored, either
    #'              locally or in S3 (default: NULL). If specified, the estimator
    #'              will create a channel pointing to the model so the training job
    #'              can download it. This model can be a 'model.tar.gz' from a
    #'              previous training job, or other artifacts coming from a
    #'              different source.
    #'              In local mode, this should point to the path in which the model
    #'              is located and not the file itself, as local Docker containers
    #'              will try to mount the URI as a volume.
    #'              More information:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html#td-deserialization
    #' @param model_channel_name (str): Name of the channel where 'model_uri' will
    #'              be downloaded (default: 'model').
    #' @param metric_definitions (list[dict]): A list of dictionaries that defines
    #'              the metric(s) used to evaluate the training jobs. Each
    #'              dictionary contains two keys: 'Name' for the name of the metric,
    #'              and 'Regex' for the regular expression used to extract the
    #'              metric from the logs. This should be defined only for jobs that
    #'              don't use an Amazon algorithm.
    #' @param encrypt_inter_container_traffic (bool): Specifies whether traffic
    #'              between training containers is encrypted for the training job
    #'              (default: ``False``).
    #' @param use_spot_instances (bool): Specifies whether to use SageMaker
    #'              Managed Spot instances for training. If enabled then the
    #'              `max_wait` arg should also be set.
    #'              More information:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
    #'              (default: ``False``).
    #' @param max_wait (int): Timeout in seconds waiting for spot training
    #'              instances (default: NULL). After this amount of time Amazon
    #'              SageMaker will stop waiting for Spot instances to become
    #'              available (default: ``NULL``).
    #' @param checkpoint_s3_uri (str): The S3 URI in which to persist checkpoints
    #'              that the algorithm persists (if any) during training. (default:
    #'              ``NULL``).
    #' @param checkpoint_local_path (str): The local path that the algorithm
    #'              writes its checkpoints to. SageMaker will persist all files
    #'              under this path to `checkpoint_s3_uri` continually during
    #'              training. On job startup the reverse happens - data from the
    #'              s3 location is downloaded to this path before the algorithm is
    #'              started. If the path is unset then SageMaker assumes the
    #'              checkpoints will be provided under `/opt/ml/checkpoints/`.
    #'              (default: ``NULL``).
    #' @param rules (list[:class:`~sagemaker.debugger.Rule`]): A list of
    #'              :class:`~sagemaker.debugger.Rule` objects used to define
    #'              rules for continuous analysis with SageMaker Debugger
    #'              (default: ``NULL``). For more, see
    #'              https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html#continuous-analyses-through-rules
    #' @param debugger_hook_config (:class:`~sagemaker.debugger.DebuggerHookConfig` or bool):
    #'              Configuration for how debugging information is emitted with
    #'              SageMaker Debugger. If not specified, a default one is created using
    #'              the estimator's ``output_path``, unless the region does not
    #'              support SageMaker Debugger. To disable SageMaker Debugger,
    #'              set this parameter to ``False``. For more, see
    #'              https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html
    #' @param tensorboard_output_config (:class:`~sagemaker.debugger.TensorBoardOutputConfig`):
    #'              Configuration for customizing debugging visualization using TensorBoard
    #'              (default: ``NULL``). For more, see
    #'              https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html#capture-real-time-tensorboard-data-from-the-debugging-hook
    #' @param enable_sagemaker_metrics (bool): Enables SageMaker Metrics Time
    #'              Series. For more information see:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_AlgorithmSpecification.html#SageMaker-Type-AlgorithmSpecification-EnableSageMakerMetricsTimeSeries
    #'              (default: ``NULL``).
    #' @param enable_network_isolation (bool): Specifies whether container will
    #'              run in network isolation mode (default: ``False``). Network
    #'              isolation mode restricts the container access to outside networks
    #'              (such as the Internet). The container does not make any inbound or
    #'              outbound network calls. Also known as Internet-free mode.
    initialize = function(role,
                          instance_count,
                          instance_type,
                          volume_size = 30,
                          volume_kms_key = NULL,
                          max_run = 24 * 60 * 60,
                          input_mode = "File",
                          output_path = NULL,
                          output_kms_key = NULL,
                          base_job_name = NULL,
                          sagemaker_session = NULL,
                          tags = NULL,
                          subnets = NULL,
                          security_group_ids = NULL,
                          model_uri = NULL,
                          model_channel_name = "model",
                          metric_definitions = NULL,
                          encrypt_inter_container_traffic = FALSE,
                          use_spot_instances =FALSE,
                          max_wait = NULL,
                          checkpoint_s3_uri = NULL,
                          checkpoint_local_path = NULL,
                          rules = NULL,
                          debugger_hook_config = NULL,
                          tensorboard_output_config = NULL,
                          enable_sagemaker_metrics = NULL,
                          enable_network_isolation = FALSE) {
      self$role = role
      self$instance_count = instance_count
      self$instance_type = instance_type
      self$volume_size = volume_size
      self$volume_kms_key = volume_kms_key
      self$max_run = max_run
      self$input_mode = input_mode
      self$tags = tags
      self$metric_definitions = metric_definitions
      self$model_uri = model_uri
      self$model_channel_name = model_channel_name
      self$code_uri = NULL
      self$code_channel_name = "code"

      if (self$instance_type %in% c("local", "local_gpu")) {
        if (self$instance_type == "local_gpu" && self$instance_count > 1) stop("Distributed Training in Local GPU is not supported", call. = FALSE)
        stop("Currently don't support local sagemaker", call. = F)
        self$sagemaker_session = sagemaker_session #  LocalSession()
        if (!inherist(self$sagemaker_session, "Session")) stop("instance_type local or local_gpu is only supported with an instance of LocalSession", call. = FALSE)
      } else  {self$sagemaker_session = sagemaker_session %||% Session$new()}


      self$base_job_name = base_job_name
      self$.current_job_name = NULL

      if (inherits(self$sagemaker_session, "Session") # need to change this part to local mode :S
        && !is.null(output_path)
        && startsWith(output_path, "file://")) {
        stop("file:// output paths are only supported in Local Mode", call. = F)}

      self$output_path = output_path
      self$output_kms_key = output_kms_key
      self$latest_training_job = NULL
      self$jobs = list()
      self$deploy_instance_type = NULL

      self$.compiled_models = list()

      # VPC configurations
      self$subnets = subnets
      self$security_group_ids = security_group_ids

      self$encrypt_inter_container_traffic = encrypt_inter_container_traffic
      self$use_spot_instances = use_spot_instances
      self$max_wait = max_wait
      self$checkpoint_s3_uri = checkpoint_s3_uri
      self$checkpoint_local_path = checkpoint_local_path

      self$rules = rules
      self$debugger_hook_config = debugger_hook_config
      self$tensorboard_output_config = tensorboard_output_config

      self$debugger_rule_configs = NULL
      self$collection_configs = NULL

      self$enable_sagemaker_metrics = enable_sagemaker_metrics
      self$.enable_network_isolation = enable_network_isolation
    },

    #' @description Return the Docker image to use for training.
    #'              The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does
    #'              the model training, calls this method to find the image to use for model
    #'              training.
    #' @return str: The URI of the Docker image.
    training_image_uri = function() {stop("I'm an abstract interface method", call. = F)},

    #' @description Return the hyperparameters as a dictionary to use for training.
    #'              The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which
    #'              trains the model, calls this method to find the hyperparameters.
    #' @return dict[str, str]: The hyperparameters.
    hyperparameters = function() {stop("I'm an abstract interface method", call. = F)},

    #' @description Return True if this Estimator will need network isolation to run.
    #' @return bool: Whether this Estimator needs network isolation or not.
    enable_network_isolation = function() {
      return(self$.enable_network_isolation)
      },

    #' @description Calls _prepare_for_training. Used when setting up a workflow.
    #' @param job_name (str): Name of the training job to be created. If not
    #'              specified, one is generated, using the base name given to the
    #'              constructor if applicable.
    prepare_workflow_for_training = function(job_name = NULL){
      self$.prepare_for_training(job_name=job_name)
    },

    #' @description Gets the path to the DebuggerHookConfig output artifacts.
    #' @return str: An S3 path to the output artifacts.
    latest_job_debugger_artifacts_path = function(){
      private$.ensure_latest_training_job(
        error_message="Cannot get the Debugger artifacts path. The Estimator is not associated with a training job."
      )
      if (!is.null(self$debugger_hook_config))
        return(file.path(self$debugger_hook_config$s3_output_path,
                         self$latest_training_job.name,
                         "debug-output"))
      return(NULL)
    },

    #' @description Gets the path to the TensorBoardOutputConfig output artifacts.
    #' @return str: An S3 path to the output artifacts.
    latest_job_tensorboard_artifacts_path = function(){
      private$.ensure_latest_training_job(
        error_message= "Cannot get the TensorBoard artifacts path. The Estimator is not associated with a training job.")
      if (!is.null(self$debugger_hook_config))
        return(file.path(self$tensorboard_output_config$s3_output_path,
          self$latest_training_job$name,
          "tensorboard-output"))
      return(NULL)
    },

    #' @description Train a model using the input training dataset.
    #'              The API calls the Amazon SageMaker CreateTrainingJob API to start
    #'              model training. The API uses configuration you provided to create the
    #'              estimator and the specified input training data to send the
    #'              CreatingTrainingJob request to Amazon SageMaker.
    #'              This is a synchronous operation. After the model training
    #'              successfully completes, you can call the ``deploy()`` method to host the
    #'              model using the Amazon SageMaker hosting services.
    #' @param inputs (str or dict or TrainingInput): Information
    #'              about the training data. This can be one of three types:
    #'              \itemize{
    #'                \item{\strong{(str)} the S3 location where training data is saved, or a file:// path in
    #'              local mode.}
    #'                \item{\strong{(dict[str, str]} or dict[str, TrainingInput]) If using multiple
    #'              channels for training data, you can specify a dict mapping channel names to
    #'              strings or :func:`~TrainingInput` objects.}
    #'                \item{\strong{(TrainingInput)} - channel configuration for S3 data sources that can
    #'              provide additional information as well as the path to the training dataset.
    #'              See :func:`TrainingInput` for full details.}
    #'                \item{\strong{(sagemaker.session.FileSystemInput)} - channel configuration for
    #'              a file system data source that can provide additional information as well as
    #'              the path to the training dataset.}}
    #' @param wait (bool): Whether the call should wait until the job completes (default: True).
    #' @param logs ([str]): A list of strings specifying which logs to print. Acceptable
    #'              strings are "All", "NULL", "Training", or "Rules". To maintain backwards
    #'              compatibility, boolean values are also accepted and converted to strings.
    #'              Only meaningful when wait is True.
    #' @param job_name (str): Training job name. If not specified, the estimator generates
    #'              a default job name, based on the training image name and current timestamp.
    #' @param experiment_config (dict[str, str]): Experiment management configuration.
    #'              Dictionary contains three optional keys,
    #'              'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
    fit = function(inputs=NULL,
                   wait=TRUE,
                   logs = "All",
                   job_name=NULL,
                   experiment_config=NULL){

      self$.prepare_for_training(job_name=job_name)

      # return job only
      self$latest_training_job = gsub(".*/","", private$.start_new(inputs, experiment_config)$TrainingJobArn)

      self$jobs = c(self$jobs, self$latest_training_job)

      if (wait){
        self$wait(logs = logs)}
    },

    #' @description Wait for an Amazon SageMaker job to complete.
    #' @param logs ([str]): A list of strings specifying which logs to print. Acceptable
    #'              strings are "All", "NULL", "Training", or "Rules". To maintain backwards
    #'              compatibility, boolean values are also accepted and converted to strings.
    wait = function(logs = "All"){
      if(inherits(logs, "logical")) logs = ifelse(logs, "All", "None")

      if(logs != "None"){
          self$sagemaker_session$logs_for_job(job_name = self$latest_training_job, wait=TRUE, log_type=logs)
      } else {
          self$sagemaker_session$wait_for_job(job = self$latest_training_job)}
    },

    #' @description Compile a Neo model using the input model.
    #' @param target_instance_family (str): Identifies the device that you want to
    #'              run your model after compilation, for example: ml_c5. For allowed
    #'              strings see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
    #' @param input_shape (dict): Specifies the name and shape of the expected
    #'              inputs for your trained model in json dictionary form, for
    #'              example: {'data':[1,3,1024,1024]}, or {'var1': [1,1,28,28],
    #'              'var2':[1,1,28,28]}
    #' @param output_path (str): Specifies where to store the compiled model
    #' @param framework (str): The framework that is used to train the original
    #'              model. Allowed values: 'mxnet', 'tensorflow', 'keras', 'pytorch',
    #'              'onnx', 'xgboost'
    #' @param framework_version (str): The version of the framework
    #' @param compile_max_run (int): Timeout in seconds for compilation (default:
    #'              3 * 60). After this amount of time Amazon SageMaker Neo
    #'              terminates the compilation job regardless of its current status.
    #' @param tags (list[dict]): List of tags for labeling a compilation job. For
    #'              more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @param target_platform_os (str): Target Platform OS, for example: 'LINUX'.
    #'              For allowed strings see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
    #'              It can be used instead of target_instance_family.
    #' @param target_platform_arch (str): Target Platform Architecture, for example: 'X86_64'.
    #'              For allowed strings see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
    #'              It can be used instead of target_instance_family.
    #' @param target_platform_accelerator (str, optional): Target Platform Accelerator,
    #'              for example: 'NVIDIA'. For allowed strings see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
    #'              It can be used instead of target_instance_family.
    #' @param compiler_options (dict, optional): Additional parameters for compiler.
    #'              Compiler Options are TargetPlatform / target_instance_family specific. See
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html for details.
    #' @param ... : Passed to invocation of ``create_model()``.
    #'              Implementations may customize ``create_model()`` to accept
    #'              ``**kwargs`` to customize model creation during deploy. For
    #'              more, see the implementation docs.
    #' @return
    #'              sagemaker.model.Model: A SageMaker ``Model`` object. See
    #'              :func:`~sagemaker.model.Model` for full details.
    compile_model = function(target_instance_family,
                             input_shape,
                             output_path,
                             framework=NULL,
                             framework_version=NULL,
                             compile_max_run=15 * 60,
                             tags=NULL,
                             target_platform_os=NULL,
                             target_platform_arch=NULL,
                             target_platform_accelerator=NULL,
                             compiler_options=NULL,
                             ...){

      if (!islistempty(framework)
          && !(framework %in% NEO_ALLOWED_FRAMEWORKS)){
        stop(sprintf("Please use valid framework, allowed values: %s",
                     paste0(NEO_ALLOWED_FRAMEWORKS, collapse = ", ")),
             call. = F)}

      if (islistempty(framework)
        && islistempty(framework_version)){
        stop("You should provide framework and framework_version at the same time.",
             call.= F)}

      model = self$create_model(...)

      self$.compiled_models[["target_instance_family"]] = model$compile(
        target_instance_family,
        input_shape,
        output_path,
        self.role,
        tags,
        private$.compilation_job_name(),
        compile_max_run,
        framework=framework,
        framework_version=framework_version,
        target_platform_os=target_platform_os,
        target_platform_arch=target_platform_arch,
        target_platform_accelerator=target_platform_accelerator,
        compiler_options=compiler_options)

      return(self$.compiled_models$target_instance_family)
    },

    #' @description Attach to an existing training job.
    #'              Create an Estimator bound to an existing training job, each subclass
    #'              is responsible to implement
    #'              ``_prepare_init_params_from_job_description()`` as this method delegates
    #'              the actual conversion of a training job description to the arguments
    #'              that the class constructor expects. After attaching, if the training job
    #'              has a Complete status, it can be ``deploy()`` ed to create a SageMaker
    #'              Endpoint and return a ``Predictor``.
    #'              If the training job is in progress, attach will block and display log
    #'              messages from the training job, until the training job completes.
    #'              Examples:
    #'              >>> my_estimator.fit(wait=False)
    #'              >>> training_job_name = my_estimator.latest_training_job.name
    #'              Later on:
    #'              >>> attached_estimator = Estimator.attach(training_job_name)
    #'              >>> attached_estimator.deploy()
    #' @param training_job_name (str): The name of the training job to attach to.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    #' @param model_channel_name (str): Name of the channel where pre-trained
    #'              model data will be downloaded (default: 'model'). If no channel
    #'              with the same name exists in the training job, this option will
    #'              be ignored.
    #' @return Instance of the calling ``Estimator`` Class with the attached
    #'              training job.
    attach = function(training_job_name,
                      sagemaker_session=NULL,
                      model_channel_name="model"){

      sagemaker_session = sagemaker_session %||% Session$new()

      job_details = sagemaker_session$sagemaker$describe_training_job(
        TrainingJobName=training_job_name)

      init_params = private$.prepare_init_params_from_job_description(job_details, model_channel_name)
      tags = sagemaker_session$sagemaker$list_tags(ResourceArn=job_details$TrainingJobArn)$Tags
      init_params[["tags"]] = tags

      # clone current class
      estimator = self$clone()
      do.call(estimator$initialize, init_params)

      # update estimator class variables
      estimator$latest_training_job = init_params$base_job_name
      estimator$.current_job_name = estimator$latest_training_job

      estimator$wait(logs = "None")

      return(estimator)
    },

    #' @description Display the logs for Estimator's training job.
    #'              If the output is a tty or a Jupyter cell, it will be color-coded based
    #'              on which instance the log entry is from.
    logs = function(){
      self$sagemaker_session$logs_for_job(self$latest_training_job, wait=TRUE)
    },

    #' @description Deploy the trained model to an Amazon SageMaker endpoint and return a
    #'              ``sagemaker.Predictor`` object.
    #'              More information:
    #'              http://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html
    #' @param initial_instance_count (int): Minimum number of EC2 instances to
    #'              deploy to an endpoint for prediction.
    #' @param instance_type (str): Type of EC2 instance to deploy to an endpoint
    #'              for prediction, for example, 'ml.c4.xlarge'.
    #' @param accelerator_type (str): Type of Elastic Inference accelerator to
    #'              attach to an endpoint for model loading and inference, for
    #'              example, 'ml.eia1.medium'. If not specified, no Elastic
    #'              Inference accelerator will be attached to the endpoint. For more
    #'              information:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
    #' @param endpoint_name (str): Name to use for creating an Amazon SageMaker
    #'              endpoint. If not specified, the name of the training job is
    #'              used.
    #' @param use_compiled_model (bool): Flag to select whether to use compiled
    #'              (optimized) model. Default: False.
    #' @param update_endpoint (bool): Flag to update the model in an existing
    #'              Amazon SageMaker endpoint. If True, this will deploy a new
    #'              EndpointConfig to an already existing endpoint and delete
    #'              resources corresponding to the previous EndpointConfig. Default:
    #'              False
    #' @param wait (bool): Whether the call should wait until the deployment of
    #'              model completes (default: True).
    #' @param model_name (str): Name to use for creating an Amazon SageMaker
    #'              model. If not specified, the name of the training job is used.
    #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
    #'              data on the storage volume attached to the instance hosting the
    #'              endpoint.
    #' @param data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
    #'              configuration related to Endpoint data capture for use with
    #'              Amazon SageMaker Model Monitoring. Default: NULL.
    #' @param tags (List[dict[str, str]]): Optional. The list of tags to attach to this specific
    #'              endpoint. Example:
    #'              >>> tags = [{'Key': 'tagname', 'Value': 'tagvalue'}]
    #'              For more information about tags, see
    #'              https://boto3.amazonaws.com/v1/documentation\
    #'              /api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags
    #' @param ... : Passed to invocation of ``create_model()``.
    #'              Implementations may customize ``create_model()`` to accept
    #'              ``...`` to customize model creation during deploy.
    #'              For more, see the implementation docs.
    #' @return sagemaker.predictor.Predictor: A predictor that provides a ``predict()`` method,
    #'              which can be used to send requests to the Amazon SageMaker
    #'              endpoint and obtain inferences.
    deploy = function(initial_instance_count,
                      instance_type,
                      accelerator_type=NULL,
                      endpoint_name=NULL,
                      use_compiled_model=FALSE,
                      update_endpoint=FALSE,
                      wait=TRUE,
                      model_name=NULL,
                      kms_key=NULL,
                      data_capture_config=NULL,
                      tags=NULL,
                      ...){

      create_model_args = list(...)

      private$.ensure_latest_training_job()
      private$.ensure_base_job_name()
      default_name = name_from_base(self$base_job_name)
      endpoint_name = endpoint_name %||% default_name
      model_name = model_name %||% default_name

      self$deploy_instance_type = instance_type
      if (use_compiled_model){
        family = gsub("\\.", "_", instance_type)
        if (!(family %in% self$.compiled_models)){
          stop(sprintf("No compiled model for %s. ",family),
               "Please compile one with compile_model before deploying.", call. = F)
          }
        model = self$.compiled_models[[family]]
      } else{
        create_model_args$model_kms_key = self$output_kms_key
        model = do.call(self$create_model, create_model_args)
      }
      model$name = model_name

      return (model$deploy(
                  instance_type=instance_type,
                  initial_instance_count=initial_instance_count,
                  accelerator_type=accelerator_type,
                  endpoint_name=endpoint_name,
                  update_endpoint=update_endpoint,
                  tags=tags %||% self$tags,
                  wait=wait,
                  kms_key=kms_key,
                  data_capture_config=data_capture_config))
    },

    #' @description Create a SageMaker ``Model`` object that can be deployed to an
    #'              ``Endpoint``.
    #' @param ... : Keyword arguments used by the implemented method for
    #'              creating the ``Model``.
    #' @return sagemaker.model.Model: A SageMaker ``Model`` object. See
    #'              :func:`~sagemaker.model.Model` for full details.
    create_model = function(...) {stop("I'm an abstract interface method", call. = F)},

    #' @description Delete an Amazon SageMaker ``Endpoint``.
    delete_endpoint = function(){
      private$.ensure_latest_training_job(error_message="Endpoint was not created yet")
      self$sagemaker_session$delete_endpoint(self$latest_training_job)
    },

    #' @description Return a ``Transformer`` that uses a SageMaker Model based on the
    #'              training job. It reuses the SageMaker Session and base job name used by
    #'              the Estimator.
    #' @param instance_count (int): Number of EC2 instances to use.
    #' @param instance_type (str): Type of EC2 instance to use, for example,
    #'              'ml.c4.xlarge'.
    #' @param strategy (str): The strategy used to decide how to batch records in
    #'              a single request (default: NULL). Valid values: 'MultiRecord'
    #'              and 'SingleRecord'.
    #' @param assemble_with (str): How the output is assembled (default: NULL).
    #'              Valid values: 'Line' or 'NULL'.
    #' @param output_path (str): S3 location for saving the transform result. If
    #'              not specified, results are stored to a default bucket.
    #' @param output_kms_key (str): Optional. KMS key ID for encrypting the
    #'              transform output (default: NULL).
    #' @param accept (str): The accept header passed by the client to
    #'              the inference endpoint. If it is supported by the endpoint,
    #'              it will be the format of the batch transform output.
    #' @param env (dict): Environment variables to be set for use during the
    #'              transform job (default: NULL).
    #' @param max_concurrent_transforms (int): The maximum number of HTTP requests
    #'              to be made to each individual transform container at one time.
    #' @param max_payload (int): Maximum size of the payload in a single HTTP
    #'              request to the container in MB.
    #' @param tags (list[dict]): List of tags for labeling a transform job. If
    #'              NULL specified, then the tags used for the training job are used
    #'              for the transform job.
    #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
    #'              which is also used during transform jobs. If not specified, the
    #'              role from the Estimator will be used.
    #' @param volume_kms_key (str): Optional. KMS key ID for encrypting the volume
    #'              attached to the ML compute instance (default: NULL).
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for the
    #'              VpcConfig set on the model.
    #'              Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param enable_network_isolation (bool): Specifies whether container will
    #'              run in network isolation mode. Network isolation mode restricts
    #'              the container access to outside networks (such as the internet).
    #'              The container does not make any inbound or outbound network
    #'              calls. If True, a channel named "code" will be created for any
    #'              user entry script for inference. Also known as Internet-free mode.
    #'              If not specified, this setting is taken from the estimator's
    #'              current configuration.
    #' @param model_name (str): Name to use for creating an Amazon SageMaker
    #'              model. If not specified, the name of the training job is used.
    transformer = function(instance_count,
                           instance_type,
                           strategy=NULL,
                           assemble_with=NULL,
                           output_path=NULL,
                           output_kms_key=NULL,
                           accept=NULL,
                           env=NULL,
                           max_concurrent_transforms=NULL,
                           max_payload=NULL,
                           tags=NULL,
                           role=NULL,
                           volume_kms_key=NULL,
                           vpc_config_override="VPC_CONFIG_DEFAULT",
                           enable_network_isolation=NULL,
                           model_name=NULL){

      tags = tags %||% self$tags
      model_name = private$.get_or_create_name(model_name)

      if (is.null(self$latest_training_job)){
        log_warn(paste("No finished training job found associated with this estimator. Please make sure",
                "this estimator is only used for building workflow config"))
      } else {
        if (is.null(enable_network_isolation))
          enable_network_isolation = self$enable_network_isolation()

        model = self$create_model(
          vpc_config_override=vpc_config_override,
          model_kms_key=self.output_kms_key,
          enable_network_isolation=enable_network_isolation)

          # not all create_model() implementations have the same kwargs
          model$name = model_name
          if (!is.null(role))
            model$role = role

          model$create_sagemaker_model(instance_type, tags=tags)
          }

      return(Transformer$new(
        model_name,
        instance_count,
        instance_type,
        strategy=strategy,
        assemble_with=assemble_with,
        output_path=output_path,
        output_kms_key=output_kms_key,
        accept=accept,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        env=env,
        tags=tags,
        base_transform_job_name=self$base_job_name,
        volume_kms_key=volume_kms_key,
        sagemaker_session=self$sagemaker_session))
      },

    #' @description Returns VpcConfig dict either from this Estimator's subnets and
    #'              security groups, or else validate and return an optional override value.
    #' @param vpc_config_override :
    get_vpc_config = function(vpc_config_override="VPC_CONFIG_DEFAULT"){
      if (vpc_config_override == "VPC_CONFIG_DEFAULT"){
        return(vpc_to_list(self$subnets, self$security_group_ids))}
      return (vpc_sanitize(vpc_config_override))
    },

    #' @description Set any values in the estimator that need to be set before training.
    #' @param job_name (str): Name of the training job to be created. If not
    #'              specified, one is generated, using the base name given to the
    #'              constructor if applicable.
    .prepare_for_training = function(job_name = NULL) {
      self$.current_job_name = private$.get_or_create_name(job_name)

      # if output_path was specified we use it otherwise initialize here.
      # For Local Mode with local_code=True we don't need an explicit output_path
      if(is.null(self$output_path)) {
        local_code = get_config_value("local.local_code", self$sagemaker_session$config)
        if (self$sagemaker_session$local_mode && !is.null(local_code)) {
          self$output_path = ""
        } else {self$output_path = sprintf("s3://%s",self$sagemaker_session$default_bucket())}
      }

      # Prepare rules and debugger configs for training.
      if (!is.null(self$rules) && is.null(self$debugger_hook_config)) {
        self$debugger_hook_config = DebuggerHookConfig$new(s3_output_path=self$output_path)}
      # If an object was provided without an S3 URI is not provided, default it for the customer.
      if (!is.null(self$debugger_hook_config) && is.null(self$debugger_hook_config$s3_output_path))
        self$debugger_hook_config$s3_output_path = self$output_path
      private$.prepare_rules()
      private$.prepare_collection_configs()
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      return(print_class(self))
    }
    ),
  private = list(

    # Set ``self.base_job_name`` if it is not set already.
    .ensure_base_job_name = function(){
      # honor supplied base_job_name or generate it
      if (is.null(self$base_job_name))
        self$base_job_name = base_name_from_image(self$training_image_uri())
    },

    # Generate a name based on the base job name or training image if needed.
    # Args:
    #   name (str): User-supplied name. If not specified, a name is generated from
    # the base job name or training image.
    # Returns:
    #   str: Either the user-supplied name or a generated name.
    .get_or_create_name = function(name = NULL){
      if (!is.null(name))
        return(name)

      private$.ensure_base_job_name()
      return(name_from_base(self$base_job_name))
    },

    # Set any necessary values in debugger rules, if they are provided.
    .prepare_rules = function(){
      self$debugger_rule_configs = list()
      if (!is.null(self$rules)){
        # Iterate through each of the provided rules.
        for (rule in self$rules){
          # Set the image URI using the default rule evaluator image and the region.
          if (rule$image_uri == "DEFAULT_RULE_EVALUATOR_IMAGE"){
              rule$image_uri = get_rule_container_image_uri(self$sagemaker_session$paws_region())
              rule$instance_type = NULL
              rule$volume_size_in_gb = NULL
          }
          # If source was provided as a rule parameter, upload to S3 and save the S3 uri.
          if ("source_s3_uri" %in% rule$rule_parameters){
            parse_result = url_parse(rule$rule_parameters$source_s3_uri)
            if (parse_result$scheme != "s3"){
              desired_s3_uri = file.path(
                "s3://",
                self$sagemaker_session$default_bucket(),
                rule$name,
                UUIDgenerate())
              s3_uri = S3Uploader$new()$upload(
                local_path=rule$rule_parameters$source_s3_uri,
                desired_s3_uri=desired_s3_uri,
                session=self$sagemaker_session)
              rule$rule_parameters$source_s3_uri = s3_uri
            }
          }
        # Save the request dictionary for the rule.
        self$debugger_rule_configs = c(self$debugger_rule_configs, rule$to_debugger_rule_config_dict())
        }
      }
    },

    .prepare_collection_configs = function(){
      # Create a set to de-duplicate CollectionConfigs.
      self$collection_configs = list()
      # Iterate through the rules and add their respective CollectionConfigs to the set.
      if (!is.null(self$rules)) {
        for (rule in self$rules)
          self$collection_configs = c(self$collection_configs, rule$collection_configs)
      }
      # Add the CollectionConfigs from DebuggerHookConfig to the set.
      if (!islistempty(self$debugger_hook_config))
        self$collection_configs = c(
          self$collection_configs, self$debugger_hook_config$collection_configs %||% list()
        )
    },

    .ensure_latest_training_job = function(error_message = "Estimator is not associated with a training job"){
      if (is.null(self$latest_training_job))
        stop(error_message, call. =F)
    },


    # ------------------------ incorporate _TrainingJob calls -------------------

    # Create a new Amazon SageMaker training job from the estimator.
    # Args:
    #   estimator (sagemaker.estimator.EstimatorBase): Estimator object
    # created by the user.
    # inputs (str): Parameters used when called
    # :meth:`~sagemaker.estimator.EstimatorBase.fit`.
    # experiment_config (dict[str, str]): Experiment management configuration used when called
    # :meth:`~sagemaker.estimator.EstimatorBase.fit`.  Dictionary contains
    # three optional keys, 'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
    # Returns:
    #   sagemaker.estimator._TrainingJob: Constructed object that captures
    # all information about the started training job.
    .start_new = function(inputs,
                          experiment_config = NULL){
      train_args= private$.get_train_args(inputs, experiment_config)

      do.call(self$sagemaker_session$train, train_args)
    },

    # Constructs a dict of arguments for an Amazon SageMaker training job from the estimator.
    # Args:
    #  estimator (sagemaker.estimator.EstimatorBase): Estimator object
    # created by the user.
    # inputs (str): Parameters used when called
    # :meth:`~sagemaker.estimator.EstimatorBase.fit`.
    # experiment_config (dict[str, str]): Experiment management configuration used when called
    # :meth:`~sagemaker.estimator.EstimatorBase.fit`.  Dictionary contains
    # three optional keys, 'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
    # Returns:
    #  Dict: dict for `sagemaker.session.Session.train` method
    .get_train_args = function(inputs,
                               experiment_config) {
      local_mode = self$sagemaker_session$local_mode
      model_uri = self$model_uri

      # Allow file:// input only in local mode
      if (private$.is_local_channel(inputs) || private$.is_local_channel(model_uri)){
        if (!local_mode) stop("File URIs are supported in local mode only. Please use a S3 URI instead.", call. = F)
      }

      config = .Job$new()$.__enclos_env__$private$.load_config(inputs, self)

      if (!islistempty(self$hyperparameters())){
        hyperparameters = self$hyperparameters()}

      train_args = config
      train_args[["input_mode"]] = self$input_mode
      train_args[["job_name"]] = self$.current_job_name
      train_args[["hyperparameters"]] = hyperparameters
      train_args[["tags"]] = self$tags
      train_args[["metric_definitions"]] = self$metric_definitions
      train_args[["experiment_config"]] = experiment_config

      if (inherits(inputs, c("TrainingInputs"))){
        if ("InputMode" %in% inputs$config){
          log_debug("Selecting TrainingInput's input_mode (%s) for TrainingInputMode.",
                    inputs$config$InputMode)
          train_args[["input_mode"]] = inputs$config$InputMod}
      }

      if (self$enable_network_isolation()){
        train_args[["enable_network_isolation"]] = TRUE}

      if (self$encrypt_inter_container_traffic){
        train_args[["encrypt_inter_container_traffic"]] = TRUE}

      if (inherits(self, "AlgorithmEstimator")){
        train_args[["algorithm_arn"]] = self$algorithm_arn
      } else {
        train_args[["image"]] = self$training_image_uri()}

      if (!islistempty(self$debugger_rule_configs))
        train_args[["debugger_rule_configs"]] = self$debugger_rule_configs

      if (!islistempty(self$debugger_hook_config)){
        self$debugger_hook_config[["collection_configs"]] = self$collection_configs
        train_args[["debugger_hook_config"]] = self$debugger_hook_config$to_request_list()}

      if (!islistempty(self$tensorboard_output_config))
        train_args[["tensorboard_output_config"]] = self$tensorboard_output_config$to_request_list()

      train_args = c(train_args, private$.add_spot_checkpoint_args(local_mode, train_args))

      if (!islistempty(self$enable_sagemaker_metrics))
        train_args[["enable_sagemaker_metrics"]] = self$enable_sagemaker_metrics

      return(train_args)
    },

    .add_spot_checkpoint_args = function(local_mode,
                                         train_args){
      train_args = list()
      if (self$use_spot_instances){
        if (local_mode){
          stop("Spot training is not supported in local mode.", call. = F)}
        train_args[["use_spot_instances"]] = TRUE
      }

      if (!islistempty(self$checkpoint_s3_uri)){
        if (local_mode){
          stop("Setting checkpoint_s3_uri is not supported in local mode.", call. = F)}
        train_args[["checkpoint_s3_uri"]] = self$checkpoint_s3_uri
      }

      if (!islistempty(self$checkpoint_local_path)){
        if (local_mode){
        stop("Setting checkpoint_local_path is not supported in local mode.", call. = F)}
        train_args[["checkpoint_local_path"]] = self$checkpoint_local_path
      }
      return(train_args)
    },

    .is_local_channel = function(input_uri){
      return(inherits(input_uri, "character") && startsWith(input_uri,"file://"))
    },

    # ---------------------------------------------------------------------------
    .compilation_job_name = function(){
      base_name = self$base_job_name %||% base_name_from_image(self$training_image_uri())
      return (name_from_base(paste0("compilation-", base_name)))
    },

    # Convert the job description to init params that can be handled by the
    # class constructor
    # Args:
    #   job_details: the returned job details from a describe_training_job
    # API call.
    # model_channel_name (str): Name of the channel where pre-trained
    # model data will be downloaded.
    # Returns:
    #   dictionary: The transformed init_params
    .prepare_init_params_from_job_description = function(job_details,
                                                         model_channel_name=NULL){
      init_params = list()

      init_params[["role"]] = job_details$RoleArn
      init_params[["instance_count"]] = job_details$ResourceConfig$InstanceCount
      init_params[["instance_type"]] = job_details$ResourceConfig$InstanceType
      init_params[["volume_size"]] = job_details$ResourceConfig$VolumeSizeInGB
      init_params[["max_run"]] = job_details$StoppingCondition$MaxRuntimeInSeconds
      init_params[["input_mode"]] = job_details$AlgorithmSpecification$TrainingInputMode
      init_params[["base_job_name"]] = job_details$TrainingJobName
      init_params[["output_path"]] = job_details$OutputDataConfig$S3OutputPath
      init_params[["output_kms_key"]] = job_details$OutputDataConfig$KmsKeyId
      if ("EnableNetworkIsolation" %in% names(job_details))
        init_params[["enable_network_isolation"]] = job_details$EnableNetworkIsolation

      has_hps = !islistempty(job_details$HyperParameters)
      init_params[["hyperparameters"]] = if (has_hps) job_details$HyperParameters else list()

      if (!islistempty(job_details$AlgorithmSpecification$AlgorithmName)) {
        init_params[["algorithm_arn"]] = job_details$AlgorithmSpecification$AlgorithmName
      }

      if (!islistempty(job_details$AlgorithmSpecification$TrainingImage)) {
        init_params[["image"]] = job_details$AlgorithmSpecification$TrainingImage
      } else {
        stop("Invalid AlgorithmSpecification. Either TrainingImage or ",
          "AlgorithmName is expected. NULL was found.", call. = F)}

      if (!islistempty(job_details$AlgorithmSpecification$MetricDefinitons))
        init_params[["metric_definitions"]] = job_details$AlgorithmSpecification$MetricsDefinition

      if (!islistempty(job_details$EnableInterContainerTrafficEncryption))
        init_params[["encrypt_inter_container_traffic"]] = job_details$EnableInterContainerTrafficEncryption

      vpc_list = vpc_from_list(job_details$VpcConfig)
      if (!islistempty(vpc_list$Subnets)){
        init_params[["subnets"]] = vpc_list$Subnets}

      if (!islistempty(vpc_list$SecurityGroupIds)){
        init_params[["security_group_ids"]] = vpc_list$SecurityGroupIds}

      if ("InputDataConfig" %in% names(job_details) && !is.null(model_channel_name)){
        for(channel in job_details$InputDataConfig){
         if (channel$ChannelName == model_channel_name){
            init_params[["model_channel_name"]] = model_channel_name
            init_params[["model_uri"]] = channel$DataSource$S3DataSource$S3Uri
            break}
          }
      }

      return(init_params)
    }

  ),

  active = list(
    #' @field model_data
    #'        The model location in S3. Only set if Estimator has been ``fit()``.
    model_data = function(){
      if (!is.null(self$latest_training_job)){
        model_uri = self$sagemaker_session$sagemaker$describe_training_job(
          TrainingJobName=self$latest_training_job)$ModelArtifacts$S3ModelArtifacts
      } else {
        log_warn(
          "No finished training job found associated with this estimator. Please make sure this estimator is only used for building workflow config")
        model_uri = file.path(self$output_path, self$.current_job_name, "output", "model.tar.gz")
      }

      return(model_uri)
    },

    #' @field training_job_analytics
    #'        Return a ``TrainingJobAnalytics`` object for the current training job.
    training_job_analytics = function() {
      if (is.null(self$.current_job_name))
        stop("Estimator is not associated with a TrainingJob", call. = F)

      return(TrainingJobAnalytics$new(
        self$.current_job_name, sagemaker_session=self$sagemaker_session))
    }
  ),
  lock_objects = F
)

#' @title Sagemaker Estimator Class
#' @description A generic Estimator to train using any supplied algorithm. This class is
#'              designed for use with algorithms that don't have their own, custom class.
#' @export
Estimator = R6Class("Estimator",
  inherit = EstimatorBase,
  public = list(

    #' @description Initialize an ``Estimator`` instance.
    #' @param image_uri (str): The container image to use for training.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if it needs to access an AWS resource.
    #' @param instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.c4.xlarge'.
    #' @param volume_size (int): Size in GB of the EBS volume to use for
    #'              storing input data during training (default: 30). Must be large
    #'              enough to store training data if File Mode is used (which is the
    #'              default).
    #' @param volume_kms_key (str): Optional. KMS key ID for encrypting EBS
    #'              volume attached to the training instance (default: NULL).
    #' @param max_run (int): Timeout in seconds for training (default: 24 *
    #'              60 * 60). After this amount of time Amazon SageMaker terminates
    #'              the job regardless of its current status.
    #' @param input_mode (str): The input mode that the algorithm supports
    #'              (default: 'File'). Valid modes:
    #'              * 'File' - Amazon SageMaker copies the training dataset from the
    #'              S3 location to a local directory.
    #'              * 'Pipe' - Amazon SageMaker streams data directly from S3 to the
    #'              container via a Unix-named pipe.
    #'              This argument can be overriden on a per-channel basis using
    #'              ``TrainingInput.input_mode``.
    #' @param output_path (str): S3 location for saving the training result (model
    #'              artifacts and output files). If not specified, results are
    #'              stored to a default bucket. If the bucket with the specific name
    #'              does not exist, the estimator creates the bucket during the
    #'              :meth:`~sagemaker.estimator.EstimatorBase.fit` method execution.
    #' @param output_kms_key (str): Optional. KMS key ID for encrypting the
    #'              training output (default: NULL).
    #' @param base_job_name (str): Prefix for training job name when the
    #'              :meth:`~sagemaker.estimator.EstimatorBase.fit` method launches.
    #'              If not specified, the estimator generates a default job name,
    #'              based on the training image name and current timestamp.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    #' @param hyperparameters (dict): Dictionary containing the hyperparameters to
    #'              initialize this estimator with.
    #' @param tags (list[dict]): List of tags for labeling a training job. For
    #'              more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @param subnets (list[str]): List of subnet ids. If not specified training
    #'              job will be created without VPC config.
    #' @param security_group_ids (list[str]): List of security group ids. If not
    #'              specified training job will be created without VPC config.
    #' @param model_uri (str): URI where a pre-trained model is stored, either
    #'              locally or in S3 (default: NULL). If specified, the estimator
    #'              can download it. This model can be a 'model.tar.gz' from a
    #'              previous training job, or other artifacts coming from a
    #'              different source.
    #'              In local mode, this should point to the path in which the model
    #'              is located and not the file itself, as local Docker containers
    #'              will try to mount the URI as a volume.
    #'              More information:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html#td-deserialization
    #' @param model_channel_name (str): Name of the channel where 'model_uri' will
    #'              be downloaded (default: 'model').
    #' @param metric_definitions (list[dict]): A list of dictionaries that defines
    #'              the metric(s) used to evaluate the training jobs. Each
    #'              dictionary contains two keys: 'Name' for the name of the metric,
    #'              and 'Regex' for the regular expression used to extract the
    #'              metric from the logs. This should be defined only for jobs that
    #'              don't use an Amazon algorithm.
    #' @param encrypt_inter_container_traffic (bool): Specifies whether traffic
    #'              between training containers is encrypted for the training job
    #'              (default: ``False``).
    #' @param use_spot_instances (bool): Specifies whether to use SageMaker
    #'              Managed Spot instances for training. If enabled then the
    #'              `max_wait` arg should also be set.
    #'              More information:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
    #'              (default: ``False``).
    #' @param max_wait (int): Timeout in seconds waiting for spot training
    #'              instances (default: NULL). After this amount of time Amazon
    #'              SageMaker will stop waiting for Spot instances to become
    #'              available (default: ``NULL``).
    #' @param checkpoint_s3_uri (str): The S3 URI in which to persist checkpoints
    #'              that the algorithm persists (if any) during training. (default:
    #'              ``NULL``).
    #' @param checkpoint_local_path (str): The local path that the algorithm
    #'              writes its checkpoints to. SageMaker will persist all files
    #'              under this path to `checkpoint_s3_uri` continually during
    #'              training. On job startup the reverse happens - data from the
    #'              s3 location is downloaded to this path before the algorithm is
    #'              started. If the path is unset then SageMaker assumes the
    #'              checkpoints will be provided under `/opt/ml/checkpoints/`.
    #'              (default: ``NULL``).
    #' @param enable_network_isolation (bool): Specifies whether container will
    #'              run in network isolation mode (default: ``False``). Network
    #'              isolation mode restricts the container access to outside networks
    #'              (such as the Internet). The container does not make any inbound or
    #'              outbound network calls. Also known as Internet-free mode.
    #' @param rules (list[:class:`~sagemaker.debugger.Rule`]): A list of
    #'              :class:`~sagemaker.debugger.Rule` objects used to define
    #'              rules for continuous analysis with SageMaker Debugger
    #'              (default: ``NULL``). For more, see
    #'              https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html#continuous-analyses-through-rules
    #' @param debugger_hook_config (:class:`~sagemaker.debugger.DebuggerHookConfig` or bool):
    #'              Configuration for how debugging information is emitted with
    #'              SageMaker Debugger. If not specified, a default one is created using
    #'              the estimator's ``output_path``, unless the region does not
    #'              support SageMaker Debugger. To disable SageMaker Debugger,
    #'              set this parameter to ``False``. For more, see
    #'              https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html
    #' @param tensorboard_output_config (:class:`~sagemaker.debugger.TensorBoardOutputConfig`):
    #'              Configuration for customizing debugging visualization using TensorBoard
    #'              (default: ``NULL``). For more, see
    #'              https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html#capture-real-time-tensorboard-data-from-the-debugging-hook
    #' @param enable_sagemaker_metrics (bool): enable SageMaker Metrics Time
    #'              Series. For more information see:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_AlgorithmSpecification.html#SageMaker-Type-AlgorithmSpecification-EnableSageMakerMetricsTimeSeries
    #'              (default: ``NULL``).
    initialize = function(image_uri,
                          role,
                          instance_count,
                          instance_type,
                          volume_size=30,
                          volume_kms_key=NULL,
                          max_run=24 * 60 * 60,
                          input_mode="File",
                          output_path=NULL,
                          output_kms_key=NULL,
                          base_job_name=NULL,
                          sagemaker_session=NULL,
                          hyperparameters=NULL,
                          tags=NULL,
                          subnets=NULL,
                          security_group_ids=NULL,
                          model_uri=NULL,
                          model_channel_name="model",
                          metric_definitions=NULL,
                          encrypt_inter_container_traffic=FALSE,
                          use_spot_instances=FALSE,
                          max_wait=NULL,
                          checkpoint_s3_uri=NULL,
                          checkpoint_local_path=NULL,
                          enable_network_isolation=FALSE,
                          rules=NULL,
                          debugger_hook_config=NULL,
                          tensorboard_output_config=NULL,
                          enable_sagemaker_metrics=NULL){

      self$image_uri = image_uri
      self$hyperparam_list = if (!islistempty(hyperparameters)) hyperparameters else list()
      super$initialize(
        role,
        instance_count,
        instance_type,
        volume_size,
        volume_kms_key,
        max_run,
        input_mode,
        output_path,
        output_kms_key,
        base_job_name,
        sagemaker_session,
        tags,
        subnets,
        security_group_ids,
        model_uri=model_uri,
        model_channel_name=model_channel_name,
        metric_definitions=metric_definitions,
        encrypt_inter_container_traffic=encrypt_inter_container_traffic,
        use_spot_instances=use_spot_instances,
        max_wait=max_wait,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        rules=rules,
        debugger_hook_config=debugger_hook_config,
        tensorboard_output_config=tensorboard_output_config,
        enable_sagemaker_metrics=enable_sagemaker_metrics,
        enable_network_isolation=enable_network_isolation)
      attr(self, "__module__") = environmentName(Estimator$parent_env)
    },

    #' @description Returns the docker image to use for training.
    #'              The fit() method, that does the model training, calls this method to
    #'              find the image to use for model training.
    training_image_uri = function(){
      return(self$image_uri)
    },

    #' @description formats hyperparameters for model tunning
    #' @param ... model hyperparameters
    set_hyperparameters = function(...){
      args = list(...)
      for(x in names(args)){
        self$hyperparam_list[[x]] = args[[x]]
      }
    },

    #' @description Returns the hyperparameters as a dictionary to use for training.
    #'              The fit() method, that does the model training, calls this method to
    #'              find the hyperparameters you specified.

    hyperparameters = function(){
      return (self$hyperparam_list)
    },

    #' @description Create a model to deploy.
    #'              The serializer, deserializer, content_type, and accept arguments are only used to define a
    #'              default Predictor. They are ignored if an explicit predictor class is passed in.
    #'              Other arguments are passed through to the Model class.
    #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
    #'              which is also used during transform jobs. If not specified, the
    #'              role from the Estimator will be used.
    #' @param image_uri (str): An container image to use for deploying the model.
    #'              Defaults to the image used for training.
    #' @param predictor_cls (Predictor): The predictor class to use when
    #'              deploying the model.
    #' @param serializer (callable): Should accept a single argument, the input
    #'              data, and return a sequence of bytes. May provide a content_type
    #'              attribute that defines the endpoint request content type
    #' @param deserializer (callable): Should accept two arguments, the result
    #'              data and the response content type, and return a sequence of
    #'              bytes. May provide a content_type attribute that defines th
    #'              endpoint response Accept content type.
    #' @param content_type (str): The invocation ContentType, overriding any
    #'              content_type from the serializer
    #' @param accept (str): The invocation Accept, overriding any accept from the
    #'              deserializer.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
    #'              the model.
    #'              Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param ... : Additional parameters passed to :class:`~sagemaker.model.Model`
    #'              .. tip::
    #'              You can find additional parameters for using this method at
    #'              :class:`~sagemaker.model.Model`.
    #' @return (sagemaker.model.Model) a Model ready for deployment.
    create_model = function(role=NULL,
                            image_uri=NULL,
                            predictor_cls=NULL,
                            serializer=NULL,
                            deserializer=NULL,
                            content_type=NULL,
                            accept=NULL,
                            vpc_config_override="VPC_CONFIG_DEFAULT",
                            ...){
      args = list(role = role %||% self$role,
                  image_uri = image_uri %||% self$training_image_uri(),
                  vpc_config = self$get_vpc_config(vpc_config_override),
                  sagemaker_session = self$sagemaker_session,
                  model_data = self$model_data,
                  ...)

      if(is.null(predictor_cls)){

        predict_wrapper = function(endpoint, session){
          return(Predictor$new(
            endpoint, session, serializer, deserializer, content_type, accept))
        }

        predictor_cls = predict_wrapper
      }

      args$predictor_cls = predictor_cls

      if (!("enable_network_isolation" %in% names(args)))
          args$enable_network_isolation = self$enable_network_isolation()

      return (do.call(Model$new, args))
    }
  ),

  private = list(
    # Convert the job description to init params that can be handled by the
    # class constructor
    # Args:
    #   job_details: the returned job details from a describe_training_job
    # API call.
    # model_channel_name (str): Name of the channel where pre-trained
    # model data will be downloaded
    # Returns:
    #   dictionary: The transformed init_params
    .prepare_init_params_from_job_description = function(job_details, model_channel_name=NULL){
      init_params = super$.prepare_init_params_from_job_description(
        job_details, model_channel_name)

      init_params$image_uri = init_params$image
      init_params$image = NULL
      return(init_params)
    }
  ),
  lock_objects = F
)


#' @title FrameWork Class
#' @description Base class that cannot be instantiated directly.
#'              Subclasses define functionality pertaining to specific ML frameworks,
#'              such as training/deployment images and predictor instances.
#' @export
Framework = R6Class("Framework",
  inherit = EstimatorBase,
  public = list(
    #' @field LAUNCH_PS_ENV_NAME
    #' class metadata
    LAUNCH_PS_ENV_NAME = "sagemaker_parameter_server_enabled",

    #' @field LAUNCH_MPI_ENV_NAME
    #' class metadata
    LAUNCH_MPI_ENV_NAME = "sagemaker_mpi_enabled",

    #' @field MPI_NUM_PROCESSES_PER_HOST
    #' class metadata
    MPI_NUM_PROCESSES_PER_HOST = "sagemaker_mpi_num_of_processes_per_host",

    #' @field MPI_CUSTOM_MPI_OPTIONS
    #' class metadata
    MPI_CUSTOM_MPI_OPTIONS = "sagemaker_mpi_custom_mpi_options",

    #' @field CONTAINER_CODE_CHANNEL_SOURCEDIR_PATH
    #' class metadata
    CONTAINER_CODE_CHANNEL_SOURCEDIR_PATH = "/opt/ml/input/data/code/sourcedir.tar.gz",

    #' @description Base class initializer. Subclasses which override ``__init__`` should
    #'              invoke ``super()``
    #' @param entry_point (str): Path (absolute or relative) to the local Python
    #'              source file which should be executed as the entry point to
    #'              training. If ``source_dir`` is specified, then ``entry_point``
    #'              must point to a file located at the root of ``source_dir``.
    #'              If 'git_config' is provided, 'entry_point' should be
    #'              a relative location to the Python source file in the Git repo.
    #'              Example:
    #'              With the following GitHub repo directory structure:
    #'              >>> |----- README.md
    #'              >>> |----- src
    #'              >>>         |----- train.py
    #'              >>>         |----- test.py
    #'              You can assign entry_point='src/train.py'.
    #' @param source_dir (str): Path (absolute, relative or an S3 URI) to a directory
    #'              with any other training source code dependencies aside from the entry
    #'              point file (default: None). If ``source_dir`` is an S3 URI, it must
    #'              point to a tar.gz file. Structure within this directory are preserved
    #'              when training on Amazon SageMaker. If 'git_config' is provided,
    #'              'source_dir' should be a relative location to a directory in the Git
    #'              repo.
    #'              .. admonition:: Example
    #'              With the following GitHub repo directory structure:
    #'              >>> |----- README.md
    #'              >>> |----- src
    #'              >>>         |----- train.py
    #'              >>>         |----- test.py
    #'              and you need 'train.py' as entry point and 'test.py' as
    #'              training source code as well, you can assign
    #'              entry_point='train.py', source_dir='src'.
    #' @param hyperparameters (dict): Hyperparameters that will be used for
    #'              training (default: None). The hyperparameters are made
    #'              accessible as a dict[str, str] to the training code on
    #'              SageMaker. For convenience, this accepts other types for keys
    #'              and values, but ``str()`` will be called to convert them before
    #'              training.
    #' @param container_log_level (str): Log level to use within the container
    #'              (default: "INFO")
    #' @param code_location (str): The S3 prefix URI where custom code will be
    #'              uploaded (default: None) - don't include a trailing slash since
    #'              a string prepended with a "/" is appended to ``code_location``. The code
    #'              file uploaded to S3 is 'code_location/job-name/source/sourcedir.tar.gz'.
    #'              If not specified, the default ``code location`` is s3://output_bucket/job-name/.
    #' @param image_uri (str): An alternate image name to use instead of the
    #'              official Sagemaker image for the framework. This is useful to
    #'              run one of the Sagemaker supported frameworks with an image
    #'              containing custom dependencies.
    #' @param dependencies (list[str]): A list of paths to directories (absolute
    #'              or relative) with any additional libraries that will be exported
    #'              to the container (default: []). The library folders will be
    #'              copied to SageMaker in the same folder where the entrypoint is
    #'              copied. If 'git_config' is provided, 'dependencies' should be a
    #'              list of relative locations to directories with any additional
    #'              libraries needed in the Git repo.
    #'              .. admonition:: Example
    #'              The following call
    #'              >>> Estimator(entry_point='train.py',
    #'              ...           dependencies=['my/libs/common', 'virtual-env'])
    #'              results in the following inside the container:
    #'              >>> $ ls
    #'              >>> opt/ml/code
    #'              >>>     |------ train.py
    #'              >>>     |------ common
    #'              >>>     |------ virtual-env
    #'              This is not supported with "local code" in Local Mode.
    #' @param enable_network_isolation (bool): Specifies whether container will
    #'              run in network isolation mode. Network isolation mode restricts
    #'              the container access to outside networks (such as the internet).
    #'              The container does not make any inbound or outbound network
    #'              calls. If True, a channel named "code" will be created for any
    #'              user entry script for training. The user entry script, files in
    #'              source_dir (if specified), and dependencies will be uploaded in
    #'              a tar to S3. Also known as internet-free mode (default: `False`).
    #' @param git_config (dict[str, str]): Git configurations used for cloning
    #'              files, including ``repo``, ``branch``, ``commit``,
    #'              ``2FA_enabled``, ``username``, ``password`` and ``token``. The
    #'              ``repo`` field is required. All other fields are optional.
    #'              ``repo`` specifies the Git repository where your training script
    #'              is stored. If you don't provide ``branch``, the default value
    #'              'master' is used. If you don't provide ``commit``, the latest
    #'              commit in the specified branch is used. .. admonition:: Example
    #'              The following config:
    #'              >>> git_config = {'repo': 'https://github.com/aws/sagemaker-python-sdk.git',
    #'              >>>               'branch': 'test-branch-git-config',
    #'              >>>               'commit': '329bfcf884482002c05ff7f44f62599ebc9f445a'}
    #'              results in cloning the repo specified in 'repo', then
    #'              checkout the 'master' branch, and checkout the specified
    #'              commit.
    #'              ``2FA_enabled``, ``username``, ``password`` and ``token`` are
    #'              used for authentication. For GitHub (or other Git) accounts, set
    #'              ``2FA_enabled`` to 'True' if two-factor authentication is
    #'              enabled for the account, otherwise set it to 'False'. If you do
    #'              not provide a value for ``2FA_enabled``, a default value of
    #'              'False' is used. CodeCommit does not support two-factor
    #'              authentication, so do not provide "2FA_enabled" with CodeCommit
    #'              repositories.
    #'              For GitHub and other Git repos, when SSH URLs are provided, it
    #'              doesn't matter whether 2FA is enabled or disabled; you should
    #'              either have no passphrase for the SSH key pairs, or have the
    #'              ssh-agent configured so that you will not be prompted for SSH
    #'              passphrase when you do 'git clone' command with SSH URLs. When
    #'              HTTPS URLs are provided: if 2FA is disabled, then either token
    #'              or username+password will be used for authentication if provided
    #'              (token prioritized); if 2FA is enabled, only token will be used
    #'              for authentication if provided. If required authentication info
    #'              is not provided, python SDK will try to use local credentials
    #'              storage to authenticate. If that fails either, an error message
    #'              will be thrown.
    #'              For CodeCommit repos, 2FA is not supported, so '2FA_enabled'
    #'              should not be provided. There is no token in CodeCommit, so
    #'              'token' should not be provided too. When 'repo' is an SSH URL,
    #'              the requirements are the same as GitHub-like repos. When 'repo'
    #'              is an HTTPS URL, username+password will be used for
    #'              authentication if they are provided; otherwise, python SDK will
    #'              try to use either CodeCommit credential helper or local
    #'              credential storage for authentication.
    #' @param checkpoint_s3_uri (str): The S3 URI in which to persist checkpoints
    #'              that the algorithm persists (if any) during training. (default:
    #'              ``None``).
    #' @param checkpoint_local_path (str): The local path that the algorithm
    #'              writes its checkpoints to. SageMaker will persist all files
    #'              under this path to `checkpoint_s3_uri` continually during
    #'              training. On job startup the reverse happens - data from the
    #'              s3 location is downloaded to this path before the algorithm is
    #'              started. If the path is unset then SageMaker assumes the
    #'              checkpoints will be provided under `/opt/ml/checkpoints/`.
    #'              (default: ``None``).
    #' @param enable_sagemaker_metrics (bool): enable SageMaker Metrics Time
    #'              Series. For more information see:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_AlgorithmSpecification.html#SageMaker-Type-AlgorithmSpecification-EnableSageMakerMetricsTimeSeries
    #'              (default: ``None``).
    #' @param ... : Additional kwargs passed to the ``EstimatorBase``
    #'              constructor.
    #'              .. tip::
    #'              You can find additional parameters for initializing this class at
    #'              :class:`~sagemaker.estimator.EstimatorBase`.
    initialize = function(entry_point,
                          source_dir=NULL,
                          hyperparameters=NULL,
                          container_log_level=c("INFO", "WARN", "ERROR", "FATAL", "CRITICAL"),
                          code_location=NULL,
                          image_uri=NULL,
                          dependencies=NULL,
                          enable_network_isolation=FALSE,
                          git_config=NULL,
                          checkpoint_s3_uri=NULL,
                          checkpoint_local_path=NULL,
                          enable_sagemaker_metrics=NULL,
                          ...){
      super$initialize(enable_network_isolation=enable_network_isolation, ...)
      if (startsWith(entry_point,"s3://")){
        stop(sprintf("Invalid entry point script: %s. Must be a path to a local file.",
            entry_point))}

      self$entry_point = entry_point
      self$git_config = git_config
      self$source_dir = source_dir
      self$dependencies = dependencies %||% list()
      self$uploaded_code = NULL

      # Align logging level with python logging
      container_log_level = match.arg(container_log_level)
      container_log_level = switch(container_log_level,
                                   "INFO" = 20,
                                   "WARN" = 30,
                                   "ERROR" = 40,
                                   "FATAL" = 50,
                                   "CRITICAL" = 50)
      self$container_log_level = container_log_level
      self$code_location = code_location
      self$image_uri = image_uri

      self$.hyperparameters = hyperparameters %||% list()
      self$checkpoint_s3_uri = checkpoint_s3_uri
      self$checkpoint_local_path = checkpoint_local_path
      self$enable_sagemaker_metrics = enable_sagemaker_metrics

      attr(self, "_framework_name") = NULL
    },

    #' @description Set hyperparameters needed for training. This method will also
    #'              validate ``source_dir``.
    #' @param job_name (str): Name of the training job to be created. If not
    #'              specified, one is generated, using the base name given to the
    #'              constructor if applicable.
    .prepare_for_training = function(job_name=NULL){

      super$.prepare_for_training(job_name = job_name)

      if (!islistempty(self$git_config)){
        updated_paths = git_clone_repo(
          self$git_config, self$entry_point, self$source_dir, self$dependencies)
        self$entry_point = updated_paths$entry_point
        self$source_dir = updated_paths$source_dir
        self$dependencies = updated_paths$dependencies
      }

      # validate source dir will raise a ValueError if there is something wrong with the
      # source directory. We are intentionally not handling it because this is a critical error.
      if (!is.null(self$source_dir) && !startsWith(tolower(self$source_dir),"s3://"))
        validate_source_dir(self$entry_point, self$source_dir)

      # if we are in local mode with local_code=True. We want the container to just
      # mount the source dir instead of uploading to S3.
      local_code = get_config_value("local.local_code", self$sagemaker_session$config)
      if (self$sagemaker_session$local_mode && local_code){
        # if there is no source dir, use the directory containing the entry point.
        if (is.null(self$source_dir))
          self$source_dir = dirname(self$entry_point)
        self.entry_point = basename(self$entry_point)

        code_dir = paste0("file://", self$source_dir)
        script = self$entry_point
      } else if (self$enable_network_isolation() && !is.null(self$entry_point)){
        self$uploaded_code = private$.stage_user_code_in_s3()
        code_dir = self$CONTAINER_CODE_CHANNEL_SOURCEDIR_PATH
        script = self$uploaded_code$script_name
        self$code_uri = self$uploaded_code$s3_prefix
      } else{
        self$uploaded_code = private$.stage_user_code_in_s3()
        code_dir = self$uploaded_code$s3_prefix
        script = self$uploaded_code$script_name
      }

      # Modify hyperparameters in-place to point to the right code directory and script URIs
      self$.hyperparameters[[DIR_PARAM_NAME]] = code_dir
      self$.hyperparameters[[SCRIPT_PARAM_NAME]] = script
      self$.hyperparameters[[CONTAINER_LOG_LEVEL_PARAM_NAME]] = self$container_log_level
      self$.hyperparameters[[JOB_NAME_PARAM_NAME]] = self$.current_job_name
      self$.hyperparameters[[SAGEMAKER_REGION_PARAM_NAME]] = self$sagemaker_session$paws_region_name

      private$.validate_and_set_debugger_configs()
    },

    #' @description Return the hyperparameters as a dictionary to use for training.
    #'              The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which
    #'              trains the model, calls this method to find the hyperparameters.
    #' @return dict[str, str]: The hyperparameters.
    hyperparameters = function(){
      return(self$.hyperparameters)
    },

    #' @description Return the Docker image to use for training.
    #'              The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does
    #'              the model training, calls this method to find the image to use for model
    #'              training.
    #' @return str: The URI of the Docker image.
    training_image_uri = function(){
      if (!is.null(self$image_uri))
        return (self$image_uri)
      return (ImageUris$new()$retrieve(
        attributes(self)$`_framework_name`,
        self$sagemaker_session$paws_region_name,
        instance_type=self$instance_type,
        version=self$framework_version,
        py_version=self$py_version,
        image_scope="training")
      )
    },

    #' @description Attach to an existing training job.
    #'              Create an Estimator bound to an existing training job, each subclass
    #'              is responsible to implement
    #'              ``_prepare_init_params_from_job_description()`` as this method delegates
    #'              the actual conversion of a training job description to the arguments
    #'              that the class constructor expects. After attaching, if the training job
    #'              has a Complete status, it can be ``deploy()`` ed to create a SageMaker
    #'              Endpoint and return a ``Predictor``.
    #'              If the training job is in progress, attach will block and display log
    #'              messages from the training job, until the training job completes.
    #'              Examples:
    #'              >>> my_estimator.fit(wait=False)
    #'              >>> training_job_name = my_estimator.latest_training_job.name
    #'              Later on:
    #'              >>> attached_estimator = Estimator.attach(training_job_name)
    #'              >>> attached_estimator.deploy()
    #' @param training_job_name (str): The name of the training job to attach to.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    #' @param model_channel_name (str): Name of the channel where pre-trained
    #'              model data will be downloaded (default: 'model'). If no channel
    #'              with the same name exists in the training job, this option will
    #'              be ignored.
    #' @return Instance of the calling ``Estimator`` Class with the attached
    #'              training job.
    attach = function(training_job_name,
                      sagemaker_session=NULL,
                      model_channel_name="model"){
      estimator = super$attach(
        training_job_name, sagemaker_session, model_channel_name)

      UploadedCode$s3_prefix=estimator$source_dir
      UploadedCode$script_name= estimator$entry_point

      estimator$uploaded_code = UploadedCode
      return(estimator)
    },

    #' @description Return a ``Transformer`` that uses a SageMaker Model based on the
    #'              training job. It reuses the SageMaker Session and base job name used by
    #'              the Estimator.
    #' @param instance_count (int): Number of EC2 instances to use.
    #' @param instance_type (str): Type of EC2 instance to use, for example,
    #'              ml.c4.xlarge'.
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
    #' @param env (dict): Environment variables to be set for use during the
    #'              transform job (default: None).
    #' @param max_concurrent_transforms (int): The maximum number of HTTP requests
    #'              to be made to each individual transform container at one time.
    #' @param max_payload (int): Maximum size of the payload in a single HTTP
    #'              request to the container in MB.
    #' @param tags (list[dict]): List of tags for labeling a transform job. If
    #'              none specified, then the tags used for the training job are used
    #'              for the transform job.
    #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
    #'              which is also used during transform jobs. If not specified, the
    #'              role from the Estimator will be used.
    #' @param model_server_workers (int): Optional. The number of worker processes
    #'              used by the inference server. If None, server will use one
    #'              worker per vCPU.
    #' @param volume_kms_key (str): Optional. KMS key ID for encrypting the volume
    #'              attached to the ML compute instance (default: None).
    #' @param entry_point (str): Path (absolute or relative) to the local Python source file which
    #'              should be executed as the entry point to training. If ``source_dir`` is specified,
    #'              then ``entry_point`` must point to a file located at the root of ``source_dir``.
    #'              If not specified, the training entry point is used.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for
    #'              the VpcConfig set on the model.
    #'              Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param enable_network_isolation (bool): Specifies whether container will
    #'              run in network isolation mode. Network isolation mode restricts
    #'              the container access to outside networks (such as the internet).
    #'              The container does not make any inbound or outbound network
    #'              calls. If True, a channel named "code" will be created for any
    #'              user entry script for inference. Also known as Internet-free mode.
    #'              If not specified, this setting is taken from the estimator's
    #'              current configuration.
    #' @param model_name (str): Name to use for creating an Amazon SageMaker
    #'              model. If not specified, the name of the training job is used.
    #' @return sagemaker.transformer.Transformer: a ``Transformer`` object that can be used to start a
    #'              SageMaker Batch Transform job.
    transformer = function(instance_count,
                           instance_type,
                           strategy=NULL,
                           assemble_with=NULL,
                           output_path=NULL,
                           output_kms_key=NULL,
                           accept=NULL,
                           env=NULL,
                           max_concurrent_transforms=NULL,
                           max_payload=NULL,
                           tags=NULL,
                           role=NULL,
                           model_server_workers=NULL,
                           volume_kms_key=NULL,
                           entry_point=NULL,
                           vpc_config_override=VPC_CONFIG_DEFAULT,
                           enable_network_isolation=NULL,
                           model_name=NULL){
      role = role %||% self$role
      tags = tags %||% self$tags
      model_name = private$.get_or_create_name(model_name)

      if (!is.null(self$latest_training_job)){
        if (is.null(enable_network_isolation))
          enable_network_isolation = self$enable_network_isolation()

        model = self$create_model(
          role=role,
          model_server_workers=model_server_workers,
          entry_point=entry_point,
          vpc_config_override=vpc_config_override,
          model_kms_key=self$output_kms_key,
          enable_network_isolation=enable_network_isolation,
          name=model_name)
        model$.create_sagemaker_model(instance_type, tags=tags)

        transform_env = model$env
        if (!islistempty(env))
          transform_env = c(transform_env, env)
      } else {
        log_warn(paste0(
          "No finished training job found associated with this estimator. Please make sure ",
          "this estimator is only used for building workflow config"))
        model_name = model_name %||% self$.current_job_name
        transform_env = env %||% list()
      }

      return(Transformer$new(
        model_name,
        instance_count,
        instance_type,
        strategy=strategy,
        assemble_with=assemble_with,
        output_path=output_path,
        output_kms_key=output_kms_key,
        accept=accept,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        env=transform_env,
        tags=tags,
        base_transform_job_name=self$base_job_name,
        volume_kms_key=volume_kms_key,
        sagemaker_session=self$sagemaker_session)
      )
    }

  ),
  private = list(
    # Upload the user training script to s3 and return the location.
    # Returns: s3 uri
    .stage_user_code_in_s3 = function(){
      local_mode = startsWith(self$output_path, "file://")

      if (is.null(self$code_location) && local_mode){
        parsed_s3 = list()
        parsed_s3$bucket = self$sagemaker_session$default_bucket()
        parsed_s3$key = sprintf("%s/%s",self$.current_job_name, "source")
        kms_key = NULL
      } else if(is.null(self$code_location)){
        parsed_s3 = split_s3_uri(self$output_path)
        code_s3_prefix = sprintf("%s/%s",self$.current_job_name, "source")
        kms_key = self$output_kms_key
      } else if (local_mode) {
        parsed_s3 = split_s3_uri(self.code_location)
        code_s3_prefix = paste(Filter(Negate(is.null), c(key_prefix, self$.current_job_name, "source")), collapse = "/")
        kms_key = NULL
      } else {
        parsed_s3 = split_s3_uri(self.code_location)
        code_s3_prefix = paste(Filter(Negate(is.null), c(key_prefix, self$.current_job_name, "source")), collapse = "/")

        parsed_s3 = split_s3_uri(self.output_path)
        kms_key = if (parsed_s3$bucket == output_bucket) self$output_kms_key else NULL
      }
      return (tar_and_upload_dir(
        sagemaker_session=self$sagemaker_session,
        bucket=parsed_s3$bucket,
        s3_key_prefix=parsed_s3$key,
        script=self$entry_point,
        directory=self$source_dir,
        dependencies=self$dependencies,
        kms_key=kms_key)
        )
      },

    # Set defaults for debugging
    .validate_and_set_debugger_configs = function(){
      if (!islistempty(self$debugger_hook_config)
          && .region_supports_debugger(self$sagemaker_session$paws_region_name))
        self$debugger_hook_config = DebuggerHookConfig$new(s3_output_path=self$output_path)
      else if(islistempty(self$debugger_hook_config))
        self$debugger_hook_config = NULL
      },

    # Get the appropriate value to pass as source_dir to model constructor
    # on deploying
    # Returns:
    #   str: Either a local or an S3 path pointing to the source_dir to be
    # used for code by the model to be deployed
    .model_source_dir = function(){
      return (if(self$sagemaker_session$local_mode) self$source_dir
              else self$uploaded_code$s3_prefix)
      },

    # Convert the job description to init params that can be handled by the
    # class constructor
    # Args:
    #   job_details: the returned job details from a describe_training_job
    # API call.
    # model_channel_name (str): Name of the channel where pre-trained
    # model data will be downloaded
    # Returns:
    #   dictionary: The transformed init_params
    .prepare_init_params_from_job_description = function(job_details,
                                                         model_channel_name=NULL){
      init_params = super$.prepare_init_params_from_job_description(
        job_details, model_channel_name)

      init_params$entry_point = init_params$hyperparameters[[SCRIPT_PARAM_NAME]]

      init_params$source_dir = init_params$hyperparameters[[DIR_PARAM_NAME]]
      init_params$container_log_level = init_params$hyperparameters[[CONTAINER_LOG_LEVEL_PARAM_NAME]]

      hyperparameters = list()
      for (i in seq_along(init_params$hyperparameters)) {
        k = names(init_params$hyperparameters)[i]
        v = init_params$hyperparameters[[i]]
        # Tuning jobs add this special hyperparameter which is not JSON serialized
        if (k == "_tuning_objective_metric"){
          if (startsWith(v, '"') && endswith(v, '"'))
            v = gsub('"', '', v)
          hyperparameters[[k]] = v
        } else {
          hyperparameters[[k]] = v}
      }

      init_params$hyperparameters = hyperparameters

      return(init_params)
    },

    # Get the appropriate value to pass as ``entry_point`` to a model constructor.
    # Returns:
    #   str: The path to the entry point script. This can be either an absolute path or
    # a path relative to ``self._model_source_dir()``.
    .model_entry_point = function(){
      if (self$sagemaker_session$local_mode || is.null(private$.model_source_dir()))
        return(self$entry_point)

      return(self$uploaded_code$script_name)
    }
  ),
  lock_objects = F
)
