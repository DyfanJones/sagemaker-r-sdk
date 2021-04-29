# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/estimator.py

#' @include image_uris.R
#' @include s3.R
#' @include utils.R
#' @include debugger.R
#' @include deprecations.R
#' @include estimator.R
#' @include fw_utils.R
#' @include tensorflow_defaults.R
#' @include tensorflow_model.R
#' @include transformer.R
#' @include vpc_utils.R

#' @import R6
#' @import lgr

#' @title TensorFlow Class
#' @description Handle end-to-end training and deployment of user-provided TensorFlow code.
#' @export
TensorFlow = R6Class("TensorFlow",
  inherit = Framework,
  public = list(

    #' @description Initialize a ``TensorFlow`` estimator.
    #' @param py_version (str): Python version you want to use for executing your model training
    #'              code. Defaults to ``None``. Required unless ``image_uri`` is provided.
    #' @param framework_version (str): TensorFlow version you want to use for executing your model
    #'              training code. Defaults to ``None``. Required unless ``image_uri`` is provided.
    #'              List of supported versions:
    #'              https://github.com/aws/sagemaker-python-sdk#tensorflow-sagemaker-estimators.
    #' @param model_dir (str): S3 location where the checkpoint data and models can be exported to
    #'              during training (default: None). It will be passed in the training script as one of
    #'              the command line arguments. If not specified, one is provided based on
    #'              your training configuration:
    #'              * *distributed training with SMDistributed or MPI with Horovod* - ``/opt/ml/model``
    #'              * *single-machine training or distributed training without MPI* - \
    #'              ``s3://{output_path}/model``
    #'              * *Local Mode with local sources (file:// instead of s3://)* - \
    #'              ``/opt/ml/shared/model``
    #'              To disable having ``model_dir`` passed to your training script,
    #'              set ``model_dir=False``.
    #' @param image_uri (str): If specified, the estimator will use this image for training and
    #'              hosting, instead of selecting the appropriate SageMaker official image based on
    #'              framework_version and py_version. It can be an ECR url or dockerhub image and tag.
    #'              Examples:
    #'              123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
    #'              custom-image:latest.
    #'              If ``framework_version`` or ``py_version`` are ``None``, then
    #'              ``image_uri`` is required. If also ``None``, then a ``ValueError``
    #'              will be raised.
    #' @param distribution (dict): A dictionary with information on how to run distributed training
    #'              (default: None). Currently, the following are supported:
    #'              distributed training with parameter servers, SageMaker Distributed (SMD) Data
    #'              and Model Parallelism, and MPI. SMD Model Parallelism can only be used with MPI.
    #'              To enable parameter server use the following setup:
    #'              .. code:: python
    #'              {
    #'              "parameter_server": {
    #'              "enabled": True
    #'              }
    #'              }
    #'              To enable MPI:
    #'              .. code:: python
    #'              {
    #'              "mpi": {
    #'              "enabled": True
    #'              }
    #'              }
    #'              To enable SMDistributed Data Parallel or Model Parallel:
    #'              .. code:: python
    #'              {
    #'              "smdistributed": {
    #'              "dataparallel": {
    #'              "enabled": True
    #'              },
    #'              "modelparallel": {
    #'              "enabled": True,
    #'              "parameters": {}
    #'              }
    #'              }
    #'              }
    #' @param ... : Additional kwargs passed to the Framework constructor.
    initialize = function(py_version=NULL,
                          framework_version=NULL,
                          model_dir=NULL,
                          image_uri=NULL,
                          distribution=NULL,
                          ...){
      kwargs = list(...)
      distribution = renamed_kwargs("distributions", "distribution", distribution, kwargs)
      instance_type = renamed_kwargs(
        "train_instance_type", "instance_type", kwargs$instance_type, kwargs
      )
      validate_version_or_image_args(framework_version, py_version, image_uri)

      self$framework_version = framework_version
      self$py_version = py_version
      self$instance_type = instance_type

      if (!is.null(distribution)){
        warn_if_parameter_server_with_multi_gpu(
          training_instance_type=instance_type, distribution=distribution
        )

        validate_smdistributed(
          instance_type=instance_type,
          framework_name=self._framework_name,
          framework_version=framework_version,
          py_version=py_version,
          distribution=distribution,
          image_uri=image_uri)
      }

      if (!("enable_sagemaker_metrics" %in% names(kwargs))){
        # enable sagemaker metrics for TF v1.15 or greater:
        if (!is.null(framework_version) && package_version(framework_version) >= package_version("1.15"))
          kwargs$enable_sagemaker_metrics = TRUE
      }

      kwargs = c(image_uri = image_uri,
                 kwargs)
      do.call(super$initialize, kwargs)

      self$model_dir = model_dir
      self$distribution = distribution %||% list()

      attr(self, "_framework_name") = "tensorflow"

      if (identical(py_version, "py2"))
        LOGGER$warn(
          python_deprecation_warning(attr(self, "_framework_name"), TENSORFLOW_LATEST_PY2_VERSION)
        )

      private$.validate_args(py_version=py_version)
    },

    #' @description Create a ``TensorFlowModel`` object that can be used for creating
    #'              SageMaker model entities, deploying to a SageMaker endpoint, or
    #'              starting SageMaker Batch Transform jobs.
    #' @param role (str): The ``TensorFlowModel``, which is also used during transform jobs.
    #'              If not specified, the role from the Estimator is used.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the
    #'              model. Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param entry_point (str): Path (absolute or relative) to the local Python source file which
    #'              should be executed as the entry point to training. If ``source_dir`` is specified,
    #'              then ``entry_point`` must point to a file located at the root of ``source_dir``.
    #'              If not specified and ``endpoint_type`` is 'tensorflow-serving',
    #'              no entry point is used. If ``endpoint_type`` is also ``None``,
    #'              then the training entry point is used.
    #' @param source_dir (str): Path (absolute or relative or an S3 URI) to a directory with any other
    #'              serving source code dependencies aside from the entry point file (default: None).
    #' @param dependencies (list[str]): A list of paths to directories (absolute or relative) with
    #'              any additional libraries that will be exported to the container (default: None).
    #' @param ... : Additional kwargs passed to
    #'              :class:`~sagemaker.tensorflow.model.TensorFlowModel`.
    #' @return sagemaker.tensorflow.model.TensorFlowModel: A ``TensorFlowModel`` object.
    #'              See :class:`~sagemaker.tensorflow.model.TensorFlowModel` for full details.
    create_model = function(role=NULL,
                            vpc_config_override="VPC_CONFIG_DEFAULT",
                            entry_point=NULL,
                            source_dir=NULL,
                            dependencies=NULL,
                            ...){
      kwargs = list(...)
      kwargs$name = private$.get_or_create_name(kwargs$name)

      if (!("image_uri" %in% names(kwargs)))
        kwargs$image_uri = self$image_uri

      if (!("enable_network_isolation" %in% names(kwargs)))
        kwargs$enable_network_isolation = self$enable_network_isolation()

      kwargs = c(model_data=self$model_data,
                 role=role %||% self$role,
                 container_log_level=self$container_log_level,
                 framework_version=self$framework_version,
                 sagemaker_session=self$sagemaker_session,
                 vpc_config=list(self$get_vpc_config(vpc_config_override)),
                 entry_point=entry_point,
                 source_dir=source_dir,
                 dependencies=dependencies,
                 kwargs)
      return(do.call(TensorFlowModel$new, kwargs))
    },

    #' @description Return hyperparameters used by your custom TensorFlow code during model training.
    hyperparameters = function(){
      hyperparameters = super$hyperparameters()
      additional_hyperparameters = private$.distribution_configuration(self$distribution)

      if (!isFALSE(self$model_dir)) {
        self$model_dir = self$model_dir %||% private$.default_s3_path(
          "model", mpi=(additional_hyperparameters[[self$LAUNCH_MPI_ENV_NAME]] %||% FALSE)
        )
        additional_hyperparameters$model_dir = self$model_dir
      }
      hyperparameters = c(hyperparameters, additional_hyperparameters)
      return(hyperparameters)
    },

    #' @description Return a ``Transformer`` that uses a SageMaker Model based on the training job. It
    #'              reuses the SageMaker Session and base job name used by the Estimator.
    #' @param instance_count (int): Number of EC2 instances to use.
    #' @param instance_type (str): Type of EC2 instance to use, for example, 'ml.c4.xlarge'.
    #' @param strategy (str): The strategy used to decide how to batch records in a single request
    #'              (default: None). Valid values: 'MultiRecord' and 'SingleRecord'.
    #' @param assemble_with (str): How the output is assembled (default: None). Valid values: 'Line'
    #'              or 'None'.
    #' @param output_path (str): S3 location for saving the transform result. If not specified,
    #'              results are stored to a default bucket.
    #' @param output_kms_key (str): Optional. KMS key ID for encrypting the transform output
    #'              (default: None).
    #' @param accept (str): The accept header passed by the client to
    #'              the inference endpoint. If it is supported by the endpoint,
    #'              it will be the format of the batch transform output.
    #' @param env (dict): Environment variables to be set for use during the transform job
    #'              (default: None).
    #' @param max_concurrent_transforms (int): The maximum number of HTTP requests to be made to
    #'              each individual transform container at one time.
    #' @param max_payload (int): Maximum size of the payload in a single HTTP request to the
    #'              container in MB.
    #' @param tags (list[dict]): List of tags for labeling a transform job. If none specified, then
    #'              the tags used for the training job are used for the transform job.
    #' @param role (str): The IAM Role ARN for the ``TensorFlowModel``, which is also used
    #'              during transform jobs. If not specified, the role from the Estimator is used.
    #' @param volume_kms_key (str): Optional. KMS key ID for encrypting the volume attached to the ML
    #'              compute instance (default: None).
    #' @param entry_point (str): Path (absolute or relative) to the local Python source file which
    #'              should be executed as the entry point to training. If ``source_dir`` is specified,
    #'              then ``entry_point`` must point to a file located at the root of ``source_dir``.
    #'              If not specified and ``endpoint_type`` is 'tensorflow-serving',
    #'              no entry point is used. If ``endpoint_type`` is also ``None``,
    #'              then the training entry point is used.
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
    #'              model. If not specified, the estimator generates a default job name
    #'              based on the training image name and current timestamp.
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
                           entry_point=NULL,
                           vpc_config_override="VPC_CONFIG_DEFAULT",
                           enable_network_isolation=NULL,
                           model_name=NULL){
      role = role %||% self$role
      model_name = private$.get_or_create_name(model_name)

      if (is.null(self$latest_training_job)) {
        LOGGER$warn(paste(
          "No finished training job found associated with this estimator. Please make sure",
          "this estimator is only used for building workflow config"))
        return (Transformer$new(
          model_name=model_name,
          instance_count=instance_count,
          instance_type=instance_type,
          strategy=strategy,
          assemble_with=assemble_with,
          output_path=output_path,
          output_kms_key=output_kms_key,
          accept=accept,
          max_concurrent_transforms=max_concurrent_transforms,
          max_payload=max_payload,
          env=env %||% list(),
          tags=tags,
          base_transform_job_name=self$base_job_name,
          volume_kms_key=volume_kms_key,
          sagemaker_session=self$sagemaker_session)
        )
      }

      if (is.null(enable_network_isolation))
        enable_network_isolation = self$enable_network_isolation()

      model = self$create_model(
        role=role,
        vpc_config_override=vpc_config_override,
        entry_point=entry_point,
        enable_network_isolation=enable_network_isolation,
        name=model_name)

      return(model$transformer(
        instance_count,
        instance_type,
        strategy=strategy,
        assemble_with=assemble_with,
        output_path=output_path,
        output_kms_key=output_kms_key,
        accept=accept,
        env=env,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        tags=tags,
        volume_kms_key=volume_kms_key)
      )
    }
  ),
  private = list(
    .HIGHEST_LEGACY_MODE_ONLY_VERSION = "1.10.0",
    .HIGHEST_PYTHON_2_VERSION = "2.1.1",

    .validate_args = function(py_version){

      if (identical(py_version, "py2") && private$.only_python_3_supported()){
        msg = paste0(
          sprintf("Python 2 containers are only available with %s and lower versions. ", TENSORFLOW_LATEST_PY2_VERSION),
          "Please use a Python 3 container.")
        AttributeError$new(msg)
      }

      if (is.null(self$image_uri) && private$.only_legacy_mode_supported()){
        legacy_image_uri = ImageUris$new()$retrieve(
          "tensorflow",
          self$sagemaker_session$paws_region_name,
          instance_type=self$instance_type,
          version=self$framework_version,
          py_version=self$py_version,
          image_scope="training")
        msg = paste0(
          sprintf("TF %s supports only legacy mode. Please supply the image URI directly with ",self$framework_version),
          sprintf("'image_uri=%s' and set 'model_dir=FALSE'. If you are using any legacy parameters ", legacy_image_uri),
          "(training_steps, evaluation_steps, checkpoint_path, requirements_file), ",
          "make sure to pass them directly as hyperparameters instead. For more, see ",
          "https://sagemaker.readthedocs.io/en/v2.0.0.rc0/frameworks/tensorflow/upgrade_from_legacy.html.")
        ValueError$new(msg)
      }
    },

    .only_legacy_mode_supported = function(){
      return(package_version(self$framework_version) <= package_version(private$.HIGHEST_LEGACY_MODE_ONLY_VERSION))
    },

    .only_python_3_supported = function(){
      return(package_version(self$framework_version) > package_version(private$.HIGHEST_PYTHON_2_VERSION))
    },

    # Convert the job description to init params that can be handled by the class constructor
    # Args:
    #   job_details: the returned job details from a describe_training_job API call.
    # Returns:
    #   dictionary: The transformed init_params
    .prepare_init_params_from_job_description = function(job_details,
                                                         model_channel_name=NULL){
      init_params = super$.prepare_init_params_from_job_description(
        job_details, model_channel_name)

      image_uri = init_params$image_uri
      init_params$image_uri = NULL
      img_split = framework_name_from_image(image_uri)
      names(img_split) = c("framework", "py_version", "tag", "script_mode")

      if (islistempty(img_split$framework)){
        # If we were unable to parse the framework name from the image, it is not one of our
        # officially supported images, so just add the image to the init params.
        init_params$image_uri = image_uri
        return(init_params)
      }

      model_dir = init_params$hyperparameters$model_dir
      init_params$hyperparameters$model_dir = NULL
      if (!is.null(model_dir)) {
        init_params$model_dir = model_dir
      } else if (islistempty(img_split$script_mode)) {
        init_params$model_dir = FALSE
      }

      init_params$py_version = img_split$py_version

      # We switched image tagging scheme from regular image version (e.g. '1.0') to more
      # expressive containing framework version, device type and python version
      # (e.g. '1.5-gpu-py2'). For backward compatibility map deprecated image tag '1.0' to a
      # '1.4' framework version otherwise extract framework version from the tag itself.
      init_params$framework_version = if (img_split$tag == "1.0") "1.4" else framework_version_from_tag(img_split$tag)

      # Legacy images are required to be passed in explicitly.
      if (islistempty(img_split$script_mode))
        init_params$image_uri = image_uri

      if (img_split$framework != attr(self, "_framework_name"))
        ValueError$new(sprintf("Training job: %s didn't use image for requested framework",
                     job_details$TrainingJobName))

      return(init_params)
    },

    .default_s3_path = function(directory,
                                mpi=FALSE){
      local_code = get_config_value("local.local_code", self$sagemaker_session$config)
      if (self$sagemaker_session$local_mode && local_code)
        return (sprintf("/opt/ml/shared/%s", directory))
      if (mpi)
        return("/opt/ml/model")
      if (!is.null(self$.current_job_name))
        return(s3_path_join(self$output_path, self$.current_job_name, directory))
      return(NULL)
    },

    # Disable Debugger Hook Config for ParameterServer (PS) as it is not
    # supported in smdebug.
    # Else, set default HookConfig
    .validate_and_set_debugger_configs = function(){
      ps_enabled = "parameter_server" %in% names(self$distribution) && (self$distribution$parameter_server$enabled %||% FALSE)
      if (ps_enabled){
        if (!islistempty(self$debugger_hook_config) %||% !islistempty(self$debugger_rule_configs)){
          LOGGER$info(paste(
            "Amazon SageMaker Debugger does not currently support",
            "Parameter Server distribution"))
        }
        self$debugger_hook_config = NULL
        self$debugger_rule_configs = NULL
      } else if (
        is.null(self$debugger_hook_config) &&
        .region_supports_debugger(self$sagemaker_session$paws_region_name)){
        # Set defaults for debugging.
        self$debugger_hook_config = DebuggerHookConfig$new(s3_output_path=self$output_path)
      }
    }
  ),
  lock_object = F
)
