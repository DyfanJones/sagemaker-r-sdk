# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/mxnet/estimator.py

#' @include deprecations.R
#' @include estimator.R
#' @include fw_utils.R
#' @include mxnet_default.R
#' @include mxnet_model.R
#' @include vpc_utils.R

#' @import R6
#' @import logger

#' @title MXNet Class
#' @description Handle end-to-end training and deployment of custom MXNet code.
#' @export
MXNet = R6Class("MXNet",
  inherit = Framework,
  public = list(

    #' @field .LOWEST_SCRIPT_MODE_VERSION
    #' Lowest MXNet version that can be executed
    .LOWEST_SCRIPT_MODE_VERSION = "1.3",

    #' @description This ``Estimator`` executes an MXNet script in a managed MXNet
    #'              execution environment, within a SageMaker Training Job. The managed
    #'              MXNet environment is an Amazon-built Docker container that executes
    #'              functions defined in the supplied ``entry_point`` Python script.
    #'              Training is started by calling
    #'              :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
    #'              After training is complete, calling
    #'              :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
    #'              SageMaker endpoint and returns an
    #'              :class:`~sagemaker.amazon.mxnet.model.MXNetPredictor` instance that can
    #'              be used to perform inference against the hosted model.
    #'              Technical documentation on preparing MXNet scripts for SageMaker
    #'              training and using the MXNet Estimator is available on the project
    #'              home-page: https://github.com/aws/sagemaker-python-sdk
    #' @param entry_point (str): Path (absolute or relative) to the Python source
    #'              file which should be executed as the entry point to training.
    #'              If ``source_dir`` is specified, then ``entry_point``
    #'              must point to a file located at the root of ``source_dir``.
    #' @param framework_version (str): MXNet version you want to use for executing
    #'              your model training code. Defaults to `None`. Required unless
    #'              ``image_uri`` is provided. List of supported versions.
    #'              https://github.com/aws/sagemaker-python-sdk#mxnet-sagemaker-estimators.
    #' @param py_version (str): Python version you want to use for executing your
    #'              model training code. One of 'py2' or 'py3'. Defaults to ``None``. Required
    #'              unless ``image_uri`` is provided.
    #' @param source_dir (str): Path (absolute, relative or an S3 URI) to a directory
    #'              with any other training source code dependencies aside from the entry
    #'              point file (default: None). If ``source_dir`` is an S3 URI, it must
    #'              point to a tar.gz file. Structure within this directory are preserved
    #'              when training on Amazon SageMaker.
    #' @param hyperparameters (dict): Hyperparameters that will be used for
    #'              training (default: None). The hyperparameters are made
    #'              accessible as a dict[str, str] to the training code on
    #'              SageMaker. For convenience, this accepts other types for keys
    #'              and values, but ``str()`` will be called to convert them before
    #'              training.
    #' @param image_uri (str): If specified, the estimator will use this image for training and
    #'              hosting, instead of selecting the appropriate SageMaker official image based on
    #'              framework_version and py_version. It can be an ECR url or dockerhub image and tag.
    #'              Examples:
    #'              * ``123412341234.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0``
    #'              * ``custom-image:latest``
    #'              If ``framework_version`` or ``py_version`` are ``None``, then
    #'              ``image_uri`` is required. If also ``None``, then a ``ValueError``
    #'              will be raised.
    #' @param distribution (dict): A dictionary with information on how to run distributed
    #'              training (default: None). Currently we support distributed training with
    #'              parameter server and MPI [Horovod].
    #' @param  ... : Additional kwargs passed to the
    #'              :class:`~sagemaker.estimator.Framework` constructor.
    initialize = function(entry_point,
                          framework_version=NULL,
                          py_version=NULL,
                          source_dir=NULL,
                          hyperparameters=NULL,
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

      if (!("enable_sagemaker_metrics" %in% names(kwargs))){
        # enable sagemaker metrics for MXNet v1.6 or greater:
        if (!is.null(self$framework_version) && package_version(self$framework_version) >= package_version("1.6"))
        kwargs$enable_sagemaker_metrics = TRUE
      }

      kwargs = c(entry_point = entry_point,
                 source_dir = source_dir,
                 hyperparameters = hyperparameters,
                 image_uri = image_uri,
                 kwargs)

      do.call(super$initialize, kwargs)

      attr(self, "_framework_name") = "mxnet"

      if (identical(py_version, "py2"))
        log_warn(
          python_deprecation_warning(attr(self, "_framework_name"), MXNET_LATEST_PY2_VERSION)
        )

      if (!is.null(distribution))
        warn_if_parameter_server_with_multi_gpu(
          training_instance_type=instance_type, distribution=distribution
        )

      private$.configure_distribution(distribution)
    },

    #' @description Create a SageMaker ``MXNetModel`` object that can be deployed to an
    #'              ``Endpoint``.
    #' @param model_server_workers (int): Optional. The number of worker processes
    #'              used by the inference server. If None, server will use one
    #'              worker per vCPU.
    #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
    #'              which is also used during transform jobs. If not specified, the
    #'              role from the Estimator will be used.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
    #'              the model. Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param entry_point (str): Path (absolute or relative) to the local Python source file which
    #'              should be executed as the entry point to training. If ``source_dir`` is specified,
    #'              then ``entry_point`` must point to a file located at the root of ``source_dir``.
    #'              If not specified, the training entry point is used.
    #' @param source_dir (str): Path (absolute or relative) to a directory with any other serving
    #'              source code dependencies aside from the entry point file.
    #'              If not specified, the model source directory from training is used.
    #' @param dependencies (list[str]): A list of paths to directories (absolute or relative) with
    #'              any additional libraries that will be exported to the container.
    #'              If not specified, the dependencies from training are used.
    #'              This is not supported with "local code" in Local Mode.
    #' @param image_uri (str): If specified, the estimator will use this image for hosting, instead
    #'              of selecting the appropriate SageMaker official image based on framework_version
    #'              and py_version. It can be an ECR url or dockerhub image and tag.
    #'              Examples:
    #'              * ``123412341234.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0``
    #'              * ``custom-image:latest``
    #' @param ... : Additional kwargs passed to the :class:`~sagemaker.mxnet.model.MXNetModel`
    #'              constructor.
    #' @return sagemaker.mxnet.model.MXNetModel: A SageMaker ``MXNetModel`` object.
    #'              See :func:`~sagemaker.mxnet.model.MXNetModel` for full details.
    create_model = function(model_server_workers=NULL,
                            role=NULL,
                            vpc_config_override="VPC_CONFIG_DEFAULT",
                            entry_point=NULL,
                            source_dir=NULL,
                            dependencies=NULL,
                            image_uri=NULL,
                            ...){
      kwargs = list(...)
      if (!("image_uri" %in% names(kwargs)))
        kwargs$image_uri = image_uri %||% self$image_uri

      kwargs$name = private$.get_or_create_name(kwargs$name)

      kwargs = c(list(
        model_data = self$model_data,
        role = role %||% self$role,
        entry_point = entry_point,
        framework_version=self$framework_version,
        py_version=self$py_version,
        source_dir=source_dir %||% private$.model_source_dir(),
        container_log_level=self$container_log_level,
        code_location=self$code_location,
        model_server_workers=model_server_workers,
        sagemaker_session=self$sagemaker_session,
        vpc_config=self$get_vpc_config(vpc_config_override),
        dependencies=dependencies %||% self$dependencies),
        kwargs
      )

      model = do.call(MXNetModel$new, kwargs)

      if (is.null(entry_point))
        model$entry_point = (
          if (model$.__enclos_env__$private$.is_mms_version())
            self$entry_point
          else private$.model_entry_point()
        )

      return(model)
    }
  ),
  private = list(
    .configure_distribution = function(distribution){
      if (is.null(distribution))
        return(invisible(NULL))

      if (!is.null(self$framework_version) &&
          package_version(self$framework_version) < package_version(self$.LOWEST_SCRIPT_MODE_VERSION))
        stop(sprintf("The distribution option is valid for only versions %s and higher",
                     self$.LOWEST_SCRIPT_MODE_VERSION), call. = F)

      enabled = distribution$parameter_server$enabled %||% FALSE
      self$.hyperparameters[[self$LAUNCH_PS_ENV_NAME]] = enabled

      mpi_dict = distribution$mpi
      mpi_enabled = mpi_dict$enabled %||% FALSE
      self$.hyperparameters[[self$LAUNCH_MPI_ENV_NAME]] = mpi_enabled

      self$.hyperparameters[[self$MPI_NUM_PROCESSES_PER_HOST]] = mpi_dict$processes_per_host

      self$.hyperparameters[[self$MPI_CUSTOM_MPI_OPTIONS]] = mpi_dict$custom_mpi_options %||% ""
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
      init_params = super$.prepare_init_params_from_job_description(
        job_details, model_channel_name
      )
      image_uri = init_params$image_uri
      init_params$image_uri = NULL
      img_split = framework_name_from_image(image_uri)
      names(img_split) = c("framework", "py_version", "tag", "scriptmode")
      # framework, py_version, tag, _ = framework_name_from_image(image_uri)

      # We switched image tagging scheme from regular image version (e.g. '1.0') to more
      # expressive containing framework version, device type and python version
      # (e.g. '0.12-gpu-py2'). For backward compatibility map deprecated image tag '1.0' to a
      # '0.12' framework version otherwise extract framework version from the tag itself.
      if (is.null(img_split$tag)) {
        framework_version = NULL
      } else if (img_split$tag == "1.0") {
        framework_version = "0.12"
      } else {
        framework_version = framework_version_from_tag(img_split$tag)}
      init_params$framework_version = framework_version
      init_params$py_version = img_split$py_version

      if (is.null(img_split$framework)) {
        # If we were unable to parse the framework name from the image it is not one of our
        # officially supported images, in this case just add the image to the init params.
        init_params$image_uri = image_uri
        return(init_params)}

      if (img_split$framework != attr(self, "_framework_name"))
        stop(sprintf(
          "Training job: %s didn't use image for requested framework",
            job_details$TrainingJobName),
          call. = F)

      return(init_params)
    }
  ),
  lock_object = F
)
