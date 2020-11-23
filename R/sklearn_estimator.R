# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/sklearn/estimator.py

#' @include image_uris.R
#' @include deprecations.R
#' @include estimator.R
#' @include fw_utils.R
#' @include sklearn_model.R
#' @include vpc_utils.R
#' @include utils.R

#' @import R6
#' @import logger

#' @title Scikit-learn Class
#' @description Handle end-to-end training and deployment of custom Scikit-learn code.
#' @export
SKLearn = R6Class("SKLearn",
  inherit = Framework,
  public = list(

    #' @description This ``Estimator`` executes an Scikit-learn script in a managed
    #'              Scikit-learn execution environment, within a SageMaker Training Job. The
    #'              managed Scikit-learn environment is an Amazon-built Docker container
    #'              that executes functions defined in the supplied ``entry_point`` Python
    #'              script.
    #'              Training is started by calling
    #'              :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
    #'              After training is complete, calling
    #'              :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
    #'              SageMaker endpoint and returns an
    #'              :class:`~sagemaker.amazon.sklearn.model.SKLearnPredictor` instance that
    #'              can be used to perform inference against the hosted model.
    #'              Technical documentation on preparing Scikit-learn scripts for
    #'              SageMaker training and using the Scikit-learn Estimator is available on
    #'              the project home-page: https://github.com/aws/sagemaker-python-sdk
    #' @param entry_point (str): Path (absolute or relative) to the Python source
    #'              file which should be executed as the entry point to training.
    #'              If ``source_dir`` is specified, then ``entry_point``
    #'              must point to a file located at the root of ``source_dir``.
    #' @param framework_version (str): Scikit-learn version you want to use for
    #'              executing your model training code. Defaults to ``None``. Required
    #'              unless ``image_uri`` is provided. List of supported versions:
    #'              https://github.com/aws/sagemaker-python-sdk#sklearn-sagemaker-estimators
    #' @param py_version (str): Python version you want to use for executing your
    #'              model training code (default: 'py3'). Currently, 'py3' is the only
    #'              supported version. If ``None`` is passed in, ``image_uri`` must be
    #'              provided.
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
    #' @param image_uri (str): If specified, the estimator will use this image
    #'              for training and hosting, instead of selecting the appropriate
    #'              SageMaker official image based on framework_version and
    #'              py_version. It can be an ECR url or dockerhub image and tag.
    #'              Examples:
    #'              123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
    #'              custom-image:latest.
    #'              If ``framework_version`` or ``py_version`` are ``None``, then
    #'              ``image_uri`` is required. If also ``None``, then a ``ValueError``
    #'              will be raised.
    #' @param ... : Additional kwargs passed to the
    #'              :class:`~sagemaker.estimator.Framework` constructor.
    initialize = function(entry_point,
                          framework_version=NULL,
                          py_version="py3",
                          source_dir=NULL,
                          hyperparameters=NULL,
                          image_uri=NULL,
                          ...){
      kwargs = list(...)
      instance_type = renamed_kwargs(
        "train_instance_type", "instance_type", kwargs$instance_type, kwargs
      )
      instance_count = renamed_kwargs(
        "train_instance_count", "instance_count", kwargs$instance_count, kwargs
      )
      validate_version_or_image_args(framework_version, py_version, image_uri)
      if (!is.null(py_version) && py_version != "py3")
        stop("Scikit-learn image only supports Python 3. Please use 'py3' for py_version.",
             call. = F)
      self$framework_version = framework_version
      self$py_version = py_version

      # SciKit-Learn does not support distributed training or training on GPU instance types.
      # Fail fast.
      private$.validate_not_gpu_instance_type(instance_type)

      if (!is.null(instance_count)){
        if (instance_count != 1)
          stop("Scikit-Learn does not support distributed training. Please remove the ",
               "'instance_count' argument or set 'instance_count=1' when initializing SKLearn.",
               call. = F)
      }

      kwargs = c(entry_point = entry_point,
                 source_dir = source_dir,
                 hyperparameters = list(hyperparameters),
                 image_uri = image_uri,
                 instance_count = 1,
                 kwargs)

      do.call(super$initialize, kwargs)

      attr(self, "_framework_name") = "sklearn"

      if (is.null(image_uri))
        self$image_uri = ImageUris$new()$retrieve(
          attr(self, "_framework_name"),
          self$sagemaker_session$paws_region_name,
          version=self$framework_version,
          py_version=self$py_version,
          instance_type=instance_type)
    },

    #' @description Create a SageMaker ``SKLearnModel`` object that can be deployed to an
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
    #' @param ... : Additional kwargs passed to the :class:`~sagemaker.sklearn.model.SKLearnModel`
    #'              constructor.
    #' @return sagemaker.sklearn.model.SKLearnModel: A SageMaker ``SKLearnModel``
    #'              object. See :func:`~sagemaker.sklearn.model.SKLearnModel` for full details.
    create_model = function(model_server_workers=NULL,
                            role=NULL,
                            vpc_config_override="VPC_CONFIG_DEFAULT",
                            entry_point=NULL,
                            source_dir=NULL,
                            dependencies=NULL,
                            ...){
      kwargs = list(...)
      role = role %||% self$role
      kwargs$name = private$.get_or_create_name(kwargs$name)

      if (!("image_uri" %in% kwargs))
          kwargs$image_uri = self$image_uri

      if (!("enable_network_isolation" %in% kwargs))
        kwargs$enable_network_isolation = self$enable_network_isolation()

      kwargs = c(list(
        model_data = self$model_data,
        role = role,
        entry_point = entry_point %||% private$.model_entry_point(),
        source_dir=(source_dir %||% self._model_source_dir()),
        container_log_level=self$container_log_level,
        code_location=self$code_location,
        py_version=self$py_version,
        framework_version=self$framework_version,
        model_server_workers=model_server_workers,
        sagemaker_session=self$sagemaker_session,
        vpc_config=self$get_vpc_config(vpc_config_override),
        dependencies=(dependencies %||% self$dependencies)),
        kwargs)
      return (do.call(SKLearnModel$new, kwargs))
    }
  ),
  private = list(

    # Convert the job description to init params that can be handled by the
    # class constructor
    # Args:
    #   job_details: the returned job details from a describe_training_job
    # API call.
    # model_channel_name (str): Name of the channel where pre-trained
    # model data will be downloaded (default: None).
    # Returns:
    #   dictionary: The transformed init_params
    .prepare_init_params_from_job_description = function(job_details,
                                                         model_channel_name=NULL){
      init_params = super$.prepare_init_params_from_job_description(
        job_details, model_channel_name)

      image_uri = init_params$image_uri
      init_params$image_uri = NULL
      img_split = framework_name_from_image(image_uri)
      names(img_split) = c("framework", "py_version", "tag", "scriptmode")

      if (islistempty(img_split$tag))
        framework_version = NULL
      else
        framework_version = framework_version_from_tag(img_split$tag)
      init_params$framework_version = framework_version
      init_params$py_version = img_split$py_version

      if (islistempty(img_split$framework)){
        # If we were unable to parse the framework name from the image it is not one of our
        # officially supported images, in this case just add the image to the init params.
        init_params$image_uri = image_uri
        return(init_params)}

      if (!islistempty(img_split$framework) && framework != "scikit-learn")
        stop(sprintf(
          "Training job: %s didn't use image for requested framework",
          job_details$TrainingJobName),
          call. = F)

      return(init_params)
    },

    .validate_not_gpu_instance_type = function(training_instance_type){
      gpu_instance_types = c("ml.p2.xlarge",
                             "ml.p2.8xlarge",
                             "ml.p2.16xlarge",
                             "ml.p3.xlarge",
                             "ml.p3.8xlarge",
                             "ml.p3.16xlarge")

      if (training_instance_type %in% gpu_instance_types)
        stop("GPU training in not supported for Scikit-Learn. ",
             "Please pick a different instance type from here: ",
             "https://aws.amazon.com/ec2/instance-types/",
             call. = F)
    }
  ),
  lock_objects = F
)
