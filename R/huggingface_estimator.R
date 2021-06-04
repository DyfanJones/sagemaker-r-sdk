# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/huggingface/estimator.py

#' @import lgr
#' @import R6
#' @import R6sagemaker.common

#' @title HuggingFace estimator class
#' @description Handle training of custom HuggingFace code.
#' @export
HuggingFace = R6Class("HuggingFace",
  inherit = R6sagemaker.common::Framework,
  public = list(

    #' @description This ``Estimator`` executes a HuggingFace script in a managed execution environment.
    #'              The managed HuggingFace environment is an Amazon-built Docker container that executes
    #'              functions defined in the supplied ``entry_point`` Python script within a SageMaker
    #'              Training Job.
    #'              Training is started by calling
    #'              :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
    #' @param py_version (str): Python version you want to use for executing your model training
    #'              code. Defaults to ``None``. Required unless ``image_uri`` is provided.  List
    #'              of supported versions:
    #'              https://github.com/aws/sagemaker-python-sdk#huggingface-sagemaker-estimators
    #' @param entry_point (str): Path (absolute or relative) to the Python source
    #'              file which should be executed as the entry point to training.
    #'              If ``source_dir`` is specified, then ``entry_point``
    #'              must point to a file located at the root of ``source_dir``.
    #' @param transformers_version (str): Transformers version you want to use for
    #'              executing your model training code. Defaults to ``None``. Required unless
    #'              ``image_uri`` is provided. List of supported versions:
    #'              https://github.com/aws/sagemaker-python-sdk#huggingface-sagemaker-estimators.
    #' @param tensorflow_version (str): TensorFlow version you want to use for
    #'              executing your model training code. Defaults to ``None``. Required unless
    #'              ``pytorch_version`` is provided. List of supported versions:
    #'              https://github.com/aws/sagemaker-python-sdk#huggingface-sagemaker-estimators.
    #' @param pytorch_version (str): PyTorch version you want to use for
    #'              executing your model training code. Defaults to ``None``. Required unless
    #'              ``tensorflow_version`` is provided. List of supported versions:
    #'              https://github.com/aws/sagemaker-python-sdk#huggingface-sagemaker-estimators.
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
    #'              * ``123412341234.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0``
    #'              * ``custom-image:latest``
    #'              If ``framework_version`` or ``py_version`` are ``None``, then
    #'              ``image_uri`` is required. If also ``None``, then a ``ValueError``
    #'              will be raised.
    #' @param distribution (dict): A dictionary with information on how to run distributed training
    #'              (default: None).  Currently, the following are supported:
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
    #' @param ... : Additional kwargs passed to the :class:`~sagemaker.estimator.Framework`
    #'              constructor.
    initialize = function(py_version,
                          entry_point,
                          transformers_version=NULL,
                          tensorflow_version=NULL,
                          pytorch_version=NULL,
                          source_dir=NULL,
                          hyperparameters=NULL,
                          image_uri=NULL,
                          distribution=NULL,
                          ...){
      kwargs = list(...)
      self$framework_version = transformers_version
      self$py_version = py_version
      self$tensorflow_version = tensorflow_version
      self$pytorch_version = pytorch_version

      private$.validate_args(image_uri=image_uri)

      if (!is.null(distribution)){
        instance_type = renamed_kwargs(
          "train_instance_type", "instance_type", kwargs[["instance_type"]], kwargs
        )

        base_framework_name = if (!is.null(tensorflow_version)) "tensorflow" else "pytorch"
        base_framework_version = (if (!is.null(tensorflow_version))
          tensorflow_version else pytorch_version)

        validate_smdistributed(
          instance_type=instance_type,
          framework_name=base_framework_name,
          framework_version=base_framework_version,
          py_version=self$py_version,
          distribution=distribution,
          image_uri=image_ur)

        warn_if_parameter_server_with_multi_gpu(
          training_instance_type=instance_type, distribution=distribution
        )
      }
      if (!("enable_sagemaker_metrics" %in% names(kwargs)))
        kwargs[["enable_sagemaker_metrics"]] = TRUE

      kwargs = c(
        entry_point=entry_point,
        source_dir=source_dir,
        hyperparameters=list(hyperparameters),
        image_uri=image_uri,
        kwargs)

      do.call(super$initialize, kwargs)

      self$distribution = distribution %||% list()

      attr(self, "_framework_name") = "huggingface"
    },

    #' @description Return hyperparameters used by your custom PyTorch code during model training.
    hyperparameters = function(){
      hyperparameters = super$hyperparameters()
      additional_hyperparameters = private$.distribution_configuration(
        distribution=self$distribution)
      # Currently don't use .json_encode_hyperparameters
      # hyperparameters = modifyList(
      #   hyperparameters,
      #   private$.json_encode_hyperparameters(additional_hyperparameters))
      hyperparameters = modifyList(
        hyperparameters,
        additional_hyperparameters)
      return(hyperparameters)
    },

    #' @description Create a model to deploy.
    #'              The serializer, deserializer, content_type, and accept arguments are only used to define a
    #'              default Predictor. They are ignored if an explicit predictor class is passed in.
    #'              Other arguments are passed through to the Model class.
    #'              Creating model with HuggingFace training job is not supported.
    #' @param model_server_workers (int): Optional. The number of worker processes
    #'              used by the inference server. If None, server will use one
    #'              worker per vCPU.
    #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
    #'              which is also used during transform jobs. If not specified, the
    #'              role from the Estimator will be used.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
    #'              the model.
    #'              Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param entry_point (str): Path (absolute or relative) to the local Python
    #'              source file which should be executed as the entry point to
    #'              training. If ``source_dir`` is specified, then ``entry_point``
    #'              must point to a file located at the root of ``source_dir``.
    #'              If 'git_config' is provided, 'entry_point' should be
    #'              a relative location to the Python source file in the Git repo.
    #' @param source_dir (str): Path (absolute, relative or an S3 URI) to a directory
    #'              with any other training source code dependencies aside from the entry
    #'              point file (default: None). If ``source_dir`` is an S3 URI, it must
    #'              point to a tar.gz file. Structure within this directory are preserved
    #'              when training on Amazon SageMaker. If 'git_config' is provided,
    #'              'source_dir' should be a relative location to a directory in the Git
    #'              repo.
    #' @param dependencies (list[str]): A list of paths to directories (absolute
    #'              or relative) with any additional libraries that will be exported
    #'              to the container (default: []). The library folders will be
    #'              copied to SageMaker in the same folder where the entrypoint is
    #'              copied. If 'git_config' is provided, 'dependencies' should be a
    #'              list of relative locations to directories with any additional
    #'              libraries needed in the Git repo.
    #' @param ... : Additional parameters passed to :class:`~sagemaker.model.Model`
    #'              .. tip::
    #'              You can find additional parameters for using this method at
    #'              :class:`~sagemaker.model.Model`.
    #' @return (sagemaker.model.Model) a Model ready for deployment.
    create_model = function(model_server_workers=NULL,
                            role=NULL,
                            vpc_config_override="VPC_CONFIG_DEFAULT",
                            entry_point=NULL,
                            source_dir=NULL,
                            dependencies=NULL,
                            ...){
      NotImplementedError$new("Creating model with HuggingFace training job is not supported.")
    }
  ),
  private = list(
    .validate_args = function(image_uri=NULL){
      if (!is.null(image_uri))
        return(NULL)

      if (is.null(self$framework_version) && is.null(image_uri))
        ValueError$new(
          "transformers_version, and image_uri are both NULL. ",
          "Specify either transformers_version or image_uri")
      if (!is.null(self$tensorflow_version) && !is.null(self$pytorch_version))
        ValueError$new(
          "tensorflow_version and pytorch_version are both not NULL. ",
          "Specify only tensorflow_version or pytorch_version.")
      if (is.null(self$tensorflow_version) && is.null(self$pytorch_version))
        ValueError$new(
          "tensorflow_version and pytorch_version are both NULL. ",
          "Specify either tensorflow_version or pytorch_version.")
      base_framework_version_len = (
        if (!is.null(self$tensorflow_version))
          length(split_str(self$tensorflow_version, "\\."))
        else length(split_str(self$pytorch_version, "\\."))
      )
      transformers_version_len = length(split_str(self$framework_version,"\\."))
      if (transformers_version_len != base_framework_version_len)
        ValueError$new(
          "Please use either full version or shortened version for both ",
          "transformers_version, tensorflow_version and pytorch_version.")
    },

    # Convert the job description to init params that can be handled by the class constructor.
    # Args:
    #   job_details: The returned job details from a describe_training_job
    # API call.
    # model_channel_name (str): Name of the channel where pre-trained
    # model data will be downloaded.
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

      if (is.null(img_split$tag)){
        framework_version = NULL
      } else {
        fmwk = split_str(img_split$framework,"-")
        names(fmwk) = c("framework", "pt_or_tf")
        tag_pattern = "^(.*)-transformers(.*)-(cpu|gpu)-(py2|py3[67]?)$"
        m = regexec(tag_pattern, img_split$tag)
        tag_match = unlist(regmatches(img_split$tag, m))
        pt_or_tf_version = tag_match[2]
        framework_version = tag_match[3]
        if (fmwk[["pt_or_tf"]] == "pytorch"){
          init_params[["pytorch_version"]] = pt_or_tf_version
        } else {
          init_params[["tensorflow_version"]] = pt_or_tf_version
        }
      }
      init_params[["transformers_version"]] = framework_version
      init_params = append(init_params, list("py_version"=img_split$py_version))

      if (is.null(img_split$framework)){
        # If we were unable to parse the framework name from the image it is not one of our
        # officially supported images, in this case just add the image to the init params.
        init_params[["image_uri"]] = image_uri
        return(init_params)
      }
      if (fmwk[["framework"]] != attr(self, "_framework_name"))
        ValueError$new(
          sprintf("Training job: %s didn't use image for requested framework",
            job_details[["TrainingJobName"]])
        )

      return(init_params)
    }
  ),
  lock_objects=FALSE
)
