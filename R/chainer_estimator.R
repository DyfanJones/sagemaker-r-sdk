# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/chainer/estimator.py

#' @import R6
#' @import logger

#' @include session.R
#' @include fw_utils.R
#' @include chainer_default.R
#' @include chainer_model.R
#' @include vpc_utils.R
#' @include utils.R

#' @title Chainer Class
#' @description Handle end-to-end training and deployment of custom Chainer code.
#' @export
Chainer = R6Class("Chainer",
  inherit = Framework,
  public = list(

    # Hyperparameters
    #' @field .use_mpi
    #' Entry point is run as an MPI script.
    .use_mpi = "sagemaker_use_mpi",

    #' @field .num_processes
    #' Total number of processes to run the entry
    #'        point with
    .num_processes = "sagemaker_num_processes",

    #' @field .process_slots_per_host
    #' The number of processes that can run
    #'        on each instance.
    .process_slots_per_host = "sagemaker_process_slots_per_host",

    #' @field .additional_mpi_options
    #' String of options to the 'mpirun' command used
    #'        to run the entry point.
    .additional_mpi_options = "sagemaker_additional_mpi_options",

    #' @description This ``Estimator`` executes an Chainer script in a managed Chainer
    #'              execution environment, within a SageMaker Training Job. The managed
    #'              Chainer environment is an Amazon-built Docker container that executes
    #'              functions defined in the supplied ``entry_point`` Python script.
    #'              Training is started by calling
    #'              :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
    #'              After training is complete, calling
    #'              :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
    #'              SageMaker endpoint and returns an
    #'              :class:`~sagemaker.amazon.chainer.model.ChainerPredictor` instance that
    #'              can be used to perform inference against the hosted model.
    #'              Technical documentation on preparing Chainer scripts for SageMaker
    #'              training and using the Chainer Estimator is available on the project
    #'              home-page: https://github.com/aws/sagemaker-python-sdk
    #' @param entry_point (str): Path (absolute or relative) to the Python source
    #'              file which should be executed as the entry point to training.
    #'              If ``source_dir`` is specified, then ``entry_point``
    #'              must point to a file located at the root of ``source_dir``.
    #' @param use_mpi (bool): If true, entry point is run as an MPI script. By
    #'              default, the Chainer Framework runs the entry point with
    #'              'mpirun' if more than one instance is used.
    #' @param num_processes (int): Total number of processes to run the entry
    #'              point with. By default, the Chainer Framework runs one process
    #'              per GPU (on GPU instances), or one process per host (on CPU
    #'              instances).
    #' @param process_slots_per_host (int): The number of processes that can run
    #'              on each instance. By default, this is set to the number of GPUs
    #'              on the instance (on GPU instances), or one (on CPU instances).
    #' @param additional_mpi_options (str): String of options to the 'mpirun'
    #'              command used to run the entry point. For example, '-X
    #'              NCCL_DEBUG=WARN' will pass that option string to the mpirun
    #'              command.
    #' @param source_dir (str): Path (absolute or relative) to a directory with
    #'              any other training source code dependencies aside from the entry
    #'              point file (default: None). Structure within this directory are
    #'              preserved when training on Amazon SageMaker.
    #' @param hyperparameters (dict): Hyperparameters that will be used for
    #'              training (default: None). The hyperparameters are made
    #'              accessible as a dict[str, str] to the training code on
    #'              SageMaker. For convenience, this accepts other types for keys
    #'              and values, but ``str()`` will be called to convert them before
    #'              training.
    #' @param py_version (str): Python version you want to use for executing your
    #'              model training code. Defaults to ``None``. Required unless ``image_uri``
    #'              is provided.
    #' @param framework_version (str): Chainer version you want to use for
    #'              executing your model training code. Defaults to ``None``. Required unless
    #'              ``image_uri`` is provided. List of supported versions:
    #'              https://github.com/aws/sagemaker-python-sdk#chainer-sagemaker-estimators.
    #' @param image_uri (str): If specified, the estimator will use this image
    #'              for training and hosting, instead of selecting the appropriate
    #'              SageMaker official image based on framework_version and
    #'              py_version. It can be an ECR url or dockerhub image and tag.
    #'              Examples
    #'              * ``123412341234.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0``
    #'              * ``custom-image:latest``
    #'              If ``framework_version`` or ``py_version`` are ``None``, then
    #'              ``image_uri`` is required. If also ``None``, then a ``ValueError``
    #'              will be raised.
    #' @param ... : Additional kwargs passed to the
    #'              :class:`~sagemaker.estimator.Framework` constructor.
    initialize = function(entry_point,
                          use_mpi=NULL,
                          num_processes=NULL,
                          process_slots_per_host=NULL,
                          additional_mpi_options=NULL,
                          source_dir=NULL,
                          hyperparameters=NULL,
                          framework_version=NULL,
                          py_version=NULL,
                          image_uri=NULL,
                          ...){

      validate_version_or_image_args(framework_version, py_version, image_uri)

      self$framework_version = framework_version
      self$py_version = py_version

      super$initialize(
        entry_point, source_dir, hyperparameters, image_uri=image_uri, ...)

      attr(self, "_framework_name") = "chainer"

      if (identical(py_version, "py2"))
        log_warn(
          python_deprecation_warning(attr(self, "_framework_name"), CHAINER_LATEST_PY2_VERSION)
        )

      self$use_mpi = use_mpi
      self$num_processes = num_processes
      self$process_slots_per_host = process_slots_per_host
      self$additional_mpi_options = additional_mpi_options
    },

    #' @description Return hyperparameters used by your custom Chainer code during
    #'              training.
    hyperparameters = function(){
      hyperparameters = super$hyperparameters()

      additional_hyperparameters = list(
        self$use_mpi,
        self$num_processes,
        self$process_slots_per_host,
        self$additional_mpi_options
        )
      names(additional_hyperparameters) = c(
        self$.use_mpi, self$.num_processes, self$.process_slots_per_host, self$.additional_mpi_options)

      # remove unset keys.
      additional_hyperparameters = Filter(Negate(is.null), additional_hyperparameters)

      return(c(hyperparameters, additional_hyperparameters))
    },

    #' @description Create a SageMaker ``ChainerModel`` object that can be deployed to an
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
    #' @param ... : Additional kwargs passed to the ChainerModel constructor.
    #' @return sagemaker.chainer.model.ChainerModel: A SageMaker ``ChainerModel``
    #'              object. See :func:`~sagemaker.chainer.model.ChainerModel` for full details.
    create_model = function(model_server_workers=NULL,
                            role=NULL,
                            vpc_config_override="VPC_CONFIG_DEFAULT",
                            entry_point=NULL,
                            source_dir=NULL,
                            dependencies=NULL,
                            ...){
      kwargs = list(...)
      kwargs$name = private$.get_or_create_name(kwargs$name)

      if (!("image_uri" %in% names(kwargs)))
        kwargs$image_uri = self$image_uri

      vpc_config = self$get_vpc_config(vpc_config_override)
      model_data=self$model_data
      kwargs = c(
        model_data=model_data,
        role=role %||% self$role,
        entry_point=entry_point %||% private$.model_entry_point(),
        source_dir=list(source_dir %||% private$.model_source_dir()),
        container_log_level=self$container_log_level,
        code_location=self$code_location,
        py_version=self$py_version,
        framework_version=self$framework_version,
        model_server_workers=model_server_workers,
        sagemaker_session=self$sagemaker_session,
        vpc_config= if(inherits(vpc_config, "list")) list(vpc_config) else vpc_config,
        dependencies=list(dependencies %||% self$dependencies),
        kwargs)
     return(do.call(ChainerModel$new, kwargs))
    }
  ),
  private = list(

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

      for (argument in list(
        self$.use_mpi,
        self$.num_processes,
        self$.process_slots_per_host,
        self$.additional_mpi_options)){

        value = init_params$hyperparameters[[argument]]
        init_params$hyperparameters[[argument]] = NULL

        if (!islistempty(value)){
          init_params[[gsub("sagemaker_", "", argument)]] = value
        }
      }
      image_uri = init_params$image_uri
      init_params$image_uri = NULL
      img_split = framework_name_from_image(image_uri)
      names(img_split) = c("framework", "py_version", "tag", "scriptmode")

      if (is.null(img_split$tag))
        img_split$framework_version = NULL
      else
        img_split$framework_version = framework_version_from_tag(img_split$tag)
      init_params$framework_version = img_split$framework_version
      init_params$py_version = img_split$py_version

      if (islistempty(img_split$framework)){
        # If we were unable to parse the framework name from the image it is not one of our
        # officially supported images, in this case just add the image to the init params.
        init_params$image_uri = image_uri
        return(init_params)}

      if (img_split$framework != attr(self, "_framework_name"))
        stop(sprintf(
          "Training job: %s didn't use image for requested framework",
            job_details$TrainingJobName),
          call. = F)
      return (init_params)
    }
  ),
  lock_objects = F
)
