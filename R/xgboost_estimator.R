# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/xgboost/estimator.py

#' @include fw_registry.R
#' @include estimator.R
#' @include image_uris.R
#' @include fw_utils.R
#' @include session.R
#' @include vpc_utils.R
#' @include xgboost_default.R
#' @include xgboost_model.R
#' @include utils.R

#' @import R6
#' @import logger

#' @title XGBoost Class
#' @description Handle end-to-end training and deployment of XGBoost booster training or training using
#'              customer provided XGBoost entry point script.
#' @export
XGBoost = R6Class("XGBoost",
  inherit = Framework,
  public = list(

    #' @description This ``Estimator`` executes an XGBoost based SageMaker Training Job.
    #'              The managed XGBoost environment is an Amazon-built Docker container thatexecutes functions
    #'              defined in the supplied ``entry_point`` Python script.
    #'              Training is started by calling :meth:`~sagemaker.amazon.estimator.Framework.fit` on this
    #'              Estimator. After training is complete, calling
    #'              :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted SageMaker endpoint
    #'              and returns an :class:`~sagemaker.amazon.xgboost.model.XGBoostPredictor` instance that
    #'              can be used to perform inference against the hosted model.
    #'              Technical documentation on preparing XGBoost scripts for SageMaker training and using the
    #'              XGBoost Estimator is available on the project home-page:
    #'              https://github.com/aws/sagemaker-python-sdk
    #' @param entry_point (str): Path (absolute or relative) to the Python source file which should
    #'              be executed as the entry point to training.  If ``source_dir`` is specified,
    #'              then ``entry_point`` must point to a file located at the root of ``source_dir``.
    #' @param framework_version (str): XGBoost version you want to use for executing your model
    #'              training code.
    #' @param source_dir (str): Path (absolute, relative or an S3 URI) to a directory
    #'              with any other training source code dependencies aside from the entry
    #'              point file (default: None). If ``source_dir`` is an S3 URI, it must
    #'              point to a tar.gz file. Structure within this directory are preserved
    #'              when training on Amazon SageMaker.
    #' @param hyperparameters (dict): Hyperparameters that will be used for training (default: None).
    #'              The hyperparameters are made accessible as a dict[str, str] to the training code
    #'              on SageMaker. For convenience, this accepts other types for keys and values, but
    #'              ``str()`` will be called to convert them before training.
    #' @param py_version (str): Python version you want to use for executing your model
    #'              training code (default: 'py3').
    #' @param image_uri (str): If specified, the estimator will use this image for training and
    #'              hosting, instead of selecting the appropriate SageMaker official image
    #'              based on framework_version and py_version. It can be an ECR url or
    #'              dockerhub image and tag.
    #'              Examples:
    #'              123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
    #'              custom-image:latest.
    #' @param ... : Additional kwargs passed to the
    #'              :class:`~sagemaker.estimator.Framework` constructor.
    initialize = function(entry_point,
                          framework_version,
                          source_dir=NULL,
                          hyperparameters=NULL,
                          py_version="py3",
                          image_uri=NULL,
                          ...){
      super$initialize(
        entry_point, source_dir, hyperparameters, image_uri=image_uri, ...
      )

      kwargs = list(...)

      self$py_version = py_version
      self$framework_version = framework_version

      if (is.null(image_uri)){
        self.image_uri = ImageUris$new()$retrieve(
          self$.framework_name,
          self$sagemaker_session$paws_region_name,
          version=framework_version,
          py_version=self$py_version,
          instance_type=kwargs$instance_type,
          image_scope="training"
        )
      }
      .framework_name = XGBOOST_NAME
    },

    #' @description Create a SageMaker ``XGBoostModel`` object that can be deployed to an ``Endpoint``.
    #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also used
    #'              during transform jobs. If not specified, the role from the Estimator will be used.
    #' @param model_server_workers (int): Optional. The number of worker processes used by the
    #'              inference server. If None, server will use one worker per vCPU.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the
    #'              model.
    #'              Default: use subnets and security groups from this Estimator.
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
    #' @param ... : Additional kwargs passed to the :class:`~sagemaker.xgboost.model.XGBoostModel`
    #'              constructor.
    #' @return sagemaker.xgboost.model.XGBoostModel: A SageMaker ``XGBoostModel`` object.
    #'              See :func:`~sagemaker.xgboost.model.XGBoostModel` for full details.
    create_model = function(model_server_workers=NULL,
                            role=NULL,
                            vpc_config_override=VPC_CONFIG_DEFAULT,
                            entry_point=NULL,
                            source_dir=NULL,
                            dependencies=NULL,
                            ...){
      role = role %||% self$role
      kwargs = list(...)
      kwargs$name = private$.get_or_create_name(kwargs$name)

      if (!("image_uri" %in% names(kwargs)))
        kwargs$image_uri = self$image_uri

      kwargs$model_data = self$model_data
      kwargs$role = role
      kwargs$entry_point = entry_point %||% private$.model_entry_point()
      kwargs$framework_version = self$framework_version
      kwargs$source_dir = source_dir %||% private$.model_source_dir()
      kwargs$container_log_level = self$container_log_level
      kwargs$code_location = self$code_location
      kwargs$py_version = self$py_version
      kwargs$model_server_workers = model_server_workers
      kwargs$sagemaker_session = self$sagemaker_session
      kwargs$vpc_config = self$get_vpc_config(vpc_config_override)
      kwargs$dependencies = dependencies %||% self$dependencies

      return (do.call(XGBoostModel$new, kwargs))
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

      job_details = sagemaker_session$sagemaker_client$describe_training_job(
        TrainingJobName=training_job_name
      )
      init_params = private$.prepare_init_params_from_job_description(job_details, model_channel_name)
      tags = sagemaker_session$sagemaker_client$list_tags(
        ResourceArn=job_details$TrainingJobArn
      )$Tags
      init_params[["tags"]] = tags
      init_params$sagemaker_session = sagemaker_session

      # clone current class
      estimator = self$clone()
      do.call(estimator$initialize, init_params)

      # update estimator class variables
      estimator$latest_training_job = init_params$base_job_name
      estimator$.current_job_name = estimator$latest_training_job

      estimator$wait()

      UploadedCode$s3_prefix=estimator$source_dir
      UploadedCode$script_name= estimator$entry_point
      estimator$uploaded_code = UploadedCode

      return(estimator)
    }
  ),
  private = list(

    # Convert the job description to init params that can be handled by the class constructor
    # Args:
    #     job_details: the returned job details from a describe_training_job API call.
    # Returns:
    #     dictionary: The transformed init_params
    .prepare_init_params_from_job_description = function(job_details, model_channel_name=NULL){
      init_params = super$.prepare_init_params_from_job_description(job_details)

      image_uri = init_params$image_uri
      init_params$image_uri = NULL

      fw_name= framework_name_from_image(image_uri)
      init_params$py_version = fw_name[[2]]

      if (!is.null(fw_name[[1]]) && fw_name[[1]] != attr(self, "_framework_name"))
        stop(sprintf("Training job: %s didn't use image for requested framework",
            job_details$TrainingJobName
          ), call. = F
        )

      init_params$framework_version = framework_version_from_tag(fw_name[[3]])

      if (islistempty(fw_name[[1]])){
        # If we were unable to parse the framework name from the image it is not one of our
        # officially supported images, in this case just add the image to the init params.
        init_params$image_uri = image_uri}
      return(init_params)
    }
  ),
  lock_objects = F
)
