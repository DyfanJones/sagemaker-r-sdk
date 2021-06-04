# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/mxnet/estimator.py

#' @include mxnet_default.R
#' @include serializers.R
#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common
#' @import lgr

#' @title MXNetPredictor Class
#' @description A Predictor for inference against MXNet Endpoints.
#'              This is able to serialize Python lists, dictionaries, and numpy arrays to
#'              multidimensional tensors for MXNet inference.
#' @export
MXNetPredictor = R6Class("MXNetPredictor",
  inherit = Predictor,
  public = list(

    #' @description Initialize an ``MXNetPredictor``.
    #' @param endpoint_name (str): The name of the endpoint to perform inference
    #'              on.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    initialize = function(endpoint_name,
                          sagemaker_session=NULL){
      super$initialize(
        endpoint_name, sagemaker_session, JSONSerializer$new(), JSONDeserializer$new()
      )
    }
  ),
  lock_objects = F
)

#' @title MXNetModel Class
#' @description An MXNet SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``.
#' @export
MXNetModel = R6Class("MXNetModel",
  inherit = R6sagemaker.common::FrameworkModel,
  public = list(

    #' @field .LOWEST_MMS_VERSION
    #' Lowest Multi Model Server MXNet version that can be executed
    .LOWEST_MMS_VERSION = "1.4.0",

    #' @description Initialize an MXNetModel.
    #' @param model_data (str): The S3 location of a SageMaker model data
    #'              ``.tar.gz`` file.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if it needs to access an AWS resource.
    #' @param entry_point (str): Path (absolute or relative) to the Python source
    #'              file which should be executed as the entry point to model
    #'              hosting. If ``source_dir`` is specified, then ``entry_point``
    #'              must point to a file located at the root of ``source_dir``.
    #' @param framework_version (str): MXNet version you want to use for executing
    #'              your model training code. Defaults to ``None``. Required unless
    #'              ``image_uri`` is provided.
    #' @param py_version (str): Python version you want to use for executing your
    #'              model training code. Defaults to ``None``. Required unless
    #'              ``image_uri`` is provided.
    #' @param image_uri (str): A Docker image URI (default: None). If not specified, a
    #'              default image for MXNet will be used.
    #'              If ``framework_version`` or ``py_version`` are ``None``, then
    #'              ``image_uri`` is required. If also ``None``, then a ``ValueError``
    #'              will be raised.
    #' @param predictor_cls (callable[str, sagemaker.session.Session]): A function
    #'              to call to create a predictor with an endpoint name and
    #'              SageMaker ``Session``. If specified, ``deploy()`` returns the
    #'              result of invoking this function on the created endpoint name.
    #' @param model_server_workers (int): Optional. The number of worker processes
    #'              used by the inference server. If None, server will use one
    #'              worker per vCPU.
    #' @param ... : Keyword arguments passed to the superclass
    #'              :class:`~sagemaker.model.FrameworkModel` and, subsequently, its
    #'              superclass :class:`~sagemaker.model.Model`.
    initialize = function(model_data,
                          role,
                          entry_point,
                          framework_version=NULL,
                          py_version=NULL,
                          image_uri=NULL,
                          predictor_cls=MXNetPredictor,
                          model_server_workers=NULL,
                          ...){
      validate_version_or_image_args(framework_version, py_version, image_uri)

      self$framework_version = framework_version
      self$py_version = py_version

      super$initialize(
        model_data=model_data,
        image_uri=image_uri,
        role=role,
        entry_point=entry_point,
        predictor_cls=predictor_cls,
        ...)

      self$model_server_workers = model_server_workers
      attr(self, "_framework_name") = "mxnet"

      if (identical(py_version, "py2"))
        LOGGER$warn(
          python_deprecation_warning(attr(self, "_framework_name"), MXNET_LATEST_PY2_VERSION)
        )
    },

    #' @description Return a container definition with framework configuration set in
    #'              model environment variables.
    #' @param instance_type (str): The EC2 instance type to deploy this Model to.
    #'              For example, 'ml.p2.xlarge'.
    #' @param accelerator_type (str): The Elastic Inference accelerator type to
    #'              deploy to the instance for loading and making inferences to the
    #'              model. For example, 'ml.eia1.medium'.
    #' @return dict[str, str]: A container definition object usable with the
    #'              CreateModel API.
    prepare_container_def = function(instance_type=NULL,
                                     accelerator_type=NULL){
      deploy_image = self$image_uri
      if (is.null(deploy_image)){
        if (is.null(instance_type))
          stop(
            "Must supply either an instance type (for choosing CPU vs GPU) or an image URI.",
            call. = F)

        region_name = self$sagemaker_session$paws_region_name
        deploy_image = self$serving_image_uri(
          region_name, instance_type, accelerator_type=accelerator_type)
      }

      deploy_key_prefix = model_code_key_prefix(self$key_prefix, self$name, deploy_image)
      private$.upload_code(deploy_key_prefix, private$.is_mms_version())
      deploy_env = self$env
      deploy_env = c(deploy_env, private$.framework_env_vars())

      if (!islistempty(self$model_server_workers))
        deploy_env[[toupper(MODEL_SERVER_WORKERS_PARAM_NAME)]] = as.character(self$model_server_workers)
      return (container_def(
        deploy_image,
        self$repacked_model_data %||% self$model_data,
        deploy_env)
      )
    },

    #' @description Create a URI for the serving image.
    #' @param region_name (str): AWS region where the image is uploaded.
    #' @param instance_type (str): SageMaker instance type. Used to determine device type
    #'              (cpu/gpu/family-specific optimized).
    #' @param accelerator_type (str): The Elastic Inference accelerator type to
    #'              deploy to the instance for loading and making inferences to the
    #'              model (default: None). For example, 'ml.eia1.medium'.
    #' @return str: The appropriate image URI based on the given parameters.
    serving_image_uri = function(region_name,
                                 instance_type,
                                 accelerator_type=NULL){
      return(ImageUris$new()$retrieve(
        attr(self, "_framework_name"),
        region_name,
        version=self$framework_version,
        py_version=self$py_version,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        image_scope="inference")
      )
    }
  ),
  private = list(

    # Whether the framework version corresponds to an inference image using
    # the Multi-Model Server (https://github.com/awslabs/multi-model-server).
    # Returns:
    #   bool: If the framework version corresponds to an image using MMS.
    .is_mms_version=function(){
      lowest_mms_version = package_version(self$.LOWEST_MMS_VERSION)
      framework_version = package_version(self$framework_version)
      return (framework_version >= lowest_mms_version)
    }
  ),
  lock_objects = F
)
