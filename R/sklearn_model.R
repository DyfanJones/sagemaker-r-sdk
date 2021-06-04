# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/sklearn/model.py

#' @include deserializers.R
#' @include predictor.R
#' @include serializers.R
#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common
#' @import lgr

#' @title A Predictor for inference against Scikit-learn Endpoints.
#' @description This is able to serialize Python lists, dictionaries, and numpy arrays to
#'              multidimensional tensors for Scikit-learn inference.
#' @export
SKLearnPredictor = R6Class("SKLearnPredictor",
  inherit = Predictor,
  public = list(

    #' @description Initialize an ``SKLearnPredictor``.
    #' @param endpoint_name (str): The name of the endpoint to perform inference
    #'              on.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    initialize = function(endpoint_name,
                          sagemaker_session=NULL){
      super$initialize(
        endpoint_name, sagemaker_session, NumpySerializer$new(), NumpyDeserializer$new())
    }
  ),
  lock_objects = F
)

#' @title SKLearnModel Class
#' @description An Scikit-learn SageMaker ``Model`` that can be deployed to a SageMaker
#'              ``Endpoint``.
#' @export
SKLearnModel = R6Class("SKLearnModel",
  inherit = R6sagemaker.common::FrameworkModel,
  public = list(

    #' @description Initialize an SKLearnModel.
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
    #' @param framework_version (str): Scikit-learn version you want to use for
    #'              executing your model training code. Defaults to ``None``. Required
    #'              unless ``image_uri`` is provided.
    #' @param py_version (str): Python version you want to use for executing your
    #'              model training code (default: 'py3'). Currently, 'py3' is the only
    #'              supported version. If ``None`` is passed in, ``image_uri`` must be
    #'              provided.
    #' @param image_uri (str): A Docker image URI (default: None). If not specified, a
    #'              default image for Scikit-learn will be used.
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
    #' @param ... : Keyword arguments passed to the ``FrameworkModel``
    #'              initializer.
    initialize = function(model_data,
                          role,
                          entry_point,
                          framework_version=NULL,
                          py_version="py3",
                          image_uri=NULL,
                          predictor_cls=SKLearnPredictor,
                          model_server_workers=NULL,
                          ...){
      validate_version_or_image_args(framework_version, py_version, image_uri)
      if (!is.null(py_version) && py_version != "py3")
        stop("Scikit-learn image only supports Python 3. Please use 'py3' for py_version.",
             call. = F)
      self$framework_version = framework_version
      self$py_version = py_version

      super$initialize(
        model_data, image_uri, role, entry_point, predictor_cls=predictor_cls, ...)

      attr(self, "_framework_name") = "sklearn"

      self$model_server_workers = model_server_workers
    },

    #' @description Return a container definition with framework configuration set in
    #'              model environment variables.
    #' @param instance_type (str): The EC2 instance type to deploy this Model to.
    #'              This parameter is unused because Scikit-learn supports only CPU.
    #' @param accelerator_type (str): The Elastic Inference accelerator type to
    #'              deploy to the instance for loading and making inferences to the
    #'              model. This parameter is unused because accelerator types
    #'              are not supported by SKLearnModel.
    #' @return dict[str, str]: A container definition object usable with the
    #'              CreateModel API.
    prepare_container_def = function(instance_type=NULL,
                                     accelerator_type=NULL){
      if (!is.null(accelerator_type))
        stop("Accelerator types are not supported for Scikit-Learn.",
             call. = F)

      deploy_image = self$image_uri
      if (is.null(deploy_image))
        deploy_image = self$serving_image_uri(
          self$sagemaker_session$paws_region_name, instance_type
        )

      deploy_key_prefix = model_code_key_prefix(self$key_prefix, self$name, deploy_image)
      private$.upload_code(key_prefix=deploy_key_prefix, repack=self$enable_network_isolation())
      deploy_env = self$env
      deploy_env = c(deploy_env, private$.framework_env_vars())

      if (!is.null(self$model_server_workers))
        deploy_env[[toupper(MODEL_SERVER_WORKERS_PARAM_NAME)]] = as.character(self$model_server_workers)
      model_data_uri = if (self$enable_network_isolation()) self$repacked_model_data  else self$model_data
      return(container_def(deploy_image, model_data_uri, deploy_env))
    },

    #' @description Create a URI for the serving image.
    #' @param region_name (str): AWS region where the image is uploaded.
    #' @param instance_type (str): SageMaker instance type.
    #' @return str: The appropriate image URI based on the given parameters.
    serving_image_uri = function(region_name,
                                 instance_type){
      return(ImageUris$new()$retrieve(
        attr(self, "_framework_name"),
        region_name,
        version=self$framework_version,
        py_version=self$py_version,
        instance_type=instance_type)
      )
    }
  ),
  lock_objects = F
)
