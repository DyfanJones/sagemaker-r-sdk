# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/xgboost/model.py

#' @include deserializers.R
#' @include serializers.R
#' @include xgboost_default.R

#' @import R6
#' @import R6sagemaker.common
#' @import lgr

#' @title XGBoostPredictor Class
#' @description Predictor for inference against XGBoost Endpoints.
#'              This is able to serialize Python lists, dictionaries, and numpy arrays to xgb.DMatrix
#'              for XGBoost inference.
#' @export
XGBoostPredictor = R6Class("XGBoostPredictor",
  inherit = Predictor,
  public = list(

    #' @description Initialize an ``XGBoostPredictor``.
    #' @param endpoint_name (str): The name of the endpoint to perform inference on.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which manages
    #'              interactions with Amazon SageMaker APIs and any other AWS services needed.
    #'              If not specified, the estimator creates one using the default AWS configuration
    #'              chain.
    initialize = function(endpoint_name,
                          sagemaker_session=NULL){
      super$initialize(
        endpoint_name, sagemaker_session, LibSVMSerializer$new(), CSVDeserializer$new()
        )
    }
  )
)

#' @title XGBoostModel Class
#' @description An XGBoost SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``.
#' @export
XGBoostModel = R6Class("XGBoostModel",
  inherit = R6sagemaker.common::FrameworkModel,
  public = list(

    #' @description Initialize an XGBoostModel.
    #' @param model_data (str): The S3 location of a SageMaker model data ``.tar.gz`` file.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training
    #'              jobs and APIs that create Amazon SageMaker endpoints use this role to access
    #'              training data and model artifacts. After the endpoint is created, the inference
    #'              code might use the IAM role, if it needs to access an AWS resource.
    #' @param entry_point (str): Path (absolute or relative) to the Python source file which should
    #'              be executed  as the entry point to model hosting. If ``source_dir`` is specified,
    #'              then ``entry_point`` must point to a file located at the root of ``source_dir``.
    #' @param image_uri (str): A Docker image URI (default: None). If not specified, a default image
    #'              for XGBoost is be used.
    #' @param py_version (str): Python version you want to use for executing your model training code
    #'              (default: 'py3').
    #' @param framework_version (str): XGBoost version you want to use for executing your model
    #'              training code.
    #' @param predictor_cls (callable[str, sagemaker.session.Session]): A function to call to create
    #'              a predictor with an endpoint name and SageMaker ``Session``.
    #'              If specified, ``deploy()`` returns the result of invoking this function on the
    #'              created endpoint name.
    #' @param model_server_workers (int): Optional. The number of worker processes used by the
    #'              inference server. If None, server will use one worker per vCPU.
    #' @param ... : Keyword arguments passed to the ``FrameworkModel`` initializer.
    initialize = function(model_data,
                          role,
                          entry_point,
                          framework_version,
                          image_uri=NULL,
                          py_version="py3",
                          predictor_cls=XGBoostPredictor,
                          model_server_workers=NULL,
                          ...){

      super$initialize(
        model_data=model_data, image_uri=image_uri, role=role, entry_point=entry_point, predictor_cls=predictor_cls, ...
      )

      self$py_version = py_version
      self$framework_version = framework_version
      self$model_server_workers = model_server_workers

      attr(self, "_framework_name") = XGBOOST_NAME
    },

    #' @description Return a container definition with framework configuration
    #'              set in model environment variables.
    #' @param instance_type (str): The EC2 instance type to deploy this Model to.
    #'              This parameter is unused because XGBoost supports only CPU.
    #' @param accelerator_type (str): The Elastic Inference accelerator type to deploy to the
    #'              instance for loading and making inferences to the model. This parameter is
    #'              unused because accelerator types are not supported by XGBoostModel.
    #' @return dict[str, str]: A container definition object usable with the CreateModel API.
    prepare_container_def = function(instance_type=NULL,
                                     accelerator_type=NULL){
      deploy_image = self$image_uri
      if (is.null(deploy_image)){
        deploy_image = self$serving_image_uri(
          self$sagemaker_session$paws_region_name, instance_type
        )
      }

      deploy_key_prefix = model_code_key_prefix(self$key_prefix, self$name, deploy_image)
      private$.upload_code(deploy_key_prefix)
      deploy_env = list(self$env)
      deploy_env = c(deploy_env, private$.framework_env_vars())

      if (!is.null(self$model_server_workers))
        deploy_env[[toupper(MODEL_SERVER_WORKERS_PARAM_NAME)]] = as.character(self$model_server_workers)
      return(container_def(deploy_image, self$model_data, deploy_env))
    },

    #' @description Create a URI for the serving image.
    #' @param region_name (str): AWS region where the image is uploaded.
    #' @param instance_type (str): SageMaker instance type. Must be a CPU instance type.
    #' @return str: The appropriate image URI based on the given parameters.
    serving_image_uri = function(region_name,
                                 instance_type){
      if(missing(region_name)) region_name = self$sagemaker_session$paws_region_name
      return(ImageUris$new()$retrieve(
        attr(self, "_framework_name"),
        region_name,
        version=self$framework_version,
        py_version=self$py_version,
        instance_type=instance_type,
        image_scope="inference"
        )
      )
    }
  ),
  lock_objects = F
)
