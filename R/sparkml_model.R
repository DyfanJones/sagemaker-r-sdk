# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/sparkml/model.py

#' @include model.R
#' @include predictor.R
#' @include session.R
#' @include serializers.R
#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common
#' @import lgr
#' @importFrom urltools url_parse

#' @title Performs predictions against an MLeap serialized SparkML model.
#' @description The implementation of
#'              :meth:`~sagemaker.predictor.Predictor.predict` in this
#'              `Predictor` requires a json as input. The input should follow the
#'              json format as documented.
#'              ``predict()`` returns a csv output, comma separated if the output is a
#'              list.
#' @export
SparkMLPredictor = R6Class("SparkMLPredictor",
  inherit = Predictor,
  public = list(

    #' @description Initializes a SparkMLPredictor which should be used with SparkMLModel
    #'              to perform predictions against SparkML models serialized via MLeap. The
    #'              response is returned in text/csv format which is the default response
    #'              format for SparkML Serving container.
    #' @param endpoint_name (str): The name of the endpoint to perform inference on.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    #' @param serializer (sagemaker.serializers.BaseSerializer): Optional. Default
    #'              serializes input data to text/csv.
    #' @param ... : Additional parameters passed to the
    #'              :class:`~sagemaker.Predictor` constructor.
    initialize = function(endpoint_name,
                          sagemaker_session=NULL,
                          serializer=CSVSerializer$new(),
                          ...){
      sagemaker_session = sagemaker_session %||% Session$new()
      super$initialize(
        endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=serializer,
        ...)
    }
  ),
  lock_objects = F
)

#' @title SparkMLModel class
#' @description Model data and S3 location holder for MLeap serialized SparkML model.
#'              Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return
#'              a Predictor to performs predictions against an MLeap serialized SparkML
#'              model .
#' @export
SparkMLModel = R6Class("SparkMLModel",
  inherit = Model,
  public = list(

    #' @description Initialize a SparkMLModel.
    #' @param model_data (str): The S3 location of a SageMaker model data
    #'              ``.tar.gz`` file. For SparkML, this will be the output that has
    #'              been produced by the Spark job after serializing the Model via
    #'              MLeap.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if it needs to access an AWS resource.
    #' @param spark_version (str): Spark version you want to use for executing the
    #'              inference (default: '2.4').
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain. For local mode,
    #'              please do not pass this variable.
    #' @param ... : Additional parameters passed to the
    #'              :class:`~sagemaker.model.Model` constructor.
    initialize = function(model_data,
                         role=NULL,
                         spark_version=2.4,
                         sagemaker_session=NULL,
                         ...){
      # For local mode, sagemaker_session should be passed as None but we need a session to get
      # paws_region_name
      region_name = (sagemaker_session %||% Session$new())$paws_region_name
      image_uri = ImageUris$new()$retrieve("sparkml-serving", region_name, version=spark_version)
      super$initialize(
        image_uri,
        model_data,
        role,
        predictor_cls=SparkMLPredictor,
        sagemaker_session=sagemaker_session,
        ...)
     }
  ),
  lock_objects = F
)
