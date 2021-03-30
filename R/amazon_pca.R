# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/pca.py

#' @include image_uris.R
#' @include amazon_estimator.R
#' @include amazon_common.R
#' @include amazon_hyperparameter.R
#' @include amazon_validation.R
#' @include predictor.R
#' @include model.R
#' @include session.R
#' @include vpc_utils.R
#' @include utils.R

#' @import R6
#' @import lgr

#' @title An unsupervised machine learning algorithm to reduce feature dimensionality.
#' @description As a result, number of features within a dataset is reduced but the dataset still
#'              retain as much information as possible.
#' @export
PCA = R6Class("PCA",
  inherit = AmazonAlgorithmEstimatorBase,
  public = list(

    #' @field repo_name
    #' sagemaker repo name for framework
    repo_name = "pca",

    #' @field repo_version
    #' version of framework
    repo_version = 1,

    #' @field DEFAULT_MINI_BATCH_SIZE
    #' The size of each mini-batch to use when training.
    DEFAULT_MINI_BATCH_SIZE = 500,

    #' @description A Principal Components Analysis (PCA)
    #'              :class:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase`.
    #'              This Estimator may be fit via calls to
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit_ndarray`
    #'              or
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
    #'              The former allows a PCA model to be fit on a 2-dimensional numpy array.
    #'              The latter requires Amazon :class:`~sagemaker.amazon.record_pb2.Record`
    #'              protobuf serialized data to be stored in S3.
    #'              To learn more about the Amazon protobuf Record class and how to
    #'              prepare bulk data in this format, please consult AWS technical
    #'              documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html
    #'              After this Estimator is fit, model data is stored in S3. The model
    #'              may be deployed to an Amazon SageMaker Endpoint by invoking
    #'              :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
    #'              deploying an Endpoint, deploy returns a
    #'              :class:`~sagemaker.amazon.pca.PCAPredictor` object that can be used to
    #'              project input vectors to the learned lower-dimensional representation,
    #'              using the trained PCA model hosted in the SageMaker Endpoint.
    #'              PCA Estimators can be configured by setting hyperparameters. The
    #'              available hyperparameters for PCA are documented below. For further
    #'              information on the AWS PCA algorithm, please consult AWS technical
    #'              documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/pca.html
    #'              This Estimator uses Amazon SageMaker PCA to perform training and host
    #'              deployed models. To learn more about Amazon SageMaker PCA, please read:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/how-pca-works.html
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if accessing AWS resource.
    #' @param instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.c4.xlarge'.
    #' @param num_components (int): The number of principal components. Must be
    #'              greater than zero.
    #' @param algorithm_mode (str): Mode for computing the principal components.
    #'              One of 'regular' or 'randomized'.
    #' @param subtract_mean (bool): Whether the data should be unbiased both
    #'              during train and at inference.
    #' @param extra_components (int): As the value grows larger, the solution
    #'              becomes more accurate but the runtime and memory consumption
    #'              increase linearly. If this value is unset or set to -1, then a
    #'              default value equal to the maximum of 10 and num_components will
    #'              be used. Valid for randomized mode only.
    #' @param ... : base class keyword argument values.
    initialize = function(role,
                          instance_count,
                          instance_type,
                          num_components,
                          algorithm_mode=NULL,
                          subtract_mean=NULL,
                          extra_components=NULL,
                          ...){

      private$.num_components = Hyperparameter$new("num_components", Validation$new()$gt(0), "Value must be an integer greater than zero", DataTypes$new()$int, obj = self)
      private$.algorithm_mode = Hyperparameter$new(
        "algorithm_mode",
        Validation$new()$isin(c("regular", "randomized")),
        'Value must be one of "regular" and "randomized"',
        DataTypes$new()$str,
        obj = self
      )
      private$.subtract_mean = Hyperparameter$new(
        name="subtract_mean", validation_message="Value must be a boolean", data_type=DataTypes$new()$bool, obj = self
      )
      private$.extra_components = Hyperparameter$new(
        name="extra_components",
        validation_message="Value must be an integer greater than or equal to 0, or -1.",
        data_type=DataTypes$new()$int,
        obj = self
      )

      super$initialize(role, instance_count, instance_type, ...)
      self$num_components = num_components
      self$algorithm_mode = algorithm_mode
      self$subtract_mean = subtract_mean
      self$extra_components = extra_components
    },

    #' @description Return a :class:`~sagemaker.amazon.pca.PCAModel` referencing the
    #'              latest s3 model data produced by this Estimator.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
    #'              the model. Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param ... : Additional kwargs passed to the PCAModel constructor.
    create_model = function(vpc_config_override="VPC_CONFIG_DEFAULT",
                            ...){
      return(PCAModel$new(
        self$model_data,
        self$role,
        sagemaker_session=self$sagemaker_session,
        vpc_config=self$get_vpc_config(vpc_config_override),
        ...
        )
      )
    },

    #' @description Set hyperparameters needed for training.
    #' @param records (:class:`~RecordSet`): The records to train this ``Estimator`` on.
    #' @param mini_batch_size (int or None): The size of each mini-batch to use when
    #'              training. If ``None``, a default value will be used.
    #' @param job_name (str): Name of the training job to be created. If not
    #'              specified, one is generated, using the base name given to the
    #'              constructor if applicable.
    .prepare_for_training = function(records,
                                     mini_batch_size=NULL,
                                     job_name=NULL){
      num_records = NULL
      if (inherits(records, "list")){
        for (record in records){
          if (record$channel == "train"){
            num_records = record$num_records
            break}
          if (is.null(num_records))
            stop("Must provide train channel.", call. = F)
          }
        } else {
          num_records = records$num_records}
      # mini_batch_size is a required parameter
      default_mini_batch_size = min(
        self$DEFAULT_MINI_BATCH_SIZE, max(1, as.integer(num_records / self$instance_count))
        )
      use_mini_batch_size = mini_batch_size %||% default_mini_batch_size
      super$.prepare_for_training(
        records=records, mini_batch_size=use_mini_batch_size, job_name=job_name
        )
    }
  ),
  private = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    .num_components=NULL,
    .algorithm_mode=NULL,
    .subtract_mean=NULL,
    .extra_components=NULL
  ),
  active = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    #' @field num_components
    #' The number of principal components. Must be greater than zero.
    num_components = function(value){
      if(missing(value))
        return(private$.num_components$descriptor)
      private$.num_components$descriptor = value
    },

    #' @field algorithm_mode
    #' Mode for computing the principal components.
    algorithm_mode = function(value){
      if(missing(value))
        return(private$.algorithm_mode$descriptor)
      private$.algorithm_mode$descriptor = value
    },

    #' @field subtract_mean
    #' Whether the data should be unbiased both during train and at inference.
    subtract_mean = function(value){
      if(missing(value))
        return(private$.subtract_mean$descriptor)
      private$.subtract_mean$descriptor = value
    },

    #' @field extra_components
    #' As the value grows larger, the solution becomes more accurate
    #'        but the runtime and memory consumption increase linearly.
    extra_components = function(value){
      if(missing(value))
        return(private$.extra_components$descriptor)
      private$.extra_components$descriptor = value
    }
  ),
  lock_objects = F
)

#' @title Transforms input vectors to lower-dimesional representations.
#' @description The implementation of
#'              :meth:`~sagemaker.predictor.Predictor.predict` in this
#'              `Predictor` requires a numpy ``ndarray`` as input. The array should
#'              contain the same number of columns as the feature-dimension of the data used
#'              to fit the model this Predictor performs inference on.
#'              :meth:`predict()` returns a list of
#'              :class:`~sagemaker.amazon.record_pb2.Record` objects, one for each row in
#'              the input ``ndarray``. The lower dimension vector result is stored in the
#'              ``projection`` key of the ``Record.label`` field.
#' @export
PCAPredictor = R6Class("PCAPredictor",
  inherit = Predictor,
  public = list(

    #' @description Initialize PCAPredictor Class
    #' @param endpoint_name (str): Name of the Amazon SageMaker endpoint to which
    #'              requests are sent.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    initialize = function(endpoint_name,
                          sagemaker_session=NULL){
      super$initialize(
        endpoint_name,
        sagemaker_session,
        serializer=RecordSerializer$new(),
        deserializer=RecordDeserializer$new()
      )
    }
  ),
  lock_objects = F
)

#' @title Reference PCA s3 model data.
#' @description Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return a
#'              Predictor that transforms vectors to a lower-dimensional representation.
#' @export
PCAModel = R6Class("PCAModel",
  inherit = Model,
  public = list(

    #' @description initialize PCAModel Class
    #' @param model_data (str): The S3 location of a SageMaker model data
    #'              ``.tar.gz`` file.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if it needs to access an AWS resource.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    #' @param ... : Keyword arguments passed to the ``FrameworkModel``
    #'              initializer.
    initialize = function(model_data,
                          role,
                          sagemaker_session=NULL,
                          ...){
      sagemaker_session = sagemaker_session %||% Session$new()
      image_uri = ImageUris$new()$retrieve(
        PCA$public_fields$repo_name,
        sagemaker_session$paws_region_name,
        version=PCA$public_fields$repo_version
      )
      super$initialize(
        image_uri,
        model_data,
        role,
        predictor_cls=PCAPredictor,
        sagemaker_session=sagemaker_session,
        ...
      )
    }
  ),
  lock_objects = F
)
