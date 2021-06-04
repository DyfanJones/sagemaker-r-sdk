# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/lda.py

#' @include amazon_estimator.R
#' @include amazon_common.R
#' @include amazon_hyperparameter.R
#' @include amazon_validation.R
#' @include predictor.R
#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common
#' @import lgr

#' @title An unsupervised learning algorithm attempting to describe data as distinct categories.
#' @description LDA is most commonly used to discover a
#'              user-specified number of topics shared by documents within a text corpus. Here each
#'              observation is a document, the features are the presence (or occurrence count) of each
#'              word, and the categories are the topics.
#' @export
LDA = R6Class("LDA",
  inherit = AmazonAlgorithmEstimatorBase,
  public = list(

    #' @field repo_name
    #' sagemaker repo name for framework
    repo_name = "lda",

    #' @field repo_version
    #' version of framework
    repo_version = 1,

    #' @description Latent Dirichlet Allocation (LDA) is :class:`Estimator` used for
    #'              unsupervised learning.
    #'              Amazon SageMaker Latent Dirichlet Allocation is an unsupervised
    #'              learning algorithm that attempts to describe a set of observations as a
    #'              mixture of distinct categories. LDA is most commonly used to discover a
    #'              user-specified number of topics shared by documents within a text
    #'              corpus. Here each observation is a document, the features are the
    #'              presence (or occurrence count) of each word, and the categories are the
    #'              topics.
    #'              This Estimator may be fit via calls to
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
    #'              It requires Amazon :class:`~sagemaker.amazon.record_pb2.Record` protobuf
    #'              serialized data to be stored in S3. There is an utility
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.record_set`
    #'              that can be used to upload data to S3 and creates
    #'              :class:`~sagemaker.amazon.amazon_estimator.RecordSet` to be passed to
    #'              the `fit` call.
    #'              To learn more about the Amazon protobuf Record class and how to
    #'              prepare bulk data in this format, please consult AWS technical
    #'              documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html
    #'              After this Estimator is fit, model data is stored in S3. The model
    #'              may be deployed to an Amazon SageMaker Endpoint by invoking
    #'              :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
    #'              deploying an Endpoint, deploy returns a
    #'              :class:`~sagemaker.amazon.lda.LDAPredictor` object that can be used for
    #'              inference calls using the trained model hosted in the SageMaker
    #'              Endpoint.
    #'              LDA Estimators can be configured by setting hyperparameters. The
    #'              available hyperparameters for LDA are documented below.
    #'              For further information on the AWS LDA algorithm, please consult AWS
    #'              technical documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/lda.html
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if accessing AWS resource.
    #' @param instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.c4.xlarge'.
    #' @param num_topics (int): The number of topics for LDA to find within the
    #'              data.
    #' @param alpha0 (float): Optional. Initial guess for the concentration
    #'              parameter
    #' @param max_restarts (int): Optional. The number of restarts to perform
    #'              during the Alternating Least Squares (ALS) spectral
    #'              decomposition phase of the algorithm.
    #' @param max_iterations (int): Optional. The maximum number of iterations to
    #'              perform during the ALS phase of the algorithm.
    #' @param tol (float): Optional. Target error tolerance for the ALS phase of
    #'              the algorithm.
    #' @param ... : base class keyword argument values.
    initialize = function(role,
                          instance_type,
                          num_topics,
                          alpha0=NULL,
                          max_restarts=NULL,
                          max_iterations=NULL,
                          tol=NULL,
                          ...){
      private$.num_topics = Hyperparameter$new("num_topics", Validation$new()$gt(0), "An integer greater than zero", DataTypes$new()$int, obj = self)
      private$.alpha0 = Hyperparameter$new("alpha0", Validation$new()$gt(0), "A positive float", DataTypes$new()$float, obj = self)
      private$.max_restarts = Hyperparameter$new("max_restarts", Validation$new()$gt(0), "An integer greater than zero", DataTypes$new()$int, obj = self)
      private$.max_iterations = Hyperparameter$new("max_iterations", Validation$new()$gt(0), "An integer greater than zero", DataTypes$new()$int, obj = self)
      private$.tol = Hyperparameter$new("tol", Validation$new()$gt(0), "A positive float", DataTypes$new()$float, obj = self)

      args = list(...)

      # this algorithm only supports single instance training
      if ("instance_count" %in% names(args) && args$instance_count != 1){
        LOGGER$info("LDA only supports single instance training. Defaulting to 1 %s.",
            instance_type)
        args$instance_count = NULL
      }

      args=c(role = role,
             instance_count = 1,
             instance_type = instance_type,
             args)
      do.call(super$initialize, args)

      self$num_topics = num_topics
      self$alpha0 = alpha0
      self$max_restarts = max_restarts
      self$max_iterations = max_iterations
      self$tol = tol
    },

    #' @description Return a :class:`~sagemaker.amazon.LDAModel` referencing the latest
    #'              s3 model data produced by this Estimator.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for
    #'              VpcConfig set on the model.
    #'              Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param ... : Additional kwargs passed to the LDAModel constructor.
    create_model = function(vpc_config_override="VPC_CONFIG_DEFAULT",
                            ...){
      return(LDAModel$new(
        model_data = self$model_data,
        role = self$role,
        sagemaker_session=self$sagemaker_session,
        vpc_config=self$get_vpc_config(vpc_config_override),
        ...)
        )
    },

    #' @description Set hyperparameters needed for training. This method will also
    #'              validate ``source_dir``.
    #' @param records (RecordSet) – The records to train this Estimator on.
    #' @param mini_batch_size (int or None) – The size of each mini-batch to use
    #'              when training. If None, a default value will be used.
    #' @param job_name (str): Name of the training job to be created. If not
    #'              specified, one is generated, using the base name given to the
    #'              constructor if applicable.
    .prepare_for_training = function(records,
                                     mini_batch_size=NULL,
                                     job_name=NULL){
      if (is.null(mini_batch_size))
        stop("mini_batch_size must be set", call. = F)

      super$.prepare_for_training(
        records, mini_batch_size=mini_batch_size, job_name=job_name
      )
    }
  ),
  private = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    .num_topics = NULL,
    .alpha0 = NULL,
    .max_restarts = NULL,
    .max_iterations = NULL,
    .tol = NULL
  ),
  active = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    #' @field num_topics
    #' The number of topics for LDA to find within the data
    num_topics = function(value){
      if(missing(value))
        return(private$.num_topics$descriptor)
      private$.num_topics$descriptor = value
    },

    #' @field alpha0
    #' Initial guess for the concentration parameter
    alpha0 = function(value){
      if(missing(value))
        return(private$.alpha0$descriptor)
      private$.alpha0$descriptor = value
    },

    #' @field max_restarts
    #' The number of restarts to perform during the Alternating Least Squares
    max_restarts = function(value){
      if(missing(value))
        return(private$.max_restarts$descriptor)
      private$.max_restarts$descriptor = value
    },

    #' @field max_iterations
    #' The maximum number of iterations to perform during the ALS phase of the algorithm.
    max_iterations = function(value){
      if(missing(value))
        return(private$.max_iterations$descriptor)
      private$.max_iterations$descriptor = value
    },

    #' @field tol
    #' Target error tolerance for the ALS phase of the algorithm.
    tol = function(value){
      if(missing(value))
        return(private$.tol$descriptor)
      private$.tol$descriptor = value
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
LDAPredictor = R6Class("LDAPredictor",
  inherit = Predictor,
  public = list(

    #' @description Initialize LDAPredictor class
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

#' @title Reference LDA s3 model data  created by LDA estimator.
#' @description Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return a
#'              Predictor that transforms vectors to a lower-dimensional representation.
#' @export
LDAModel = R6Class("LDAModel",
  inherit = R6sagemaker.common::Model,
  public = list(

    #' @description Initialize LDAModel class
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
        LDA$public_fields$repo_name,
        sagemaker_session$paws_region_name,
        version=LDA$public_fields$repo_version
      )
      super$initialize(
        image_uri,
        model_data,
        role,
        predictor_cls=LDAPredictor,
        sagemaker_session=sagemaker_session,
        ...
      )
    }
  ),
  lock_objects = F
)
