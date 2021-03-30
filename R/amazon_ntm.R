# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/ntm.py

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

#' @title An unsupervised learning algorithm used to organize a corpus of documents into topics
#' @description The resulting topics contain word groupings based on their statistical distribution.
#'              Documents that contain frequent occurrences of words such as "bike", "car", "train",
#'              "mileage", and "speed" are likely to share a topic on "transportation" for example.
#' @export
NTM = R6Class("NTM",
  inherit =  AmazonAlgorithmEstimatorBase,
  public = list(

    #' @field repo_name
    #' sagemaker repo name for framework
    repo_name = "ntm",

    #' @field repo_version
    #' version of framework
    repo_version = 1,

    #' @description Neural Topic Model (NTM) is :class:`Estimator` used for unsupervised
    #'              learning.
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
    #'              :class:`~sagemaker.amazon.ntm.NTMPredictor` object that can be used for
    #'              inference calls using the trained model hosted in the SageMaker
    #'              Endpoint.
    #'              NTM Estimators can be configured by setting hyperparameters. The
    #'              available hyperparameters for NTM are documented below.
    #'              For further information on the AWS NTM algorithm, please consult AWS
    #'              technical documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/ntm.html
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if accessing AWS resource.
    #' @param instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.c4.xlarge'.
    #' @param num_topics (int): Required. The number of topics for NTM to find
    #'              within the data.
    #' @param encoder_layers (list): Optional. Represents number of layers in the
    #'              encoder and the output size of each layer.
    #' @param epochs (int): Optional. Maximum number of passes over the training
    #'              data.
    #' @param encoder_layers_activation (str): Optional. Activation function to
    #'              use in the encoder layers.
    #' @param optimizer (str): Optional. Optimizer to use for training.
    #' @param tolerance (float): Optional. Maximum relative change in the loss
    #'              function within the last num_patience_epochs number of epochs
    #'              below which early stopping is triggered.
    #' @param num_patience_epochs (int): Optional. Number of successive epochs
    #'              over which early stopping criterion is evaluated.
    #' @param batch_norm (bool): Optional. Whether to use batch normalization
    #'              during training.
    #' @param rescale_gradient (float): Optional. Rescale factor for gradient.
    #' @param clip_gradient (float): Optional. Maximum magnitude for each gradient
    #'              component.
    #' @param weight_decay (float): Optional. Weight decay coefficient. Adds L2
    #'              regularization.
    #' @param learning_rate (float): Optional. Learning rate for the optimizer.
    #' @param ... : base class keyword argument values.
    initialize = function(role,
                          instance_count,
                          instance_type,
                          num_topics,
                          encoder_layers=NULL,
                          epochs=NULL,
                          encoder_layers_activation=NULL,
                          optimizer=NULL,
                          tolerance=NULL,
                          num_patience_epochs=NULL,
                          batch_norm=NULL,
                          rescale_gradient=NULL,
                          clip_gradient=NULL,
                          weight_decay=NULL,
                          learning_rate=NULL,
                          ...){

      private$.num_topics = Hyperparameter$new("num_topics", list(Validation$new()$ge(2), Validation$new()$le(1000)), "An integer in [2, 1000]", DataTypes$new()$int, obj = self)
      private$.encoder_layers = Hyperparameter$new(
        name="encoder_layers",
        validation_message='A comma separated list of " "positive integers',
        data_type=as.list,
        obj = self
      )
      private$.epochs = Hyperparameter$new("epochs", list(Validation$new()$ge(1), Validation$new()$le(100)), "An integer in [1, 100]", DataTypes$new()$int, obj = self)
      private$.encoder_layers_activation = Hyperparameter$new(
        "encoder_layers_activation",
        Validation$new()$isin(c("sigmoid", "tanh", "relu")),
        'One of "sigmoid", "tanh" or "relu"',
        DataTypes$new()$str,
        obj = self
      )
      private$.optimizer = Hyperparameter$new(
        "optimizer",
        Validation$new()$isin(c("adagrad", "adam", "rmsprop", "sgd", "adadelta")),
        'One of "adagrad", "adam", "rmsprop", "sgd" and "adadelta"',
        DataTypes$new()$str,
        obj = self
      )
      private$.tolerance = Hyperparameter$new("tolerance", list(Validation$new()$ge(1e-6), Validation$new()$le(0.1)), "A float in [1e-6, 0.1]", DataTypes$new()$float, obj = self)
      private$.num_patience_epochs = Hyperparameter$new("num_patience_epochs", list(Validation$new()$ge(1), Validation$new()$le(10)), "An integer in [1, 10]", DataTypes$new()$int, obj = self)
      private$.batch_norm = Hyperparameter$new(name="batch_norm", validation_message="Value must be a boolean", data_type=DataTypes$new()$bool, obj = self)
      private$.rescale_gradient = Hyperparameter$new("rescale_gradient", list(Validation$new()$ge(1e-3), Validation$new()$le(1.0)), "A float in [1e-3, 1.0]", DataTypes$new()$float, obj = self)
      private$.clip_gradient = Hyperparameter$new("clip_gradient", Validation$new()$ge(1e-3), "A float greater equal to 1e-3", DataTypes$new()$float, obj = self)
      private$.weight_decay = Hyperparameter$new("weight_decay", list(Validation$new()$ge(0.0), Validation$new()$le(1.0)), "A float in [0.0, 1.0]", DataTypes$new()$float, obj = self)
      private$.learning_rate = Hyperparameter$new("learning_rate", list(Validation$new()$ge(1e-6), Validation$new()$le(1.0)), "A float in [1e-6, 1.0]", DataTypes$new()$float, obj = self)

      super$initialize(role, instance_count, instance_type, ...)
      self$num_topics = num_topics
      self$encoder_layers = encoder_layers
      self$epochs = epochs
      self$encoder_layers_activation = encoder_layers_activation
      self$optimizer = optimizer
      self$tolerance = tolerance
      self$num_patience_epochs = num_patience_epochs
      self$batch_norm = batch_norm
      self$rescale_gradient = rescale_gradient
      self$clip_gradient = clip_gradient
      self$weight_decay = weight_decay
      self$learning_rate = learning_rate
    },

    #' @description Return a :class:`~sagemaker.amazon.NTMModel` referencing the latest
    #'              s3 model data produced by this Estimator.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
    #'              the model. Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param ... : Additional kwargs passed to the NTMModel constructor.
    create_model = function(vpc_config_override="VPC_CONFIG_DEFAULT",
                            ...){
      return(NTMModel$new(
        self$model_data,
        self$role,
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
                                     mini_batch_size,
                                     job_name=NULL){
      if (!is.null(mini_batch_size) && (mini_batch_size < 1 | mini_batch_size > 10000))
        stop("mini_batch_size must be in [1, 10000]", call. = F)
      super$.prepare_for_training(
        records, mini_batch_size=mini_batch_size, job_name=job_name
      )
    }
  ),
  private = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    .num_topics=NULL,
    .encoder_layers=NULL,
    .epochs=NULL,
    .encoder_layers_activation=NULL,
    .optimizer=NULL,
    .tolerance=NULL,
    .num_patience_epochs=NULL,
    .batch_norm=NULL,
    .rescale_gradient=NULL,
    .clip_gradient=NULL,
    .weight_decay=NULL,
    .learning_rate=NULL
  ),
  active = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    #' @field num_topics
    #' The number of topics for NTM to find within the data
    num_topics = function(value){
      if(missing(value))
        return(private$.num_topics$descriptor)
      private$.num_topics$descriptor = value
    },

    #' @field encoder_layers
    #' Represents number of layers in the encoder and the output size of each layer
    encoder_layers = function(value){
      if(missing(value))
        return(private$.encoder_layers$descriptor)
      private$.encoder_layers$descriptor = value
    },

    #' @field epochs
    #' Maximum number of passes over the training data.
    epochs = function(value){
      if(missing(value))
        return(private$.epochs$descriptor)
      private$.epochs$descriptor = value
    },

    #' @field encoder_layers_activation
    #' Activation function to use in the encoder layers.
    encoder_layers_activation = function(value){
      if(missing(value))
        return(private$.encoder_layers_activation$descriptor)
      private$.encoder_layers_activation$descriptor = value
    },

    #' @field optimizer
    #' Optimizer to use for training.
    optimizer = function(value){
      if(missing(value))
        return(private$.optimizer$descriptor)
      private$.optimizer$descriptor = value
    },

    #' @field tolerance
    #' Maximum relative change in the loss function within the
    #'        last num_patience_epochs number of epochs below which
    #'        early stopping is triggered.
    tolerance = function(value){
      if(missing(value))
        return(private$.tolerance$descriptor)
      private$.tolerance$descriptor = value
    },

    #' @field num_patience_epochs
    #' Number of successive epochs over which early stopping criterion is evaluated.
    num_patience_epochs = function(value){
      if(missing(value))
        return(private$.num_patience_epochs$descriptor)
      private$.num_patience_epochs$descriptor = value
    },

    #' @field batch_norm
    #' Whether to use batch normalization during training.
    batch_norm = function(value){
      if(missing(value))
        return(private$.batch_norm$descriptor)
      private$.batch_norm$descriptor = value
    },

    #' @field rescale_gradient
    #' Rescale factor for gradient
    rescale_gradient = function(value){
      if(missing(value))
        return(private$.rescale_gradient$descriptor)
      private$.rescale_gradient$descriptor = value
    },

    #' @field clip_gradient
    #' Maximum magnitude for each gradient component.
    clip_gradient = function(value){
      if(missing(value))
        return(private$.clip_gradient$descriptor)
      private$.clip_gradient$descriptor = value
    },

    #' @field weight_decay
    #' Weight decay coefficient.
    weight_decay = function(value){
      if(missing(value))
        return(private$.weight_decay$descriptor)
      private$.weight_decay$descriptor = value
    },

    #' @field learning_rate
    #' Learning rate for the optimizer.
    learning_rate = function(value){
      if(missing(value))
        return(private$.learning_rate$descriptor)
      private$.learning_rate$descriptor = value
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
NTMPredictor = R6Class("NTMPredictor",
  inherit = Predictor,
  public = list(

    #' @description Initialize NTMPredictor class
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

#' @title Reference NTM s3 model data.
#' @description Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return a
#'              Predictor that transforms vectors to a lower-dimensional representation.
#' @export
NTMModel = R6Class("NTMModel",
  inherit = Model,
  public = list(

    #' @description Initialize NTMModel class
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
        NTM$public_fields$repo_name,
        sagemaker_session$paws_region_name,
        version=NTM$public_fields$repo_version
      )
      super$initialize(
        image_uri,
        model_data,
        role,
        predictor_cls=NTMPredictor,
        sagemaker_session=sagemaker_session,
        ...
      )
    }
  ),
  lock_objects = F
)
