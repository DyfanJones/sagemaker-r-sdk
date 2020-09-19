# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/ipinsights.py

#' @include image_uris.R
#' @include amazon_estimator.R
#' @include amazon_hyperparameter.R
#' @include deserializers.R
#' @include model.R
#' @include serializers.R
#' @include session.R
#' @include vpc_utils.R

#' @import R6

#' @title An unsupervised learning algorithm that learns the usage patterns for IPv4 addresses.
#' @description It is designed to capture associations between IPv4 addresses and various entities, such
#'              as user IDs or account numbers.
#' @export
IPInsights = R6Class("IPInsights",
  inherit = AmazonAlgorithmEstimatorBase,
  public = list(

    #' @field repo_name
    #' sagemaker repo name for framework
    repo_name = "ipinsights",

    #' @field repo_version
    #' version of framework
    repo_version = 1,

    #' @field MINI_BATCH_SIZE
    #' The size of each mini-batch to use when training. If None, a default value will be used.
    MINI_BATCH_SIZE = 10000,

    #' @description This estimator is for IP Insights, an unsupervised algorithm that
    #'              learns usage patterns of IP addresses.
    #'              This Estimator may be fit via calls to
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
    #'              It requires CSV data to be stored in S3.
    #'              After this Estimator is fit, model data is stored in S3. The model
    #'              may be deployed to an Amazon SageMaker Endpoint by invoking
    #'              :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
    #'              deploying an Endpoint, deploy returns a
    #'              :class:`~sagemaker.amazon.IPInsightPredictor` object that can be used
    #'              for inference calls using the trained model hosted in the SageMaker
    #'              Endpoint.
    #'              IPInsights Estimators can be configured by setting hyperparamters.
    #'              The available hyperparamters are documented below.
    #'              For further information on the AWS IPInsights algorithm, please
    #'              consult AWS technical documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights-hyperparameters.html
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if accessing AWS resource.
    #' @param instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.m5.xlarge'.
    #' @param num_entity_vectors (int): Required. The number of embeddings to
    #'              train for entities accessing online resources. We recommend 2x
    #'              the total number of unique entity IDs.
    #' @param vector_dim (int): Required. The size of the embedding vectors for
    #'              both entity and IP addresses.
    #' @param batch_metrics_publish_interval (int): Optional. The period at which
    #'              to publish metrics (batches).
    #' @param epochs (int): Optional. Maximum number of passes over the training
    #'              data.
    #' @param learning_rate (float): Optional. Learning rate for the optimizer.
    #' @param num_ip_encoder_layers (int): Optional. The number of fully-connected
    #'              layers to encode IP address embedding.
    #' @param random_negative_sampling_rate (int): Optional. The ratio of random
    #'              negative samples to draw during training. Random negative
    #'              samples are randomly drawn IPv4 addresses.
    #' @param shuffled_negative_sampling_rate (int): Optional. The ratio of
    #'              shuffled negative samples to draw during training. Shuffled
    #'              negative samples are IP addresses picked from within a batch.
    #' @param weight_decay (float): Optional. Weight decay coefficient. Adds L2
    #'              regularization.
    #' @param ... : base class keyword argument values.
    initialize = function(role,
                          instance_count,
                          instance_type,
                          num_entity_vectors,
                          vector_dim,
                          batch_metrics_publish_interval=NULL,
                          epochs=NULL,
                          learning_rate=NULL,
                          num_ip_encoder_layers=NULL,
                          random_negative_sampling_rate=NULL,
                          shuffled_negative_sampling_rate=NULL,
                          weight_decay=NULL,
                          ...){

      private$.num_entity_vectors = Hyperparameter$new(
        "num_entity_vectors", list(Validation$new()$ge(1), Validation$new()$le(250000000)), "An integer in [1, 250000000]", DataTypes$new()$int, obj = self
      )
      private$.vector_dim = Hyperparameter$new("vector_dim", list(Validation$new()$ge(4), Validation$new()$le(4096)), "An integer in [4, 4096]", DataTypes$new()$int, obj = self)

      private$.batch_metrics_publish_interval = Hyperparameter$new(
        "batch_metrics_publish_interval", Validation$new()$ge(1), "An integer greater than 0", DataTypes$new()$int, obj = self
      )
      private$.epochs = Hyperparameter$new("epochs", Validation$new()$ge(1), "An integer greater than 0", DataTypes$new()$int, obj = self)
      private$.learning_rate = Hyperparameter$new("learning_rate", list(Validation$new()$ge(1e-6), Validation$new()$le(10.0)), "A float in [1e-6, 10.0]", DataTypes$new()$float, obj = self)
      private$.num_ip_encoder_layers = Hyperparameter$new(
        "num_ip_encoder_layers", list(Validation$new()$ge(0), Validation$new()$le(100)), "An integer in [0, 100]", DataTypes$new()$int, obj = self
      )
      private$.random_negative_sampling_rate = Hyperparameter$new(
        "random_negative_sampling_rate", list(Validation$new()$ge(0), Validation$new()$le(500)), "An integer in [0, 500]", DataTypes$new()$int, obj = self
      )
      private$.shuffled_negative_sampling_rate = Hyperparameter$new(
        "shuffled_negative_sampling_rate", list(Validation$new()$ge(0), Validation$new()$le(500)), "An integer in [0, 500]", DataTypes$new()$int, obj = self
      )
      private$.weight_decay = Hyperparameter$new("weight_decay", list(Validation$new()$ge(0.0), Validation$new()$le(10.0)), "A float in [0.0, 10.0]", DataTypes$new()$float, obj = self)

      super$initialize(role, instance_count, instance_type, ...)

      self$num_entity_vectors = num_entity_vectors
      self$vector_dim = vector_dim
      self$batch_metrics_publish_interval = batch_metrics_publish_interval
      self$epochs = epochs
      self$learning_rate = learning_rate
      self$num_ip_encoder_layers = num_ip_encoder_layers
      self$random_negative_sampling_rate = random_negative_sampling_rate
      self$shuffled_negative_sampling_rate = shuffled_negative_sampling_rate
      self$weight_decay = weight_decay
    },

    #' @description Create a model for the latest s3 model produced by this estimator.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
    #'              the model.
    #'              Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param ... : Additional kwargs passed to the IPInsightsModel constructor.
    #' @return :class:`~sagemaker.amazon.IPInsightsModel`: references the latest s3 model
    #'              data produced by this estimator.
    create_model = function(vpc_config_override="VPC_CONFIG_DEFAULT", ...){
      return(IPInsightsModel$new(
        self$model_data,
        self$role,
        sagemaker_session=self$sagemaker_session,
        vpc_config=self$get_vpc_config(vpc_config_override),
        ...
        )
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
      if (!is.null(mini_batch_size) && (mini_batch_size < 1 || mini_batch_size > 500000))
        stop("mini_batch_size must be in [1, 500000]", call. = F)
      super$.prepare_for_training(
        records, mini_batch_size=mini_batch_size, job_name=job_name
      )
    }
  ),
  private = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    .num_entity_vectors = NULL,
    .vector_dim = NULL,
    .batch_metrics_publish_interval = NULL,
    .epochs = NULL,
    .learning_rate = NULL,
    .num_ip_encoder_layers = NULL,
    .random_negative_sampling_rate = NULL,
    .shuffled_negative_sampling_rate = NULL,
    .weight_decay = NULL
  ),
  active = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    #' @field num_entity_vectors
    #' The number of embeddings to train for entities accessing online resources
    num_entity_vectors = function(value){
      if(missing(value))
        return(private$.num_entity_vectors$descriptor)
      private$.num_entity_vectors$descriptor = value
    },

    #' @field vector_dim
    #' The size of the embedding vectors for both entity and IP addresses
    vector_dim = function(value){
      if(missing(value))
        return(private$.vector_dim$descriptor)
      private$.vector_dim$descriptor = value
    },

    #' @field batch_metrics_publish_interval
    #' The period at which to publish metrics
    batch_metrics_publish_interval = function(value){
      if(missing(value))
        return(private$.batch_metrics_publish_interval$descriptor)
      private$.batch_metrics_publish_interval$descriptor = value
    },

    #' @field epochs
    #' Maximum number of passes over the training data.
    epochs = function(value){
      if(missing(value))
        return(private$.epochs$descriptor)
      private$.epochs$descriptor = value
    },

    #' @field learning_rate
    #' Learning rate for the optimizer.
    learning_rate = function(value){
      if(missing(value))
        return(private$.learning_rate$descriptor)
      private$.learning_rate$descriptor = value
    },

    #' @field num_ip_encoder_layers
    #' The number of fully-connected layers to encode IP address embedding.
    num_ip_encoder_layers = function(value){
      if(missing(value))
        return(private$.num_ip_encoder_layers$descriptor)
      private$.num_ip_encoder_layers$descriptor = value
    },

    #' @field random_negative_sampling_rate
    #' The ratio of random negative samples to draw during training.
    random_negative_sampling_rate = function(value){
      if(missing(value))
        return(private$.random_negative_sampling_rate$descriptor)
      private$.random_negative_sampling_rate$descriptor = value
    },

    #' @field shuffled_negative_sampling_rate
    #' The ratio of shuffled negative samples to draw during training.
    shuffled_negative_sampling_rate = function(value){
      if(missing(value))
        return(private$.shuffled_negative_sampling_rate$descriptor)
      private$.shuffled_negative_sampling_rate$descriptor = value
    },

    #' @field weight_decay
    #' Weight decay coefficient. Adds L2 regularization
    weight_decay = function(value){
      if(missing(value))
        return(private$.weight_decay$descriptor)
      private$.weight_decay$descriptor = value
    }
  ),
  lock_objects = F
)


#' @title Returns dot product of entity and IP address embeddings as a score for
#'              compatibility.
#' @description The implementation of
#'              :meth:`~sagemaker.predictor.Predictor.predict` in this
#'              `Predictor` requires a numpy ``ndarray`` as input. The array should
#'              contain two columns. The first column should contain the entity ID. The
#'              second column should contain the IPv4 address in dot notation.
#' @export
IPInsightsPredictor = R6Class("IPInsightsPredictor",
  inherit = Predictor,
  public = list(

    #' @description Initialize IPInsightsPredictor class
    #' @param endpoint_name (str): Name of the Amazon SageMaker endpoint to which
    #'              requests are sent.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    initialize = function(endpoint_name, sagemaker_session=NULL){
      super$initialize(
        endpoint_name,
        sagemaker_session,
        serializer=CSVSerializer$new(),
        deserializer=JSONDeserializer$new()
      )
    }
  ),
  lock_objects = F
)

#' @title Reference IPInsights s3 model data.
#' @description Calling :meth:`~sagemaker.model.Model.deploy`
#'              creates an Endpoint and returns a
#'              Predictor that calculates anomaly scores for data points.
#' @export
IPInsightsModel = R6Class("IPInsightsModel",
  inherit = Model,
  public = list(

    #' @description Initialize IPInsightsModel class
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
        IPInsights$public_fields$repo_name,
        sagemaker_session$paws_region_name,
        version=IPInsights$public_fields$repo_version,
      )
      super$initialize(
        image_uri,
        model_data,
        role,
        predictor_cls=IPInsightsPredictor,
        sagemaker_session=sagemaker_session,
        ...
      )
    }
  ),
  lock_objects = F
)
