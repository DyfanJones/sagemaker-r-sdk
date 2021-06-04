# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/kmeans.py

#' @include amazon_estimator.R
#' @include amazon_hyperparameter.R
#' @include deserializers.R
#' @include model.R
#' @include serializers.R
#' @include session.R
#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common

#' @title An unsupervised learning algorithm that attempts to find discrete groupings within data.
#' @description As the result of KMeans, members of a group are as similar as possible to one another and as
#'              different as possible from members of other groups. You define the attributes that you want
#'              the algorithm to use to determine similarity.
#' @export
KMeans = R6Class("KMeans",
  inherit = AmazonAlgorithmEstimatorBase,
  public = list(

    #' @field repo_name
    #' sagemaker repo name for framework
    repo_name = "kmeans",

    #' @field repo_version
    #' version of framework
    repo_version = 1,

    #' @description A k-means clustering
    #'              :class:`~sagemaker.amazon.AmazonAlgorithmEstimatorBase`. Finds k
    #'              clusters of data in an unlabeled dataset.
    #'              This Estimator may be fit via calls to
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit_ndarray`
    #'              or
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
    #'              The former allows a KMeans model to be fit on a 2-dimensional numpy
    #'              array. The latter requires Amazon
    #'              :class:`~sagemaker.amazon.record_pb2.Record` protobuf serialized data to
    #'              be stored in S3.
    #'              To learn more about the Amazon protobuf Record class and how to
    #'              prepare bulk data in this format, please consult AWS technical
    #'              documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html.
    #'              After this Estimator is fit, model data is stored in S3. The model
    #'              may be deployed to an Amazon SageMaker Endpoint by invoking
    #'              :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
    #'              deploying an Endpoint, ``deploy`` returns a
    #'              :class:`~sagemaker.amazon.kmeans.KMeansPredictor` object that can be
    #'              used to k-means cluster assignments, using the trained k-means model
    #'              hosted in the SageMaker Endpoint.
    #'              KMeans Estimators can be configured by setting hyperparameters. The
    #'              available hyperparameters for KMeans are documented below. For further
    #'              information on the AWS KMeans algorithm, please consult AWS technical
    #'              documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if accessing AWS resource.
    #' @param instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.c4.xlarge'.
    #' @param k (int): The number of clusters to produce.
    #' @param init_method (str): How to initialize cluster locations. One of
    #'              'random' or 'kmeans++'.
    #' @param max_iterations (int): Maximum iterations for Lloyds EM procedure in
    #'              the local kmeans used in finalize stage.
    #' @param tol (float): Tolerance for change in ssd for early stopping in local
    #'              kmeans.
    #' @param num_trials (int): Local version is run multiple times and the one
    #'              with the best loss is chosen. This determines how many times.
    #' @param local_init_method (str): Initialization method for local version.
    #'              One of 'random', 'kmeans++'
    #' @param half_life_time_size (int): The points can have a decayed weight.
    #'              When a point is observed its weight, with regard to the
    #'              computation of the cluster mean is 1. This weight will decay
    #'              exponentially as we observe more points. The exponent
    #'              coefficient is chosen such that after observing
    #'              ``half_life_time_size`` points after the mentioned point, its
    #'              weight will become 1/2. If set to 0, there will be no decay.
    #' @param epochs (int): Number of passes done over the training data.
    #' @param center_factor (int): The algorithm will create
    #'              ``num_clusters * extra_center_factor`` as it runs and reduce the
    #'              number of centers to ``k`` when finalizing
    #' @param eval_metrics (list): JSON list of metrics types to be used for
    #'              reporting the score for the model. Allowed values are "msd"
    #'              Means Square Error, "ssd": Sum of square distance. If test data
    #'              is provided, the score shall be reported in terms of all
    #'              requested metrics.
    #' @param ... : base class keyword argument values.
    initialize = function(role,
                          instance_count,
                          instance_type,
                          k,
                          init_method=NULL,
                          max_iterations=NULL,
                          tol=NULL,
                          num_trials=NULL,
                          local_init_method=NULL,
                          half_life_time_size=NULL,
                          epochs=NULL,
                          center_factor=NULL,
                          eval_metrics=NULL,
                          ...){

      private$.k = Hyperparameter$new("k", Validation$new()$gt(1), "An integer greater-than 1", DataTypes$new()$int, obj = self)
      private$.init_method = Hyperparameter$new("init_method", Validation$new()$isin(c("random", "kmeans++")), 'One of "random", "kmeans++"', DataTypes$new()$str, obj = self)
      private$.max_iterations = Hyperparameter$new("local_lloyd_max_iter", Validation$new()$gt(0), "An integer greater-than 0", DataTypes$new()$int, obj = self)
      private$.tol = Hyperparameter$new("local_lloyd_tol", list(Validation$new()$ge(0), Validation$new()$le(1)), "An float in [0, 1]", DataTypes$new()$float, obj = self)
      private$.num_trials = Hyperparameter$new("local_lloyd_num_trials", Validation$new()$gt(0), "An integer greater-than 0", DataTypes$new()$int, obj = self)
      private$.local_init_method = Hyperparameter$new(
        "local_lloyd_init_method", Validation$new()$isin(c("random", "kmeans++")), 'One of "random", "kmeans++"', DataTypes$new()$str, obj = self
      )
      private$.half_life_time_size = Hyperparameter$new(
        "half_life_time_size", Validation$new()$ge(0), "An integer greater-than-or-equal-to 0", DataTypes$new()$int, obj = self
      )
      private$.epochs = Hyperparameter$new("epochs", Validation$new()$gt(0), "An integer greater-than 0", DataTypes$new()$int, obj = self)
      private$.center_factor = Hyperparameter$new("extra_center_factor", Validation$new()$gt(0), "An integer greater-than 0", DataTypes$new()$int, obj = self)
      private$.eval_metrics = Hyperparameter$new(
        name="eval_metrics",
        validation_message='A comma separated list of "msd" or "ssd"',
        data_type=as.list,
        obj = self
      )

      super$initialize(role, instance_count, instance_type, ...)
      self$k = k
      self$init_method = init_method
      self$max_iterations = max_iterations
      self$tol = tol
      self$num_trials = num_trials
      self$local_init_method = local_init_method
      self$half_life_time_size = half_life_time_size
      self$epochs = epochs
      self$center_factor = center_factor
      self$eval_metrics = eval_metrics
    },

    #' @description Return a :class:`~sagemaker.amazon.kmeans.KMeansModel` referencing
    #'              the latest s3 model data produced by this Estimator.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for
    #'              VpcConfig set on the model.
    #'              Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param ... : Additional kwargs passed to the KMeansModel constructor.
    create_model = function(vpc_config_override="VPC_CONFIG_DEFAULT",
                            ...){
      return (KMeansModel$new(
        self$model_data,
        self$role,
        self$sagemaker_session,
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
                                     mini_batch_size=5000,
                                     job_name=NULL){
      super$.prepare_for_training(
        records, mini_batch_size=mini_batch_size, job_name=job_name
      )
    },

    #' @description Return the SageMaker hyperparameters for training this KMeans
    #'              Estimator
    hyperparameters = function(){
      hp_dict = list(force_dense="True")  # KMeans requires this hp to fit on Record objects
      return(c(super$hyperparameters(),hp_dict))
    }
  ),
  private = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    .k = NULL,
    .init_method = NULL,
    .max_iterations = NULL,
    .tol = NULL,
    .num_trials = NULL,
    .local_init_method = NULL,
    .half_life_time_size = NULL,
    .epochs = NULL,
    .center_factor = NULL,
    .eval_metrics = NULL
  ),
  active = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    #' @field k
    #' The number of clusters to produce.
    k = function(value){
      if(missing(value))
        return(private$.k$descriptor)
      private$.k$descriptor = value
    },

    #' @field init_method
    #' How to initialize cluster locations.
    init_method = function(value){
      if(missing(value))
        return(private$.init_method$descriptor)
      private$.init_method$descriptor = value
    },

    #' @field max_iterations
    #' Maximum iterations for Lloyds EM procedure in the local kmeans used in finalize stage.
    max_iterations = function(value){
      if(missing(value))
        return(private$.max_iterations$descriptor)
      private$.max_iterations$descriptor = value
    },

    #' @field tol
    #' Tolerance for change in ssd for early stopping in local kmeans.
    tol = function(value){
      if(missing(value))
        return(private$.tol$descriptor)
      private$.tol$descriptor = value
    },

    #' @field num_trials
    #' Local version is run multiple times and the one with the best loss is chosen.
    num_trials = function(value){
      if(missing(value))
        return(private$.num_trials$descriptor)
      private$.num_trials$descriptor = value
    },

    #' @field local_init_method
    #' Initialization method for local version.
    local_init_method = function(value){
      if(missing(value))
        return(private$.local_init_method$descriptor)
      private$.local_init_method$descriptor = value
    },

    #' @field half_life_time_size
    #' The points can have a decayed weight.
    half_life_time_size = function(value){
      if(missing(value))
        return(private$.half_life_time_size$descriptor)
      private$.half_life_time_size$descriptor = value
    },

    #' @field epochs
    #' Number of passes done over the training data.
    epochs = function(value){
      if(missing(value))
        return(private$.epochs$descriptor)
      private$.epochs$descriptor = value
    },

    #' @field center_factor
    #' The algorithm will create ``num_clusters * extra_center_factor`` as it runs.
    center_factor = function(value){
      if(missing(value))
        return(private$.center_factor$descriptor)
      private$.center_factor$descriptor = value
    },

    #' @field eval_metrics
    #' JSON list of metrics types to be used for reporting the score for the model.
    eval_metrics = function(value){
      if(missing(value))
        return(private$.eval_metrics$descriptor)
      private$.eval_metrics$descriptor = value
    }
  ),
  lock_objects = F
)

#' @title Assigns input vectors to their closest cluster in a KMeans model.
#' @description The implementation of
#'              :meth:`~sagemaker.predictor.Predictor.predict` in this
#'              `Predictor` requires a numpy ``ndarray`` as input. The array should
#'              contain the same number of columns as the feature-dimension of the data used
#'              to fit the model this Predictor performs inference on.
#'              ``predict()`` returns a list of
#'              :class:`~sagemaker.amazon.record_pb2.Record` objects, one for each row in
#'              the input ``ndarray``. The nearest cluster is stored in the
#'              ``closest_cluster`` key of the ``Record.label`` field.
#' @export
KMeansPredictor = R6Class("KMeansPredictor",
  inherit = Predictor,
  public = list(

    #' @description Initialize KMeansPredictor Class
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

#' @title Reference KMeans s3 model data.
#' @description Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return a
#'              Predictor to performs k-means cluster assignment.
#' @export
KMeansModel = R6Class("KMeansModel",
  inherit = Model,
  public = list(

    #' @description Initialize KMeansPredictor Class
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
        KMeans$public_fields$repo_name,
        sagemaker_session$paws_region_name,
        version=KMeans$public_fields$repo_version
      )
      super$initialize(
        image_uri,
        model_data,
        role,
        predictor_cls=KMeansPredictor,
        sagemaker_session=sagemaker_session,
        ...
      )
    }
  ),
  lock_objects = F
)

