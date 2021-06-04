# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/randomcutforest.py

#' @include amazon_estimator.R
#' @include amazon_common.R
#' @include amazon_hyperparameter.R
#' @include amazon_validation.R
#' @include predictor.R
#' @include model.R
#' @include session.R
#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common

#' @title An unsupervised algorithm for detecting anomalous data points within a data set.
#' @description These are observations which diverge from otherwise well-structured or patterned data.
#'              Anomalies can manifest as unexpected spikes in time series data, breaks in periodicity,
#'              or unclassifiable data points.
#' @export
RandomCutForest = R6Class("RandomCutForest",
  inherit = AmazonAlgorithmEstimatorBase,
  public = list(

    #' @field repo_name
    #' sagemaker repo name for framework
    repo_name = "randomcutforest",

    #' @field repo_version
    #' version of framework
    repo_version = 1,

    #' @field MINI_BATCH_SIZE
    #' The size of each mini-batch to use when training.
    MINI_BATCH_SIZE = 1000,

    #' @description An `Estimator` class implementing a Random Cut Forest.
    #'              Typically used for anomaly detection, this Estimator may be fit via calls to
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
    #'              :class:`~sagemaker.amazon.ntm.RandomCutForestPredictor` object that can
    #'              be used for inference calls using the trained model hosted in the
    #'              SageMaker Endpoint.
    #'              RandomCutForest Estimators can be configured by setting
    #'              hyperparameters. The available hyperparameters for RandomCutForest are
    #'              documented below.
    #'              For further information on the AWS Random Cut Forest algorithm,
    #'              please consult AWS technical documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if accessing AWS resource.
    #' @param instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.c4.xlarge'.
    #' @param num_samples_per_tree (int): Optional. The number of samples used to
    #'              build each tree in the forest. The total number of samples drawn
    #'              from the train dataset is num_trees * num_samples_per_tree.
    #' @param num_trees (int): Optional. The number of trees used in the forest.
    #' @param eval_metrics (list): Optional. JSON list of metrics types to be used
    #'              for reporting the score for the model. Allowed values are
    #'              "accuracy", "precision_recall_fscore": positive and negative
    #'              precision, recall, and f1 scores. If test data is provided, the
    #'              score shall be reported in terms of all requested metrics.
    #' @param ... : base class keyword argument values.
    initialize = function(role,
                          instance_count,
                          instance_type,
                          num_samples_per_tree=NULL,
                          num_trees=NULL,
                          eval_metrics=NULL,
                          ...){
      super$initialize(role, instance_count, instance_type, ...)

      private$.eval_metrics = Hyperparameter$new(
        name="eval_metrics",
        validation_message='A comma separated list of "accuracy" or "precision_recall_fscore"',
        data_type=as.list,
        obj = self
      )
      private$.num_trees = Hyperparameter$new("num_trees", list(Validation$new()$ge(50), Validation$new()$le(1000)), "An integer in [50, 1000]", DataTypes$new()$int, obj = self)
      private$.num_samples_per_tree = Hyperparameter$new(
        "num_samples_per_tree", list(Validation$new()$ge(1), Validation$new()$le(2048)), "An integer in [1, 2048]", DataTypes$new()$int, obj = self
      )
      private$.feature_dim = Hyperparameter$new("feature_dim", list(Validation$new()$ge(1), Validation$new()$le(10000)), "An integer in [1, 10000]", DataTypes$new()$int, obj = self)

      self$num_samples_per_tree = num_samples_per_tree
      self$num_trees = num_trees
      self$eval_metrics = eval_metrics
    },

    #' @description Return a :class:`~sagemaker.amazon.RandomCutForestModel` referencing
    #'              the latest s3 model data produced by this Estimator.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
    #'              the model. Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param ... : Additional kwargs passed to the RandomCutForestModel constructor.
    create_model = function(vpc_config_override="VPC_CONFIG_DEFAULT", ...){
      return(RandomCutForestModel$new(
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
      if (is.null(mini_batch_size)){
        mini_batch_size = self$MINI_BATCH_SIZE
      } else if (mini_batch_size != self$MINI_BATCH_SIZE){
          stop(sprintf(
            "Random Cut Forest uses a fixed mini_batch_size of %s", self$MINI_BATCH_SIZE)
          )
      }
        super$.prepare_for_training(
          records, mini_batch_size=mini_batch_size, job_name=job_name
        )
    }
  ),

  active = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    #' @field eval_metrics
    #' JSON list of metrics types to be used for reporting the score for the model
    eval_metrics = function(value){
      if(missing(value))
        return(private$.eval_metrics$descriptor)
      private$.eval_metrics$descriptor = value
    },

    #' @field num_trees
    #' The number of trees used in the forest.
    num_trees = function(value){
      if(missing(value))
        return(private$.num_trees$descriptor)
      private$.num_trees$descriptor = value
    },

    #' @field num_samples_per_tree
    #' The number of samples used to build each tree in the forest.
    num_samples_per_tree = function(value){
      if(missing(value))
        return(private$.num_samples_per_tree$descriptor)
      private$.num_samples_per_tree$descriptor = value
    },

    #' @field feature_dim
    #' Doc string place
    feature_dim = function(value){
      if(missing(value))
        return(private$.feature_dim$descriptor)
      private$.feature_dim$descriptor = value
    }
  ),
  private = list(
    # --------- initializing private objects of r python descriptor class ---------
    .eval_metrics = NULL,
    .num_trees = NULL,
    .num_samples_per_tree = NULL,
    .feature_dim = NULL
  ),
  lock_objects = F
)

#' @title Assigns an anomaly score to each of the datapoints provided.
#' @description The implementation of
#'              :meth:`~sagemaker.predictor.Predictor.predict` in this
#'              `Predictor` requires a numpy ``ndarray`` as input. The array should
#'              contain the same number of columns as the feature-dimension of the data used
#'              to fit the model this Predictor performs inference on.
#' @export
RandomCutForestPredictor = R6Class("RandomCutForestPredictor",
  inherit = Predictor,
  public = list(

    #' @description Initialize RandomCutForestPredictor class
    #' @param endpoint_name (str): Name of the Amazon SageMaker endpoint to which
    #'              requests are sent.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: NULL). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    initialize = function(endpoint_name, sagemaker_session=NULL){
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

#' @title Reference RandomCutForest s3 model data.
#' @description Calling :meth:`~sagemaker.model.Model.deploy` creates
#'              an Endpoint and returns a
#'              Predictor that calculates anomaly scores for datapoints.
#' @export
RandomCutForestModel = R6Class("RandomCutForestModel",
  inherit = Model,
  public = list(

    #' @description Initialize RandomCutForestModel class
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
        RandomCutForest$public_fields$repo_name,
        sagemaker_session$paws_region_name,
        version=RandomCutForest$public_fields$repo_version
      )
      super$initialize(
        image_uri,
        model_data,
        role,
        predictor_cls=RandomCutForestPredictor,
        sagemaker_session=sagemaker_session,
        ...
      )
    }
  ),
  lock_objects = F
)
