# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/knn.py

#' @include amazon_estimator.R
#' @include amazon_common.R
#' @include amazon_hyperparameter.R
#' @include amazon_validation.R
#' @include predictor.R
#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common

#' @title An index-based algorithm. It uses a non-parametric method for classification or regression.
#' @description For classification problems, the algorithm queries the k points that are closest to the sample
#'              point and returns the most frequently used label of their class as the predicted label. For
#'              regression problems, the algorithm queries the k closest points to the sample point and returns
#'              the average of their feature values as the predicted value.
#' @export
KNN = R6Class("KNN",
  inherit = AmazonAlgorithmEstimatorBase,
  public = list(

    #' @field repo_name
    #' sagemaker repo name for framework
    repo_name = "knn",

    #' @field repo_version
    #' version of framework
    repo_version = 1,

    #' @description k-nearest neighbors (KNN) is :class:`Estimator` used for
    #'              classification and regression. This Estimator may be fit via calls to
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
    #'              It requires Amazon :class:`~sagemaker.amazon.record_pb2.Record` protobuf
    #'              serialized data to be stored in S3. There is an utility
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.record_set`
    #'              that can be used to upload data to S3 and creates
    #'              :class:`~sagemaker.amazon.amazon_estimator.RecordSet` to be passed to
    #'              the `fit` call. To learn more about the Amazon protobuf Record class and
    #'              how to prepare bulk data in this format, please consult AWS technical
    #'              documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html After
    #'              this Estimator is fit, model data is stored in S3. The model may be
    #'              deployed to an Amazon SageMaker Endpoint by invoking
    #'              :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
    #'              deploying an Endpoint, deploy returns a
    #'              :class:`~sagemaker.amazon.knn.KNNPredictor` object that can be used for
    #'              inference calls using the trained model hosted in the SageMaker
    #'              Endpoint. KNN Estimators can be configured by setting hyperparameters.
    #'              The available hyperparameters for KNN are documented below. For further
    #'              information on the AWS KNN algorithm, please consult AWS technical
    #'              documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/knn.html
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if accessing AWS resource.
    #' @param instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.c4.xlarge'.
    #' @param k (int): Required. Number of nearest neighbors.
    #' @param sample_size (int): Required. Number of data points to be sampled
    #'              from the training data set.
    #' @param predictor_type (str): Required. Type of inference to use on the
    #'              data's labels, allowed values are 'classifier' and 'regressor'.
    #' @param dimension_reduction_type (str): Optional. Type of dimension
    #'              reduction technique to use. Valid values: "sign", "fjlt"
    #' @param dimension_reduction_target (int): Optional. Target dimension to
    #'              reduce to. Required when dimension_reduction_type is specified.
    #' @param index_type (str): Optional. Type of index to use. Valid values are
    #'              "faiss.Flat", "faiss.IVFFlat", "faiss.IVFPQ".
    #' @param index_metric (str): Optional. Distance metric to measure between
    #'              points when finding nearest neighbors. Valid values are
    #'              "COSINE", "INNER_PRODUCT", "L2"
    #' @param faiss_index_ivf_nlists (str): Optional. Number of centroids to
    #'              construct in the index if index_type is "faiss.IVFFlat" or
    #'              "faiss.IVFPQ".
    #' @param faiss_index_pq_m (int): Optional. Number of vector sub-components to
    #'              construct in the index, if index_type is "faiss.IVFPQ".
    #' @param ... : base class keyword argument values.
    initialize = function(role,
                          instance_count,
                          instance_type,
                          k,
                          sample_size,
                          predictor_type,
                          dimension_reduction_type=NULL,
                          dimension_reduction_target=NULL,
                          index_type=NULL,
                          index_metric=NULL,
                          faiss_index_ivf_nlists=NULL,
                          faiss_index_pq_m=NULL,
                          ...){
      private$.k = Hyperparameter$new("k", list(Validation$new()$ge(1)), "An integer greater than 0", DataTypes$new()$int, obj = self)
      private$.sample_size = Hyperparameter$new("sample_size", list(Validation$new()$ge(1)), "An integer greater than 0", DataTypes$new()$int, obj = self)
      private$.predictor_type = Hyperparameter$new(
        "predictor_type", Validation$new()$isin(c("classifier", "regressor")), 'One of "classifier" or "regressor"', DataTypes$new()$str, obj = self
      )
      private$.dimension_reduction_target = Hyperparameter$new(
        "dimension_reduction_target",
        list(Validation$new()$ge(1)),
        "An integer greater than 0 and less than feature_dim",
        DataTypes$new()$int,
        obj = self
      )
      private$.dimension_reduction_type = Hyperparameter$new(
        "dimension_reduction_type", Validation$new()$isin(c("sign", "fjlt")), 'One of "sign" or "fjlt"', DataTypes$new()$str, obj = self
      )
      private$.index_metric = Hyperparameter$new(
        "index_metric",
        Validation$new()$isin(c("COSINE", "INNER_PRODUCT", "L2")),
        'One of "COSINE", "INNER_PRODUCT", "L2"',
        DataTypes$new()$str,
        obj = self
      )
      private$.index_type = Hyperparameter$new(
        "index_type",
        Validation$new()$isin(c("faiss.Flat", "faiss.IVFFlat", "faiss.IVFPQ")),
        'One of "faiss.Flat", "faiss.IVFFlat", "faiss.IVFPQ"',
        DataTypes$new()$str,
        obj = self
      )
      private$.faiss_index_ivf_nlists = Hyperparameter$new("faiss_index_ivf_nlists", list(), '"auto" or an integer greater than 0', DataTypes$new()$str, obj = self)
      private$.faiss_index_pq_m = Hyperparameter$new("faiss_index_pq_m", list(Validation$new()$ge(1)), "An integer greater than 0", DataTypes$new()$int, obj = self)

      super$initialize(role, instance_count, instance_type, ...)
      self$k = k
      self$sample_size = sample_size
      self$predictor_type = predictor_type
      self$dimension_reduction_type = dimension_reduction_type
      self$dimension_reduction_target = dimension_reduction_target
      self$index_type = index_type
      self$index_metric = index_metric
      self$faiss_index_ivf_nlists = faiss_index_ivf_nlists
      self$faiss_index_pq_m = faiss_index_pq_m

      if (!is.null(dimension_reduction_type) && is.null(dimension_reduction_target))
        stop('"dimension_reduction_target" is required when "dimension_reduction_type" is set.',
             call. = F)
    },

    #' @description Return a :class:`~sagemaker.amazon.KNNModel` referencing the latest
    #'              s3 model data produced by this Estimator.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
    #'              the model. Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param ... : Additional kwargs passed to the KNNModel constructor.
    create_model = function(vpc_config_override="VPC_CONFIG_DEFAULT",
                            ...){
      return(KNNModel$new(
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
      super$.prepare_for_training(
        records, mini_batch_size=mini_batch_size, job_name=job_name
      )
    }
  ),
  private = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    .k = NULL,
    .sample_size = NULL,
    .predictor_type = NULL,
    .dimension_reduction_target = NULL,
    .dimension_reduction_type = NULL,
    .index_metric = NULL,
    .index_type = NULL,
    .faiss_index_ivf_nlists = NULL,
    .faiss_index_pq_m = NULL
  ),
  active = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    #' @field k
    #' Number of nearest neighbors.
    k = function(value){
      if(missing(value))
        return(private$.k$descriptor)
      private$.k$descriptor = value
    },

    #' @field sample_size
    #' Number of data points to be sampled from the training data set
    sample_size = function(value){
      if(missing(value))
        return(private$.sample_size$descriptor)
      private$.sample_size$descriptor = value
    },

    #' @field predictor_type
    #' Type of inference to use on the data's labels
    predictor_type = function(value){
      if(missing(value))
        return(private$.predictor_type$descriptor)
      private$.predictor_type$descriptor = value
    },

    #' @field dimension_reduction_target
    #' Target dimension to reduce to
    dimension_reduction_target = function(value){
      if(missing(value))
        return(private$.dimension_reduction_target$descriptor)
      private$.dimension_reduction_target$descriptor = value
    },

    #' @field dimension_reduction_type
    #' Type of dimension reduction technique to use
    dimension_reduction_type = function(value){
      if(missing(value))
        return(private$.dimension_reduction_type$descriptor)
      private$.dimension_reduction_type$descriptor = value
    },

    #' @field index_metric
    #' Distance metric to measure between points when finding nearest neighbors
    index_metric = function(value){
      if(missing(value))
        return(private$.index_metric$descriptor)
      private$.index_metric$descriptor = value
    },

    #' @field index_type
    #' Type of index to use. Valid values are "faiss.Flat", "faiss.IVFFlat", "faiss.IVFPQ".
    index_type = function(value){
      if(missing(value))
        return(private$.index_type$descriptor)
      private$.index_type$descriptor = value
    },

    #' @field faiss_index_ivf_nlists
    #' Number of centroids to construct in the index
    faiss_index_ivf_nlists = function(value){
      if(missing(value))
        return(private$.faiss_index_ivf_nlists$descriptor)
      private$.faiss_index_ivf_nlists$descriptor = value
    },

    #' @field faiss_index_pq_m
    #' Number of vector sub-components to construct in the index
    faiss_index_pq_m = function(value){
      if(missing(value))
        return(private$.faiss_index_pq_m$descriptor)
      private$.faiss_index_pq_m$descriptor = value
    }
  ),
  lock_objects = F
)

#' @title Performs classification or regression prediction from input vectors.
#' @description The implementation of
#'              :meth:`~sagemaker.predictor.Predictor.predict` in this
#'              `Predictor` requires a numpy ``ndarray`` as input. The array should
#'              contain the same number of columns as the feature-dimension of the data used
#'              to fit the model this Predictor performs inference on.
#'              :func:`predict` returns a list of
#'              :class:`~sagemaker.amazon.record_pb2.Record` objects, one for each row in
#'              the input ``ndarray``. The prediction is stored in the ``"predicted_label"``
#'              key of the ``Record.label`` field.
#' @export
KNNPredictor = R6Class("KNNPredictor",
  inherit = Predictor,
  public = list(

    #' @description Initialize KNNPredictor class
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

#' @title Reference S3 model data created by KNN estimator.
#' @description Calling :meth:`~sagemaker.model.Model.deploy`
#'              creates an Endpoint and returns :class:`KNNPredictor`.
#' @export
KNNModel = R6Class("KNNModel",
  inherit = R6sagemaker.common::Model,
  public= list(

    #' @description Initialize KNNModel Class
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
        KNN$public_fields$repo_name,
        sagemaker_session$paws_region_name,
        version=KNN$public_fields$repo_version
      )
      super$initialize(
        image_uri,
        model_data,
        role,
        predictor_cls=KNNPredictor,
        sagemaker_session=sagemaker_session,
        ...
      )
    }
  ),
  lock_objects = F
)
