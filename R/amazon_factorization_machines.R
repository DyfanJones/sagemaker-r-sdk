# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/factorization_machines.py

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

#' @title A supervised learning algorithm used in classification and regression.
#' @description Factorization Machines combine the advantages of Support Vector Machines
#'              with factorization models. It is an extension of a linear model that is
#'              designed to capture interactions between features within high dimensional
#'              sparse datasets economically.
#' @export
FactorizationMachines = R6Class("FactorizationMachines",
  inherit = AmazonAlgorithmEstimatorBase,
  public = list(

    #' @field repo_name
    #' sagemaker repo name for framework
    repo_name = "factorization-machines",

    #' @field repo_version
    #' version of framework
    repo_version = 1,

    #' @description Factorization Machines is :class:`Estimator` for general-purpose
    #'              supervised learning.
    #'              Amazon SageMaker Factorization Machines is a general-purpose
    #'              supervised learning algorithm that you can use for both classification
    #'              and regression tasks. It is an extension of a linear model that is
    #'              designed to parsimoniously capture interactions between features within
    #'              high dimensional sparse datasets.
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
    #'              :class:`~sagemaker.amazon.pca.FactorizationMachinesPredictor` object
    #'              that can be used for inference calls using the trained model hosted in
    #'              the SageMaker Endpoint.
    #'              FactorizationMachines Estimators can be configured by setting
    #'              hyperparameters. The available hyperparameters for FactorizationMachines
    #'              are documented below.
    #'              For further information on the AWS FactorizationMachines algorithm,
    #'              please consult AWS technical documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if accessing AWS resource.
    #' @param instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.c4.xlarge'.
    #' @param num_factors (int): Dimensionality of factorization.
    #' @param predictor_type (str): Type of predictor 'binary_classifier' or
    #'              'regressor'.
    #' @param epochs (int): Number of training epochs to run.
    #' @param clip_gradient (float): Optimizer parameter. Clip the gradient by
    #'              projecting onto the box [-clip_gradient, +clip_gradient]
    #' @param eps (float): Optimizer parameter. Small value to avoid division by
    #'              0.
    #' @param rescale_grad (float): Optimizer parameter. If set, multiplies the
    #'              gradient with rescale_grad before updating. Often choose to be
    #'              1.0/batch_size.
    #' @param bias_lr (float): Non-negative learning rate for the bias term.
    #' @param linear_lr (float): Non-negative learning rate for linear terms.
    #' @param factors_lr (float): Non-negative learning rate for factorization
    #'              terms.
    #' @param bias_wd (float): Non-negative weight decay for the bias term.
    #' @param linear_wd (float): Non-negative weight decay for linear terms.
    #' @param factors_wd (float): Non-negative weight decay for factorization
    #'              terms.
    #' @param bias_init_method (string): Initialization method for the bias term:
    #'              'normal', 'uniform' or 'constant'.
    #' @param bias_init_scale (float): Non-negative range for initialization of
    #'              the bias term that takes effect when bias_init_method parameter
    #'              is 'uniform'
    #' @param bias_init_sigma (float): Non-negative standard deviation for
    #'              initialization of the bias term that takes effect when
    #'              bias_init_method parameter is 'normal'.
    #' @param bias_init_value (float): Initial value of the bias term that takes
    #'              effect when bias_init_method parameter is 'constant'.
    #' @param linear_init_method (string): Initialization method for linear term:
    #'              'normal', 'uniform' or 'constant'.
    #' @param linear_init_scale (float): Non-negative range for initialization of
    #'              linear terms that takes effect when linear_init_method parameter
    #'              is 'uniform'.
    #' @param linear_init_sigma (float): Non-negative standard deviation for
    #'              initialization of linear terms that takes effect when
    #'              linear_init_method parameter is 'normal'.
    #' @param linear_init_value (float): Initial value of linear terms that takes
    #'              effect when linear_init_method parameter is 'constant'.
    #' @param factors_init_method (string): Initialization method for
    #'              factorization term: 'normal', 'uniform' or 'constant'.
    #' @param factors_init_scale (float): Non-negative range for initialization of
    #'              factorization terms that takes effect when factors_init_method
    #'              parameter is 'uniform'.
    #' @param factors_init_sigma (float): Non-negative standard deviation for
    #'              initialization of factorization terms that takes effect when
    #'              factors_init_method parameter is 'normal'.
    #' @param factors_init_value (float): Initial value of factorization terms
    #'              that takes effect when factors_init_method parameter is
    #'              'constant'.
    #' @param ... : base class keyword argument values. You can find additional
    #'              parameters for initializing this class at
    #'              :class:`~sagemaker.estimator.amazon_estimator.AmazonAlgorithmEstimatorBase` and
    #'              :class:`~sagemaker.estimator.EstimatorBase`.
    initialize = function(role,
                         instance_count,
                         instance_type,
                         num_factors,
                         predictor_type,
                         epochs=NULL,
                         clip_gradient=NULL,
                         eps=NULL,
                         rescale_grad=NULL,
                         bias_lr=NULL,
                         linear_lr=NULL,
                         factors_lr=NULL,
                         bias_wd=NULL,
                         linear_wd=NULL,
                         factors_wd=NULL,
                         bias_init_method=NULL,
                         bias_init_scale=NULL,
                         bias_init_sigma=NULL,
                         bias_init_value=NULL,
                         linear_init_method=NULL,
                         linear_init_scale=NULL,
                         linear_init_sigma=NULL,
                         linear_init_value=NULL,
                         factors_init_method=NULL,
                         factors_init_scale=NULL,
                         factors_init_sigma=NULL,
                         factors_init_value=NULL,
                         ...){

      super$initialize(role = role, instance_count = instance_count, instance_type = instance_type, ...)

      private$.num_factors = Hyperparameter$new("num_factors", Validation$new()$gt(0), "An integer greater than zero", DataTypes$new()$int, self)
      private$.predictor_type = Hyperparameter$new(
        "predictor_type",
        Validation$new()$isin(c("binary_classifier", "regressor")),
        'Value "binary_classifier" or "regressor"',
        DataTypes$new()$str, self)
      private$.epochs = Hyperparameter$new("epochs", Validation$new()$gt(0), "An integer greater than 0", DataTypes$new()$int, self)
      private$.clip_gradient = Hyperparameter$new("clip_gradient", list(), "A float value", DataTypes$new()$float, self)
      private$.eps = Hyperparameter$new("eps", list(), "A float value", DataTypes$new()$float, self)
      private$.rescale_grad = Hyperparameter$new("rescale_grad", list(), "A float value", DataTypes$new()$float, self)
      private$.bias_lr = Hyperparameter$new("bias_lr", Validation$new()$ge(0), "A non-negative float", DataTypes$new()$float, self)
      private$.linear_lr = Hyperparameter$new("linear_lr", Validation$new()$ge(0), "A non-negative float", DataTypes$new()$float, self)
      private$.factors_lr = Hyperparameter$new("factors_lr", Validation$new()$ge(0), "A non-negative float", DataTypes$new()$float, self)
      private$.bias_wd = Hyperparameter$new("bias_wd", Validation$new()$ge(0), "A non-negative float", DataTypes$new()$float, self)
      private$.linear_wd = Hyperparameter$new("linear_wd", Validation$new()$ge(0), "A non-negative float", DataTypes$new()$float, self)
      private$.factors_wd = Hyperparameter$new("factors_wd", Validation$new()$ge(0), "A non-negative float", DataTypes$new()$float, self)
      private$.bias_init_method = Hyperparameter$new(
        "bias_init_method",
        Validation$new()$isin(c("normal", "uniform", "constant")),
        'Value "normal", "uniform" or "constant"',
        DataTypes$new()$str, self)
      private$.bias_init_scale = Hyperparameter$new("bias_init_scale", Validation$new()$ge(0), "A non-negative float", DataTypes$new()$float, self)
      private$.bias_init_sigma = Hyperparameter$new("bias_init_sigma", Validation$new()$ge(0), "A non-negative float", DataTypes$new()$float, self)
      private$.bias_init_value = Hyperparameter$new("bias_init_value", list(), "A float value", DataTypes$new()$float, self)
      private$.linear_init_method = Hyperparameter$new(
        "linear_init_method",
        Validation$new()$isin(c("normal", "uniform", "constant")),
        'Value "normal", "uniform" or "constant"',
        DataTypes$new()$str, self)
      private$.linear_init_scale = Hyperparameter$new("linear_init_scale", Validation$new()$ge(0), "A non-negative float", DataTypes$new()$float, self)
      private$.linear_init_sigma = Hyperparameter$new("linear_init_sigma", Validation$new()$ge(0), "A non-negative float", DataTypes$new()$float, self)
      private$.linear_init_value = Hyperparameter$new("linear_init_value", list(), "A float value", DataTypes$new()$float, self)
      private$.factors_init_method = Hyperparameter$new(
        "factors_init_method",
        Validation$new()$isin(c("normal", "uniform", "constant")),
        'Value "normal", "uniform" or "constant"',
        DataTypes$new()$str, self)
      private$.factors_init_scale = Hyperparameter$new("factors_init_scale", Validation$new()$ge(0), "A non-negative float", DataTypes$new()$float, self)
      private$.factors_init_sigma = Hyperparameter$new("factors_init_sigma", Validation$new()$ge(0), "A non-negative float", DataTypes$new()$float, self)
      private$.factors_init_value = Hyperparameter$new("factors_init_value", list(), "A float value", DataTypes$new()$float, self)

      self$num_factors = num_factors
      self$predictor_type = predictor_type
      self$epochs = epochs
      self$clip_gradient = clip_gradient
      self$eps = eps
      self$rescale_grad = rescale_grad
      self$bias_lr = bias_lr
      self$linear_lr = linear_lr
      self$factors_lr = factors_lr
      self$bias_wd = bias_wd
      self$linear_wd = linear_wd
      self$factors_wd = factors_wd
      self$bias_init_method = bias_init_method
      self$bias_init_scale = bias_init_scale
      self$bias_init_sigma = bias_init_sigma
      self$bias_init_value = bias_init_value
      self$linear_init_method = linear_init_method
      self$linear_init_scale = linear_init_scale
      self$linear_init_sigma = linear_init_sigma
      self$linear_init_value = linear_init_value
      self$factors_init_method = factors_init_method
      self$factors_init_scale = factors_init_scale
      self$factors_init_sigma = factors_init_sigma
      self$factors_init_value = factors_init_value
    },

    #' @description Return a :class:`~sagemaker.amazon.FactorizationMachinesModel`
    #'              referencing the latest s3 model data produced by this Estimator.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
    #'              the model. Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param ... : Additional kwargs passed to the FactorizationMachinesModel constructor.
    creat_model = function(vpc_config_override="VPC_CONFIG_DEFAULT", ...){
      return(FactorizationMachinesModel$new(
        self$model_data,
        self$role,
        sagemaker_session=self$sagemaker_session,
        vpc_config=self$get_vpc_config(vpc_config_override),
        ...)
      )
    }
  ),
  active = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    #' @field num_factors
    #' Dimensionality of factorization.
    num_factors = function(value){
      if(missing(value))
        return(private$.num_factors$descriptor)
      private$.num_factors$descriptor = value
    },

    #' @field predictor_type
    #' Type of predictor 'binary_classifier' or 'regressor'.
    predictor_type = function(value){
      if(missing(value))
        return(private$.predictor_type$descriptor)
      private$.predictor_type$descriptor = value
    },

    #' @field epochs
    #' Number of training epochs to run.
    epochs = function(value){
      if(missing(value))
        return(private$.epochs$descriptor)
      private$.epochs$descriptor = value
    },

    #' @field clip_gradient
    #' Clip the gradient by projecting onto the box [-clip_gradient, +clip_gradient]
    clip_gradient = function(value){
      if(missing(value))
        return(private$.clip_gradient$descriptor)
      private$.clip_gradient$descriptor = value
    },

    #' @field eps
    #' Small value to avoid division by 0.
    eps = function(value){
      if(missing(value))
        return(private$.eps$descriptor)
      private$.eps$descriptor = value
    },

    #' @field rescale_grad
    #' If set, multiplies the gradient with rescale_grad before updating
    rescale_grad = function(value){
      if(missing(value))
        return(private$.rescale_grad$descriptor)
      private$.rescale_grad$descriptor = value
    },

    #' @field bias_lr
    #' Non-negative learning rate for the bias term.
    bias_lr = function(value){
      if(missing(value))
        return(private$.bias_lr$descriptor)
      private$.bias_lr$descriptor = value
    },

    #' @field linear_lr
    #' Non-negative learning rate for linear terms.
    linear_lr = function(value){
      if(missing(value))
        return(private$.linear_lr$descriptor)
      private$.linear_lr$descriptor = value
    },

    #' @field factors_lr
    #' Non-negative learning rate for factorization terms.
    factors_lr = function(value){
      if(missing(value))
        return(private$.factors_lr$descriptor)
      private$.factors_lr$descriptor = value
    },

    #' @field bias_wd
    #' Non-negative weight decay for the bias term.
    bias_wd = function(value){
      if(missing(value))
        return(private$.bias_wd$descriptor)
      private$.bias_wd$descriptor = value
    },

    #' @field linear_wd
    #' Non-negative weight decay for linear terms.
    linear_wd = function(value){
      if(missing(value))
        return(private$.linear_wd$descriptor)
      private$.linear_wd$descriptor = value
    },

    #' @field factors_wd
    #' Non-negative weight decay for factorization terms.
    factors_wd = function(value){
      if(missing(value))
        return(private$.factors_wd$descriptor)
      private$.factors_wd$descriptor = value
    },

    #' @field bias_init_method
    #' Initialization method for the bias term:
    #' 'normal', 'uniform' or 'constant'.
    bias_init_method = function(value){
      if(missing(value))
        return(private$.bias_init_method$descriptor)
      private$.bias_init_method$descriptor = value
    },

    #' @field bias_init_scale
    #' Non-negative range for initialization of
    #' the bias term that takes effect when bias_init_method parameter
    #' is 'uniform'
    bias_init_scale = function(value){
      if(missing(value))
        return(private$.bias_init_scale$descriptor)
      private$.bias_init_scale$descriptor = value
    },

    #' @field bias_init_sigma
    #' Non-negative standard deviation for
    #' initialization of the bias term that takes effect when
    #' bias_init_method parameter is 'normal'.
    bias_init_sigma = function(value){
      if(missing(value))
        return(private$.bias_init_sigma$descriptor)
      private$.bias_init_sigma$descriptor = value
    },

    #' @field bias_init_value
    #' Initial value of the bias term that takes
    #' effect when bias_init_method parameter is 'constant'.
    bias_init_value = function(value){
      if(missing(value))
        return(private$.bias_init_value$descriptor)
      private$.bias_init_value$descriptor = value
    },

    #' @field linear_init_method
    #' Initialization method for linear term:
    #' normal', 'uniform' or 'constant'.
    linear_init_method = function(value){
      if(missing(value))
        return(private$.linear_init_method$descriptor)
      private$.linear_init_method$descriptor = value
    },

    #' @field linear_init_scale
    #' on-negative range for initialization of
    #' linear terms that takes effect when linear_init_method parameter
    #' is 'uniform'.
    linear_init_scale = function(value){
      if(missing(value))
        return(private$.linear_init_scale$descriptor)
      private$.linear_init_scale$descriptor = value
    },

    #' @field linear_init_sigma
    #' Non-negative standard deviation for
    #' initialization of linear terms that takes effect when
    #' linear_init_method parameter is 'normal'.
    linear_init_sigma = function(value){
      if(missing(value))
        return(private$.linear_init_sigma$descriptor)
      private$.linear_init_sigma$descriptor = value
    },

    #' @field linear_init_value
    #' Initial value of linear terms that takes
    #' effect when linear_init_method parameter is 'constant'.
    linear_init_value = function(value){
      if(missing(value))
        return(private$.linear_init_value$descriptor)
      private$.linear_init_value$descriptor = value
    },

    #' @field factors_init_method
    #' Initialization method for
    #' factorization term: 'normal', 'uniform' or 'constant'.
    factors_init_method = function(value){
      if(missing(value))
        return(private$.factors_init_method$descriptor)
      private$.factors_init_method$descriptor = value
    },

    #' @field factors_init_scale
    #' Non-negative range for initialization of
    #' factorization terms that takes effect when factors_init_method
    #' parameter is 'uniform'.
    factors_init_scale = function(value){
      if(missing(value))
        return(private$.factors_init_scale$descriptor)
      private$.factors_init_scale$descriptor = value
    },

    #' @field factors_init_sigma
    #' Non-negative standard deviation for
    #' initialization of factorization terms that takes effect when
    #' factors_init_method parameter is 'normal'.
    factors_init_sigma = function(value){
      if(missing(value))
        return(private$.factors_init_sigma$descriptor)
      private$.factors_init_sigma$descriptor = value
    },

    #' @field factors_init_value
    #' Initial value of factorization terms
    #' that takes effect when factors_init_method parameter is
    #' constant'.
    factors_init_value = function(value){
      if(missing(value))
        return(private$.factors_init_value$descriptor)
      private$.factors_init_value$descriptor = value
    }
  ),
  private = list(
    # --------- initializing private objects of r python descriptor class ---------
    .num_factors = NULL,
    .predictor_type = NULL,
    .epochs = NULL,
    .clip_gradient = NULL,
    .eps = NULL,
    .rescale_grad = NULL,
    .bias_lr = NULL,
    .linear_lr = NULL,
    .factors_lr = NULL,
    .bias_wd = NULL,
    .linear_wd = NULL,
    .factors_wd = NULL,
    .bias_init_method = NULL,
    .bias_init_scale = NULL,
    .bias_init_sigma = NULL,
    .bias_init_value = NULL,
    .linear_init_method = NULL,
    .linear_init_scale = NULL,
    .linear_init_sigma = NULL,
    .linear_init_value = NULL,
    .factors_init_method = NULL,
    .factors_init_scale = NULL,
    .factors_init_sigma = NULL,
    .factors_init_value = NULL
  ),
  lock_objects = F
)

#' @title Performs binary-classification or regression prediction from input
#'              vectors.
#' @description The implementation of
#'              :meth:`~sagemaker.predictor.Predictor.predict` in this
#'              `Predictor` requires a numpy ``ndarray`` as input. The array should
#'              contain the same number of columns as the feature-dimension of the data used
#'              to fit the model this Predictor performs inference on.
#'              :meth:`predict()` returns a list of
#'              :class:`~sagemaker.amazon.record_pb2.Record` objects, one for each row in
#'              the input ``ndarray``. The prediction is stored in the ``"score"`` key of
#'              the ``Record.label`` field. Please refer to the formats details described:
#'              https://docs.aws.amazon.com/sagemaker/latest/dg/fm-in-formats.html
#' @export
FactorizationMachinesPredictor = R6Class(
  inherit = Predictor,
  public = list(

    #' @description Initialize FactorizationMachinesPredictor class
    #' @param endpoint_name (str): Name of the Amazon SageMaker endpoint to which
    #'              requests are sent.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: NULL). If not
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
  )
)



#' @title Amazon FactorizationMachinesModel Class
#' @description Reference S3 model data created by FactorizationMachines estimator.
#'              Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and
#'              returns :class:`FactorizationMachinesPredictor`.
#' @export
FactorizationMachinesModel = R6Class("FactorizationMachinesModel",
  inherit = Model,
  public = list(

    #' @description Initialize FactorizationMachinesModel class
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
        FactorizationMachines$public_fields$repo_name,
        sagemaker_session$paws_region_name,
        version=FactorizationMachines$public_fields$repo_version
      )
      super$initialize(
        image_uri,
        model_data,
        role,
        predictor_cls=FactorizationMachinesPredictor,
        sagemaker_session=sagemaker_session,
        ...
      )
    }
  )
)
