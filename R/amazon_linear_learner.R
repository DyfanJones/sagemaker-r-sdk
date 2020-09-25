# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/linear_learner.py

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

#' @title A supervised learning algorithms used for solving classification or regression problems.
#' @description For input, you give the model labeled examples (x, y). x is a high-dimensional vector and
#'              y is a numeric label. For binary classification problems, the label must be either 0 or 1.
#'              For multiclass classification problems, the labels must be from 0 to num_classes - 1. For
#'              regression problems, y is a real number. The algorithm learns a linear function, or, for
#'              classification problems, a linear threshold function, and maps a vector x to an approximation
#'              of the label y
#' @export
LinearLearner = R6Class("LinearLearner",
  inherit = AmazonAlgorithmEstimatorBase,
  public = list(

    #' @field repo_name
    #' sagemaker repo name for framework
    repo_name = "linear-learner",

    #' @field repo_version
    #' version of framework
    repo_version = 1,

    #' @field DEFAULT_MINI_BATCH_SIZE
    #' The size of each mini-batch to use when training.
    DEFAULT_MINI_BATCH_SIZE = 1000,

    #' @description An :class:`Estimator` for binary classification and regression.
    #'              Amazon SageMaker Linear Learner provides a solution for both
    #'              classification and regression problems, allowing for exploring different
    #'              training objectives simultaneously and choosing the best solution from a
    #'              validation set. It allows the user to explore a large number of models
    #'              and choose the best, which optimizes either continuous objectives such
    #'              as mean square error, cross entropy loss, absolute error, etc., or
    #'              discrete objectives suited for classification such as F1 measure,
    #'              precision@@recall, accuracy. The implementation provides a significant
    #'              speedup over naive hyperparameter optimization techniques and an added
    #'              convenience, when compared with solutions providing a solution only to
    #'              continuous objectives.
    #'              This Estimator may be fit via calls to
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit_ndarray`
    #'              or
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
    #'              The former allows a LinearLearner model to be fit on a 2-dimensional
    #'              numpy array. The latter requires Amazon
    #'              :class:`~sagemaker.amazon.record_pb2.Record` protobuf serialized data to
    #'              be stored in S3.
    #'              To learn more about the Amazon protobuf Record class and how to
    #'              prepare bulk data in this format, please consult AWS technical
    #'              documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html
    #'              After this Estimator is fit, model data is stored in S3. The model
    #'              may be deployed to an Amazon SageMaker Endpoint by invoking
    #'              :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
    #'              deploying an Endpoint, ``deploy`` returns a
    #'              :class:`~sagemaker.amazon.linear_learner.LinearLearnerPredictor` object
    #'              that can be used to make class or regression predictions, using the
    #'              trained model.
    #'              LinearLearner Estimators can be configured by setting
    #'              hyperparameters. The available hyperparameters for LinearLearner are
    #'              documented below. For further information on the AWS LinearLearner
    #'              algorithm, please consult AWS technical documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if accessing AWS resource.
    #' @param instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.c4.xlarge'.
    #' @param predictor_type (str): The type of predictor to learn. Either
    #'              "binary_classifier" or "multiclass_classifier" or "regressor".
    #' @param binary_classifier_model_selection_criteria (str): One of 'accuracy',
    #'              'f1', 'f_beta', 'precision_at_target_recall', 'recall_at_target_precision',
    #'              'cross_entropy_loss', 'loss_function'
    #' @param target_recall (float): Target recall. Only applicable if
    #'              binary_classifier_model_selection_criteria is
    #'              precision_at_target_recall.
    #' @param target_precision (float): Target precision. Only applicable if
    #'              binary_classifier_model_selection_criteria is
    #'              recall_at_target_precision.
    #' @param positive_example_weight_mult (float): The importance weight of
    #'              positive examples is multiplied by this constant. Useful for
    #'              skewed datasets. Only applies for classification tasks.
    #' @param epochs (int): The maximum number of passes to make over the training
    #'              data.
    #' @param use_bias (bool): Whether to include a bias field
    #' @param num_models (int): Number of models to train in parallel. If not set,
    #'              the number of parallel models to train will be decided by the
    #'              algorithm itself. One model will be trained according to the
    #'              given training parameter (regularization, optimizer, loss) and
    #'              the rest by close by parameters.
    #' @param num_calibration_samples (int): Number of observations to use from
    #'              validation dataset for doing model calibration (finding the best threshold).
    #' @param init_method (str): Function to use to set the initial model weights.
    #'              One of "uniform" or "normal"
    #' @param init_scale (float): For "uniform" init, the range of values.
    #' @param init_sigma (float): For "normal" init, the standard-deviation.
    #' @param init_bias (float): Initial weight for bias term
    #' @param optimizer (str): One of 'sgd', 'adam', 'rmsprop' or 'auto'
    #' @param loss (str): One of 'logistic', 'squared_loss', 'absolute_loss',
    #'              'hinge_loss', 'eps_insensitive_squared_loss', 'eps_insensitive_absolute_loss',
    #'              'quantile_loss', 'huber_loss' or
    #'              'softmax_loss' or 'auto'.
    #' @param wd (float): L2 regularization parameter i.e. the weight decay
    #'              parameter. Use 0 for no L2 regularization.
    #' @param l1 (float): L1 regularization parameter. Use 0 for no L1
    #'              regularization.
    #' @param momentum (float): Momentum parameter of sgd optimizer.
    #' @param learning_rate (float): The SGD learning rate
    #' @param beta_1 (float): Exponential decay rate for first moment estimates.
    #'              Only applies for adam optimizer.
    #' @param beta_2 (float): Exponential decay rate for second moment estimates.
    #'              Only applies for adam optimizer.
    #' @param bias_lr_mult (float): Allows different learning rate for the bias
    #'              term. The actual learning rate for the bias is learning rate times bias_lr_mult.
    #' @param bias_wd_mult (float): Allows different regularization for the bias
    #'              term. The actual L2 regularization weight for the bias is wd times bias_wd_mult.
    #'              By default there is no regularization on the bias term.
    #' @param use_lr_scheduler (bool): If true, we use a scheduler for the
    #'              learning rate.
    #' @param lr_scheduler_step (int): The number of steps between decreases of
    #'              the learning rate. Only applies to learning rate scheduler.
    #' @param lr_scheduler_factor (float): Every lr_scheduler_step the learning
    #'              rate will decrease by this quantity. Only applies for learning
    #'              rate scheduler.
    #' @param lr_scheduler_minimum_lr (float): The learning rate will never
    #'              decrease to a value lower than this. Only applies for learning rate scheduler.
    #' @param normalize_data (bool): Normalizes the features before training to
    #'              have standard deviation of 1.0.
    #' @param normalize_label (bool): Normalizes the regression label to have a
    #'              standard deviation of 1.0. If set for classification, it will be
    #'              ignored.
    #' @param unbias_data (bool): If true, features are modified to have mean 0.0.
    #' @param unbias_label (bool): If true, labels are modified to have mean 0.0.
    #' @param num_point_for_scaler (int): The number of data points to use for
    #'              calculating the normalizing and unbiasing terms.
    #' @param margin (float): The margin for hinge_loss.
    #' @param quantile (float): Quantile for quantile loss. For quantile q, the
    #'              model will attempt to produce predictions such that true_label < prediction with
    #'              probability q.
    #' @param loss_insensitivity (float): Parameter for epsilon insensitive loss
    #'              type. During training and metric evaluation, any error smaller than this is
    #'              considered to be zero.
    #' @param huber_delta (float): Parameter for Huber loss. During training and
    #'              metric evaluation, compute L2 loss for errors smaller than delta and L1 loss for
    #'              errors larger than delta.
    #' @param early_stopping_patience (int): The number of epochs to wait before ending training
    #'              if no improvement is made. The improvement is training loss if validation data is
    #'              not provided, or else it is the validation loss or the binary classification model
    #'              selection criteria like accuracy, f1-score etc. To disable early stopping,
    #'              set early_stopping_patience to a value larger than epochs.
    #' @param early_stopping_tolerance (float): Relative tolerance to measure an
    #'              improvement in loss. If the ratio of the improvement in loss divided by the
    #'              previous best loss is smaller than this value, early stopping will
    #'              consider the improvement to be zero.
    #' @param num_classes (int): The number of classes for the response variable.
    #'              Required when predictor_type is multiclass_classifier and ignored otherwise. The
    #'              classes are assumed to be labeled 0, ..., num_classes - 1.
    #' @param accuracy_top_k (int): The value of k when computing the Top K
    #'              Accuracy metric for multiclass classification. An example is scored as correct
    #'              if the model assigns one of the top k scores to the true
    #'              label.
    #' @param f_beta (float): The value of beta to use when calculating F score
    #'              metrics for binary or multiclass classification. Also used if
    #'              binary_classifier_model_selection_criteria is f_beta.
    #' @param balance_multiclass_weights (bool): Whether to use class weights
    #'              which give each class equal importance in the loss function. Only used when
    #'              predictor_type is multiclass_classifier.
    #' @param ... : base class keyword argument values.
    initialize = function(role,
                          instance_count,
                          instance_type,
                          predictor_type,
                          binary_classifier_model_selection_criteria=NULL,
                          target_recall=NULL,
                          target_precision=NULL,
                          positive_example_weight_mult=NULL,
                          epochs=NULL,
                          use_bias=NULL,
                          num_models=NULL,
                          num_calibration_samples=NULL,
                          init_method=NULL,
                          init_scale=NULL,
                          init_sigma=NULL,
                          init_bias=NULL,
                          optimizer=NULL,
                          loss=NULL,
                          wd=NULL,
                          l1=NULL,
                          momentum=NULL,
                          learning_rate=NULL,
                          beta_1=NULL,
                          beta_2=NULL,
                          bias_lr_mult=NULL,
                          bias_wd_mult=NULL,
                          use_lr_scheduler=NULL,
                          lr_scheduler_step=NULL,
                          lr_scheduler_factor=NULL,
                          lr_scheduler_minimum_lr=NULL,
                          normalize_data=NULL,
                          normalize_label=NULL,
                          unbias_data=NULL,
                          unbias_label=NULL,
                          num_point_for_scaler=NULL,
                          margin=NULL,
                          quantile=NULL,
                          loss_insensitivity=NULL,
                          huber_delta=NULL,
                          early_stopping_patience=NULL,
                          early_stopping_tolerance=NULL,
                          num_classes=NULL,
                          accuracy_top_k=NULL,
                          f_beta=NULL,
                          balance_multiclass_weights=NULL,
                          ...){

      private$.binary_classifier_model_selection_criteria = Hyperparameter$new(
        "binary_classifier_model_selection_criteria",
        Validation$new()$isin(c(
          "accuracy",
          "f1",
          "f_beta",
          "precision_at_target_recall",
          "recall_at_target_precision",
          "cross_entropy_loss",
          "loss_function")
        ),
        data_type=DataTypes$new()$str,
        obj = self
      )
      private$.target_recall = Hyperparameter$new("target_recall", list(Validation$new()$gt(0), Validation$new()$lt(1)), "A float in (0,1)", DataTypes$new()$float, obj = self)
      private$.target_precision = Hyperparameter$new("target_precision", list(Validation$new()$gt(0), Validation$new()$lt(1)), "A float in (0,1)", DataTypes$new()$float , obj = self)
      private$.positive_example_weight_mult = Hyperparameter$new(
        "positive_example_weight_mult",list(), "A float greater than 0 or 'auto' or 'balanced'", DataTypes$new()$str,obj = self
      )
      private$.epochs = Hyperparameter$new("epochs", Validation$new()$gt(0), "An integer greater-than 0", DataTypes$new()$int, obj = self)
      private$.predictor_type = Hyperparameter$new(
        "predictor_type",
        Validation$new()$isin(c("binary_classifier", "regressor", "multiclass_classifier")),
        'One of "binary_classifier" or "multiclass_classifier" or "regressor"',
        DataTypes$new()$str,
        obj = self)
      private$.use_bias = Hyperparameter$new("use_bias", list(), "Either True or False", DataTypes$new()$bool, obj = self)
      private$.num_models = Hyperparameter$new("num_models", Validation$new()$gt(0), "An integer greater-than 0", DataTypes$new()$int, obj = self)
      private$.num_calibration_samples = Hyperparameter$new("num_calibration_samples", Validation$new()$gt(0), "An integer greater-than 0", DataTypes$new()$int, obj = self)
      private$.init_method = Hyperparameter$new("init_method", Validation$new()$isin(c("uniform", "normal")), 'One of "uniform" or "normal"', DataTypes$new()$str, obj = self)
      private$.init_scale = Hyperparameter$new("init_scale", Validation$new()$gt(0), "A float greater-than 0", DataTypes$new()$float, obj = self)
      private$.init_sigma = Hyperparameter$new("init_sigma", Validation$new()$gt(0), "A float greater-than 0", DataTypes$new()$float, obj = self)
      private$.init_bias = Hyperparameter$new("init_bias", list(), "A number", DataTypes$new()$float, obj = self)
      private$.optimizer = Hyperparameter$new(
        "optimizer",
        Validation$new()$isin(c("sgd", "adam", "rmsprop", "auto")),
        'One of "sgd", "adam", "rmsprop" or "auto',
        DataTypes$new()$str,
        obj = self)
      private$.loss = Hyperparameter$new(
        "loss",
        Validation$new()$isin(c(
          "logistic",
          "squared_loss",
          "absolute_loss",
          "hinge_loss",
          "eps_insensitive_squared_loss",
          "eps_insensitive_absolute_loss",
          "quantile_loss",
          "huber_loss",
          "softmax_loss",
          "auto")
        ),
        paste0(
          '"logistic", "squared_loss", "absolute_loss", "hinge_loss", "eps_insensitive_squared_loss",',
          ' "eps_insensitive_absolute_loss", "quantile_loss", "huber_loss", "softmax_loss" or "auto"'),
        DataTypes$new()$str,
        obj = self)
      private$.wd = Hyperparameter$new("wd", Validation$new()$ge(0), "A float greater-than or equal to 0", DataTypes$new()$float, obj = self)
      private$.l1 = Hyperparameter$new("l1", Validation$new()$ge(0), "A float greater-than or equal to 0", DataTypes$new()$float, obj = self)
      private$.momentum = Hyperparameter$new("momentum", list(Validation$new()$ge(0), Validation$new()$lt(1)), "A float in [0,1)", DataTypes$new()$float, obj = self)
      private$.learning_rate = Hyperparameter$new("learning_rate", Validation$new()$gt(0), "A float greater-than 0", DataTypes$new()$float, obj = self)
      private$.beta_1 = Hyperparameter$new("beta_1", list(Validation$new()$ge(0), Validation$new()$lt(1)), "A float in [0,1)", DataTypes$new()$float, obj = self)
      private$.beta_2 = Hyperparameter$new("beta_2",list(Validation$new()$ge(0), Validation$new()$lt(1)), "A float in [0,1)", DataTypes$new()$float, obj = self)
      private$.bias_lr_mult = Hyperparameter$new("bias_lr_mult", Validation$new()$gt(0), "A float greater-than 0", DataTypes$new()$float, obj = self)
      private$.bias_wd_mult = Hyperparameter$new("bias_wd_mult", Validation$new()$ge(0), "A float greater-than or equal to 0", DataTypes$new()$float, obj = self)
      private$.use_lr_scheduler = Hyperparameter$new("use_lr_scheduler", list(), "A boolean", DataTypes$new()$bool, obj = self)
      private$.lr_scheduler_step = Hyperparameter$new("lr_scheduler_step", Validation$new()$gt(0), "An integer greater-than 0", DataTypes$new()$int, obj = self)
      private$.lr_scheduler_factor = Hyperparameter$new("lr_scheduler_factor", list(Validation$new()$gt(0), Validation$new()$lt(1)), "A float in (0,1)", DataTypes$new()$float, obj = self)
      private$.lr_scheduler_minimum_lr = Hyperparameter$new("lr_scheduler_minimum_lr", Validation$new()$gt(0), "A float greater-than 0", DataTypes$new()$float, obj = self)
      private$.normalize_data = Hyperparameter$new("normalize_data", list(), "A boolean", DataTypes$new()$bool, obj = self)
      private$.normalize_label = Hyperparameter$new("normalize_label", list(), "A boolean", DataTypes$new()$bool, obj = self)
      private$.unbias_data = Hyperparameter$new("unbias_data", list(), "A boolean", DataTypes$new()$bool, obj = self)
      private$.unbias_label = Hyperparameter$new("unbias_label", list(), "A boolean", DataTypes$new()$bool, obj = self)
      private$.num_point_for_scaler = Hyperparameter$new("num_point_for_scaler", Validation$new()$gt(0), "An integer greater-than 0", DataTypes$new()$int, obj = self)
      private$.margin = Hyperparameter$new("margin", Validation$new()$ge(0), "A float greater-than or equal to 0", DataTypes$new()$float, obj = self)
      private$.quantile = Hyperparameter$new("quantile", list(Validation$new()$gt(0), Validation$new()$lt(1)), "A float in (0,1)", DataTypes$new()$float, obj = self)
      private$.loss_insensitivity = Hyperparameter$new("loss_insensitivity", Validation$new()$gt(0), "A float greater-than 0", DataTypes$new()$float, obj = self)
      private$.huber_delta = Hyperparameter$new("huber_delta", Validation$new()$ge(0), "A float greater-than or equal to 0", DataTypes$new()$float, obj = self)
      private$.early_stopping_patience = Hyperparameter$new("early_stopping_patience", Validation$new()$gt(0), "An integer greater-than 0", DataTypes$new()$int, obj = self)
      private$.early_stopping_tolerance = Hyperparameter$new(
        "early_stopping_tolerance", Validation$new()$gt(0), "A float greater-than 0", DataTypes$new()$float, obj = self
      )
      private$.num_classes = Hyperparameter$new("num_classes", list(Validation$new()$gt(0), Validation$new()$le(1000000)), "An integer in [1,1000000]", DataTypes$new()$int, obj = self)
      private$.accuracy_top_k = Hyperparameter$new("accuracy_top_k", list(Validation$new()$gt(0), Validation$new()$le(1000000)), "An integer in [1,1000000]", DataTypes$new()$int, obj = self)
      private$.f_beta = Hyperparameter$new("f_beta", Validation$new()$gt(0), "A float greater-than 0", DataTypes$new()$float, obj = self)
      private$.balance_multiclass_weights = Hyperparameter$new("balance_multiclass_weights", list(), "A boolean", DataTypes$new()$bool, obj = self)


      super$initialize(role, instance_count, instance_type, ...)
      self$predictor_type = predictor_type
      self$binary_classifier_model_selection_criteria = binary_classifier_model_selection_criteria
      self$target_recall = target_recall
      self$target_precision = target_precision
      self$positive_example_weight_mult = positive_example_weight_mult
      self$epochs = epochs
      self$use_bias = use_bias
      self$num_models = num_models
      self$num_calibration_samples = num_calibration_samples
      self$init_method = init_method
      self$init_scale = init_scale
      self$init_sigma = init_sigma
      self$init_bias = init_bias
      self$optimizer = optimizer
      self$loss = loss
      self$wd = wd
      self$l1 = l1
      self$momentum = momentum
      self$learning_rate = learning_rate
      self$beta_1 = beta_1
      self$beta_2 = beta_2
      self$bias_lr_mult = bias_lr_mult
      self$bias_wd_mult = bias_wd_mult
      self$use_lr_scheduler = use_lr_scheduler
      self$lr_scheduler_step = lr_scheduler_step
      self$lr_scheduler_factor = lr_scheduler_factor
      self$lr_scheduler_minimum_lr = lr_scheduler_minimum_lr
      self$normalize_data = normalize_data
      self$normalize_label = normalize_label
      self$unbias_data = unbias_data
      self$unbias_label = unbias_label
      self$num_point_for_scaler = num_point_for_scaler
      self$margin = margin
      self$quantile = quantile
      self$loss_insensitivity = loss_insensitivity
      self$huber_delta = huber_delta
      self$early_stopping_patience = early_stopping_patience
      self$early_stopping_tolerance = early_stopping_tolerance
      self$num_classes = num_classes
      self$accuracy_top_k = accuracy_top_k
      self$f_beta = f_beta
      self$balance_multiclass_weights = balance_multiclass_weights

      if(self$predictor_type == "multiclass_classifier" &&
         (is.null(num_classes) || as.integer(num_classes) < 3))
        stop("For predictor_type 'multiclass_classifier', 'num_classes' should be set to a value greater than 2.",
             call. = F)
    },

    #' @description Return a :class:`~sagemaker.amazon.LinearLearnerModel` referencing
    #'              the latest s3 model data produced by this Estimator.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
    #'              the model. Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param ... : Additional kwargs passed to the LinearLearnerModel constructor.
    create_model = function(vpc_config_override="VPC_CONFIG_DEFAULT",
                            ...){
      return(LinearLearnerModel$new(
        self$model_data,
        self$role,
        self$sagemaker_session,
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
      num_records = NULL
      if (inherits(records, "list")){
        for(record in records){
          if(record$channel == "train"){
            num_records = record$num_records
            break}
        }
        if(is.null(num_records))
          stop("Must provide train channel.", call. = F)
      } else{
          num_records = records$num_records
      }
      # mini_batch_size can't be greater than number of records or training job fails
      default_mini_batch_size = min(
        self$DEFAULT_MINI_BATCH_SIZE, max(1, as.integer(num_records / self$instance_count))
        )
      mini_batch_size = mini_batch_size %||% default_mini_batch_size
      super$.prepare_for_training(
        records, mini_batch_size=mini_batch_size, job_name=job_name
        )
    }
  ),
  private = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    .binary_classifier_model_selection_criteria=NULL,
    .target_recall=NULL,
    .target_precision=NULL,
    .positive_example_weight_mult=NULL,
    .epochs=NULL,
    .use_bias=NULL,
    .num_models=NULL,
    .num_calibration_samples=NULL,
    .init_method=NULL,
    .init_scale=NULL,
    .init_sigma=NULL,
    .init_bias=NULL,
    .optimizer=NULL,
    .loss=NULL,
    .wd=NULL,
    .l1=NULL,
    .momentum=NULL,
    .learning_rate=NULL,
    .beta_1=NULL,
    .beta_2=NULL,
    .bias_lr_mult=NULL,
    .bias_wd_mult=NULL,
    .use_lr_scheduler=NULL,
    .lr_scheduler_step=NULL,
    .lr_scheduler_factor=NULL,
    .lr_scheduler_minimum_lr=NULL,
    .normalize_data=NULL,
    .normalize_label=NULL,
    .unbias_data=NULL,
    .unbias_label=NULL,
    .num_point_for_scaler=NULL,
    .margin=NULL,
    .quantile=NULL,
    .loss_insensitivity=NULL,
    .huber_delta=NULL,
    .early_stopping_patience=NULL,
    .early_stopping_tolerance=NULL,
    .num_classes=NULL,
    .accuracy_top_k=NULL,
    .f_beta=NULL,
    .balance_multiclass_weights=NULL
  ),
  active = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    #' @field binary_classifier_model_selection_criteria
    #' One of 'accuracy', 'f1', 'f_beta', 'precision_at_target_recall', 'recall_at_target_precision', 'cross_entropy_loss', 'loss_function'
    binary_classifier_model_selection_criteria = function(value){
      if(missing(value))
        return(private$.binary_classifier_model_selection_criteria$descriptor)
      private$.binary_classifier_model_selection_criteria$descriptor = value
    },

    #' @field target_recall
    #' Only applicable if binary_classifier_model_selection_criteria is precision_at_target_recall
    target_recall = function(value){
      if(missing(value))
        return(private$.target_recall$descriptor)
      private$.target_recall$descriptor = value
    },

    #' @field target_precision
    #' Only applicable if binary_classifier_model_selection_criteria is recall_at_target_precision.
    target_precision = function(value){
      if(missing(value))
        return(private$.target_precision$descriptor)
      private$.target_precision$descriptor = value
    },

    #' @field positive_example_weight_mult
    #' The importance weight of positive examples is multiplied by this constant.
    positive_example_weight_mult = function(value){
      if(missing(value))
        return(private$.positive_example_weight_mult$descriptor)
      private$.positive_example_weight_mult$descriptor = value
    },

    #' @field epochs
    #' The maximum number of passes to make over the training data.
    epochs = function(value){
      if(missing(value))
        return(private$.epochs$descriptor)
      private$.epochs$descriptor = value
    },

    #' @field use_bias
    #' Whether to include a bias field
    use_bias = function(value){
      if(missing(value))
        return(private$.use_bias$descriptor)
      private$.use_bias$descriptor = value
    },

    #' @field num_models
    #' Number of models to train in parallel
    num_models = function(value){
      if(missing(value))
        return(private$.num_models$descriptor)
      private$.num_models$descriptor = value
    },

    #' @field num_calibration_samples
    #' Number of observations to use from validation dataset for doing model calibration
    num_calibration_samples = function(value){
      if(missing(value))
        return(private$.num_calibration_samples$descriptor)
      private$.num_calibration_samples$descriptor = value
    },

    #' @field init_method
    #' Function to use to set the initial model weights.
    init_method = function(value){
      if(missing(value))
        return(private$.init_method$descriptor)
      private$.init_method$descriptor = value
    },

    #' @field init_scale
    #' For "uniform" init, the range of values.
    init_scale = function(value){
      if(missing(value))
        return(private$.init_scale$descriptor)
      private$.init_scale$descriptor = value
    },

    #' @field init_sigma
    #' For "normal" init, the standard-deviation.
    init_sigma = function(value){
      if(missing(value))
        return(private$.init_sigma$descriptor)
      private$.init_sigma$descriptor = value
    },

    #' @field init_bias
    #' Initial weight for bias term
    init_bias = function(value){
      if(missing(value))
        return(private$.init_bias$descriptor)
      private$.init_bias$descriptor = value
    },

    #' @field optimizer
    #' One of 'sgd', 'adam', 'rmsprop' or 'auto'
    optimizer = function(value){
      if(missing(value))
        return(private$.optimizer$descriptor)
      private$.optimizer$descriptor = value
    },

    #' @field loss
    #' One of 'logistic', 'squared_loss', 'absolute_loss',
    #'        'hinge_loss', 'eps_insensitive_squared_loss', 'eps_insensitive_absolute_loss',
    #'        'quantile_loss', 'huber_loss' or
    #'        'softmax_loss' or 'auto'.
    loss = function(value){
      if(missing(value))
        return(private$.loss$descriptor)
      private$.loss$descriptor = value
    },

    #' @field wd
    #' L2 regularization parameter
    wd = function(value){
      if(missing(value))
        return(private$.wd$descriptor)
      private$.wd$descriptor = value
    },

    #' @field l1
    #' L1 regularization parameter.
    l1 = function(value){
      if(missing(value))
        return(private$.l1$descriptor)
      private$.l1$descriptor = value
    },

    #' @field momentum
    #' Momentum parameter of sgd optimizer.
    momentum = function(value){
      if(missing(value))
        return(private$.momentum$descriptor)
      private$.momentum$descriptor = value
    },

    #' @field learning_rate
    #' The SGD learning rate
    learning_rate = function(value){
      if(missing(value))
        return(private$.learning_rate$descriptor)
      private$.learning_rate$descriptor = value
    },

    #' @field beta_1
    #' Exponential decay rate for first moment estimates.
    beta_1 = function(value){
      if(missing(value))
        return(private$.beta_1$descriptor)
      private$.beta_1$descriptor = value
    },

    #' @field beta_2
    #' Exponential decay rate for second moment estimates.
    beta_2 = function(value){
      if(missing(value))
        return(private$.beta_2$descriptor)
      private$.beta_2$descriptor = value
    },

    #' @field bias_lr_mult
    #' Allows different learning rate for the bias term.
    bias_lr_mult = function(value){
      if(missing(value))
        return(private$.bias_lr_mult$descriptor)
      private$.bias_lr_mult$descriptor = value
    },

    #' @field bias_wd_mult
    #' Allows different regularization for the bias term.
    bias_wd_mult = function(value){
      if(missing(value))
        return(private$.bias_wd_mult$descriptor)
      private$.bias_wd_mult$descriptor = value
    },

    #' @field use_lr_scheduler
    #' If true, we use a scheduler for the learning rate.
    use_lr_scheduler = function(value){
      if(missing(value))
        return(private$.use_lr_scheduler$descriptor)
      private$.use_lr_scheduler$descriptor = value
    },

    #' @field lr_scheduler_step
    #' The number of steps between decreases of the learning rate
    lr_scheduler_step = function(value){
      if(missing(value))
        return(private$.lr_scheduler_step$descriptor)
      private$.lr_scheduler_step$descriptor = value
    },

    #' @field lr_scheduler_factor
    #' Every lr_scheduler_step the learning rate will decrease by this quantity.
    lr_scheduler_factor = function(value){
      if(missing(value))
        return(private$.lr_scheduler_factor$descriptor)
      private$.lr_scheduler_factor$descriptor = value
    },

    #' @field lr_scheduler_minimum_lr
    #' Every lr_scheduler_step the learning rate will decrease by this quantity.
    lr_scheduler_minimum_lr = function(value){
      if(missing(value))
        return(private$.lr_scheduler_minimum_lr$descriptor)
      private$.lr_scheduler_minimum_lr$descriptor = value
    },

    #' @field normalize_data
    #' Normalizes the features before training to have standard deviation of 1.0.
    normalize_data = function(value){
      if(missing(value))
        return(private$.normalize_data$descriptor)
      private$.normalize_data$descriptor = value
    },

    #' @field normalize_label
    #' Normalizes the regression label to have a standard deviation of 1.0.
    normalize_label = function(value){
      if(missing(value))
        return(private$.normalize_label$descriptor)
      private$.normalize_label$descriptor = value
    },

    #' @field unbias_data
    #' If true, features are modified to have mean 0.0.
    unbias_data = function(value){
      if(missing(value))
        return(private$.unbias_data$descriptor)
      private$.unbias_data$descriptor = value
    },

    #' @field unbias_label
    #' If true, labels are modified to have mean 0.0.
    unbias_label = function(value){
      if(missing(value))
        return(private$.unbias_label$descriptor)
      private$.unbias_label$descriptor = value
    },

    #' @field num_point_for_scaler
    #' The number of data points to use for calculating the normalizing and unbiasing terms.
    num_point_for_scaler = function(value){
      if(missing(value))
        return(private$.num_point_for_scaler$descriptor)
      private$.num_point_for_scaler$descriptor = value
    },

    #' @field margin
    #' The margin for hinge_loss.
    margin = function(value){
      if(missing(value))
        return(private$.margin$descriptor)
      private$.margin$descriptor = value
    },

    #' @field quantile
    #' Quantile for quantile loss.
    quantile = function(value){
      if(missing(value))
        return(private$.quantile$descriptor)
      private$.quantile$descriptor = value
    },

    #' @field loss_insensitivity
    #' Parameter for epsilon insensitive loss type.
    loss_insensitivity = function(value){
      if(missing(value))
        return(private$.loss_insensitivity$descriptor)
      private$.loss_insensitivity$descriptor = value
    },

    #' @field huber_delta
    #' Parameter for Huber loss.
    huber_delta = function(value){
      if(missing(value))
        return(private$.huber_delta$descriptor)
      private$.huber_delta$descriptor = value
    },

    #' @field early_stopping_patience
    #' The number of epochs to wait before ending training if no improvement is made.
    early_stopping_patience = function(value){
      if(missing(value))
        return(private$.early_stopping_patience$descriptor)
      private$.early_stopping_patience$descriptor = value
    },

    #' @field early_stopping_tolerance
    #' Relative tolerance to measure an improvement in loss.
    early_stopping_tolerance = function(value){
      if(missing(value))
        return(private$.early_stopping_tolerance$descriptor)
      private$.early_stopping_tolerance$descriptor = value
    },

    #' @field num_classes
    #' The number of classes for the response variable.
    num_classes = function(value){
      if(missing(value))
        return(private$.num_classes$descriptor)
      private$.num_classes$descriptor = value
    },

    #' @field accuracy_top_k
    #' The value of k when computing the Top K
    accuracy_top_k = function(value){
      if(missing(value))
        return(private$.accuracy_top_k$descriptor)
      private$.accuracy_top_k$descriptor = value
    },

    #' @field f_beta
    #' The value of beta to use when calculating F score
    #'        metrics for binary or multiclass classification.
    f_beta = function(value){
      if(missing(value))
        return(private$.f_beta$descriptor)
      private$.f_beta$descriptor = value
    },

    #' @field balance_multiclass_weights
    #' Whether to use class weights which give each class equal importance
    #'        in the loss function.
    balance_multiclass_weights = function(value){
      if(missing(value))
        return(private$.balance_multiclass_weights$descriptor)
      private$.balance_multiclass_weights$descriptor = value
    }
  ),
  lock_objects = F
)

#' @title Performs binary-classification or regression prediction from input vectors.
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
LinearLearnerPredictor = R6Class("LinearLearnerPredictor",
  inherit = Predictor,
  public = list(

    #' @description Initialize LinearLearnerPredictor Class
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

#' @title Reference LinearLearner s3 model data.
#' @description Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and returns a
#'              :class:`LinearLearnerPredictor`
#' @export
LinearLearnerModel = R6Class("LinearLearnerModel",
  inherit = Model,
  public = list(

    #' @description Initialize LinearLearnerModel class
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
        LinearLearner$public_fields$repo_name,
        sagemaker_session$paws_region_name,
        version=LinearLearner$public_fields$repo_version
      )
      super$initialize(
        image_uri,
        model_data,
        role,
        predictor_cls=LinearLearnerPredictor,
        sagemaker_session=sagemaker_session,
        ...
      )
    }
  ),
  lock_objects = F
)
