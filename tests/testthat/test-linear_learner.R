# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_linear_learner.py
context("linear_learner")

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"

PREDICTOR_TYPE = "binary_classifier"

COMMON_TRAIN_ARGS = list(
  "role"= ROLE,
  "instance_count"= INSTANCE_COUNT,
  "instance_type"= INSTANCE_TYPE
)

ALL_REQ_ARGS = c(list("predictor_type" = PREDICTOR_TYPE), COMMON_TRAIN_ARGS)

REGION = "us-west-2"
BUCKET_NAME = "Some-Bucket"

DESCRIBE_TRAINING_JOB_RESULT = list("ModelArtifacts"= list("S3ModelArtifacts"= "s3://bucket/model.tar.gz"))

ENDPOINT_DESC = list("EndpointConfigName"= "test-endpoint")

ENDPOINT_CONFIG_DESC = list("ProductionVariants"= list(list("ModelName"= "model-1"), list("ModelName"= "model-2")))

paws_mock <- Mock$new(name = "PawsCredentials", region_name = REGION)
sagemaker_session <- Mock$new(
  name = "Session",
  paws_credentials = paws_mock,
  paws_region_name=REGION,
  config=NULL,
  local_mode=FALSE,
  s3 = NULL)

sagemaker_session$default_bucket <- Mock$new()$return_value(BUCKET_NAME, .min_var = 0)
sagemaker_session$sagemaker$describe_training_job <- Mock$new()$return_value(DESCRIBE_TRAINING_JOB_RESULT)
sagemaker_session$sagemaker$describe_endpoint <- Mock$new()$return_value(ENDPOINT_DESC)
sagemaker_session$sagemaker$describe_endpoint_config <- Mock$new()$return_value(ENDPOINT_CONFIG_DESC)
sagemaker_session$s3$put_object <- Mock$new()$return_value(NULL)
sagemaker_session$expand_role <- Mock$new()$return_value(ROLE)
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-lr-dummy"))
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-lr")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-lr-endpoint")
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)

test_that("test init required positional", {
  lr = LinearLearner$new(
    ROLE,
    INSTANCE_COUNT,
    INSTANCE_TYPE,
    PREDICTOR_TYPE,
    sagemaker_session=sagemaker_session
  )
  expect_equal(lr$role, COMMON_TRAIN_ARGS$role)
  expect_equal(lr$instance_count, INSTANCE_COUNT)
  expect_equal(lr$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(lr$predictor_type, PREDICTOR_TYPE)
})

test_that("test init required named", {
  lr_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lr = do.call(LinearLearner$new, lr_args)

  expect_equal(lr$role, COMMON_TRAIN_ARGS$role)
  expect_equal(lr$instance_count, INSTANCE_COUNT)
  expect_equal(lr$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(lr$predictor_type, ALL_REQ_ARGS$predictor_type)
})

test_that("test all hyperparameters", {
  lr_args = c(sagemaker_session=sagemaker_session,
              binary_classifier_model_selection_criteria="accuracy",
              target_recall=0.5,
              target_precision=0.6,
              positive_example_weight_mult=0.1,
              epochs=1,
              use_bias=TRUE,
              num_models=5,
              num_calibration_samples=6,
              init_method="uniform",
              init_scale=0.1,
              init_sigma=0.001,
              init_bias=0,
              optimizer="sgd",
              loss="logistic",
              wd=0.4,
              l1=0.04,
              momentum=0.1,
              learning_rate=0.001,
              beta_1=0.2,
              beta_2=0.03,
              bias_lr_mult=5.5,
              bias_wd_mult=6.6,
              use_lr_scheduler=FALSE,
              lr_scheduler_step=2,
              lr_scheduler_factor=0.03,
              lr_scheduler_minimum_lr=0.001,
              normalize_data=FALSE,
              normalize_label=TRUE,
              unbias_data=TRUE,
              unbias_label=FALSE,
              num_point_for_scaler=3,
              margin=1.0,
              quantile=0.5,
              loss_insensitivity=0.1,
              huber_delta=0.1,
              early_stopping_patience=3,
              early_stopping_tolerance=0.001,
              num_classes=1,
              accuracy_top_k=3,
              f_beta=1.0,
              balance_multiclass_weights=FALSE,
              ALL_REQ_ARGS)
  lr = do.call(LinearLearner$new, lr_args)

  expect_equal(lr$hyperparameters(), list(
    binary_classifier_model_selection_criteria="accuracy",
    target_recall=0.5,
    target_precision=0.6,
    positive_example_weight_mult="0.1",
    epochs=1,
    use_bias=TRUE,
    num_models=5,
    num_calibration_samples=6,
    init_method="uniform",
    init_scale=0.1,
    init_sigma=0.001,
    init_bias=0,
    optimizer="sgd",
    loss="logistic",
    wd=0.4,
    l1=0.04,
    momentum=0.1,
    learning_rate=0.001,
    beta_1=0.2,
    beta_2=0.03,
    bias_lr_mult=5.5,
    bias_wd_mult=6.6,
    use_lr_scheduler=FALSE,
    lr_scheduler_step=2,
    lr_scheduler_factor=0.03,
    lr_scheduler_minimum_lr=0.001,
    normalize_data=FALSE,
    normalize_label=TRUE,
    unbias_data=TRUE,
    unbias_label=FALSE,
    num_point_for_scaler=3,
    margin=1.0,
    quantile=0.5,
    loss_insensitivity=0.1,
    huber_delta=0.1,
    early_stopping_patience=3,
    early_stopping_tolerance=0.001,
    num_classes=1,
    accuracy_top_k=3,
    f_beta=1.0,
    balance_multiclass_weights=FALSE)
  )
})

test_that("test image", {
  lr_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lr = do.call(LinearLearner$new, lr_args)

  expect_equal(lr$training_image_uri(), ImageUris$new()$retrieve("linear-learner", REGION))
})

test_that("test required hyper parameters type", {
  lr_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("predictor_type"=0)

  for(i in seq_along(test_param)){
    test_args = c(lr_args, test_param[i])
    expect_error(do.call(LinearLearner$new, test_args))
  }
})

test_that("test num classes is required for multiclass classifier", {
  lr_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lr_args$predictor_type ="multiclass_classifier"
  expect_error(do.call(LinearLearner$new, lr_args),
               "For predictor_type 'multiclass_classifier', 'num_classes' should be set to a value greater than 2.")
})


test_that("test num classes can be string for multiclass classifier", {
  lr_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lr_args$predictor_type ="multiclass_classifier"
  lr_args$num_classes = "3"
  lr = do.call(LinearLearner$new, lr_args)

  expect_equal(lr$role, COMMON_TRAIN_ARGS$role)
  expect_equal(lr$instance_count, INSTANCE_COUNT)
  expect_equal(lr$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(lr$predictor_type, lr_args$predictor_type)
  expect_equal(lr$num_classes, as.integer(lr_args$num_classes))
})

test_that("test optional hyper parameters type", {
  lr_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("binary_classifier_model_selection_criteria"=0,
                    "target_recall"= "string",
                    "target_precision"= "string",
                    "epochs"= "string",
                    "num_models"= "string",
                    "num_calibration_samples"= "string",
                    "init_method"= 0,
                    "init_scale"= "string",
                    "init_sigma"= "string",
                    "init_bias"= "string",
                    "optimizer"= 0,
                    "loss"= 0,
                    "wd"= "string",
                    "l1"= "string",
                    "momentum"= "string",
                    "learning_rate"= "string",
                    "beta_1"= "string",
                    "beta_2"= "string",
                    "bias_lr_mult"= "string",
                    "bias_wd_mult"= "string",
                    "lr_scheduler_step"= "string",
                    "lr_scheduler_factor"= "string",
                    "lr_scheduler_minimum_lr"= "string",
                    "num_point_for_scaler"= "string",
                    "margin"= "string",
                    "quantile"= "string",
                    "loss_insensitivity"= "string",
                    "huber_delta"= "string",
                    "early_stopping_patience"= "string",
                    "early_stopping_tolerance"= "string",
                    "num_classes"= "string",
                    "accuracy_top_k"= "string",
                    "f_beta"= "string")

  for(i in seq_along(test_param)){
    test_args = c(lr_args, test_param[i])
    expect_error(do.call(LinearLearner$new, test_args))
  }
})

test_that("test optional hyper parameters type", {
  lr_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("binary_classifier_model_selection_criteria"= "string",
                    "target_recall"= 0,
                    "target_recall"= 1,
                    "target_precision"= 0,
                    "target_precision"= 1,
                    "epochs"= 0,
                    "num_models"= 0,
                    "num_calibration_samples"= 0,
                    "init_method"= "string",
                    "init_scale"= 0,
                    "init_sigma"= 0,
                    "optimizer"= "string",
                    "loss"= "string",
                    "wd"= -1,
                    "l1"= -1,
                    "momentum"= 1,
                    "learning_rate"= 0,
                    "beta_1"= 1,
                    "beta_2"= 1,
                    "bias_lr_mult"= 0,
                    "bias_wd_mult"= -1,
                    "lr_scheduler_step"= 0,
                    "lr_scheduler_factor"= 0,
                    "lr_scheduler_factor"= 1,
                    "lr_scheduler_minimum_lr"= 0,
                    "num_point_for_scaler"= 0,
                    "margin"= -1,
                    "quantile"= 0,
                    "quantile"= 1,
                    "loss_insensitivity"= 0,
                    "huber_delta"= -1,
                    "early_stopping_patience"= 0,
                    "early_stopping_tolerance"= 0,
                    "num_classes"= 0,
                    "accuracy_top_k"= 0,
                    "f_beta"= -1.0)

  for(i in seq_along(test_param)){
    test_args = c(lr_args, test_param[i])
    expect_error(do.call(LinearLearner$new, test_args))
  }
})

PREFIX = "prefix"
FEATURE_DIM = 10
DEFAULT_MINI_BATCH_SIZE = 1000

test_that("test prepare for training calculate batch size 1", {
  lr_args = c(base_job_name="lr", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lr=do.call(LinearLearner$new, lr_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  lr$.prepare_for_training(data)

  expect_equal(lr$mini_batch_size , 1)
})

test_that("test prepare for training calculate batch size 2", {
  lr_args = c(base_job_name="lr", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lr=do.call(LinearLearner$new, lr_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=DEFAULT_MINI_BATCH_SIZE,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  lr$.prepare_for_training(data)

  expect_equal(lr$mini_batch_size , DEFAULT_MINI_BATCH_SIZE)
})

test_that("test prepare for training multiple channel", {
  lr_args = c(base_job_name="lr", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lr=do.call(LinearLearner$new, lr_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=DEFAULT_MINI_BATCH_SIZE,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  lr$.prepare_for_training(list(data, data))

  expect_equal(lr$mini_batch_size , DEFAULT_MINI_BATCH_SIZE)
})


test_that("test prepare for training multiple channel no train", {
  lr_args = c(base_job_name="lr", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lr=do.call(LinearLearner$new, lr_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=DEFAULT_MINI_BATCH_SIZE,
    feature_dim=FEATURE_DIM,
    channel="mock"
  )
  expect_error(lr$.prepare_for_training(list(data, data)), "Must provide train channel.")
})

test_that("test call fit pass batch size", {
  lr_args = c(base_job_name="lr", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lr=do.call(LinearLearner$new, lr_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=DEFAULT_MINI_BATCH_SIZE,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  lr$fit(data, 10)

  expect_equal(lr$latest_training_job , "sagemaker-lr-dummy")
  expect_equal(lr$mini_batch_size , 10)
})

test_that("test model image", {
  lr_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lr=do.call(LinearLearner$new, lr_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  lr$fit(data)
  model = lr$create_model()

  expect_equal(model$image_uri, ImageUris$new()$retrieve("linear-learner", REGION))
})

test_that("test predictor type", {
  lr_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lr=do.call(LinearLearner$new, lr_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  lr$fit(data)
  model = lr$create_model()
  predictor = model$deploy(1, INSTANCE_TYPE)

  expect_true(inherits(predictor, "LinearLearnerPredictor"))
})
