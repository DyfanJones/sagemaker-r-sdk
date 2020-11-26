# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_fm.py
context("factorization machines")

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
NUM_FACTORS = 3
PREDICTOR_TYPE = "regressor"

COMMON_TRAIN_ARGS = list(
  "role"= ROLE,
  "instance_count"= INSTANCE_COUNT,
  "instance_type"= INSTANCE_TYPE
)

ALL_REQ_ARGS = c(
  list("num_factors"= NUM_FACTORS, "predictor_type"= PREDICTOR_TYPE), COMMON_TRAIN_ARGS
)

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
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-fm-dummy"))
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-fm")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-fm-endpoint")
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)


test_that("test init required positional", {
  fm = FactorizationMachines$new(
    "myrole", 1, "ml.c4.xlarge", 3, "regressor", sagemaker_session=sagemaker_session
  )
  expect_equal(fm$role, "myrole")
  expect_equal(fm$instance_count, 1)
  expect_equal(fm$instance_type, "ml.c4.xlarge")
  expect_equal(fm$num_factors, 3)
  expect_equal(fm$predictor_type, "regressor")
})

test_that("test init required named", {
  fm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  fm = do.call(FactorizationMachines$new, fm_args)

  expect_equal(fm$role, COMMON_TRAIN_ARGS$role)
  expect_equal(fm$instance_count, COMMON_TRAIN_ARGS$instance_count)
  expect_equal(fm$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(fm$num_factors, ALL_REQ_ARGS$num_factors)
  expect_equal(fm$predictor_type, ALL_REQ_ARGS$predictor_type)
})

test_that("test all hyperparameters", {
  fm_args = c(
    sagemaker_session=sagemaker_session,
    epochs=2,
    clip_gradient=1e2,
    eps=0.001,
    rescale_grad=2.2,
    bias_lr=0.01,
    linear_lr=0.002,
    factors_lr=0.0003,
    bias_wd=0.0004,
    linear_wd=1.01,
    factors_wd=1.002,
    bias_init_method="uniform",
    bias_init_scale=0.1,
    bias_init_sigma=0.05,
    bias_init_value=2.002,
    linear_init_method="constant",
    linear_init_scale=0.02,
    linear_init_sigma=0.003,
    linear_init_value=1.0,
    factors_init_method="normal",
    factors_init_scale=1.101,
    factors_init_sigma=1.202,
    factors_init_value=1.303,
    ALL_REQ_ARGS)
  fm = do.call(FactorizationMachines$new, fm_args)

  expect_equal(fm$hyperparameters(), list(
    num_factors=ALL_REQ_ARGS$num_factors,
    predictor_type=ALL_REQ_ARGS$predictor_type,
    epochs=2,
    clip_gradient=1e2,
    eps=0.001,
    rescale_grad=2.2,
    bias_lr=0.01,
    linear_lr=0.002,
    factors_lr=0.0003,
    bias_wd=0.0004,
    linear_wd=1.01,
    factors_wd=1.002,
    bias_init_method="uniform",
    bias_init_scale=0.1,
    bias_init_sigma=0.05,
    bias_init_value=2.002,
    linear_init_method="constant",
    linear_init_scale=0.02,
    linear_init_sigma=0.003,
    linear_init_value=1.0,
    factors_init_method="normal",
    factors_init_scale=1.101,
    factors_init_sigma=1.202,
    factors_init_value=1.303
    )
  )
})

test_that("test image", {
  fm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  fm = do.call(FactorizationMachines$new, fm_args)

  expect_equal(fm$training_image_uri(), ImageUris$new()$retrieve("factorization-machines", REGION))
})

test_that("test required hyper parameters type", {
  fm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  fm_args$num_factors = "Dummy"
  expect_error(do.call(FactorizationMachines$new, fm_args), "Could not convert object 'Dummy' to integer")

  fm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  fm_args$predictor_type = "Dummy"
  expect_error(do.call(FactorizationMachines$new, fm_args), 'Invalid hyperparameter value Dummy for predictor_type. Expecting: Value "binary_classifier" or "regressor"')
})

test_that("test optional hyper parameters type", {
  optional_hyper_parameters = list(
    "epochs"="string",
    "clip_gradient"="string",
    "eps"="string",
    "rescale_grad"="string",
    "bias_lr"="string",
    "linear_lr"="string",
    "factors_lr"="string",
    "bias_wd"="string",
    "linear_wd"="string",
    "factors_wd"="string",
    "bias_init_method"=0,
    "bias_init_scale"="string",
    "bias_init_sigma"="string",
    "bias_init_value"="string",
    "linear_init_method"=0,
    "linear_init_scale"="string",
    "linear_init_sigma"="string",
    "linear_init_value"="string",
    "factors_init_method"=0,
    "factors_init_scale"="string",
    "factors_init_sigma"="string",
    "factors_init_value"="string"
  )
  for(i in seq_along(optional_hyper_parameters)){
    fm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS, optional_hyper_parameters[i])
    expect_error(do.call(FactorizationMachines$new, fm_args))
  }
})

test_that("test optional hyper parameters type", {
  optional_hyper_parameters = list(
    "epochs"=0,
    "bias_lr"=-1,
    "linear_lr"=-1,
    "factors_lr"=-1,
    "bias_wd"=-1,
    "linear_wd"=-1,
    "factors_wd"=-1,
    "bias_init_method"="string",
    "bias_init_scale"=-1,
    "bias_init_sigma"=-1,
    "linear_init_method"="string",
    "linear_init_scale"=-1,
    "linear_init_sigma"=-1,
    "factors_init_method"="string",
    "factors_init_scale"=-1,
    "factors_init_sigma"=-1
  )
  for(i in seq_along(optional_hyper_parameters)){
    fm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS, optional_hyper_parameters[i])
    expect_error(do.call(FactorizationMachines$new, fm_args))
  }
})

PREFIX = "prefix"
FEATURE_DIM = 10
MINI_BATCH_SIZE = 200

test_that("test call fit", {
  fm_args = c(base_job_name="fm", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  fm = do.call(FactorizationMachines$new, fm_args)

  data = RecordSet$new(
    sprintf("s3://%s/%s", BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  fm$fit(data, MINI_BATCH_SIZE)

  expect_equal(fm$latest_training_job, "sagemaker-fm-dummy")
})

test_that("test prepare for training wrong type mini batch size", {
  fm_args = c(base_job_name="fm", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  fm = do.call(FactorizationMachines$new, fm_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s", BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(fm$.__enclos_env__$private$.prepare_for_training(data, "some"))
})

test_that("test prepare for training wrong value mini batch size", {
  fm_args = c(base_job_name="fm", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  fm = do.call(FactorizationMachines$new, fm_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s", BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(fm$.__enclos_env__$private$.prepare_for_training(data, 0))
})

test_that("test model image", {
  fm_args = c(base_job_name="fm", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  fm = do.call(FactorizationMachines$new, fm_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s", BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  fm$fit(data, MINI_BATCH_SIZE)
  model = fm$create_model()

  expect_equal(model$image_uri, ImageUris$new()$retrieve("factorization-machines", REGION))
})

test_that("test predictor type", {
  fm_args = c(base_job_name="fm", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  fm = do.call(FactorizationMachines$new, fm_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s", BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  fm$fit(data, MINI_BATCH_SIZE)
  model = fm$create_model()
  predictor = model$deploy(1, INSTANCE_TYPE)

  expect_true(inherits(predictor, "FactorizationMachinesPredictor"))
})
