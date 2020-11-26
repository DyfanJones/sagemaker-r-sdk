# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_pca.py
context("pca")

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
NUM_COMPONENTS = 5

COMMON_TRAIN_ARGS = list(
  "role"= ROLE,
  "instance_count"= INSTANCE_COUNT,
  "instance_type"= INSTANCE_TYPE
)

ALL_REQ_ARGS = c(list("num_components"=NUM_COMPONENTS), COMMON_TRAIN_ARGS)

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
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-pca-dummy"))
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-pca")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-pca-endpoint")
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)

test_that("test init required positional", {
  pca = PCA$new(
    ROLE,
    INSTANCE_COUNT,
    INSTANCE_TYPE,
    NUM_COMPONENTS,
    sagemaker_session=sagemaker_session)
  expect_equal(pca$role, COMMON_TRAIN_ARGS$role)
  expect_equal(pca$instance_count, INSTANCE_COUNT)
  expect_equal(pca$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(pca$num_components, NUM_COMPONENTS)
})

test_that("test init required named", {
  pca_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  pca = do.call(PCA$new, pca_args)

  expect_equal(pca$role, COMMON_TRAIN_ARGS$role)
  expect_equal(pca$instance_count, INSTANCE_COUNT)
  expect_equal(pca$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(pca$num_components, ALL_REQ_ARGS$num_components)
})

test_that("test all hyperparameters", {
  pca_args = c(sagemaker_session=sagemaker_session,
               algorithm_mode="regular",
               subtract_mean="True",
               extra_components=1,
               ALL_REQ_ARGS)
  pca = do.call(PCA$new, pca_args)

  expect_equal(pca$hyperparameters(), list(
    num_components=ALL_REQ_ARGS$num_components,
    algorithm_mode="regular",
    subtract_mean=TRUE,
    extra_components=1
  ))
})

test_that("test image", {
  pca_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  pca = do.call(PCA$new, pca_args)

  expect_equal(pca$training_image_uri(), ImageUris$new()$retrieve("pca", REGION))
})

test_that("test required hyper parameters type", {
  pca_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  pca_args$num_components = NULL
  test_param = list("num_components"="string")

  for(i in seq_along(test_param)){
    test_args = c(pca_args, test_param[i])
    expect_error(do.call(PCA$new, test_args))
  }
})

test_that("test required hyper parameters value", {
  pca_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  pca_args$num_components = NULL
  test_param = list("num_components"=0)

  for(i in seq_along(test_param)){
    test_args = c(pca_args, test_param[i])
    expect_error(do.call(PCA$new, test_args))
  }
})

test_that("test optional hyper parameters type", {
  pca_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("algorithm_mode"=0,
                    "extra_components"="string")

  for(i in seq_along(test_param)){
    test_args = c(pca_args, test_param[i])
    expect_error(do.call(PCA$new, test_args))
  }
})

test_that("test error optional hyper parameters value", {
  pca_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("algorithm_mode"="string")

  for(i in seq_along(test_param)){
    test_args = c(pca_args, test_param[i])
    expect_error(do.call(PCA$new, test_args))
  }
})

PREFIX = "prefix"
FEATURE_DIM = 10
MINI_BATCH_SIZE = 200

test_that("test call fit", {
  pca_args = c(base_job_name="pca", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  pca=do.call(PCA$new, pca_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  pca$fit(data, MINI_BATCH_SIZE)

  expect_equal(pca$latest_training_job , "sagemaker-pca-dummy")
  expect_equal(pca$mini_batch_size , MINI_BATCH_SIZE)
})

test_that("test prepare for training none mini batch_size", {
  pca_args = c(base_job_name="pca", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  pca=do.call(PCA$new, pca_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  pca$fit(data)

  expect_equal(pca$latest_training_job , "sagemaker-pca-dummy")
})

test_that("test prepare for training no mini batch size", {
  pca_args = c(base_job_name="pca", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  pca=do.call(PCA$new, pca_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  pca$.prepare_for_training(data)

  expect_equal(pca$mini_batch_size, 1)
})

test_that("test prepare for training wrong type mini batch size", {
  pca_args = c(base_job_name="pca", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  pca=do.call(PCA$new, pca_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(pca$fit(data, "some"))
})

test_that("test prepare for training multiple channel", {
  pca_args = c(base_job_name="pca", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  pca=do.call(PCA$new, pca_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  pca$.prepare_for_training(list(data, data))

  expect_equal(pca$mini_batch_size, 1)
})

test_that("test prepare for training multiple channel no train", {
  pca_args = c(base_job_name="pca", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  pca=do.call(PCA$new, pca_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="mock"
  )

  expect_error(pca$.prepare_for_training(list(data, data)), "Must provide train channel.")
})

test_that("test model image", {
  pca_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  pca=do.call(PCA$new, pca_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  pca$fit(data, MINI_BATCH_SIZE)
  model = pca$create_model()

  expect_equal(model$image_uri, ImageUris$new()$retrieve("pca", REGION))
})

test_that("test predictor type", {
  pca_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  pca=do.call(PCA$new, pca_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  pca$fit(data, MINI_BATCH_SIZE)
  model = pca$create_model()
  predictor = model$deploy(1, INSTANCE_TYPE)

  expect_true(inherits(predictor, "PCAPredictor"))
})
