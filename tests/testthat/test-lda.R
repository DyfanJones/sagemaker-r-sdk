# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_lda.py
context("lda")

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
NUM_TOPICS = 3

COMMON_TRAIN_ARGS = list(
  "role"= ROLE,
  "instance_type"= INSTANCE_TYPE
)

ALL_REQ_ARGS = c(list("num_topics" = NUM_TOPICS),
                 COMMON_TRAIN_ARGS)

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

sagemaker_session$default_bucket <- Mock$new()$return_value(BUCKET_NAME)
sagemaker_session$sagemaker$describe_training_job <- Mock$new()$return_value(DESCRIBE_TRAINING_JOB_RESULT)
sagemaker_session$sagemaker$describe_endpoint <- Mock$new()$return_value(ENDPOINT_DESC)
sagemaker_session$sagemaker$describe_endpoint_config <- Mock$new()$return_value(ENDPOINT_CONFIG_DESC)
sagemaker_session$s3$put_object <- Mock$new()$return_value(NULL)
sagemaker_session$expand_role <- Mock$new()$return_value(ROLE)
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-lda-dummy"))
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-lda")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-lda-endpoint")
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)

test_that("test init required positional", {
  lda = LDA$new(ROLE, INSTANCE_TYPE, NUM_TOPICS, sagemaker_session=sagemaker_session)
  expect_equal(lda$role, COMMON_TRAIN_ARGS$role)
  expect_equal(lda$instance_count, INSTANCE_COUNT)
  expect_equal(lda$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(lda$num_topics, NUM_TOPICS)
})

test_that("test init required named", {
  lda_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lda = do.call(LDA$new, lda_args)

  expect_equal(lda$role, COMMON_TRAIN_ARGS$role)
  expect_equal(lda$instance_count, INSTANCE_COUNT)
  expect_equal(lda$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(lda$num_topics, NUM_TOPICS)
})

test_that("test all hyperparameters", {
  lda_args = c(sagemaker_session=sagemaker_session,
               alpha0=2.2,
               max_restarts=3,
               max_iterations=10,
               tol=3.3,
               ALL_REQ_ARGS)
  lda = do.call(LDA$new, lda_args)

  expect_equal(lda$hyperparameters(), list(
    num_topics=ALL_REQ_ARGS$num_topic,
    alpha0=2.2,
    max_restarts=3,
    max_iterations=10,
    tol=3.3)
  )
})

test_that("test image", {
  lda_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lda = do.call(LDA$new, lda_args)

  expect_equal(lda$training_image_uri(), ImageUris$new()$retrieve("lda", REGION))
})

test_that("test required hyper parameters type", {
  lda_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list(num_topics = "string")

  for(i in seq_along(test_param)){
    test_args = c(lda_args, test_param[i])
    expect_error(do.call(LDA$new, test_args))
  }
})

test_that("test required hyper parameters value", {
  lda_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list(num_topics = 0)

  for(i in seq_along(test_param)){
    test_args = c(lda_args, test_param[i])
    expect_error(do.call(LDA$new, test_args))
  }
})

test_that("test optional hyper parameters type", {
  lda_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("alpha0"= "string",
                    "max_restarts"= "string",
                    "max_iterations"= "string",
                    "tol"="string")

  for(i in seq_along(test_param)){
    test_args = c(lda_args, test_param[i])
    expect_error(do.call(LDA$new, test_args))
  }
})

test_that("test error optional hyper parameters type", {
  lda_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("max_restarts"= 0,
                    "max_iterations"= 0,
                    "tol"= 0)

  for(i in seq_along(test_param)){
    test_args = c(lda_args, test_param[i])
    expect_error(do.call(LDA$new, test_args))
  }
})

PREFIX = "prefix"
FEATURE_DIM = 10
MINI_BATCH_SIZE = 200

test_that("test call fit", {
  lda_args = c(base_job_name="lda", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lda=do.call(LDA$new, lda_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  lda$fit(data, MINI_BATCH_SIZE)

  expect_equal(lda$latest_training_job , "sagemaker-lda-dummy")
  expect_equal(lda$mini_batch_size , MINI_BATCH_SIZE)
})

test_that("test prepare for training none mini batch_size", {
  lda_args = c(base_job_name="lda", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lda=do.call(LDA$new, lda_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  lda$fit(data)

  expect_equal(lda$latest_training_job , "sagemaker-lda-dummy")
})

test_that("test prepare for training wrong type mini batch size", {
  lda_args = c(base_job_name="lda", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lda=do.call(LDA$new, lda_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(lda$.prepare_for_training(data, "some"))
})

test_that("test prepare for training wrong value lower mini batch size", {
  lda_args = c(base_job_name="lda", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lda=do.call(LDA$new, lda_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(lda$.prepare_for_training(data, 0))
  expect_error(lda$.prepare_for_training(data, NULL))
})

test_that("test model image", {
  lda_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lda=do.call(LDA$new, lda_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  lda$fit(data, MINI_BATCH_SIZE)
  model = lda$create_model()

  expect_equal(model$image_uri, ImageUris$new()$retrieve("lda", REGION))
})

test_that("test predictor type", {
  lda_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  lda=do.call(LDA$new, lda_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  lda$fit(data, MINI_BATCH_SIZE)
  model = lda$create_model()
  predictor = model$deploy(1, INSTANCE_TYPE)

  expect_true(inherits(predictor, "LDAPredictor"))
})
