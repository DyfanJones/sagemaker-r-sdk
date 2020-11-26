# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_ipinsights.py
context("ipinsights")

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"

# Required algorithm hyperparameters
NUM_ENTITY_VECTORS = 10000
VECTOR_DIM = 128

COMMON_TRAIN_ARGS = list(
  "role"= ROLE,
  "instance_count"= INSTANCE_COUNT,
  "instance_type"= INSTANCE_TYPE
)

ALL_REQ_ARGS = c(list("num_entity_vectors"= NUM_ENTITY_VECTORS, "vector_dim"= VECTOR_DIM),  COMMON_TRAIN_ARGS)

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
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-ipinsight-dummy"))
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-ipinsight")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-ipinsight-endpoint")
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)

test_that("test init required positional", {
  ipinsights = IPInsights$new(
    ROLE,
    INSTANCE_COUNT,
    INSTANCE_TYPE,
    NUM_ENTITY_VECTORS,
    VECTOR_DIM,
    sagemaker_session=sagemaker_session
  )
  expect_equal(ipinsights$role, COMMON_TRAIN_ARGS$role)
  expect_equal(ipinsights$instance_count, INSTANCE_COUNT)
  expect_equal(ipinsights$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(ipinsights$num_entity_vectors, NUM_ENTITY_VECTORS)
  expect_equal(ipinsights$vector_dim, VECTOR_DIM)
})

test_that("test init required named", {
  ip_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ipinsights = do.call(IPInsights$new, ip_args)

  expect_equal(ipinsights$role, COMMON_TRAIN_ARGS$role)
  expect_equal(ipinsights$instance_count, INSTANCE_COUNT)
  expect_equal(ipinsights$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(ipinsights$num_entity_vectors, NUM_ENTITY_VECTORS)
  expect_equal(ipinsights$vector_dim, VECTOR_DIM)
})

test_that("test all hyperparameters", {
  ip_args = c(sagemaker_session=sagemaker_session,
              batch_metrics_publish_interval=100,
              epochs=10,
              learning_rate=0.001,
              num_ip_encoder_layers=3,
              random_negative_sampling_rate=5,
              shuffled_negative_sampling_rate=5,
              weight_decay=5.0,
              ALL_REQ_ARGS)
  ipinsights = do.call(IPInsights$new, ip_args)

  expect_equal(ipinsights$hyperparameters(), list(
    num_entity_vectors=ALL_REQ_ARGS$num_entity_vectors,
    vector_dim=ALL_REQ_ARGS$vector_dim,
    batch_metrics_publish_interval=100,
    epochs=10,
    learning_rate=0.001,
    num_ip_encoder_layers=3,
    random_negative_sampling_rate=5,
    shuffled_negative_sampling_rate=5,
    weight_decay=5.0)
  )
})

test_that("test image", {
  ip_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ipinsights = do.call(IPInsights$new, ip_args)

  expect_equal(ipinsights$training_image_uri(), ImageUris$new()$retrieve("ipinsights", REGION))
})

test_that("test required hyper parameters type", {
  ip_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("num_entity_vectors" = "string", "vector_dim" = "string")

  for(i in seq_along(test_param)){
    test_args = c(ip_args, test_param[i])
    expect_error(do.call(IPInsights$new, test_args))
  }
})

test_that("test required hyper parameters value", {
  ip_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("num_entity_vectors" = 0,
                    "num_entity_vectors"= 500000001,
                    "vector_dim" = 3,
                    "vector_dim" = 4097)

  for(i in seq_along(test_param)){
    test_args = c(ip_args, test_param[i])
    expect_error(do.call(IPInsights$new, test_args))
  }
})

test_that("test optional hyper parameters value", {
  ip_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("batch_metrics_publish_interval" = 0,
                    "epochs" = 0,
                    "learning_rate" = 0,
                    "learning_rate" = 11,
                    "num_ip_encoder_layers" = -1,
                    "num_ip_encoder_layers" = 101,
                    "random_negative_sampling_rate" = -1,
                    "random_negative_sampling_rate" = 501,
                    "shuffled_negative_sampling_rate" = -1,
                    "shuffled_negative_sampling_rate" = 501,
                    "weight_decay" = -1,
                    "weight_decay" = 11)

  for(i in seq_along(test_param)){
    test_args = c(ip_args, test_param[i])
    expect_error(do.call(IPInsights$new, test_args))
  }
})

PREFIX = "prefix"
FEATURE_DIM = NULL
MINI_BATCH_SIZE = 200

test_that("test call fit", {
  ip_args = c(base_job_name="ipinsights", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ipinsights=do.call(IPInsights$new, ip_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  ipinsights$fit(data, MINI_BATCH_SIZE)

  expect_equal(ipinsights$latest_training_job , "sagemaker-ipinsight-dummy")
  expect_equal(ipinsights$mini_batch_size , MINI_BATCH_SIZE)
})


test_that("test prepare for training none mini batch_size", {
  ip_args = c(base_job_name="ipinsights", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ipinsights=do.call(IPInsights$new, ip_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  ipinsights$fit(data)

  expect_equal(ipinsights$latest_training_job , "sagemaker-ipinsight-dummy")
})

test_that("test prepare for training wrong type mini batch size", {
  ip_args = c(base_job_name="ipinsights", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ipinsights=do.call(IPInsights$new, ip_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(ipinsights$.prepare_for_training(data, "some"))
})

test_that("test prepare for training wrong value lower mini batch size", {
  ip_args = c(base_job_name="ipinsights", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ipinsights=do.call(IPInsights$new, ip_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(ipinsights$.prepare_for_training(data, 0))
})

test_that("test prepare for training wrong value upper mini batch size", {
  ip_args = c(base_job_name="ipinsights", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ipinsights=do.call(IPInsights$new, ip_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(ipinsights$.prepare_for_training(data, 500001))
})

test_that("test model image", {
  ip_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ipinsights=do.call(IPInsights$new, ip_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  ipinsights$fit(data, MINI_BATCH_SIZE)
  model = ipinsights$create_model()

  expect_equal(model$image_uri, ImageUris$new()$retrieve("ipinsights", REGION))
})

test_that("test predictor type", {
  ip_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ipinsights=do.call(IPInsights$new, ip_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  ipinsights$fit(data, MINI_BATCH_SIZE)
  model = ipinsights$create_model()
  predictor = model$deploy(1, INSTANCE_TYPE)

  expect_true(inherits(predictor, "IPInsightsPredictor"))
})
