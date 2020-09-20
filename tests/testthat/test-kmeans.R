# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_kmeans.py
context("kmeans")

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
K = 2
EVAL_METRICS = list("msd", "ssd")

# Required algorithm hyperparameters
NUM_ENTITY_VECTORS = 10000
VECTOR_DIM = 128

COMMON_TRAIN_ARGS = list(
  "role"= ROLE,
  "instance_count"= INSTANCE_COUNT,
  "instance_type"= INSTANCE_TYPE
)

ALL_REQ_ARGS = c(list(k = K),  COMMON_TRAIN_ARGS)

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
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-kmeans-dummy"))
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-kmeans")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-kmeans-endpoint")
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)

test_that("test init required positional", {
  kmeans = KMeans$new(ROLE, INSTANCE_COUNT, INSTANCE_TYPE, K, sagemaker_session=sagemaker_session)
  expect_equal(kmeans$role, COMMON_TRAIN_ARGS$role)
  expect_equal(kmeans$instance_count, INSTANCE_COUNT)
  expect_equal(kmeans$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(kmeans$k, K)
})

test_that("test init required named", {
  km_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  kmeans = do.call(KMeans$new, km_args)

  expect_equal(kmeans$role, COMMON_TRAIN_ARGS$role)
  expect_equal(kmeans$instance_count, INSTANCE_COUNT)
  expect_equal(kmeans$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(kmeans$k, ALL_REQ_ARGS$k)
})

test_that("test all hyperparameters", {
  km_args = c(sagemaker_session=sagemaker_session,
              init_method="random",
              max_iterations=3,
              tol=0.5,
              num_trials=5,
              local_init_method="kmeans++",
              half_life_time_size=0,
              epochs=10,
              center_factor=2,
              eval_metrics=list(EVAL_METRICS),
              ALL_REQ_ARGS)
  kmeans = do.call(KMeans$new, km_args)

  expect_equal(kmeans$hyperparameters(), list(
    k=ALL_REQ_ARGS$k,
    init_method="random",
    local_lloyd_max_iter=3,
    local_lloyd_tol=0.5,
    local_lloyd_num_trials=5,
    local_lloyd_init_method="kmeans++",
    half_life_time_size=0,
    epochs=10,
    extra_center_factor=2,
    eval_metrics=EVAL_METRICS,
    force_dense="True")
  )
})

test_that("test image", {
  km_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  kmeans = do.call(KMeans$new, km_args)

  expect_equal(kmeans$training_image_uri(), ImageUris$new()$retrieve("kmeans", REGION))
})

test_that("test required hyper parameters type", {
  km_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("k" = "string")

  for(i in seq_along(test_param)){
    test_args = c(km_args, test_param[i])
    expect_error(do.call(KMeans$new, test_args))
  }
})

test_that("test required hyper parameters value", {
  km_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("k" = 0)

  for(i in seq_along(test_param)){
    test_args = c(km_args, test_param[i])
    expect_error(do.call(KMeans$new, test_args))
  }
})

test_that("test optional hyper parameters type", {
  km_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("init_method"=0,
                    "max_iterations"="string",
                    "tol"="string",
                    "num_trials"="string",
                    "local_init_method"=0,
                    "half_life_time_size"="string",
                    "epochs"="string",
                    "center_factor"="string")

  for(i in seq_along(test_param)){
    test_args = c(km_args, test_param[i])
    expect_error(do.call(KMeans$new, test_args))
  }
})

test_that("test optional hyper parameters value", {
  km_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("init_method"="string",
                    "max_iterations"=0,
                    "tol"=-.1,
                    "tol"=1.1,
                    "num_trials"=0,
                    "local_init_method"="string",
                    "half_life_time_size"=-1,
                    "epochs"="string",
                    "center_factor"=0)

  for(i in seq_along(test_param)){
    test_args = c(km_args, test_param[i])
    expect_error(do.call(KMeans$new, test_args))
  }
})

PREFIX = "prefix"
FEATURE_DIM = 10
MINI_BATCH_SIZE = 200

test_that("test call fit", {
  km_args = c(base_job_name="kmeans", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  kmeans=do.call(KMeans$new, km_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  kmeans$fit(data, MINI_BATCH_SIZE)

  expect_equal(kmeans$latest_training_job , "sagemaker-kmeans-dummy")
  expect_equal(kmeans$mini_batch_size , MINI_BATCH_SIZE)
})


test_that("test prepare for training no mini batch_size", {
  km_args = c(base_job_name="kmeans", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  kmeans=do.call(KMeans$new, km_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  kmeans$.prepare_for_training(data)

  expect_equal(kmeans$mini_batch_size, 5000)
})

test_that("test prepare for training wrong type mini batch size", {
  km_args = c(base_job_name="kmeans", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  kmeans=do.call(KMeans$new, km_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(kmeans$.prepare_for_training(data, "some"))
})

test_that("test prepare for training wrong value lower mini batch size", {
  km_args = c(base_job_name="kmeans", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  kmeans=do.call(KMeans$new, km_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(kmeans$.prepare_for_training(data, 0))
})

test_that("test model image", {
  km_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  kmeans=do.call(KMeans$new, km_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  kmeans$fit(data, MINI_BATCH_SIZE)
  model = kmeans$create_model()

  expect_equal(model$image_uri, ImageUris$new()$retrieve("kmeans", REGION))
})

test_that("test predictor type", {
  km_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  kmeans=do.call(KMeans$new, km_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  kmeans$fit(data, MINI_BATCH_SIZE)
  model = kmeans$create_model()
  predictor = model$deploy(1, INSTANCE_TYPE)

  expect_true(inherits(predictor, "KMeansPredictor"))
})
