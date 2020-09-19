# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_randomcutforest.py
context("randomcutforest")

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
NUM_SAMPLES_PER_TREE = 20
NUM_TREES = 50
EVAL_METRICS = list("accuracy", "precision_recall_fscore")

COMMON_TRAIN_ARGS = list(
  "role"= ROLE,
  "instance_count"= INSTANCE_COUNT,
  "instance_type"= INSTANCE_TYPE
)

ALL_REQ_ARGS = COMMON_TRAIN_ARGS

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
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-randomcutforest-dummy"))
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-randomcutforest")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-randomcutforest-endpoint")
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)

test_that("test init required positional", {
  randomcutforest = RandomCutForest$new(
    ROLE,
    INSTANCE_COUNT,
    INSTANCE_TYPE,
    NUM_SAMPLES_PER_TREE,
    NUM_TREES,
    EVAL_METRICS,
    sagemaker_session=sagemaker_session
  )
  expect_equal(randomcutforest$role, ROLE)
  expect_equal(randomcutforest$instance_count, INSTANCE_COUNT)
  expect_equal(randomcutforest$instance_type, INSTANCE_TYPE)
  expect_equal(randomcutforest$num_trees, NUM_TREES)
  expect_equal(randomcutforest$num_samples_per_tree, NUM_SAMPLES_PER_TREE)
  expect_equal(randomcutforest$eval_metrics, EVAL_METRICS)
})

test_that("test init required named", {
  rf_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  randomcutforest = do.call(RandomCutForest$new, rf_args)

  expect_equal(randomcutforest$role, COMMON_TRAIN_ARGS$role)
  expect_equal(randomcutforest$instance_count, INSTANCE_COUNT)
  expect_equal(randomcutforest$instance_type, COMMON_TRAIN_ARGS$instance_type)
})

test_that("test all hyperparameters", {
  rf_args = c(sagemaker_session=sagemaker_session,
              num_trees=NUM_TREES,
              num_samples_per_tree=NUM_SAMPLES_PER_TREE,
              eval_metrics=list(EVAL_METRICS),
              ALL_REQ_ARGS)
  randomcutforest = do.call(RandomCutForest$new, rf_args)

  expect_equal(randomcutforest$hyperparameters(), list(
    num_samples_per_tree=NUM_SAMPLES_PER_TREE,
    num_trees=NUM_TREES,
    eval_metrics=EVAL_METRICS))
})

test_that("test image", {
  rf_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  randomcutforest = do.call(RandomCutForest$new, rf_args)

  expect_equal(randomcutforest$training_image_uri(), ImageUris$new()$retrieve("randomcutforest", REGION))
})

test_that("test iterable hyper parameters type", {
  rf_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list(num_trees="Dummy",
                    num_samples_per_tree="DUMMY")

  for(i in seq_along(test_param)){
    test_args = c(rf_args, test_param[i])
    expect_error(do.call(RandomCutForest$new, test_args))
  }
})

test_that("test optional hyper parameters value", {
  rf_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list(num_trees=49,
                    num_trees=1001,
                    num_samples_per_tree=0,
                    num_samples_per_tree=2049)

  for(i in seq_along(test_param)){
    test_args = c(rf_args, test_param[i])
    expect_error(do.call(RandomCutForest$new, test_args))
  }
})

PREFIX = "prefix"
FEATURE_DIM = 10
MAX_FEATURE_DIM = 10000
MINI_BATCH_SIZE = 1000

test_that("test prepare for training no mini batch_size", {
  rf_args = c(base_job_name="randomcutforest", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  randomcutforest=do.call(RandomCutForest$new, rf_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  randomcutforest$.prepare_for_training(data)

  expect_equal(randomcutforest$mini_batch_size , MINI_BATCH_SIZE)
})

test_that("test prepare for training no mini batch_size", {
  rf_args = c(base_job_name="randomcutforest", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  randomcutforest=do.call(RandomCutForest$new, rf_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(randomcutforest$.prepare_for_training(data, 1234))
})

test_that("test prepare for training feature dim greater than max allowed", {
  rf_args = c(base_job_name="randomcutforest", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  randomcutforest=do.call(RandomCutForest$new, rf_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=MAX_FEATURE_DIM+1,
    channel="train"
  )
  expect_error(randomcutforest$.prepare_for_training(data))
})

test_that("test model image", {
  rf_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  randomcutforest=do.call(RandomCutForest$new, rf_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  randomcutforest$fit(data, MINI_BATCH_SIZE)
  model = randomcutforest$create_model()

  expect_equal(model$image_uri, ImageUris$new()$retrieve("randomcutforest", REGION))
})

test_that("test model image", {
  rf_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  randomcutforest=do.call(RandomCutForest$new, rf_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  randomcutforest$fit(data, MINI_BATCH_SIZE)
  model = randomcutforest$create_model()
  predictor = model$deploy(1, INSTANCE_TYPE)

  expect_true(inherits(predictor, "RandomCutForestPredictor"))
})
