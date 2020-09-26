# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_ntm.py
context("ntm")

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
NUM_TOPICS = 5

COMMON_TRAIN_ARGS = list(
  "role"= ROLE,
  "instance_count"= INSTANCE_COUNT,
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
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-ntm-dummy"))
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-ntm")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-ntm-endpoint")
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)

test_that("test init required positional", {
  ntm = NTM$new(
    ROLE,
    INSTANCE_COUNT,
    INSTANCE_TYPE,
    NUM_TOPICS,
    sagemaker_session=sagemaker_session
    )
  expect_equal(ntm$role, COMMON_TRAIN_ARGS$role)
  expect_equal(ntm$instance_count, INSTANCE_COUNT)
  expect_equal(ntm$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(ntm$num_topics, NUM_TOPICS)
})

test_that("test init required named", {
  ntm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ntm = do.call(NTM$new, ntm_args)

  expect_equal(ntm$role, COMMON_TRAIN_ARGS$role)
  expect_equal(ntm$instance_count, INSTANCE_COUNT)
  expect_equal(ntm$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(ntm$num_topics, NUM_TOPICS)
})

test_that("test all hyperparameters", {
  ntm_args = c(sagemaker_session=sagemaker_session,
               encoder_layers=list(list(1, 2, 3)),
               epochs=3,
               encoder_layers_activation="tanh",
               optimizer="sgd",
               tolerance=0.05,
               num_patience_epochs=2,
               batch_norm=FALSE,
               rescale_gradient=0.5,
               clip_gradient=0.5,
               weight_decay=0.5,
               learning_rate=0.5,
               ALL_REQ_ARGS)
  ntm = do.call(NTM$new, ntm_args)

  expect_equal(ntm$hyperparameters(), list(
    num_topics=ALL_REQ_ARGS$num_topics,
    encoder_layers=list(1, 2, 3),
    epochs=3,
    encoder_layers_activation="tanh",
    optimizer="sgd",
    tolerance=0.05,
    num_patience_epochs=2,
    batch_norm=FALSE,
    rescale_gradient=0.5,
    clip_gradient=0.5,
    weight_decay=0.5,
    learning_rate=0.5)
  )
})

test_that("test image", {
  ntm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ntm = do.call(NTM$new, ntm_args)

  expect_equal(ntm$training_image_uri(), ImageUris$new()$retrieve("ntm", REGION))
})

test_that("test required hyper parameters type", {
  ntm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ntm_args$num_topics = NULL
  test_param = list(num_topics = "string")

  for(i in seq_along(test_param)){
    test_args = c(ntm_args, test_param[i])
    expect_error(do.call(NTM$new, test_args))
  }
})

test_that("test required hyper parameters value", {
  ntm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ntm_args$num_topics = NULL
  test_param = list(num_topics = 0,
                    num_topics = 10000)

  for(i in seq_along(test_param)){
    test_args = c(ntm_args, test_param[i])
    expect_error(do.call(NTM$new, test_args))
  }
})

test_that("test optional hyper parameters type", {
  ntm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("epochs"= "string",
                    "encoder_layers_activation"= 0,
                    "optimizer"= 0,
                    "tolerance"= "string",
                    "num_patience_epochs"= "string",
                    "rescale_gradient"= "string",
                    "clip_gradient"= "string",
                    "weight_decay"= "string",
                    "learning_rate"= "string")

  for(i in seq_along(test_param)){
    test_args = c(ntm_args, test_param[i])
    expect_error(do.call(NTM$new, test_args))
  }
})

test_that("test error optional hyper parameters value", {
  ntm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("epochs"= 0,
                    "epochs"= 1000,
                    "encoder_layers_activation"= "string",
                    "optimizer"= "string",
                    "tolerance"= 0,
                    "tolerance"= 0.5,
                    "num_patience_epochs"= 0,
                    "num_patience_epochs"= 100,
                    "rescale_gradient"= 0,
                    "rescale_gradient"= 10,
                    "clip_gradient"= 0,
                    "weight_decay"= -1,
                    "weight_decay"= 2,
                    "learning_rate"= 0,
                    "learning_rate"= 2)

  for(i in seq_along(test_param)){
    test_args = c(ntm_args, test_param[i])
    expect_error(do.call(NTM$new, test_args))
  }
})

PREFIX = "prefix"
FEATURE_DIM = 10
MINI_BATCH_SIZE = 200

test_that("test call fit", {
  ntm_args = c(base_job_name="ntm", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ntm=do.call(NTM$new, ntm_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  ntm$fit(data, MINI_BATCH_SIZE)

  expect_equal(ntm$latest_training_job , "sagemaker-ntm-dummy")
  expect_equal(ntm$mini_batch_size , MINI_BATCH_SIZE)
})

test_that("test prepare for training none mini batch_size", {
  ntm_args = c(base_job_name="ntm", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ntm=do.call(NTM$new, ntm_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  ntm$fit(data)

  expect_equal(ntm$latest_training_job , "sagemaker-ntm-dummy")
})

test_that("test prepare for training wrong type mini batch size", {
  ntm_args = c(base_job_name="ntm", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ntm=do.call(NTM$new, ntm_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(ntm$.prepare_for_training(data, "some"))
})

test_that("test prepare for training wrong value lower mini batch size", {
  ntm_args = c(base_job_name="ntm", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ntm=do.call(NTM$new, ntm_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(ntm$.prepare_for_training(data, 0))
  expect_error(ntm$.prepare_for_training(data, 10001))
})

test_that("test model image", {
  ntm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ntm=do.call(NTM$new, ntm_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  ntm$fit(data, MINI_BATCH_SIZE)
  model = ntm$create_model()

  expect_equal(model$image_uri, ImageUris$new()$retrieve("ntm", REGION))
})

test_that("test predictor type", {
  ntm_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  ntm=do.call(NTM$new, ntm_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  ntm$fit(data, MINI_BATCH_SIZE)
  model = ntm$create_model()
  predictor = model$deploy(1, INSTANCE_TYPE)

  expect_true(inherits(predictor, "NTMPredictor"))
})
