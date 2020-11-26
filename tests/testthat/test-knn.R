# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_knn.py
context("knn")

ROLE = "myrole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
K = 5
SAMPLE_SIZE = 1000
PREDICTOR_TYPE_REGRESSOR = "regressor"
PREDICTOR_TYPE_CLASSIFIER = "classifier"

COMMON_TRAIN_ARGS = list(
  "role"= ROLE,
  "instance_count"= INSTANCE_COUNT,
  "instance_type"= INSTANCE_TYPE
)

ALL_REQ_ARGS = c(list(k = K,
                      "sample_size"= SAMPLE_SIZE,
                      "predictor_type" = PREDICTOR_TYPE_REGRESSOR),
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

sagemaker_session$default_bucket <- Mock$new()$return_value(BUCKET_NAME, .min_var = 0)
sagemaker_session$sagemaker$describe_training_job <- Mock$new()$return_value(DESCRIBE_TRAINING_JOB_RESULT)
sagemaker_session$sagemaker$describe_endpoint <- Mock$new()$return_value(ENDPOINT_DESC)
sagemaker_session$sagemaker$describe_endpoint_config <- Mock$new()$return_value(ENDPOINT_CONFIG_DESC)
sagemaker_session$s3$put_object <- Mock$new()$return_value(NULL)
sagemaker_session$expand_role <- Mock$new()$return_value(ROLE)
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-knn-dummy"))
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-knn")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-knn-endpoint")
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)

test_that("test init required positional", {
  knn = KNN$new(
    ROLE,
    INSTANCE_COUNT,
    INSTANCE_TYPE,
    K,
    SAMPLE_SIZE,
    PREDICTOR_TYPE_REGRESSOR,
    sagemaker_session=sagemaker_session
  )
  expect_equal(knn$role, COMMON_TRAIN_ARGS$role)
  expect_equal(knn$instance_count, INSTANCE_COUNT)
  expect_equal(knn$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(knn$k, K)
})

test_that("test init required named", {
  knn_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  knn = do.call(KNN$new, knn_args)

  expect_equal(knn$role, COMMON_TRAIN_ARGS$role)
  expect_equal(knn$instance_count, INSTANCE_COUNT)
  expect_equal(knn$instance_type, COMMON_TRAIN_ARGS$instance_type)
  expect_equal(knn$k, ALL_REQ_ARGS$k)
})

test_that("test all hyperparameters regressor", {
  knn_args = c(sagemaker_session=sagemaker_session,
               dimension_reduction_type="sign",
               dimension_reduction_target="2",
               index_type="faiss.Flat",
               index_metric="COSINE",
               faiss_index_ivf_nlists="auto",
               faiss_index_pq_m=1,
               ALL_REQ_ARGS)
  knn = do.call(KNN$new, knn_args)

  expect_equal(knn$hyperparameters(), list(
    k=ALL_REQ_ARGS$k,
    sample_size=ALL_REQ_ARGS$sample_size,
    predictor_type=ALL_REQ_ARGS$predictor_type,
    dimension_reduction_type="sign",
    dimension_reduction_target=2,
    index_type="faiss.Flat",
    index_metric="COSINE",
    faiss_index_ivf_nlists="auto",
    faiss_index_pq_m=1)
  )
})

test_that("test all hyperparameters classifier", {
  knn_args = c(sagemaker_session=sagemaker_session,
               dimension_reduction_type="fjlt",
               dimension_reduction_target="2",
               index_type="faiss.IVFFlat",
               index_metric="L2",
               faiss_index_ivf_nlists="20",
               ALL_REQ_ARGS)
  knn_args$predictor_type = PREDICTOR_TYPE_CLASSIFIER
  knn = do.call(KNN$new, knn_args)

  expect_equal(knn$hyperparameters(), list(
    k=ALL_REQ_ARGS$k,
    sample_size=ALL_REQ_ARGS$sample_size,
    predictor_type=knn_args$predictor_type,
    dimension_reduction_type="fjlt",
    dimension_reduction_target=2,
    index_type="faiss.IVFFlat",
    index_metric="L2",
    faiss_index_ivf_nlists="20")
  )
})

test_that("test image", {
  knn_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  knn = do.call(KNN$new, knn_args)

  expect_equal(knn$training_image_uri(), ImageUris$new()$retrieve("knn", REGION))
})

test_that("test required hyper parameters type", {
  knn_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("k" = "string")

  for(i in seq_along(test_param)){
    test_args = c(knn_args, test_param[i])
    expect_error(do.call(KNN$new, test_args))
  }
})

test_that("test required hyper parameters value", {
  knn_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("k"="string",
                    "sample_size"="string",
                    "predictor_type"=1)

  for(i in seq_along(test_param)){
    test_args = c(knn_args, test_param[i])
    expect_error(do.call(KNN$new, test_args))
  }
})

test_that("test optional hyper parameters type", {
  knn_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("predictor_type"="random_string")

  for(i in seq_along(test_param)){
    test_args = c(knn_args, test_param[i])
    expect_error(do.call(KNN$new, test_args))
  }
})

test_that("test error optional hyper parameters type", {
  knn_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("index_type"=1,
                    "index_metric"="string")

  for(i in seq_along(test_param)){
    test_args = c(knn_args, test_param[i])
    expect_error(do.call(KNN$new, test_args))
  }
})

test_that("test error optional hyper parameters value", {
  knn_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list("index_type"="faiss.random",
                    "index_metric"="randomstring",
                    "faiss_index_pq_m"=-1)

  for(i in seq_along(test_param)){
    test_args = c(knn_args, test_param[i])
    expect_error(do.call(KNN$new, test_args))
  }
})

test_that("test error conditional hyper parameters value", {
  knn_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  test_param = list(
        list("dimension_reduction_type"="sign"),  # errors due to missing dimension_reduction_target
        list("dimension_reduction_type"="sign", "dimension_reduction_target"= -2),
        list("dimension_reduction_type"="sign", "dimension_reduction_target"="string"),
        list("dimension_reduction_type"=2, "dimension_reduction_target"=20),
        list("dimension_reduction_type"="randomstring", "dimension_reduction_target"=20))

  for(i in seq_along(test_param)){
    test_args = c(knn_args, test_param[[i]])
    expect_error(do.call(KNN$new, test_args))
  }
})

PREFIX = "prefix"
FEATURE_DIM = 10
MINI_BATCH_SIZE = 200

test_that("test call fit", {
  knn_args = c(base_job_name="knn", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  knn=do.call(KNN$new, knn_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  knn$fit(data, MINI_BATCH_SIZE)

  expect_equal(knn$latest_training_job , "sagemaker-knn-dummy")
  expect_equal(knn$mini_batch_size , MINI_BATCH_SIZE)
})

test_that("test prepare for training none mini batch_size", {
  knn_args = c(base_job_name="knn", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  knn=do.call(KNN$new, knn_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )
  knn$fit(data)

  expect_equal(knn$latest_training_job , "sagemaker-knn-dummy")
})

test_that("test prepare for training wrong type mini batch size", {
  knn_args = c(base_job_name="knn", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  knn=do.call(KNN$new, knn_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(knn$.prepare_for_training(data, "some"))
})

test_that("test prepare for training wrong value lower mini batch size", {
  knn_args = c(base_job_name="knn", sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  knn=do.call(KNN$new, knn_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  expect_error(knn$.prepare_for_training(data, 0))
})

test_that("test model image", {
  knn_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  knn=do.call(KNN$new, knn_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  knn$fit(data, MINI_BATCH_SIZE)
  model = knn$create_model()

  expect_equal(model$image_uri, ImageUris$new()$retrieve("knn", REGION))
})

test_that("test predictor type", {
  knn_args = c(sagemaker_session=sagemaker_session, ALL_REQ_ARGS)
  knn=do.call(KNN$new, knn_args)
  data = RecordSet$new(
    sprintf("s3://%s/%s",BUCKET_NAME, PREFIX),
    num_records=1,
    feature_dim=FEATURE_DIM,
    channel="train"
  )

  knn$fit(data, MINI_BATCH_SIZE)
  model = knn$create_model()
  predictor = model$deploy(1, INSTANCE_TYPE)

  expect_true(inherits(predictor, "KNNPredictor"))
})
