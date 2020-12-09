# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_sparkml_serving.py
context("spark ml")

MODEL_DATA = "s3://bucket/model.tar.gz"
ROLE = "myrole"
TRAIN_INSTANCE_TYPE = "ml.c4.xlarge"

REGION = "us-west-2"
BUCKET_NAME = "Some-Bucket"
ENDPOINT = "some-endpoint"

ENDPOINT_DESC = list("EndpointConfigName"= ENDPOINT)

ENDPOINT_CONFIG_DESC = list("ProductionVariants"= list(list("ModelName"= "model-1"), list("ModelName"= "model-2")))

paws_mock = Mock$new(name = "PawsCredentials", region_name = REGION)
sagemaker_session = Mock$new(
  name="Session",
  paws_credentials=paws_mock,
  paws_region_name = REGION,
  config=NULL,
  local_mode=FALSE
)

sagemaker_session$sagemaker$describe_endpoint <- Mock$new()$return_value(ENDPOINT_DESC)
sagemaker_session$sagemaker$describe_endpoint_config <- Mock$new()$return_value(ENDPOINT_CONFIG_DESC)
sagemaker_session$create_model <-  Mock$new()$return_value(NULL)
sagemaker_session$s3$put_object <- Mock$new()$return_value(NULL)
sagemaker_session$expand_role <- Mock$new()$return_value(ROLE)
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-sparkml-dummy"))
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-sparkml")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-sparkml-endpoint")
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)

test_that("test sparkml model", {
  sparkml = SparkMLModel$new(sagemaker_session=sagemaker_session, model_data=MODEL_DATA, role=ROLE)
  expect_equal(sparkml$image_uri, ImageUris$new()$retrieve("sparkml-serving", REGION, version="2.4"))
})

test_that("test auto ml default channel name", {
  sparkml = SparkMLModel$new(sagemaker_session=sagemaker_session, model_data=MODEL_DATA, role=ROLE)
  predictor = sparkml$deploy(1, TRAIN_INSTANCE_TYPE)
  expect_true(inherits(predictor, "SparkMLPredictor"))
})

test_that("test auto ml default channel name", {
  sparkml = SparkMLModel$new(sagemaker_session=sagemaker_session, model_data=MODEL_DATA, role=ROLE)
  custom_serializer = Mock$new(name="BaseSerializer")
  predictor = sparkml$deploy(1, TRAIN_INSTANCE_TYPE, serializer=custom_serializer)
  expect_true(inherits(predictor, "SparkMLPredictor"))
  expect_equal(predictor$serializer, custom_serializer)
})
