# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_chainer.py

DATA_DIR = file.path(getwd(), "data")
SCRIPT_NAME = "dummy_script.py"
SCRIPT_PATH = file.path(DATA_DIR, SCRIPT_NAME)
TAR_FILE <- file.path(DATA_DIR, "test_tar.tgz")
BIN_OBJ <- readBin(con = TAR_FILE, what = "raw", n = file.size(TAR_FILE))
SERVING_SCRIPT_FILE = "another_dummy_script.py"
MODEL_DATA = "s3://some/data.tar.gz"
ENV = list("DUMMY_ENV_VAR"= "dummy_value")
TIMESTAMP = "2017-11-06-14:14:15.672"
TIME = 1507167947
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.4xlarge"
ACCELERATOR_TYPE = "ml.eia.medium"
IMAGE = "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-serving:1.4.0-gpu-py3"
COMPILATION_JOB_NAME = "compilation-sagemaker-mxnet-%s"
FRAMEWORK = "mxnet"
ROLE = "Dummy"
REGION = "us-west-2"
GPU = "ml.p2.xlarge"
CPU = "ml.c4.xlarge"
CPU_C5 = "ml.c5.xlarge"
LAUNCH_PS_DISTRIBUTION_DICT = list("parameter_server"= list("enabled"= TRUE))
mxnet_inference_version = "1.4.0"
mxnet_inference_py_version = "py3"
neo_mxnet_version = "1.7"

ENDPOINT_DESC = list("EndpointConfigName"="test-endpoint")

ENDPOINT_CONFIG_DESC = list("ProductionVariants"= list(list("ModelName"= "model-1"), list("ModelName"= "model-2")))

LIST_TAGS_RESULT = list("Tags"= list(list("Key"= "TagtestKey", "Value"= "TagtestValue")))

EXPERIMENT_CONFIG = list(
  "ExperimentName"= "exp",
  "TrialName"= "trial",
  "TrialComponentDisplayName"= "tc"
)

paws_mock = Mock$new(name = "PawsCredentials", region_name = REGION)
sagemaker_session = Mock$new(
  name="Session",
  paws_credentials=paws_mock,
  paws_region_name=REGION,
  config=NULL,
  local_mode=FALSE,
  s3 = NULL
)

describe = list("ModelArtifacts"= list("S3ModelArtifacts"= "s3://m/m.tar.gz"))
describe_compilation = list("ModelArtifacts"= list("S3ModelArtifacts"= "s3://m/model_c5.tar.gz"))
sagemaker_session$sagemaker$describe_training_job = Mock$new()$return_value(describe)
sagemaker_session$sagemaker$describe_endpoint = Mock$new()$return_value(ENDPOINT_DESC)
sagemaker_session$sagemaker$describe_endpoint_config = Mock$new()$return_value(ENDPOINT_CONFIG_DESC)
sagemaker_session$sagemaker$list_tags = Mock$new()$return_value(LIST_TAGS_RESULT)
sagemaker_session$wait_for_compilation_job = Mock$new()$return_value(describe_compilation)
sagemaker_session$default_bucket = Mock$new(name="default_bucket")$return_value(BUCKET_NAME)
sagemaker_session$expand_role = Mock$new(name="expand_role")$return_value(ROLE)
sagemaker_session$wait_for_job = Mock$new()$return_value(NULL)
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-chainer-dummy"))
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-chainer")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-chainer-endpoint")
sagemaker_session$s3$put_object <- Mock$new()$return_value(NULL)
sagemaker_session$s3$get_object <- Mock$new()$return_value(list(Body = BIN_OBJ))
sagemaker_session$call_args("compile_model")

.is_mms_version <- function(mxnet_version){
  return (package_version(MXNetModel$public_fields$.LOWEST_MMS_VERSION) <= package_version(mxnet_version))
}

.get_train_args <- function(job_name){
  return (list(
    "image_uri"= IMAGE,
    "input_mode"= "File",
    "input_config"= list(
      list(
        "ChannelName"= "training",
        "DataSource"= list(
          "S3DataSource"= list(
            "S3DataDistributionType"= "FullyReplicated",
            "S3DataType"= "S3Prefix")
          )
        )
      ),
    "role"= ROLE,
    "job_name"= job_name,
    "output_config"= list("S3OutputPath"= sprintf("s3://%s/", BUCKET_NAME)),
    "resource_config"= list(
      "InstanceType"= "ml.c4.4xlarge",
      "InstanceCount"= 1,
      "VolumeSizeInGB"= 30),
    "hyperparameters"= list(
      "sagemaker_program"= "dummy_script.py",
      "sagemaker_container_log_level"= "INFO",
      "sagemaker_job_name"= job_name,
      "sagemaker_submit_directory"= sprintf(
        "s3://%s/%s/source/sourcedir.tar.gz", BUCKET_NAME, job_name),
      "sagemaker_region"= '"us-west-2"'),
    "stop_condition"= list("MaxRuntimeInSeconds"= 24 * 60 * 60),
    "tags"= NULL,
    "vpc_config"= NULL,
    "metric_definitions"= NULL,
    "experiment_config"= NULL,
    "debugger_hook_config"= list(
      "CollectionConfigurations"= list(),
      "S3OutputPath"= sprintf("s3://%s/",BUCKET_NAME))
    )
  )
}

.get_environment <- function(submit_directory, model_url, image_uri){
  return (list(
    "Image"= image_uri,
    "Environment"= list(
      "SAGEMAKER_PROGRAM"= "dummy_script.py",
      "SAGEMAKER_SUBMIT_DIRECTORY"= submit_directory,
      "SAGEMAKER_CONTAINER_LOG_LEVEL"= "20",
      "SAGEMAKER_REGION"= "us-west-2"),
    "ModelDataUrl"= model_url)
  )
}

.create_compilation_job <- function(input_shape, output_location){
  return (list(
    "input_model_config"= list(
      "S3Uri"= "s3://m/m.tar.gz",
      "DataInputConfig"= input_shape,
      "Framework"= toupper(FRAMEWORK)),
    "output_model_config"= list("S3OutputLocation"= output_location, "TargetDevice"= "ml_c4"),
    "role"= ROLE,
    "stop_condition"= list("MaxRuntimeInSeconds"= 900),
    "tags"= NULL)
  )
}

.neo_inference_image <- function(mxnet_version){
  return (sprintf("301217895009.dkr.ecr.us-west-2.amazonaws.com/sagemaker-inference-%s:%s-cpu-py3",
    tolower(FRAMEWORK), mxnet_version)
  )
}

test_that("test create model", {
  container_log_level = 'INFO'
  source_dir = "s3://mybucket/source"
  base_job_name = "job"

  mx = MXNet$new(
    entry_point=SCRIPT_NAME,
    source_dir=source_dir,
    framework_version=mxnet_inference_version,
    py_version=mxnet_inference_py_version,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    container_log_level=container_log_level,
    base_job_name=base_job_name
  )

  mx$fit(inputs="s3://mybucket/train", job_name="new_name")

  model = mx$create_model()

  expect_equal(model$sagemaker_session, sagemaker_session)
  expect_equal(model$framework_version, mxnet_inference_version)
  expect_equal(model$py_version, mxnet_inference_py_version)
  expect_equal(model$entry_point, SCRIPT_NAME)
  expect_equal(model$role, ROLE)
  expect_equal(model$container_log_level, "20")
  expect_equal(model$source_dir, source_dir)
  expect_null(model$image_uri)
  expect_null(model$vpc_config)
})

test_that("test create model with optional params", {
  container_log_level = 'INFO'
  source_dir = "s3://mybucket/source"
  mx = MXNet$new(
    entry_point=SCRIPT_NAME,
    source_dir=source_dir,
    framework_version=mxnet_inference_version,
    py_version=mxnet_inference_py_version,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    container_log_level=container_log_level,
    base_job_name="job"
  )

  mx$fit(inputs="s3://mybucket/train", job_name="new_name")

  new_role = "role"
  model_server_workers = 2
  vpc_config = list("Subnets"= list("foo"), "SecurityGroupIds"= list("bar"))
  model_name = "model-name"
  model = mx$create_model(
    role=new_role,
    model_server_workers=model_server_workers,
    vpc_config_override=vpc_config,
    entry_point=SERVING_SCRIPT_FILE,
    env=ENV,
    name=model_name
  )

  expect_equal(model$role, new_role)
  expect_equal(model$model_server_workers, model_server_workers)
  expect_equal(model$vpc_config, vpc_config)
  expect_equal(model$entry_point, SERVING_SCRIPT_FILE)
  expect_equal(model$env, ENV)
  expect_equal(model$name, model_name)
})

test_that("test create model with custom image", {
  container_log_level = 'INFO'
  source_dir = "s3://mybucket/source"
  custom_image = "mxnet:2.0"
  base_job_name = "job"

  mx = MXNet$new(
    entry_point=SCRIPT_NAME,
    source_dir=source_dir,
    framework_version="2.0",
    py_version="py3",
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    image_uri=custom_image,
    container_log_level=container_log_level,
    base_job_name=base_job_name
  )

  mx$fit(inputs="s3://mybucket/train", job_name="new_name")

  model = mx$create_model()

  expect_equal(model$sagemaker_session, sagemaker_session)
  expect_equal(model$image_uri, custom_image)
  expect_equal(model$entry_point, SCRIPT_NAME)
  expect_equal(model$role, ROLE)
  expect_equal(model$container_log_level, "20")
  expect_equal(model$source_dir, source_dir)
})

test_that("test mxnet", {
  mx = MXNet$new(
    entry_point=SCRIPT_PATH,
    framework_version= mxnet_inference_version,
    py_version=mxnet_inference_py_version,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    enable_sagemaker_metrics=FALSE
  )
  inputs = "s3://mybucket/train"

  mx$fit(inputs=inputs, experiment_config=EXPERIMENT_CONFIG)

  model = mx$create_model()


  actual_environment = model$prepare_container_def(GPU)

  submit_directory = actual_environment$Environment$SAGEMAKER_SUBMIT_DIRECTORY
  model_url = actual_environment$ModelDataUrl
  expected_environment = .get_environment(submit_directory, model_url, IMAGE)

  expect_equal(actual_environment, expected_environment)
  expect_true(grepl("cpu", model$prepare_container_def(CPU)$Image))

  predictor = mx$deploy(1, GPU)

  expect_true(inherits(predictor, "MXNetPredictor"))
})

test_that(" test mxnet neo", {
  mx = MXNet$new(
    entry_point=SCRIPT_PATH,
    framework_version="1.6",
    py_version="py3",
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    base_job_name="sagemaker-mxnet"
  )
  mx$fit()

  input_shape = list("data"= list(100, 1, 28, 28))
  output_location = "s3://neo-sdk-test"

  compiled_model = mx$compile_model(
    target_instance_family="ml_c4",
    input_shape=input_shape,
    output_path=output_location,
    framework="mxnet",
    framework_version=neo_mxnet_version
  )

  expected_compile_model_args = .create_compilation_job(input_shape, output_location)

  actual_compile_model_args = sagemaker_session$compile_model()

  expect_equal(expected_compile_model_args[-6], actual_compile_model_args[-6])
  expect_true(grepl("^compilation-sagemaker-mxnet-[0-9-]+", actual_compile_model_args[[6]]))
  expect_equal(compiled_model$image_uri, .neo_inference_image(neo_mxnet_version))

  predictor = mx$deploy(1, CPU, use_compiled_model=TRUE)

  expect_true(inherits(predictor, "MXNetPredictor"))

  expect_error(mx$deploy(1, CPU_C5, use_compiled_model=TRUE))

  # deploy without sagemaker Neo should continue to work
  predictor = mx$deploy(1, CPU)

  expect_true(inherits(predictor, "MXNetPredictor"))
})

