# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_rl.py

DATA_DIR = file.path(getwd(), "data")
SCRIPT_PATH =file.path(DATA_DIR, "dummy_script.py")
TAR_FILE <- file.path(DATA_DIR, "test_tar.tgz")
BIN_OBJ <- readBin(con = TAR_FILE, what = "raw", n = file.size(TAR_FILE))
TIMESTAMP = "2017-11-06-14:14:15.672"
TIME = 1510006209.073025
BUCKET_NAME = "notmybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.4xlarge"
IMAGE_URI = "sagemaker-rl"
IMAGE_URI_FORMAT_STRING = "520713654638.dkr.ecr.%s.amazonaws.com/%s-%s:%s%s-%s-py3"
PYTHON_VERSION = "py3"
ROLE = "Dummy"
REGION = "us-west-2"
GPU = "ml.p2.xlarge"
CPU = "ml.c4.xlarge"
coach_tensorflow_version = "0.10"
coach_mxnet_version = "0.11.0"

ENDPOINT_DESC = list("EndpointConfigName"="test-endpoint")

ENDPOINT_CONFIG_DESC = list("ProductionVariants"=list(list("ModelName"="model-1"), list("ModelName"="model-2")))

LIST_TAGS_RESULT = list("Tags"=list(list("Key"="TagtestKey", "Value"="TagtestValue")))

EXPERIMENT_CONFIG = list(
  "ExperimentName"="exp",
  "TrialName"="trial",
  "TrialComponentDisplayName"="tc")

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
sagemaker_session$default_bucket = Mock$new(name="default_bucket")$return_value(BUCKET_NAME, .min_var = 0)
sagemaker_session$expand_role = Mock$new(name="expand_role")$return_value(ROLE)
sagemaker_session$wait_for_job = Mock$new()$return_value(NULL)
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-chainer-dummy"))
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-chainer")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-chainer-endpoint")
sagemaker_session$s3$put_object <- Mock$new()$return_value(NULL)
sagemaker_session$s3$get_object <- Mock$new()$return_value(list(Body = BIN_OBJ))
sagemaker_session$call_args("compile_model")

.get_full_cpu_image_uri = function(toolkit, toolkit_version, framework){
  return(sprintf(IMAGE_URI_FORMAT_STRING,
    REGION, IMAGE_URI, framework, toolkit, toolkit_version, "cpu")
  )
}

.create_train_job = function(toolkit, toolkit_version, framework){
    job_name = sprintf("%s-%s-%s", IMAGE_URI, framework, TIMESTAMP)
  return(list(
    "image_uri"=.get_full_cpu_image_uri(toolkit, toolkit_version, framework),
    "input_mode"="File",
    "input_config"=list(
      list(
        "ChannelName"="training",
        "DataSource"=list(
          "S3DataSource"=list(
            "S3DataDistributionType"="FullyReplicated",
            "S3DataType"="S3Prefix")
        )
      )
    ),
    "role"=ROLE,
    "job_name"=job_name,
    "output_config"=list("S3OutputPath"=sprintf("s3://%s/",BUCKET_NAME)),
    "resource_config"=list(
      "InstanceType"="ml.c4.4xlarge",
      "InstanceCount"=1,
      "VolumeSizeInGB"=30),
    "hyperparameters"=list(
      "sagemaker_program"="dummy_script.py",
      "sagemaker_estimator"='RLEstimator',
      "sagemaker_container_log_level"="INFO",
      "sagemaker_job_name"=job_name,
      "sagemaker_s3_output"=sprintf('s3://%s/',BUCKET_NAME),
      "sagemaker_submit_directory"=sprintf(
        "s3://%s/%s/source/sourcedir.tar.gz", BUCKET_NAME, job_name),
      "sagemaker_region"='us-west-2'),
    "stop_condition"=list("MaxRuntimeInSeconds"=24 * 60 * 60),
    "tags"=NULL,
    "vpc_config"=NULL,
    "metric_definitions"=list(
      list("Name"="reward-training", "Regex"="^Training>.*Total reward=(.*?),"),
      list("Name"="reward-testing", "Regex"="^Testing>.*Total reward=(.*?),")),
    "environment"=NULL,
    "experiment_config"=NULL,
    "debugger_hook_config"=list(
      "CollectionConfigurations"=list(),
      "S3OutputPath"=sprintf("s3://%s/",BUCKET_NAME)
    ),
    "profiler_rule_configs"=list(
      list(
        "RuleConfigurationName"="ProfilerReport-1510006209",
        "RuleEvaluatorImage"="895741380848.dkr.ecr.us-west-2.amazonaws.com/sagemaker-debugger-rules:latest",
        "RuleParameters"=list("rule_to_invoke"="ProfilerReport")
        )
      ),
    "profiler_config"=list(
      "S3OutputPath"=sprintf("s3://%s/",BUCKET_NAME)
      )
    )
  )
}

test_that("test create tf model", {
  container_log_level = 'INFO'
  source_dir = "s3://mybucket/source"

  rl = RLEstimator$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    toolkit=RLToolkit$COACH,
    toolkit_version=coach_tensorflow_version,
    framework=RLFramework$TENSORFLOW,
    container_log_level="WARN",
    source_dir=source_dir)

  rl$fit(inputs="s3://mybucket/train", job_name="new_name")

  model = rl$create_model()

  supported_versions = R6sagemaker:::TOOLKIT_FRAMEWORK_VERSION_MAP[[RLToolkit$COACH]]
  framework_version = supported_versions[[coach_tensorflow_version]][[RLFramework$TENSORFLOW]]

  expect_true(inherits(model, "TensorFlowModel"))
  expect_identical(model$sagemaker_session, sagemaker_session)
  expect_equal(model$framework_version, framework_version)
  expect_equal(model$role, ROLE)
  expect_equal(model$.container_log_level, 20)
  expect_null(model$vpc_config)
})

test_that("test create mxnet model", {
  container_log_level = 'INFO'
  source_dir = "s3://mybucket/source"

  rl = RLEstimator$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    toolkit=RLToolkit$COACH,
    toolkit_version=coach_mxnet_version,
    framework=RLFramework$MXNET,
    container_log_level=container_log_level,
    source_dir=source_dir)

  rl$fit(inputs="s3://mybucket/train", job_name="new_name")

  model = rl$create_model()

  model$.container_log_level

  supported_versions = R6sagemaker:::TOOLKIT_FRAMEWORK_VERSION_MAP[[RLToolkit$COACH]]
  framework_version = supported_versions[[coach_mxnet_version]][[RLFramework$MXNET]]

  expect_true(inherits(model, "MXNetModel"))
  expect_identical(model$sagemaker_session, sagemaker_session)
  expect_equal(model$framework_version, framework_version)
  expect_equal(model$role, ROLE)
  expect_equal(model$container_log_level, "20")
  expect_null(model$vpc_config)
})


test_that("test create model with optional params",{
  container_log_level = 'INFO'
  source_dir = "s3://mybucket/source"
  rl = RLEstimator$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    toolkit=RLToolkit$COACH,
    toolkit_version=coach_mxnet_version,
    framework=RLFramework$MXNET,
    container_log_level=container_log_level,
    source_dir=source_dir)

  rl$fit(job_name="new_name")

  new_role = "role"
  new_entry_point = "deploy_script.py"
  vpc_config = list("Subnets"=list("foo"), "SecurityGroupIds"=list("bar"))
  model_name = "model-name"
  model = rl$create_model(
    role=new_role, entry_point=new_entry_point, vpc_config_override=vpc_config, name=model_name
  )

  expect_equal(model$role, new_role)
  expect_equal(model$vpc_config, vpc_config)
  expect_equal(model$entry_point, new_entry_point)
  expect_equal(model$name, model_name)
})

test_that("test create model with custom image",{
  container_log_level = 'INFO'
  source_dir = "s3://mybucket/source"
  image = "selfdrivingcars:9000"
  rl = RLEstimator$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    image_uri=image,
    container_log_level=container_log_level,
    source_dir=source_dir)

  job_name = "new_name"
  rl$fit(job_name=job_name)

  new_entry_point = "deploy_script.py"
  model = rl$create_model(entry_point=new_entry_point)

  expect_equal(model$sagemaker_session, sagemaker_session)
  expect_equal(model$image_uri, image)
  expect_equal(model$entry_point, new_entry_point)
  expect_equal(model$role, ROLE)
  expect_equal(model$container_log_level,"20")
  expect_equal(model$source_dir, source_dir)
})

test_that("test rl",{
  rl = RLEstimator$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    toolkit=RLToolkit$COACH,
    toolkit_version=coach_mxnet_version,
    framework=RLFramework$MXNET)

  inputs = "s3://mybucket/train"

  rl$fit(inputs=inputs, experiment_config=EXPERIMENT_CONFIG)

  expected_train_args = .create_train_job(
    RLToolkit$COACH, coach_mxnet_version, RLFramework$MXNET)
  expected_train_args[["input_config"]][[1]][["DataSource"]][["S3DataSource"]][["S3Uri"]] = inputs
  expected_train_args[["experiment_config"]] = EXPERIMENT_CONFIG

  actual_train_args = sagemaker_session$train()$.call_args

})

