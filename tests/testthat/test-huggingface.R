# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/sagemaker/huggingface/test_estimator.py
context("huggingface")

DATA_DIR = file.path(getwd(), "data")
SCRIPT_PATH =file.path(DATA_DIR, "dummy_script.py")
SERVING_SCRIPT_FILE = "another_dummy_script.py"
SCRIPT_PATH =file.path(DATA_DIR, "dummy_script.py")
TAR_FILE <- file.path(DATA_DIR, "test_tar.tgz")
BIN_OBJ <- readBin(con = TAR_FILE, what = "raw", n = file.size(TAR_FILE))
MODEL_DATA = "s3://some/data.tar.gz"
ENV = list("DUMMY_ENV_VAR"="dummy_value")
TIMESTAMP = "2017-11-06-14:14:15.672"
TIME = 1510006209.073025
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.p2.xlarge"
ACCELERATOR_TYPE = "ml.eia.medium"
IMAGE_URI = "huggingface"
JOB_NAME = sprintf("%s-.*", IMAGE_URI)
ROLE = "Dummy"
REGION = "us-east-1"
GPU = "ml.p2.xlarge"

ENDPOINT_DESC = list("EndpointConfigName"="test-endpoint")

ENDPOINT_CONFIG_DESC = list("ProductionVariants"=list(list("ModelName"="model-1"), list("ModelName"="model-2")))

LIST_TAGS_RESULT = list("Tags"=list(list("Key"="TagtestKey", "Value"="TagtestValue")))

EXPERIMENT_CONFIG = list(
  "ExperimentName"="exp",
  "TrialName"="trial",
  "TrialComponentDisplayName"="tc"
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

.get_full_gpu_image_uri = function(version, base_framework_version){
  return(ImageUris$new()$retrieve(
    "huggingface",
    REGION,
    version=version,
    py_version="py36",
    instance_type=GPU,
    image_scope="training",
    base_framework_version=base_framework_version,
    container_version="cu110-ubuntu18.04")
  )
}

.create_train_job = function(version, base_framework_version, s3_inputs){
  return(list(
    "image_uri"=.get_full_gpu_image_uri(version, base_framework_version),
    "input_mode"="File",
    "input_config"= list(
      list(
        "DataSource"=list(
          "S3DataSource"=list(
            "S3DataType"="S3Prefix",
            "S3Uri" = s3_inputs,
            "S3DataDistributionType"="FullyReplicated"
            )
        ),
        "ChannelName"="training"
      )
    ),
    "role"=ROLE,
    "job_name"=JOB_NAME,
    "output_config"=list("S3OutputPath"=sprintf("s3://%s/",BUCKET_NAME)),
    "resource_config"=list(
      "InstanceCount"=1,
      "InstanceType"=GPU,
      "VolumeSizeInGB"=30),
    "hyperparameters"=list(
      "sagemaker_program"="dummy_script.py",
      "sagemaker_container_log_level"="20",
      "sagemaker_job_name"=JOB_NAME,
      "sagemaker_submit_directory"=sprintf(
        "s3://%s/%s/source/sourcedir.tar.gz", BUCKET_NAME, JOB_NAME),
      "sagemaker_region"='us-east-1'),
    "stop_condition"=list("MaxRuntimeInSeconds"=24 * 60 * 60),
    "vpc_config"=NULL,
    "debugger_hook_config"=list(
      "S3OutputPath"=sprintf("s3://%s/",BUCKET_NAME),
      "CollectionConfigurations"=list()
      ),
    "profiler_rule_configs"=list(
      list(
        "RuleConfigurationName"="ProfilerReport-.*",
        "RuleEvaluatorImage"="503895931360.dkr.ecr.us-east-1.amazonaws.com/sagemaker-debugger-rules:latest",
        "RuleParameters"=list("rule_to_invoke"="ProfilerReport")
      )
    ),
    "profiler_config"=list(
      "S3OutputPath"= sprintf("s3://%s/",BUCKET_NAME)
      )
    )
  )
}

test_that("test huggingface invalid args", {
  error_msg = "ValueError. Please use either full version or shortened version for both transformers_version, tensorflow_version and pytorch_version."
  expect_error(
    HuggingFace$new(
      py_version="py36",
      entry_point=SCRIPT_PATH,
      role=ROLE,
      instance_count=INSTANCE_COUNT,
      instance_type=INSTANCE_TYPE,
      transformers_version="4.2.1",
      pytorch_version="1.6",
      enable_sagemaker_metrics=FALSE
    ),
    error_msg
  )

  error_msg = "ValueError. transformers_version, and image_uri are both NULL. Specify either transformers_version or image_uri"
  expect_error(
    HuggingFace$new(
      py_version="py36",
      entry_point=SCRIPT_PATH,
      role=ROLE,
      instance_count=INSTANCE_COUNT,
      instance_type=INSTANCE_TYPE,
      pytorch_version="1.6",
      enable_sagemaker_metrics=FALSE
    ),
    error_msg
  )

  error_msg = "ValueError. tensorflow_version and pytorch_version are both NULL. Specify either tensorflow_version or pytorch_version."
  expect_error(
    HuggingFace$new(
      py_version="py36",
      entry_point=SCRIPT_PATH,
      role=ROLE,
      instance_count=INSTANCE_COUNT,
      instance_type=INSTANCE_TYPE,
      transformers_version="4.2.1",
      enable_sagemaker_metrics=FALSE
    ),
    error_msg
  )

  error_msg = "ValueError. tensorflow_version and pytorch_version are both not NULL. Specify only tensorflow_version or pytorch_version."
  expect_error(
    HuggingFace$new(
      py_version="py36",
      entry_point=SCRIPT_PATH,
      role=ROLE,
      instance_count=INSTANCE_COUNT,
      instance_type=INSTANCE_TYPE,
      transformers_version="4.2",
      pytorch_version="1.6",
      tensorflow_version="2.3",
      enable_sagemaker_metrics=FALSE
    ),
    error_msg
  )
})

test_that("test huggingface",{
  huggingface_training_version="4.4"
  huggingface_pytorch_version="1.6"
  hf = HuggingFace$new(
    py_version="py36",
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    transformers_version=huggingface_training_version,
    pytorch_version=huggingface_pytorch_version,
    enable_sagemaker_metrics=FALSE
  )

  inputs = "s3://mybucket/train"

  hf$fit(inputs=inputs, experiment_config=EXPERIMENT_CONFIG)

  expected_train_args = .create_train_job(
    huggingface_training_version, sprintf("pytorch%s",huggingface_pytorch_version),inputs
  )
  expected_train_args[["experiment_config"]] = EXPERIMENT_CONFIG
  expected_train_args[["enable_sagemaker_metrics"]] = FALSE
  actual_train_args = sagemaker_session$train()$.call_args

  expect_identical(sort(names(expected_train_args)), sort(names(actual_train_args)))

  exp_hp = expected_train_args[["hyperparameters"]]
  act_hp = actual_train_args[["hyperparameters"]]
  expected_train_args[["hyperparameters"]] = NULL
  actual_train_args[["hyperparameters"]] = NULL
  for(i in names(exp_hp)){
    expect_true(grepl(exp_hp[[i]], act_hp[[i]]))
  }

  exp_RuleConfigurationName = expected_train_args[["profiler_rule_configs"]][[1]][["RuleConfigurationName"]]
  act_RuleConfigurationName = actual_train_args[["profiler_rule_configs"]][[1]][["RuleConfigurationName"]]

  expected_train_args[["profiler_rule_configs"]][[1]][["RuleConfigurationName"]] = NULL
  actual_train_args[["profiler_rule_configs"]][[1]][["RuleConfigurationName"]] = NULL

  expect_true(grepl(exp_RuleConfigurationName, act_RuleConfigurationName))

  for (i in l_name){
    if(i != "job_name") {
      expect_identical(expected_train_args[[i]], actual_train_args[[i]])
    } else {
      expect_true(grepl(expected_train_args[[i]], actual_train_args[[i]]))
    }
  }
})

test_that("test attach", {
  training_image = sprintf(paste0(
    "1.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:%s-",
    "transformers%s-gpu-py36-cu110-ubuntu18.04"),
    huggingface_pytorch_version, huggingface_training_version)

  returned_job_description = list(
    "AlgorithmSpecification"=list("TrainingInputMode"="File", "TrainingImage"=training_image),
    "HyperParameters"=list(
      "sagemaker_submit_directory"='s3://some/sourcedir.tar.gz',
      "sagemaker_program"='iris-dnn-classifier.py',
      "sagemaker_s3_uri_training"='sagemaker-3/integ-test-data/tf_iris',
      "sagemaker_container_log_level"='INFO',
      "sagemaker_job_name"='neo',
      "training_steps"="100",
      "sagemaker_region"='us-east-1'),
    "RoleArn"="arn:aws:iam::366:role/SageMakerRole",
    "ResourceConfig"=list(
      "VolumeSizeInGB"=30,
      "InstanceCount"=1,
      "InstanceType"="ml.c4.xlarge"),
    "StoppingCondition"=list("MaxRuntimeInSeconds"=24 * 60 * 60),
    "TrainingJobName"="neo",
    "TrainingJobStatus"="Completed",
    "TrainingJobArn"="arn:aws:sagemaker:us-west-2:336:training-job/neo",
    "OutputDataConfig"=list("KmsKeyId"="", "S3OutputPath"="s3://place/output/neo"),
    "TrainingJobOutput"=list("S3TrainingJobOutput"="s3://here/output.tar.gz")
  )

  sm = sagemaker_session$clone(TRUE)
  sm$sagemaker$describe_training_job = Mock$new()$return_value(
    returned_job_description
  )

  hf=HuggingFace$new(
    py_version="py36",
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    transformers_version=huggingface_training_version,
    pytorch_version=huggingface_pytorch_version,
    enable_sagemaker_metrics=FALSE
  )

  estimator = hf$attach(training_job_name="neo", sagemaker_session=sm)
  expect_equal(estimator$latest_training_job, "neo")
  expect_equal(estimator$py_version, "py36")
  expect_equal(estimator$framework_version, huggingface_training_version)
  expect_equal(estimator$pytorch_version, huggingface_pytorch_version)
  expect_equal(estimator$role, "arn:aws:iam::366:role/SageMakerRole")
  expect_equal(estimator$instance_count, 1)
  expect_equal(estimator$max_run, 24 * 60 * 60)
  expect_equal(estimator$input_mode, "File")
  expect_equal(estimator$base_job_name, "neo")
  expect_equal(estimator$output_path, "s3://place/output/neo")
  expect_equal(estimator$output_kms_key,"")
  expect_equal(estimator$hyperparameters()[["training_steps"]], "100")
  expect_equal(estimator$source_dir, "s3://some/sourcedir.tar.gz")
  expect_equal(estimator$entry_point, "iris-dnn-classifier.py")
})

