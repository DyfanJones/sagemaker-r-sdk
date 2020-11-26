# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_chainer.py
context("chainer")

DATA_DIR = file.path(getwd(), "data")
SCRIPT_PATH = file.path(DATA_DIR, "dummy_script.py")
SERVING_SCRIPT_FILE = "another_dummy_script.py"
MODEL_DATA = "s3://some/data.tar.gz"
ENV = list("DUMMY_ENV_VAR"= "dummy_value")
TIMESTAMP = "2017-11-06-14:14:15.672"
TIME = 1507167947
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.4xlarge"
ACCELERATOR_TYPE = "ml.eia.medium"
IMAGE_URI = "sagemaker-chainer"
JOB_NAME = sprintf("%s-%s", IMAGE_URI, TIMESTAMP)
IMAGE_URI_FORMAT_STRING = "520713654638.dkr.ecr.%s.amazonaws.com/%s:%s-%s-%s"
ROLE = "Dummy"
REGION = "us-west-2"
GPU = "ml.p2.xlarge"
CPU = "ml.c4.xlarge"
chainer_version = "4.0.0"
chainer_py_version = "py3"

ENDPOINT_DESC = list("EndpointConfigName"="test-endpoint")

ENDPOINT_CONFIG_DESC = list("ProductionVariants"= list(list("ModelName"= "model-1"), list("ModelName"= "model-2")))

LIST_TAGS_RESULT = list("Tags"= list(list("Key"= "TagtestKey", "Value"= "TagtestValue")))

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
sagemaker_session$sagemaker$describe_training_job = Mock$new()$return_value(describe)
sagemaker_session$sagemaker$describe_endpoint = Mock$new()$return_value(ENDPOINT_DESC)
sagemaker_session$sagemaker$describe_endpoint_config = Mock$new()$return_value(ENDPOINT_CONFIG_DESC)
sagemaker_session$sagemaker$list_tags = Mock$new()$return_value(LIST_TAGS_RESULT)
sagemaker_session$default_bucket = Mock$new(name="default_bucket")$return_value(BUCKET_NAME, .min_var = 0)
sagemaker_session$expand_role = Mock$new(name="expand_role")$return_value(ROLE)
sagemaker_session$wait_for_job = Mock$new()$return_value(NULL)
sagemaker_session$train <- Mock$new()$return_value(list(TrainingJobArn = "sagemaker-chainer-dummy"))
sagemaker_session$logs_for_job <- Mock$new()$return_value(NULL)
sagemaker_session$create_model <- Mock$new()$return_value("sagemaker-chainer")
sagemaker_session$endpoint_from_production_variants <- Mock$new()$return_value("sagemaker-chainer-endpoint")

create_train_job=list(
    "image_uri"= sprintf(IMAGE_URI_FORMAT_STRING, REGION, IMAGE_URI, chainer_version, "cpu", chainer_py_version),
    "input_mode"= "File",
    "input_config"= list(
      list(
        "ChannelName"= "training",
        "DataSource"= list(
          "S3DataSource"= list(
            "S3DataDistributionType"= "FullyReplicated",
            "S3DataType"= "S3Prefix"
            )
          )
        )
      ),
    "role"= ROLE,
    "job_name"= JOB_NAME,
    "output_config"= list("S3OutputPath"= sprintf("s3://%s/",BUCKET_NAME)),
    "resource_config"= list(
      "InstanceType"= "ml.c4.4xlarge",
      "InstanceCount"= 1,
      "VolumeSizeInGB"= 30
    ),
    "hyperparameters"= list(
      "sagemaker_program"= "dummy_script.py",
      "sagemaker_container_log_level"= "INFO",
      "sagemaker_job_name"= JOB_NAME,
      "sagemaker_submit_directory"= sprintf("s3://%s/%s/source/sourcedir.tar.gz",BUCKET_NAME, JOB_NAME),
      "sagemaker_region"= 'us-west-2'
    ),
    "stop_condition"= list("MaxRuntimeInSeconds"= 24 * 60 * 60),
    "tags"= NULL,
    "vpc_config"= NULL,
    "metric_definitions"= NULL,
    "experiment_config"= NULL,
    "debugger_hook_config"= list(
      "CollectionConfigurations"= list(),
      "S3OutputPath"= sprintf("s3://%s/",BUCKET_NAME)
    )
  )

test_that("test additional hyperparameters", {
  chainer_args = list(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type= INSTANCE_TYPE,
    base_job_name=NULL,
    use_mpi=TRUE,
    num_processes=4,
    process_slots_per_host=10,
    additional_mpi_options="-x MY_ENVIRONMENT_VARIABLE",
    framework_version = chainer_version,
    py_version = chainer_py_version
  )
  chainer = do.call(Chainer$new, chainer_args)
  hp = chainer$hyperparameters()

  expect_true(is.logical(hp$sagemaker_use_mpi))
  expect_equal(hp$sagemaker_num_processes, 4)
  expect_equal(hp$sagemaker_process_slots_per_host, 10)
  expect_equal(hp$sagemaker_additional_mpi_options,"-x MY_ENVIRONMENT_VARIABLE")
})

test_that("test attach with additional hyperparameters", {
  training_image = sprintf("1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-chainer:%s-cpu-%s",
    chainer_version, chainer_py_version)

  returned_job_description = list(
    "AlgorithmSpecification"= list("TrainingInputMode"= "File", "TrainingImage"= training_image),
    "HyperParameters"= list(
      "sagemaker_submit_directory"= 's3://some/sourcedir.tar.gz',
      "sagemaker_program"= 'iris-dnn-classifier.py',
      "sagemaker_s3_uri_training"= 'sagemaker-3/integ-test-data/tf_iris',
      "sagemaker_container_log_level"= 'INFO',
      "sagemaker_job_name"= 'neo',
      "sagemaker_region"= 'us-west-2',
      "sagemaker_num_processes"= 4,
      "sagemaker_additional_mpi_options"= "-x MY_ENVIRONMENT_VARIABLE",
      "sagemaker_process_slots_per_host"= 10,
      "sagemaker_use_mpi"= TRUE
    ),
    "RoleArn"= "arn:aws:iam::366:role/SageMakerRole",
    "ResourceConfig"= list(
      "VolumeSizeInGB"= 30,
      "InstanceCount"= 1,
      "InstanceType"= "ml.c4.xlarge"
    ),
    "StoppingCondition"= list("MaxRuntimeInSeconds"= 24 * 60 * 60),
    "TrainingJobName"= "neo",
    "TrainingJobStatus"= "Completed",
    "TrainingJobArn"= "arn=aws=sagemaker=us-west-2=336:training-job/neo",
    "OutputDataConfig"= list("KmsKeyId"= "", "S3OutputPath"= "s3=//place/output/neo"),
    "TrainingJobOutput"= list("S3TrainingJobOutput"= "s3=//here/output.tar.gz")
  )

  sm = sagemaker_session$clone(deep = T)
  sm$sagemaker$describe_training_job = Mock$new()$return_value(returned_job_description)

  chainer_args = list(
    entry_point = SCRIPT_PATH,
    framework_version = chainer_version,
    py_version = chainer_py_version,
    role = ROLE,
    instance_count = INSTANCE_COUNT,
    instance_type = INSTANCE_TYPE)
  chainer = do.call(Chainer$new, chainer_args)

  estimator = chainer$attach(training_job_name="neo", sagemaker_session=sm)
  hp = estimator$hyperparameters()
  expect_true(is.logical(hp$sagemaker_use_mpi))
  expect_equal(hp$sagemaker_num_processes, 4)
  expect_equal(hp$sagemaker_process_slots_per_host, 10)
  expect_equal(hp$sagemaker_additional_mpi_options,"-x MY_ENVIRONMENT_VARIABLE")

  expect_true(estimator$use_mpi)
  expect_equal(estimator$num_processes, 4)
  expect_equal(estimator$process_slots_per_host, 10)
  expect_equal(estimator$additional_mpi_options, "-x MY_ENVIRONMENT_VARIABLE")
})

test_that("test create model", {
  container_log_level = 'INFO'
  source_dir = "s3://mybucket/source"
  base_job_name = "job"

  chainer = Chainer$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    framework_version=chainer_version,
    container_log_level=container_log_level,
    py_version=chainer_py_version,
    base_job_name=base_job_name,
    source_dir=source_dir
  )

  chainer$fit(inputs="s3://mybucket/train", job_name="new_name")
  chainer$latest_training_job
  chainer$model_data
  model = chainer$create_model()

  expect_equal(model$sagemaker_session, sagemaker_session)
  expect_equal(model$framework_version, chainer_version)
  expect_equal(model$py_version, chainer$py_version)
  expect_equal(model$role, ROLE)
  expect_equal(model$container_log_level, "20")
  expect_equal(model$source_dir, source_dir)
  expect_null(model$vpc_config)
})

test_that("test create model with optional params", {
  container_log_level = 'INFO'
  source_dir = "s3://mybucket/source"
  chainer = Chainer$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    container_log_level=container_log_level,
    framework_version=chainer_version,
    py_version=chainer_py_version,
    base_job_name="job",
    source_dir=source_dir
  )

  chainer$fit(inputs="s3://mybucket/train", job_name="new_name")

  new_role = "role"
  model_server_workers = 2
  vpc_config = list("Subnets"= list("foo"), "SecurityGroupIds"= list("bar"))
  model_name = "model-name"
  model = chainer$create_model(
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
  custom_image = "ubuntu:latest"
  chainer = Chainer$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    image_uri=custom_image,
    container_log_level=container_log_level,
    base_job_name="job",
    source_dir=source_dir
  )

  chainer$fit(inputs="s3://mybucket/train", job_name="new_name")
  model = chainer$create_model()

  expect_equal(model$image_uri, custom_image)
})


test_that("test chainer", {
  chainer = Chainer$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    framework_version=chainer_version,
    py_version=chainer_py_version
  )

  inputs = "s3://mybucket/train"

  chainer$fit(inputs=inputs)
  model = chainer$create_model()

  expected_image_base = "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-chainer:%s-gpu-%s"
  expect_equal(model$prepare_container_def(GPU), list(
    "Image"= sprintf(expected_image_base, chainer_version, chainer_py_version),
    "Environment"= list(
      "SAGEMAKER_PROGRAM"= "dummy_script.py",
      "SAGEMAKER_SUBMIT_DIRECTORY"= sprintf("s3://mybucket/%s/source/sourcedir.tar.gz", chainer$.current_job_name),
      "SAGEMAKER_CONTAINER_LOG_LEVEL"= "20",
      "SAGEMAKER_REGION"= "us-west-2"
      ),
    "ModelDataUrl"= "s3://m/m.tar.gz"
    )
  )

  expect_true(grepl("cpu", model$prepare_container_def(CPU)$Image))
  predictor = chainer$deploy(1, GPU)
  expect_true(inherits(predictor, "ChainerPredictor"))
})

test_that("test model", {
  model = ChainerModel$new(
    "s3://some/data.tar.gz",
    role=ROLE,
    entry_point=SCRIPT_PATH,
    sagemaker_session=sagemaker_session,
    framework_version=chainer_version,
    py_version=chainer_py_version
  )

  predictor = model$deploy(1, GPU)
  expect_true(inherits(predictor, "ChainerPredictor"))
})

test_that("test model prepare container def accelerator_error", {
  model = ChainerModel$new(
    MODEL_DATA,
    role=ROLE,
    entry_point=SCRIPT_PATH,
    sagemaker_session=sagemaker_session,
    framework_version=chainer_version,
    py_version=chainer_py_version
  )

  expect_error(model$prepare_container_def(INSTANCE_TYPE, accelerator_type=ACCELERATOR_TYPE))
})

test_that("test model prepare container def no instance type or image", {
  model = ChainerModel$new(
    MODEL_DATA,
    role=ROLE,
    entry_point=SCRIPT_PATH,
    framework_version=chainer_version,
    py_version=chainer_py_version
  )

  expect_error(model$prepare_container_def())
})

test_that("test training image default", {
  chainer = Chainer$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    framework_version=chainer_version,
    py_version=chainer_py_version
  )

  expect_equal(chainer$training_image_uri(),
               sprintf(IMAGE_URI_FORMAT_STRING, REGION, IMAGE_URI,
                       chainer_version, "cpu", chainer_py_version))
})

test_that("test attach", {
  training_image = sprintf("1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-chainer:%s-cpu-%s",
    chainer_version, chainer_py_version)

  returned_job_description = list(
    "AlgorithmSpecification"= list("TrainingInputMode"= "File", "TrainingImage"= training_image),
    "HyperParameters"= list(
      "sagemaker_submit_directory"= 's3://some/sourcedir.tar.gz',
      "sagemaker_program"= 'iris-dnn-classifier.py',
      "sagemaker_s3_uri_training"= 'sagemaker-3/integ-test-data/tf_iris',
      "sagemaker_container_log_level"= 'INFO',
      "sagemaker_job_name"= 'neo',
      "training_steps"= "100",
      "sagemaker_region"= 'us-west-2'
      ),
    "RoleArn"= "arn:aws:iam::366:role/SageMakerRole",
    "ResourceConfig"= list(
      "VolumeSizeInGB"= 30,
      "InstanceCount"= 1,
      "InstanceType"= "ml.c4.xlarge"
      ),
    "StoppingCondition"= list("MaxRuntimeInSeconds"= 24 * 60 * 60),
    "TrainingJobName"= "neo",
    "TrainingJobStatus"= "Completed",
    "TrainingJobArn"= "arn:aws:sagemaker:us-west-2:336:training-job/neo",
    "OutputDataConfig"= list("KmsKeyId"= "", "S3OutputPath"= "s3://place/output/neo"),
    "TrainingJobOutput"= list("S3TrainingJobOutput"= "s3://here/output.tar.gz")
    )

  sagemaker_session$sagemaker$describe_training_job = Mock$new()$return_value(returned_job_description)

  chainer = Chainer$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    framework_version=chainer_version,
    py_version=chainer_py_version
  )

  estimator = chainer$attach(training_job_name="neo", sagemaker_session=sagemaker_session)

  expect_equal(estimator$latest_training_job, "neo")
  expect_equal(estimator$py_version, chainer_py_version)
  expect_equal(estimator$framework_version, chainer_version)
  expect_equal(estimator$role, "arn:aws:iam::366:role/SageMakerRole")
  expect_equal(estimator$instance_count, 1)
  expect_equal(estimator$max_run, 24 * 60 * 60)
  expect_equal(estimator$input_mode, "File")
  expect_equal(estimator$base_job_name, "neo")
  expect_equal(estimator$output_path, "s3://place/output/neo")
  expect_equal(estimator$output_kms_key, "")
  expect_equal(estimator$hyperparameters()$training_steps, "100")
  expect_equal(estimator$source_dir, "s3://some/sourcedir.tar.gz")
  expect_equal(estimator$entry_point, "iris-dnn-classifier.py")
})

test_that("test attach wrong framework", {
  rjd = list(
    "AlgorithmSpecification"= list(
      "TrainingInputMode"= "File",
      "TrainingImage"= "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py2-cpu:1.0.4"
      ),
    "HyperParameters"= list(
      "sagemaker_submit_directory"= 's3://some/sourcedir.tar.gz',
      "checkpoint_path"= 's3://other/1508872349',
      "sagemaker_program"= 'iris-dnn-classifier.py',
      "sagemaker_container_log_level"= 'INFO',
      "training_steps"= "100",
      "sagemaker_region"= 'us-west-2'
      ),
    "RoleArn"= "arn:aws:iam::366:role/SageMakerRole",
    "ResourceConfig"= list(
      "VolumeSizeInGB"= 30,
      "InstanceCount"= 1,
      "InstanceType"= "ml.c4.xlarge"
      ),
    "StoppingCondition"= list("MaxRuntimeInSeconds"= 24 * 60 * 60),
    "TrainingJobName"= "neo",
    "TrainingJobStatus"= "Completed",
    "TrainingJobArn"= "arn:aws:sagemaker:us-west-2:336:training-job/neo",
    "OutputDataConfig"= list("KmsKeyId"= "", "S3OutputPath"= "s3://place/output/neo"),
    "TrainingJobOutput"= list("S3TrainingJobOutput"= "s3://here/output.tar.gz")
  )

  sagemaker_session$sagemaker$describe_training_job = Mock$new()$return_value(rjd)

  chainer = Chainer$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    framework_version=chainer_version,
    py_version=chainer_py_version
  )

  expect_error(chainer$attach(training_job_name="neo", sagemaker_session=sagemaker_session),
               "Training job: neo didn't use image for requested framework")
})

test_that("test attach wrong framework", {
  training_image = "1.dkr.ecr.us-west-2.amazonaws.com/my_custom_chainer_image:latest"
  returned_job_description = list(
    "AlgorithmSpecification"= list("TrainingInputMode"= "File", "TrainingImage"= training_image),
    "HyperParameters"= list(
      "sagemaker_submit_directory"= 's3://some/sourcedir.tar.gz',
      "sagemaker_program"= 'iris-dnn-classifier.py',
      "sagemaker_s3_uri_training"= 'sagemaker-3/integ-test-data/tf_iris',
      "sagemaker_container_log_level"= 'INFO',
      "sagemaker_job_name"= 'neo',
      "training_steps"= "100",
      "sagemaker_region"= 'us-west-2'
      ),
    "RoleArn"= "arn:aws:iam::366:role/SageMakerRole",
    "ResourceConfig"= list(
      "VolumeSizeInGB"= 30,
      "InstanceCount"= 1,
      "InstanceType"= "ml.c4.xlarge"
      ),
    "StoppingCondition"= list("MaxRuntimeInSeconds"= 24 * 60 * 60),
    "TrainingJobName"= "neo",
    "TrainingJobStatus"= "Completed",
    "TrainingJobArn"= "arn:aws:sagemaker:us-west-2:336:training-job/neo",
    "OutputDataConfig"= list("KmsKeyId"= "", "S3OutputPath"= "s3://place/output/neo"),
    "TrainingJobOutput"= list("S3TrainingJobOutput"= "s3://here/output.tar.gz")
  )

  sagemaker_session$sagemaker$describe_training_job = Mock$new()$return_value(returned_job_description)

  chainer = Chainer$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    framework_version=chainer_version,
    py_version=chainer_py_version
  )

  estimator = chainer$attach(training_job_name="neo", sagemaker_session=sagemaker_session)
  expect_equal(estimator$image_uri, training_image)
  expect_equal(estimator$training_image_uri(), training_image)
})

test_that("test estimator py2 warning", {
  estimator = Chainer$new(
    entry_point=SCRIPT_PATH,
    role=ROLE,
    sagemaker_session=sagemaker_session,
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    framework_version=chainer_version,
    py_version="py2")

  expect_equal(estimator$py_version, "py2")
})

test_that("test model py2 warning", {
  model = ChainerModel$new(
    MODEL_DATA,
    role=ROLE,
    entry_point=SCRIPT_PATH,
    sagemaker_session=sagemaker_session,
    framework_version=chainer_version,
    py_version="py2"
  )

  expect_equal(model$py_version, "py2")
})
