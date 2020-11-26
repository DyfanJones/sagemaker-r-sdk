# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_automl.py
context("automl")

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
ENTRY_POINT = "blah.py"

TIMESTAMP = "2017-11-06-14:14:15.671"
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c5.2xlarge"
RESOURCE_POOLS = list(list("InstanceType"= INSTANCE_TYPE, "PoolSize"= INSTANCE_COUNT))
ROLE = "DummyRole"
TARGET_ATTRIBUTE_NAME = "target"
REGION = "us-west-2"
DEFAULT_S3_INPUT_DATA = sprintf("s3://%s/data", BUCKET_NAME)
DEFAULT_OUTPUT_PATH = sprintf("s3://%s/", BUCKET_NAME)
LOCAL_DATA_PATH = "file://data"
DEFAULT_MAX_CANDIDATES = NULL
DEFAULT_JOB_NAME = sprintf("automl-%s", TIMESTAMP)

JOB_NAME = "default-job-name"
JOB_NAME_2 = "banana-auto-ml-job"
JOB_NAME_3 = "descriptive-auto-ml-job"
VOLUME_KMS_KEY = "volume-kms-key-id-string"
OUTPUT_KMS_KEY = "output-kms-key-id-string"
OUTPUT_PATH = "s3://my_other_bucket/"
BASE_JOB_NAME = "banana"
PROBLEM_TYPE = "BinaryClassification"
BLACKLISTED_ALGORITHM = list("xgboost")
LIST_TAGS_RESULT = list("Tags"= list(list("Key"= "key1", "Value"= "value1")))
MAX_CANDIDATES = 10
MAX_RUNTIME_PER_TRAINING_JOB = 3600
TOTAL_JOB_RUNTIME = 36000
TARGET_OBJECTIVE = "0.01"
JOB_OBJECTIVE = list("fake job objective")
TAGS = list(list("Name"= "some-tag", "Value"= "value-for-tag"))
VPC_CONFIG = list("SecurityGroupIds"= list("group"), "Subnets"= list("subnet"))
COMPRESSION_TYPE = "Gzip"
ENCRYPT_INTER_CONTAINER_TRAFFIC = FALSE
GENERATE_CANDIDATE_DEFINITIONS_ONLY = FALSE
BEST_CANDIDATE = list("best-candidate" = "best-trial")
BEST_CANDIDATE_2 = list("best-candidate" = "best-trial-2")
AUTO_ML_DESC = list("AutoMLJobName"= JOB_NAME, "BestCandidate"= BEST_CANDIDATE)
AUTO_ML_DESC_2 = list("AutoMLJobName"= JOB_NAME_2, "BestCandidate"= BEST_CANDIDATE_2)
AUTO_ML_DESC_3 = list(
  "AutoMLJobArn"= "automl_job_arn",
  "AutoMLJobConfig"= list(
    "CompletionCriteria"= list(
      "MaxAutoMLJobRuntimeInSeconds"= 3000,
      "MaxCandidates"= 28,
      "MaxRuntimePerTrainingJobInSeconds"= 100
    ),
    "SecurityConfig"= list("EnableInterContainerTrafficEncryption"= TRUE)
  ),
  "AutoMLJobName"= "mock_automl_job_name",
  "AutoMLJobObjective"= list("MetricName" = "Auto"),
  "AutoMLJobSecondaryStatus" = "Completed",
  "AutoMLJobStatus" = "Completed",
  "GenerateCandidateDefinitionsOnly" = FALSE,
  "InputDataConfig" = list(
    list(
      "DataSource" = list(
        "S3DataSource"= list("S3DataType"= "S3Prefix", "S3Uri"= "s3://input/prefix")
      ),
      "TargetAttributeName"= "y"
      )
    ),
  "OutputDataConfig"= list("KmsKeyId"= "string", "S3OutputPath"= "s3://output_prefix"),
  "ProblemType"= "Auto",
  "RoleArn"= "mock_role_arn"
)

INFERENCE_CONTAINERS = list(
  list(
    "Environment"= list("SAGEMAKER_PROGRAM"= "sagemaker_serve"),
    "Image"= "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-data-processing:1.0-cpu-py3",
    "ModelDataUrl"= "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/data-processing/output"
    ),
  list(
    "Environment"= list("MAX_CONTENT_LENGTH"= "20000000"),
    "Image"= "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-training:1.0-cpu-py3",
    "ModelDataUrl"= "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/training/output"
    ),
  list(
    "Environment"= list("INVERSE_LABEL_TRANSFORM"= "1"),
    "Image"= "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-transform:1.0-cpu-py3",
    "ModelDataUrl"= "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/transform/output"
    )
)

CLASSIFICATION_INFERENCE_CONTAINERS = list(
  list(
    "Environment"= list("SAGEMAKER_PROGRAM"= "sagemaker_serve"),
    "Image"= "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-data-processing:1.0-cpu-py3",
    "ModelDataUrl"= "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/data-processing/output"
    ),
  list(
    "Environment"= list(
      "MAX_CONTENT_LENGTH"= "20000000",
      "SAGEMAKER_INFERENCE_SUPPORTED"= "probability,probabilities,predicted_label",
      "SAGEMAKER_INFERENCE_OUTPUT"= "predicted_label"
      ),
    "Image"= "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-training:1.0-cpu-py3",
    "ModelDataUrl"= "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/training/output"
    ),
  list(
    "Environment"= list(
      "INVERSE_LABEL_TRANSFORM"= "1",
      "SAGEMAKER_INFERENCE_SUPPORTED"= "probability,probabilities,predicted_label,labels",
      "SAGEMAKER_INFERENCE_OUTPUT"= "predicted_label",
      "SAGEMAKER_INFERENCE_INPUT"= "predicted_label"
      ),
    "Image"= "account.dkr.ecr.us-west-2.amazonaws.com/sagemaker-auto-ml-transform:1.0-cpu-py3",
    "ModelDataUrl"= "s3://sagemaker-us-west-2-account/sagemaker-auto-ml-gamma/transform/output"
    )
)

CANDIDATE_STEPS = list(
  list(
    "CandidateStepName"= "training-job/sagemaker-auto-ml-gamma/data-processing",
    "CandidateStepType"= "AWS::Sagemaker::TrainingJob"
    ),
  list(
    "CandidateStepName"= "transform-job/sagemaker-auto-ml-gamma/transform",
    "CandidateStepType"= "AWS::Sagemaker::TransformJob"
    ),
  list(
    "CandidateStepName"= "training-job/sagemaker-auto-ml-gamma/training",
    "CandidateStepType"= "AWS::Sagemaker::TrainingJob"
    )
)

CANDIDATE_DICT = list(
  "CandidateName"= "candidate_mock",
  "InferenceContainers"= INFERENCE_CONTAINERS,
  "CandidateSteps"= CANDIDATE_STEPS
)

CLASSIFICATION_CANDIDATE_DICT = list(
  "CandidateName"= "candidate_mock",
  "InferenceContainers"= CLASSIFICATION_INFERENCE_CONTAINERS,
  "CandidateSteps"= CANDIDATE_STEPS
)

TRAINING_JOB = list(
  "AlgorithmSpecification"= list(
    "AlgorithmName"= "string",
    "TrainingImage"= "string",
    "TrainingInputMode"= "string"
    ),
  "CheckpointConfig"= list("LocalPath"= "string", "S3Uri"= "string"),
  "EnableInterContainerTrafficEncryption" = FALSE,
  "EnableManagedSpotTraining" = FALSE,
  "EnableNetworkIsolation" = FALSE,
  "InputDataConfig" = list(
    list("DataSource"= list("S3DataSource"= list("S3DataType"= "string", "S3Uri"= "string")))
    ),
  "OutputDataConfig"= list("KmsKeyId"= "string", "S3OutputPath"= "string"),
  "ResourceConfig"= list(),
  "RoleArn"= "string",
  "StoppingCondition"= list(),
  "TrainingJobArn"= "string",
  "TrainingJobName"= "string",
  "TrainingJobStatus"= "Completed",
  "VpcConfig"= list()
)

TRANSFORM_JOB = list(
  "BatchStrategy"= "string",
  "DataProcessing"= list(),
  "Environment"= list("string"= "string"),
  "FailureReason"= "string",
  "LabelingJobArn"= "string",
  "MaxConcurrentTransforms"= 1,
  "MaxPayloadInMB"= 2000,
  "ModelName"= "string",
  "TransformInput"= list("DataSource"= list("S3DataSource"= list("S3DataType"= "string", "S3Uri"= "string"))),
  "TransformJobStatus"= "Completed",
  "TransformJobArn"= "string",
  "TransformJobName"= "string",
  "TransformOutput"= list(),
  "TransformResources"= list()
)

describe_auto_ml_job_mock = function(job_name=NULL){
  if (is.null(job_name) || job_name == JOB_NAME)
    return(AUTO_ML_DESC)
  else if (job_name == JOB_NAME_2)
    return(AUTO_ML_DESC_2)
  else if (job_name == JOB_NAME_3)
    return(AUTO_ML_DESC_3)
}

paws_mock = Mock$new(name = "PawsCredentials", region_name = REGION)
sagemaker_session = Mock$new(
  name="Session",
  paws_credentials=paws_mock,
  config=NULL,
  local_mode=FALSE
)

sagemaker_session$default_bucket = Mock$new()$return_value(BUCKET_NAME, .min_var = 0)
sagemaker_session$upload_data = Mock$new()$return_value(DEFAULT_S3_INPUT_DATA)
sagemaker_session$expand_role = Mock$new()$return_value(ROLE)
sagemaker_session$describe_auto_ml_job = Mock$new()$side_effect(describe_auto_ml_job_mock)
sagemaker_session$sagemaker$describe_training_job = Mock$new()$return_value(TRAINING_JOB)
sagemaker_session$sagemaker$describe_transform_job = Mock$new()$return_value(TRANSFORM_JOB)
sagemaker_session$list_candidates = Mock$new()$return_value(list("Candidates"= list()))
sagemaker_session$sagemaker$list_tags = Mock$new()$return_value(LIST_TAGS_RESULT)
sagemaker_session$call_args("auto_ml")
sagemaker_session$call_args("train")
sagemaker_session$call_args("transform")
sagemaker_session$logs_for_auto_ml_job = Mock$new()$return_value(NULL)

candidate = Mock$new(
  name="candidate_mock",
  containers=INFERENCE_CONTAINERS,
  steps=CANDIDATE_STEPS,
  sagemaker_session=sagemaker_session
)

test_that("test auto ml default channel name", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )

  inputs = DEFAULT_S3_INPUT_DATA
  AutoMLJob$new(sagemaker_session)$start_new(auto_ml, inputs)

  args = auto_ml$sagemaker_session$auto_ml()
  args$input_config

  expect_equal(args$input_config,
    list(
      list(
        "DataSource"= list(
          "S3DataSource"= list("S3DataType"= "S3Prefix", "S3Uri"= DEFAULT_S3_INPUT_DATA)
        ),
        "TargetAttributeName"= TARGET_ATTRIBUTE_NAME
      )
    )
  )
})

test_that("test auto ml invalid input data format", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )

  inputs = 1

  expect_error(
    AutoMLJob$new(sagemaker_session)$start_new(auto_ml, inputs),
    sprintf("Cannot format input %s. Expecting a string or a list of strings.", inputs))
})

test_that("test auto ml only one of problem type and job objective provided", {
  msg = paste0("One of problem type and objective metric provided. Either both of them ",
               "should be provided or none of them should be provided.")
  expect_error(
    AutoML$new(
      role=ROLE,
      target_attribute_name=TARGET_ATTRIBUTE_NAME,
      sagemaker_session=sagemaker_session,
      problem_type=PROBLEM_TYPE),
    msg)
})

test_that("test auto ml fit set logs to false", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )

  inputs = DEFAULT_S3_INPUT_DATA

  expect_output(auto_ml$fit(inputs, job_name=JOB_NAME, wait=FALSE, logs=TRUE),
                "Setting `logs` to FALSE. `logs` is only meaningful when `wait` is TRUE.")
})

test_that("test auto ml additional optional params", {
  auto_ml = AutoML$new(
    role=ROLE,
    target_attribute_name=TARGET_ATTRIBUTE_NAME,
    sagemaker_session=sagemaker_session,
    volume_kms_key=VOLUME_KMS_KEY,
    vpc_config=VPC_CONFIG,
    encrypt_inter_container_traffic=ENCRYPT_INTER_CONTAINER_TRAFFIC,
    compression_type=COMPRESSION_TYPE,
    output_kms_key=OUTPUT_KMS_KEY,
    output_path=OUTPUT_PATH,
    problem_type=PROBLEM_TYPE,
    max_candidates=MAX_CANDIDATES,
    max_runtime_per_training_job_in_seconds=MAX_RUNTIME_PER_TRAINING_JOB,
    total_job_runtime_in_seconds=TOTAL_JOB_RUNTIME,
    job_objective=JOB_OBJECTIVE,
    generate_candidate_definitions_only=GENERATE_CANDIDATE_DEFINITIONS_ONLY,
    tags=TAGS)

  inputs = DEFAULT_S3_INPUT_DATA
  auto_ml$fit(inputs, job_name=JOB_NAME)
  args = sagemaker_session$auto_ml()
  expect_equal(args, list(
      "input_config"= list(
        list(
          "DataSource"= list(
            "S3DataSource"= list("S3DataType"= "S3Prefix", "S3Uri"= DEFAULT_S3_INPUT_DATA)
          ),
          "CompressionType"= COMPRESSION_TYPE,
          "TargetAttributeName"= TARGET_ATTRIBUTE_NAME
          )
        ),
      "output_config"= list("S3OutputPath"= OUTPUT_PATH, "KmsKeyId"= OUTPUT_KMS_KEY),
      "auto_ml_job_config"= list(
        "CompletionCriteria"= list(
          "MaxCandidates"= MAX_CANDIDATES,
          "MaxRuntimePerTrainingJobInSeconds"= MAX_RUNTIME_PER_TRAINING_JOB,
          "MaxAutoMLJobRuntimeInSeconds"= TOTAL_JOB_RUNTIME
          ),
        "SecurityConfig"= list(
          "EnableInterContainerTrafficEncryption"= ENCRYPT_INTER_CONTAINER_TRAFFIC,
          "VolumeKmsKeyId"= VOLUME_KMS_KEY,
          "VpcConfig"= VPC_CONFIG
          )
        ),
      "role"= ROLE,
      "generate_candidate_definitions_only"= GENERATE_CANDIDATE_DEFINITIONS_ONLY,
      "job_name"= JOB_NAME,
      "problem_type"= PROBLEM_TYPE,
      "job_objective"= JOB_OBJECTIVE,
      "tags"= TAGS
    )
  )
})

test_that("test auto ml default fit", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )
  inputs = DEFAULT_S3_INPUT_DATA
  auto_ml$fit(inputs)
  args = sagemaker_session$auto_ml()
  # remove timestamp job name
  args$job_name = NULL
  expect_equal(args, list(
    "input_config"= list(
      list(
        "DataSource"= list(
          "S3DataSource"= list("S3DataType"= "S3Prefix", "S3Uri"= DEFAULT_S3_INPUT_DATA)
          ),
        "TargetAttributeName"= TARGET_ATTRIBUTE_NAME
        )
      ),
    "output_config"= list("S3OutputPath"= DEFAULT_OUTPUT_PATH),
    "auto_ml_job_config"= list(
      "CompletionCriteria"= list("MaxCandidates"= DEFAULT_MAX_CANDIDATES),
      "SecurityConfig"= list(
        "EnableInterContainerTrafficEncryption"= ENCRYPT_INTER_CONTAINER_TRAFFIC
        )
      ),
    "role"= ROLE,
    "generate_candidate_definitions_only"= GENERATE_CANDIDATE_DEFINITIONS_ONLY
    )
  )
})

test_that("test auto ml local input", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )
  inputs = DEFAULT_S3_INPUT_DATA
  auto_ml$fit(inputs)
  args = sagemaker_session$auto_ml()
  expect_equal(args$input_config[[1]]$DataSource$S3DataSource$S3Uri,
               DEFAULT_S3_INPUT_DATA)
})

test_that("test auto ml input", {
  inputs = AutoMLInput$new(
    inputs=DEFAULT_S3_INPUT_DATA, target_attribute_name="target", compression="Gzip"
  )
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )
  auto_ml$fit(inputs)
  args = sagemaker_session$auto_ml()
  expect_equal(args$input_config, list(
    list(
      "DataSource"= list(
        "S3DataSource"= list("S3DataType"= "S3Prefix", "S3Uri"= DEFAULT_S3_INPUT_DATA)
        ),
      "TargetAttributeName"= TARGET_ATTRIBUTE_NAME,
      "CompressionType"= "Gzip")
    )
  )
})

test_that("test describe auto ml job", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )
  expect_equal(auto_ml$describe_auto_ml_job(job_name=JOB_NAME),
               AUTO_ML_DESC)
})

test_that("test list candidates default", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )

  auto_ml$current_job_name = "current_job_name"
  expect_equal(auto_ml$list_candidates(), list())
})

sagemaker_session$call_args("list_candidates")

test_that("test list candidates with optional args", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )
  auto_ml$list_candidates(
    job_name=JOB_NAME,
    status_equals="Completed",
    candidate_name="candidate-name",
    candidate_arn="candidate-arn",
    sort_order="Ascending",
    sort_by="Status",
    max_results=99
  )

  args = sagemaker_session$list_candidates()
  expect_equal(args, list(
    "job_name"= JOB_NAME,
    "status_equals"= "Completed",
    "candidate_name"= "candidate-name",
    "candidate_arn"= "candidate-arn",
    "sort_order"= "Ascending",
    "sort_by"= "Status",
    "max_results"= 99)
    )
})

test_that("test best candidate with existing best candidate", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )

  auto_ml$.best_candidate = BEST_CANDIDATE
  best_candidate = auto_ml$best_candidate()
  expect_equal(best_candidate, BEST_CANDIDATE)
})

test_that("test best candidate default job name", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )

  auto_ml$current_job_name = JOB_NAME
  auto_ml$.auto_ml_job_desc = AUTO_ML_DESC
  best_candidate = auto_ml$best_candidate()
  expect_equal(best_candidate, BEST_CANDIDATE)
})

test_that("test best candidate job no desc", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )

  auto_ml$current_job_name = JOB_NAME
  best_candidate = auto_ml$best_candidate()
  expect_equal(best_candidate, BEST_CANDIDATE)
})

test_that("test best candidate no desc no job name", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )

  best_candidate = auto_ml$best_candidate(job_name=JOB_NAME)
  expect_equal(best_candidate, BEST_CANDIDATE)
})

test_that("test best candidate job name not match", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )

  auto_ml$current_job_name = JOB_NAME
  auto_ml$.auto_ml_job_desc = AUTO_ML_DESC
  best_candidate = auto_ml$best_candidate(job_name=JOB_NAME_2)

  expect_equal(best_candidate, BEST_CANDIDATE_2)
})

# Skipping Mock Deploy tests
# Unable to overload existing methods in R6 classes

test_that("test candidate estimator get steps", {
  candidate_estimator = CandidateEstimator$new(CANDIDATE_DICT, sagemaker_session=sagemaker_session)
  steps = candidate_estimator$get_steps()

  expect_equal(length(steps), 3)
})

test_that("test validate and update inference response", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )
  cic = auto_ml$validate_and_update_inference_response(
    inference_containers=CLASSIFICATION_INFERENCE_CONTAINERS,
    inference_response_keys=c("predicted_label", "labels", "probabilities", "probability")
  )

  expect_equal(cic[[3]]$Environment$SAGEMAKER_INFERENCE_OUTPUT, "predicted_label,labels,probabilities,probability")
  expect_equal(cic[[3]]$Environment$SAGEMAKER_INFERENCE_INPUT, "predicted_label,probabilities,probability")
  expect_equal(cic[[2]]$Environment$SAGEMAKER_INFERENCE_OUTPUT, "predicted_label,probabilities,probability")
})

test_that("test validate and update inference response wrong input", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )
  expect_error(
    auto_ml$validate_and_update_inference_response(
      inference_containers=CLASSIFICATION_INFERENCE_CONTAINERS,
      inference_response_keys=c("wrong_key", "wrong_label", "probabilities", "probability")
    )
  )
})

test_that("test create model", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )

  pipeline_model = auto_ml$create_model(
    name=JOB_NAME,
    sagemaker_session=sagemaker_session,
    candidate=CLASSIFICATION_CANDIDATE_DICT,
    vpc_config=VPC_CONFIG,
    enable_network_isolation=TRUE,
    model_kms_key=NULL,
    predictor_cls=NULL,
    inference_response_keys=NULL
  )

  expect_true(inherits(pipeline_model, "PipelineModel"))
})

test_that("test attach", {
  auto_ml = AutoML$new(
    role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
  )
  aml = auto_ml$attach(auto_ml_job_name=JOB_NAME_3, sagemaker_session=sagemaker_session)

  expect_equal(aml$current_job_name, JOB_NAME_3)
  expect_equal(aml$role, "mock_role_arn")
  expect_equal(aml$target_attribute_name, "y")
  expect_equal(aml$problem_type, "Auto")
  expect_equal(aml$output_path, "s3://output_prefix")
  expect_equal(aml$tags, LIST_TAGS_RESULT$Tags)
})
