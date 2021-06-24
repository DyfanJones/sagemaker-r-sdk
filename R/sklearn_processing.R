# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/sklearn/model.py

#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common
#' @import lgr

#' @title SKLearnProcessor Class
#' @description Handles Amazon SageMaker processing tasks for jobs using scikit-learn.
#' @export
SKLearnProcessor = R6Class("SKLearnProcessor",
  inherit = R6sagemaker.common::ScriptProcessor,
  public = list(

    #' @description Initialize an ``SKLearnProcessor`` instance. The SKLearnProcessor
    #'              handles Amazon SageMaker processing tasks for jobs using scikit-learn.
    #' @param framework_version (str): The version of scikit-learn.
    #' @param role (str): An AWS IAM role name or ARN. The Amazon SageMaker training jobs
    #'              and APIs that create Amazon SageMaker endpoints use this role
    #'              to access training data and model artifacts. After the endpoint
    #'              is created, the inference code might use the IAM role, if it
    #'              needs to access an AWS resource.
    #' @param instance_type (str): Type of EC2 instance to use for
    #'              processing, for example, 'ml.c4.xlarge'.
    #' @param instance_count (int): The number of instances to run
    #'              the Processing job with. Defaults to 1.
    #' @param command ([str]): The command to run, along with any command-line flags.
    #'              Example: ["python3", "-v"]. If not provided, ["python3"] or ["python2"]
    #'              will be chosen based on the py_version parameter.
    #' @param volume_size_in_gb (int): Size in GB of the EBS volume to
    #'              use for storing data during processing (default: 30).
    #' @param volume_kms_key (str): A KMS key for the processing
    #'              volume.
    #' @param output_kms_key (str): The KMS key id for all ProcessingOutputs.
    #' @param max_runtime_in_seconds (int): Timeout in seconds.
    #'              After this amount of time Amazon SageMaker terminates the job
    #'              regardless of its current status.
    #' @param base_job_name (str): Prefix for processing name. If not specified,
    #'              the processor generates a default job name, based on the
    #'              training image name and current timestamp.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the processor creates one
    #'              using the default AWS configuration chain.
    #' @param env (dict): Environment variables to be passed to the processing job.
    #' @param tags ([dict]): List of tags to be passed to the processing job.
    #' @param network_config (sagemaker.network.NetworkConfig): A NetworkConfig
    #'              object that configures network isolation, encryption of
    #'              inter-container traffic, security group IDs, and subnets.
    initialize = function(framework_version,
                          role,
                          instance_type,
                          instance_count,
                          command=NULL,
                          volume_size_in_gb=30,
                          volume_kms_key=NULL,
                          output_kms_key=NULL,
                          max_runtime_in_seconds=NULL,
                          base_job_name=NULL,
                          sagemaker_session=NULL,
                          env=NULL,
                          tags=NULL,
                          network_config=NULL){
      if (is.null(command))
        command = list("python3")

      session = sagemaker_session %||% Session$new()
      region = session$paws_region_name
      image_uri = ImageUris$new()$retrieve(
        "sklearn", region, version=framework_version, instance_type=instance_type
      )

      super$initialize(
        role=role,
        image_uri=image_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        command=command,
        volume_size_in_gb=volume_size_in_gb,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        max_runtime_in_seconds=max_runtime_in_seconds,
        base_job_name=base_job_name,
        sagemaker_session=session,
        env=env,
        tags=tags,
        network_config=network_config
      )
    }
  ),
  lock_objects = F
)
