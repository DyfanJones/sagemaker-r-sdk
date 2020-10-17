# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/automl/candidate_estimator.py

#' @include session.R
#' @include job.R
#' @include utils.R

#' @import R6

#' @title CandidateEstimator Class
#' @description A class for SageMaker AutoML Job Candidate
#' @export
CandidateEstimator = R6Class("CandidateEstimator",
  public = list(

    #' @description Constructor of CandidateEstimator.
    #' @param candidate (dict): a dictionary of candidate returned by AutoML.list_candidates()
    #'              or AutoML.best_candidate().
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    initialize = function(candidate,
                          sagemaker_session=NULL){
      self$name = candidate$CandidateName
      self$containers = candidate$InferenceContainers
      self$steps = self._process_steps(candidate$CandidateSteps)
      self$sagemaker_session = sagemaker_session %||% Session$new()
    },

    #' @description Get the step job of a candidate so that users can construct estimators/transformers
    #' @return list: a list of dictionaries that provide information about each step job's name,
    #'              type, inputs and description
    get_steps = function(){
      candidate_steps = list()
      for (step in self$steps){
        step_type = step$type
        step_name = step$name
        if (step_type == "TrainingJob"){
          training_job = self$sagemaker_session$sagemaker$describe_training_job(
            TrainingJobName=step_name)

          inputs = training_job$InputDataConfig
          candidate_step = CandidateStep$new(step_name, inputs, step_type, training_job)
          candidate_steps = c(candidate_steps, candidate_step)
        } else if (step_type == "TransformJob") {
          transform_job = self$sagemaker_session$sagemaker$describe_transform_job(
            TransformJobName=step_name)
          inputs = transform_job$TransformInput
          candidate_step = CandidateStep$new(step_name, inputs, step_type, transform_job)
          candidate_steps = c(candidate_steps, candidate_step)
        }
      }
      return(candidate_steps)
    },

    #' @description Rerun a candidate's step jobs with new input datasets or security config.
    #' @param inputs (str or list[str]): Local path or S3 Uri where the training data is stored. If a
    #'              local path is provided, the dataset will be uploaded to an S3 location.
    #' @param candidate_name (str): name of the candidate to be rerun, if None, candidate's original
    #'              name will be used.
    #' @param volume_kms_key (str): The KMS key id to encrypt data on the storage volume attached to
    #'              the ML compute instance(s).
    #' @param encrypt_inter_container_traffic (bool): To encrypt all communications between ML compute
    #'              instances in distributed training. Default: False.
    #' @param vpc_config (dict): Specifies a VPC that jobs and hosted models have access to.
    #'              Control access to and from training and model containers by configuring the VPC
    #' @param wait (bool): Whether the call should wait until all jobs completes (default: True).
    #' @param logs (bool): Whether to show the logs produced by the job.
    #'              Only meaningful when wait is True (default: True).
    fit = function(inputs,
                   candidate_name=NULL,
                   volume_kms_key=NULL,
                   encrypt_inter_container_traffic=FALSE,
                   vpc_config=NULL,
                   wait=TRUE,
                   logs=TRUE){
      if (logs && !wait)
        stop(
          "Logs can only be shown if wait is set to True.",
          "Please either set wait to True or set logs to False.", call. = F
        )

      self$name = candidate_name %||% self$name
      running_jobs = list()

      # convert inputs to TrainingInput format
      if (inherits(inputs, "character")){
        if (!startsWith(inputs, "s3://"))
          inputs = self$sagemaker_session$upload_data(inputs, key_prefix="auto-ml-input-data")
      }

      for(step in self$steps){
        step_type = step$type
        step_name = step$name
        if(step_type == "TrainingJob"){
          # prepare inputs
          input_dict = list()
          if (inherits(inputs, "character")) {
            input_dict$train = .Job$private_methods$.format_string_uri_input(inputs)
          } else {
            msg = "Cannot format input %s. Expecting a string."
            stop(sprintf(msg,inputs), call. = F)
          }
          channels = lapply(seq_along(input_dict), function(i)
            .Job$private_methods$.convert_input_to_channel(names(input_dict)[i], input_dict[[i]]))

          desc = self$sagemaker_session$sagemaker$describe_training_job(
            TrainingJobName=step_name
          )

          base_name = "sagemaker-automl-training-rerun"
          step_name = name_from_base(base_name)
          step$name = step_name
          train_args = private$.get_train_args(
            desc,
            channels,
            step_name,
            volume_kms_key,
            encrypt_inter_container_traffic,
            vpc_config
          )

          do.call(self$sagemaker_session$train, train_args)
          running_jobs[[step_name]] = TRUE
        } else if (step_type == "TransformJob"){
          # prepare inputs
          if (!inherits(inputs, "character") || !startsWith(inputs, "s3://")){
            msg = "Cannot format input %s. Expecting a string starts with file:// or s3://"
            stop(sprintf(msg, inputs), call. = F)}
          desc = self$sagemaker_session$sagemaker$describe_transform_job(
             TransformJobName=step_name
           )
          base_name = "sagemaker-automl-transform-rerun"
          step_name = name_from_base(base_name)
          step$name = step_name
          transform_args = private$.get_transform_args(desc, inputs, step_name, volume_kms_key)
          do.call(self$sagemaker_session$transform, transform_args)
          running_jobs[[step_name]] = TRUE
        }
      }

      if (wait){
        while(True){
          for (step in self$steps){
            status = NULL
            step_type = step$type
            step_name = step$name
            if (step_type == "TrainingJob"){
              status = self$sagemaker_session$sagemaker$describe_training_job(
                TrainingJobName=step_name)$TrainingJobStatus
            } else if (step_type == "TransformJob")
              status = self$sagemaker_session$sagemaker$describe_transform_job(
                TransformJobName=step_name)$TransformJobStatus
            if (status %in% c("Completed", "Failed", "Stopped"))
              running_jobs[[step_name]] = FALSE
            if (private$.check_all_job_finished(running_jobs))
              break
          }
        }
      }
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      return(print_class(self))
    }
  ),
  private = list(

    # Check if all step jobs are finished.
    # Args:
    #   running_jobs (dict): a dictionary that keeps track of the status
    # of each step job.
    # Returns (bool): True if all step jobs are finished. False if one or
    # more step jobs are still running.
    .check_all_job_finished = function(running_jobs){
      for (i in seq_along(running_jobs)){
        if (running_jobs[[i]])
          return(FALSE)
        }
      return(TRUE)
    },

    # Format training args to pass in sagemaker_session.train.
    # Args:
    #   desc (dict): the response from DescribeTrainingJob API.
    # inputs (list): a list of input data channels.
    # name (str): the name of the step job.
    # volume_kms_key (str): The KMS key id to encrypt data on the storage volume attached to
    # the ML compute instance(s).
    # encrypt_inter_container_traffic (bool): To encrypt all communications between ML compute
    # instances in distributed training.
    # vpc_config (dict): Specifies a VPC that jobs and hosted models have access to.
    # Control access to and from training and model containers by configuring the VPC
    # Returns (dcit): a dictionary that can be used as args of
    # sagemaker_session.train method.
    .get_train_args = function(desc,
                               inputs,
                               name,
                               volume_kms_key,
                               encrypt_inter_container_traffic,
                               vpc_config){
      train_args = list(
        "input_config"= inputs,
        "job_name"= name,
        "input_mode"= desc$AlgorithmSpecification$TrainingInputMode,
        "role"= desc$RoleArn,
        "output_config"= desc$OutputDataConfig,
        "resource_config"= desc$ResourceConfig,
        "image_uri"= desc$AlgorithmSpecification$TrainingImage,
        "enable_network_isolation"= desc$EnableNetworkIsolation,
        "encrypt_inter_container_traffic"= encrypt_inter_container_traffic,
        "use_spot_instances"= desc$EnableManagedSpotTraining,
        "hyperparameters"= list(),
        "stop_condition"= list(),
        "metric_definitions"= NULL,
        "checkpoint_s3_uri"= NULL,
        "checkpoint_local_path"= NULL,
        "tags"= list(),
        "vpc_config"= NULL)
      if (!islistempty(volume_kms_key))
        train_args$resource_config$VolumeKmsKeyId = volume_kms_key
      if ("VpcConfig" %in% names(desc)) {
        train_args$vpc_config = desc$VpcConfig
      } else if (!islistempty(vpc_config))
        train_args$vpc_config = vpc_config
      if ("Hyperparameters" %in% names(desc))
        train_args$hyperparameters = desc$Hyperparameters
      if ("CheckpointConfig" %in% names(desc)){
        train_args$checkpoint_s3_uri = desc$CheckpointConfig$S3Uri
        train_args$checkpoint_local_path = desc$CheckpointConfig$LocalPath}
      if ("StoppingCondition" %in% names(desc))
        train_args$stop_condition = desc$StoppingCondition
      return(train_args)
    },

    # Format training args to pass in sagemaker_session.train.
    # Args:
    #   desc (dict): the response from DescribeTrainingJob API.
    # inputs (str): an S3 uri where new input dataset is stored.
    # name (str): the name of the step job.
    # volume_kms_key (str): The KMS key id to encrypt data on the storage volume attached to
    # the ML compute instance(s).
    # Returns (dcit): a dictionary that can be used as args of
    # sagemaker_session.transform method.
    .get_transform_args = function(desc,
                                   inputs,
                                   name,
                                   volume_kms_key){
      transform_args = list(
        job_name = name,
        model_name = desc$ModelName,
        output_config = desc$TransformOutput,
        resource_config = desc$TransformResources,
        data_processing = desc$DataProcessing,
        tags = list(),
        strategy = NULL,
        max_concurrent_transforms = NULL,
        max_payload = NULL,
        env = NULL,
        experiment_config = NULL)

      input_config = desc$TransformInput
      input_config$DataSource$S3DataSource$S3Uri = inputs
      transform_args$input_config = input_config
      transform_args$resource_config$VolumeKmsKeyId = volume_kms_key
      transform_args$strategy = desc$BatchStrategy
      transform_args$max_concurrent_transforms = desc$MaxConcurrentTransforms
      transform_args$max_payload = desc$MaxPayloadInMB
      transform_args$env = desc$Environment

      return(transform_args)
    },

    # Extract candidate's step jobs name and type.
    #     Args:
    #         steps (list): a list of a candidate's step jobs.
    # Returns (list): a list of extracted information about step jobs'
    #         name and type.
    .process_steps = function(steps){
      processed_steps = list()
      for (step in steps){
        step_name = step$CandidateStepName
        step_type = split_str(step$CandidateStepType, "::")[3]
        processed_steps= c(processed_steps, list("name"= step_name, "type"= step_type))
      }
      return(processed_steps)
    }
  ),
  lock_objects = F
)

#' @title CandidateStep Class
#' @description A class that maintains an AutoML Candidate step's name, inputs, type, and description.
#' @export
CandidateStep = R6Class("CandidateStep",
  public = list(

    #' @field name
    #' Name of the candidate step -> (str)
    name = NULL,

    #' @field inputs
    #' Inputs of the candidate step -> (dict)
    inputs = NULL,

    #' @field type
    #' Type of the candidate step, Training or Transform -> (str)
    type = NULL,

    #' @field description
    #' Description of candidate step job -> (dict)
    description = NULL,

    #' @description Initialize CandidateStep Class
    #' @param name (str): Name of the candidate step
    #' @param inputs (dict): Inputs of the candidate step
    #' @param step_type (str): Type of the candidate step, Training or Transform
    #' @param description (dict): Description of candidate step job
    initialize = function(name, inputs, step_type, description){
      self$name = name
      self$inputs = inputs
      self$type = step_type
      self$description = description
    }
  )
)
