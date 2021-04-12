# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/workflow/airflow.py

#' @include fw_utils.R
#' @include job.R
#' @include utils.R
#' @include s3.R
#' @include session.R
#' @include vpc_utils.R
#' @include amazon_estimator.R
#' @include tensorflow_estimator.R
#' @include model.R
#' @include error.R

#' @import R6

#' @title AirFlow helper class
#' @description Helper class to take sagemaker classes and format output for Airflow.
#' @export
AirFlow = R6Class("AirFlow",
  public = list(

    #' @description Prepare S3 operations and environment variables related to framework.
    #'              S3 operations specify where to upload `source_dir`.
    #' @param estimator (sagemaker.estimator.Estimator): The framework estimator to
    #'              get information from and update.
    #' @param s3_operations (dict): The dict to specify s3 operations (upload
    #'              `source_dir` ).
    prepare_framework = function(estimator,
                                 s3_operations){
      s3_split = list()
      if (!is.null(estimator$code_location)){
        s3_split = split_s3_uri(estimator$code_location)
        s3_split$key = os.path.join(s3_split$keykey, estimator$.current_job_name, "source", "sourcedir.tar.gz")
      } else if (!is.null(estimator$uploaded_code)){
        s3_split = split_s3_uri(estimator.uploaded_code.s3_prefix)
      } else {
        s3_split$bucket = estimator$sagemaker_session$.__enclos_env__$private$.default_bucket
        s3_split$key = file.path(estimator$.current_job_name, "source", "sourcedir.tar.gz")
      }

      script = basename(estimator$entry_point)

      if (!is.null(estimator$source_dir) && grepl("^s3://", tolower(estimator$source_dir))){
          code_dir = estimator$source_dir
          UploadedCode$s3_prefix=code_dir
          UploadedCode$script_name= script
          estimator$uploaded_code = UploadedCode
      } else {
          code_dir = sprintf("s3://%s/%s", s3_split$bucket, s3_split$key)
          UploadedCode$s3_prefix=code_dir
          UploadedCode$script_name= script
          estimator$uploaded_code = UploadedCode

          s3_operations[["S3Upload"]] = list(
            list(
              "Path"=(estimator$source_dir %||% estimator$entry_point),
              "Bucket"=s3_split$bucket,
              "Key"=s3_split$key,
              "Tar"=TRUE)
          )
      }
      estimator$.hyperparameters[[sagemaker$model$DIR_PARAM_NAME]] = code_dir
      estimator$.hyperparameters[[sagemaker$model$SCRIPT_PARAM_NAME]] = script
      estimator$.hyperparameters[[
        sagemaker$model$CONTAINER_LOG_LEVEL_PARAM_NAME
      ]] = estimator$container_log_level
      estimator$.hyperparameters[[sagemaker$model$JOB_NAME_PARAM_NAME]] = estimator$.current_job_name
      estimator$.hyperparameters[[
        sagemaker$model$SAGEMAKER_REGION_PARAM_NAME
      ]] = estimator$sagemaker_session$paws_region_name
    },

    #' @description Sets up amazon algorithm estimator.
    #'              This is done by adding the required `feature_dim` hyperparameter from training data.
    #' @param estimator (sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase): An estimator
    #'              for a built-in Amazon algorithm to get information from and update.
    #' @param inputs: The training data.
    #'              * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
    #'              Amazon :class:~`Record` objects serialized and stored in S3. For
    #'              use with an estimator for an Amazon algorithm.
    #'              * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
    #'              :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
    #'              where each instance is a different channel of training data.
    #' @param mini_batch_size (numeric):
    prepare_amazon_algorithm_estimator = function(estimator,
                                                  inputs,
                                                  mini_batch_size=NULL){
      if (is.list(inputs)){
        for (record in inputs){
          if (inherits(record, "RecordSet") && record$channel == "train"){
              estimator$feature_dim = record$feature_dim
              break
            }
        }
      } else if (inherits(inputs, "RecordSet")){
          estimator$feature_dim = inputs$feature_dim
      } else {
        TypeError$new("Training data must be represented in RecordSet or list of RecordSets")
      }
      estimator$mini_batch_size = mini_batch_size
    },

    #' @description Export Airflow base training config from an estimator
    #' @param estimator (sagemaker.estimator.EstimatorBase): The estimator to export
    #'              training config from. Can be a BYO estimator, Framework estimator or
    #'              Amazon algorithm estimator.
    #'              inputs: Information about the training data. Please refer to the ``fit()``
    #'              method of
    #'              the associated estimator, as this can take any of the following
    #'              forms:
    #'              * (str) - The S3 location where training data is saved.
    #'              * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput]) - If using multiple
    #'              channels for training data, you can specify a dict mapping channel names to
    #'              strings or :func:`~sagemaker.inputs.TrainingInput` objects.
    #'              * (sagemaker.inputs.TrainingInput) - Channel configuration for S3 data sources that can
    #'              provide additional information about the training dataset. See
    #'              :func:`sagemaker.inputs.TrainingInput` for full details.
    #'              * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
    #'              Amazon :class:~`Record` objects serialized and stored in S3.
    #'              For use with an estimator for an Amazon algorithm.
    #'              * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
    #'              :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
    #'              where each instance is a different channel of training data.
    #' @param job_name (str): Specify a training job name if needed.
    #' @param mini_batch_size (int): Specify this argument only when estimator is a
    #'              built-in estimator of an Amazon algorithm. For other estimators,
    #'              batch size should be specified in the estimator.
    #' @return dict: Training config that can be directly used by
    #'              SageMakerTrainingOperator in Airflow.
    training_base_config = function(estimator,
                                    inputs=NULL,
                                    job_name=NULL,
                                    mini_batch_size=NULL){
      if (inherits(estimator, "AmazonAlgorithmEstimatorBase")){
        estimator$prepare_workflow_for_training(
          records=inputs, mini_batch_size=mini_batch_size, job_name=job_name
        )
      } else {
        estimator$prepare_workflow_for_training(job_name=job_name)
      }
      s3_operations = list()

      if (!is.null(job_name)){
        estimator$.current_job_name = job_name
      } else {
        base_name = estimator.base_job_name %||% base_name_from_image(
          estimator$training_image_uri())
        estimator$.current_job_name = name_from_base(base_name)
      }
      if (is.null(estimator$output_path)){
        default_bucket = estimator$sagemaker_session$default_bucket()
        estimator$output_path = sprintf("s3://%s/",default_bucket)
      }
      if (inherits(estimator, "Framework")){
        self$prepare_framework(estimator, s3_operations)

      } else if (inherits(estimator, "AmazonAlgorithmEstimatorBase")){
        self$prepare_amazon_algorithm_estimator(estimator, inputs, mini_batch_size)
      }
      job_config = .Job$new()$.__enclos_env__$.load_config(
        inputs, estimator, expand_role=FALSE, validate_uri=FALSE)

      train_config = list(
        "AlgorithmSpecification"=list(
          "TrainingImage"=estimator$training_image_uri(),
          "TrainingInputMode"=estimator.input_mode),
        "OutputDataConfig"=job_config[["output_config"]],
        "StoppingCondition"=job_config[["stop_condition"]],
        "ResourceConfig"=job_config[["resource_config"]],
        "RoleArn"=job_config[["role"]])

      train_config[["InputDataConfig"]] = job_config[["input_config"]]
      train_config[["VpcConfig"]] = job_config[["vpc_config"]]

      if (estimator$use_spot_instances)
        train_config[["EnableManagedSpotTraining"]] = TRUE

      if (!islistempty(estimator$hyperparameters())){
        hyperparameters = estimator$hyperparameters()
        train_config[["HyperParameters"]] = hyperparameters
      }
      if (!islistempty(s3_operations))
        train_config[["S3Operations"]] = s3_operations

      return(train_config)
    },

    #' @description Export Airflow training config from an estimator
    #' @param estimator (sagemaker.estimator.EstimatorBase): The estimator to export
    #'              training config from. Can be a BYO estimator, Framework estimator or
    #'              Amazon algorithm estimator.
    #' @param inputs: Information about the training data. Please refer to the ``fit()``
    #'              method of the associated estimator, as this can take any of the following forms:
    #'              * (str) - The S3 location where training data is saved.
    #'              * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput]) - If using multiple
    #'              channels for training data, you can specify a dict mapping channel names to
    #'              strings or :func:`~sagemaker.inputs.TrainingInput` objects.
    #'              * (sagemaker.inputs.TrainingInput) - Channel configuration for S3 data sources that can
    #'              provide additional information about the training dataset. See
    #'              :func:`sagemaker.inputs.TrainingInput` for full details.
    #'              * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
    #'              Amazon :class:~`Record` objects serialized and stored in S3.
    #'              For use with an estimator for an Amazon algorithm.
    #'              * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
    #'              :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
    #'              where each instance is a different channel of training data.
    #' @param job_name (str): Specify a training job name if needed.
    #' @param mini_batch_size (int): Specify this argument only when estimator is a
    #'              built-in estimator of an Amazon algorithm. For other estimators,
    #'              batch size should be specified in the estimator.
    #' @return dict: Training config that can be directly used by
    #'              SageMakerTrainingOperator in Airflow.
    training_config = function(estimator,
                               inputs=NULL,
                               job_name=NULL,
                               mini_batch_size=NULL){
      train_config = self$training_base_config(estimator, inputs, job_name, mini_batch_size)

      train_config[["TrainingJobName"]] = estimator$.current_job_name

      if (!is.null(estimator$tags))
        train_config[["Tags"]] = estimator$tags

      if (!is.null(estimator$metric_definitions))
        train_config[["AlgorithmSpecification"]][["MetricDefinitions"]] = estimator$metric_definitions

      return(train_config)
    },

    #' @description Export Airflow tuning config from a HyperparameterTuner
    #' @param tuner (sagemaker.tuner.HyperparameterTuner): The tuner to export tuning
    #'              config from.
    #' @param inputs: Information about the training data. Please refer to the ``fit()``
    #'              method of the associated estimator in the tuner, as this can take any of the
    #'              following forms:
    #'              * (str) - The S3 location where training data is saved.
    #'              * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput]) - If using multiple
    #'              channels for training data, you can specify a dict mapping channel names to
    #'              strings or :func:`~sagemaker.inputs.TrainingInput` objects.
    #'              * (sagemaker.inputs.TrainingInput) - Channel configuration for S3 data sources that can
    #'              provide additional information about the training dataset. See
    #'              :func:`sagemaker.inputs.TrainingInput` for full details.
    #'              * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
    #'              Amazon :class:~`Record` objects serialized and stored in S3.
    #'              For use with an estimator for an Amazon algorithm.
    #'              * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
    #'              :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
    #'              where each instance is a different channel of training data.
    #'              * (dict[str, one the forms above]): Required by only tuners created via
    #'              the factory method ``HyperparameterTuner.create()``. The keys should be the
    #'              same estimator names as keys for the ``estimator_dict`` argument of the
    #'              ``HyperparameterTuner.create()`` method.
    #' @param job_name (str): Specify a tuning job name if needed.
    #' @param include_cls_metadata: It can take one of the following two forms.
    #'              * (bool) - Whether or not the hyperparameter tuning job should include information
    #'              about the estimator class (default: False). This information is passed as a
    #'              hyperparameter, so if the algorithm you are using cannot handle unknown
    #'              hyperparameters (e.g. an Amazon SageMaker built-in algorithm that does not
    #'              have a custom estimator in the Python SDK), then set ``include_cls_metadata``
    #'              to ``False``.
    #'              * (dict[str, bool]) - This version should be used for tuners created via the factory
    #'              method ``HyperparameterTuner.create()``, to specify the flag for individual
    #'              estimators provided in the ``estimator_dict`` argument of the method. The keys
    #'              would be the same estimator names as in ``estimator_dict``. If one estimator
    #'              doesn't need the flag set, then no need to include it in the dictionary. If none
    #'              of the estimators need the flag set, then an empty dictionary ``{}`` must be used.
    #' @param mini_batch_size: It can take one of the following two forms.
    #'              * (int) - Specify this argument only when estimator is a built-in estimator of an
    #'              Amazon algorithm. For other estimators, batch size should be specified in the
    #'              estimator.
    #'              * (dict[str, int]) - This version should be used for tuners created via the factory
    #'              method ``HyperparameterTuner.create()``, to specify the value for individual
    #'              estimators provided in the ``estimator_dict`` argument of the method. The keys
    #'              would be the same estimator names as in ``estimator_dict``. If one estimator
    #' doesn't need the value set, then no need to include it in the dictionary. If
    #'              none of the estimators need the value set, then an empty dictionary ``{}``
    #'              must be used.
    #' @return dict: Tuning config that can be directly used by SageMakerTuningOperator in Airflow.
    tuning_config = function(tuner,
                             inputs,
                             job_name=NULL,
                             include_cls_metadata=FALSE,
                             mini_batch_size=NULL){
      tuner$.__enclos_env__$.prepare_job_name_for_tuning(job_name=job_name)

      tune_config = list(
        "HyperParameterTuningJobName"=tuner$.current_job_name,
        "HyperParameterTuningJobConfig"= private$.extract_tuning_job_config(tuner)
      )

      if (!is.null(tuner$estimator)){
        ll = private$.extract_training_config_from_estimator(
          tuner, inputs, include_cls_metadata, mini_batch_size)
        tune_config[["TrainingJobDefinition"]] = ll$TrainingJobDefinitions
        s3_operations = ll$s3_operations
      } else {
        ll = private$.extract_training_config_list_from_estimator_dict(
          tuner, inputs, include_cls_metadata, mini_batch_size)
        tune_config[["TrainingJobDefinitions"]] = ll$TrainingJobDefinitions
        s3_operations = ll$s3_operations
      }

      if (!islistempty(s3_operations))
        tune_config[["S3Operations"]] = s3_operations

      if (!islistempty(tuner$tags))
        tune_config[["Tags"]] = tuner$tags

      if (!islistempty(tuner$warm_start_config))
        tune_config[["WarmStartConfig"]] = tuner$warm_start_config$to_input_req()

      return(tune_config)
    },


    #' @description Updated the S3 URI of the framework source directory in given estimator.
    #' @param estimator (sagemaker.estimator.Framework): The Framework estimator to
    #'              update.
    #' @param job_name (str): The new job name included in the submit S3 URI
    #' @return str: The updated S3 URI of framework source directory
    update_submit_s3_uri=function(estimator, job_name){
      if (islistempty(estimator$uploaded_code))
        return(NULL)

      pattern = "(?<=/)[^/]+?(?=/source/sourcedir.tar.gz)"

      # update the S3 URI with the latest training job.
      # s3://path/old_job/source/sourcedir.tar.gz will become s3://path/new_job/source/sourcedir.tar.gz
      submit_uri = estimator$uploaded_code$s3_prefix
      submit_uri = gsub(pattern, job_name, submit_uri)
      script_name = estimator$uploaded_code$script_name
      UploadedCode$s3_prefix=submit_uri
      UploadedCode$script_name=script_name
      estimator$uploaded_code = UploadedCode
    },

    #' @description Update training job of the estimator from a task in the DAG
    #' @param estimator (sagemaker.estimator.EstimatorBase): The estimator to update
    #' @param task_id (str): The task id of any
    #'              airflow.contrib.operators.SageMakerTrainingOperator or
    #'              airflow.contrib.operators.SageMakerTuningOperator that generates
    #'              training jobs in the DAG.
    #' @param task_type (str): Whether the task is from SageMakerTrainingOperator or
    #'              SageMakerTuningOperator. Values can be 'training', 'tuning' or None
    #'              (which means training job is not from any task).
    update_estimator_from_task = function(estimator,
                                          task_id,
                                          task_type){
      if (is.null(task_type))
        return(NULL)
      if (tolower(task_type) == "training"){
        training_job = sprintf(
          "{{ ti.xcom_pull(task_ids='%s')['Training']['TrainingJobName'] }}", task_id)
        job_name = training_job
      } else if (tolower(task_type) == "tuning"){
        training_job = sprintf(
          "{{ ti.xcom_pull(task_ids='%s')['Tuning']['BestTrainingJob']['TrainingJobName'] }}",
          task_id)
        # need to strip the double quotes in json to get the string
        job_name = sprintf(paste0(
          "{{ ti.xcom_pull(task_ids='%s')['Tuning']['TrainingJobDefinition']",
          "['StaticHyperParameters']['sagemaker_job_name'].strip('%s') }}"), task_id, '"')
      } else {
        ValueError$new("task_type must be either 'training', 'tuning' or None.")}
      estimator$.current_job_name = training_job
      if (inherits(estimator, "Framework"))
        self$update_submit_s3_uri(estimator, job_name)
    },

    #' @description This prepares the framework model container information and specifies related S3 operations.
    #'              Prepare the framework model container information. Specify related S3
    #'              operations for Airflow to perform. (Upload `source_dir` )
    #' @param model (sagemaker.model.FrameworkModel): The framework model
    #' @param instance_type (str): The EC2 instance type to deploy this Model to. For
    #'              example, 'ml.p2.xlarge'.
    #' @param s3_operations (dict): The dict to specify S3 operations (upload
    #'              `source_dir` ).
    #' @return dict: The container information of this framework model.
    prepare_framework_container_def = function(model,
                                               instance_type,
                                               s3_operations){
      deploy_image = model$image_uri
      if (islistempty(deploy_image)){
        region_name = model$sagemaker_session$paws_region_name
        deploy_image = model$serving_image_uri(region_name, instance_type)}
      base_name = base_name_from_image(deploy_image)
      model$name = model$name %||% name_from_base(base_name)

      bucket = model$bucket %||% model$sagemaker_session$.default_bucket
      if (!is.null(model$entry_point)){
        script = basename(model$entry_point)
        key = sprintf("%s/source/sourcedir.tar.gz", model$name)

        if (!islistempty(model$source_dir) && grepl("^s3://", tolower(model$source_dir))){
          code_dir = model$source_dir
          UploadedCode$s3_prefix=code_dir
          UploadedCode$script_name= script
          model$uploaded_code = UploadedCode
        } else {
          code_dir = sprintf("s3://%s/%s", bucket, key)
          UploadedCode$s3_prefix=code_dir
          UploadedCode$script_name= script
          model$uploaded_code = UploadedCode
          s3_operations[["S3Upload"]] = list(
            list("Path"=(model$source_dir %||% script), "Bucket"=bucket, "Key"=key, "Tar"=TRUE)
          )
        }
      }
      deploy_env = list(model$env)
      deploy_env = modifyList(deploy_env, model$.__enclos_env__$.framework_env_vars())

      tryCatch({
        if (!islistempty(model$model_server_workers))
          deploy_env[[toupper(MODEL_SERVER_WORKERS_PARAM_NAME)]] = as.character(
            model$model_server_workers)
      }, error = function(e) {
        # This applies to a FrameworkModel which is not SageMaker Deep Learning Framework Model
        NULL
      })

      return (container_def(deploy_image, model$model_data, deploy_env))
    },

    #' @description Export Airflow model config from a SageMaker model
    #' @param model (sagemaker.model.Model): The Model object from which to export the Airflow config
    #' @param instance_type (str): The EC2 instance type to deploy this Model to. For
    #'              example, 'ml.p2.xlarge'
    #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the model
    #' @param image_uri (str): An Docker image URI to use for deploying the model
    #' @return dict: Model config that can be directly used by SageMakerModelOperator
    #'              in Airflow. It can also be part of the config used by
    #'              SageMakerEndpointOperator and SageMakerTransformOperator in Airflow.
    model_config = function(model,
                            instance_type=NULL,
                            role=NULL,
                            image_uri=NULL){
      s3_operations = list()
      model.image_uri = image_uri %||% model$image_uri

      if (inherits(model, "FrameworkModel")){
        container_def = prepare_framework_container_def(model, instance_type, s3_operations)
      } else {
        container_def = model.prepare_container_def()
        base_name = base_name_from_image(container_def[["Image"]])
        model$name = model$name %||% name_from_base(base_name)
      }
      primary_container = session$.__enclos_env__$.expand_container_def(container_def)

      config = list(
        "ModelName"=model$name,
        "PrimaryContainer"=primary_container,
        "ExecutionRoleArn"=role %||% model$role)

      if (!islistempty(model$vpc_config))
        config[["VpcConfig"]] = model$vpc_config

      if (!islistempty(s3_operations))
        config[["S3Operations"]] = s3_operations

      return(config)
    },

    #' @description Export Airflow model config from a SageMaker estimator
    #' @param estimator (sagemaker.model.EstimatorBase): The SageMaker estimator to
    #'              export Airflow config from. It has to be an estimator associated
    #'              with a training job.
    #' @param task_id (str): The task id of any
    #'              airflow.contrib.operators.SageMakerTrainingOperator or
    #'              airflow.contrib.operators.SageMakerTuningOperator that generates
    #'              training jobs in the DAG. The model config is built based on the
    #'              training job generated in this operator.
    #' @param task_type (str): Whether the task is from SageMakerTrainingOperator or
    #'              SageMakerTuningOperator. Values can be 'training', 'tuning' or None
    #'              (which means training job is not from any task).
    #' @param instance_type (str): The EC2 instance type to deploy this Model to. For
    #'              example, 'ml.p2.xlarge'
    #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the model
    #' @param image_uri (str): A Docker image URI to use for deploying the model
    #' @param name (str): Name of the model
    #' @param model_server_workers (int): The number of worker processes used by the
    #'              inference server. If None, server will use one worker per vCPU. Only
    #'              effective when estimator is a SageMaker framework.
    #' @param vpc_config_override (dict[str, list[str]]): Override for VpcConfig set on
    #'              the model. Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @return dict: Model config that can be directly used by SageMakerModelOperator in Airflow. It can
    #'              also be part of the config used by SageMakerEndpointOperator in Airflow.
    model_config_from_estimator = function(
      estimator,
      task_id,
      task_type,
      instance_type=None,
      role=None,
      image_uri=None,
      name=None,
      model_server_workers=None,
      vpc_config_override="VPC_CONFIG_DEFAULT"){
      self$update_estimator_from_task(estimator, task_id, task_type)
      if (inherits(estimator, "Estimator")){
        model = estimator$create_model(
          role=role, image_uri=image_uri, vpc_config_override=vpc_config_override
        )
      } else if (inherits(estimator, "AmazonAlgorithmEstimatorBase")){
        model = estimator.create_model(vpc_config_override=vpc_config_override)
      } else if (inherits(estimator, "TensorFlow")){
        model = estimator$create_model(
          role=role, vpc_config_override=vpc_config_override, entry_point=estimator$entry_point
        )
      } else if (inherits(estimator, "Framework")){
        model = estimator$create_model(
          model_server_workers=model_server_workers,
          role=role,
          vpc_config_override=vpc_config_override,
          entry_point=estimator$entry_point)
      } else {
        TypeError$new(paste(
          "Estimator must be one of sagemaker.estimator.Estimator, sagemaker.estimator.Framework",
          "or AmazonAlgorithmEstimatorBase."))
      }
      model$name = name

      return (self$model_config(model, instance_type, role, image_uri))
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      print_class(self)
    }
  ),
  private = list(
    # Extract tuning job config from a HyperparameterTuner
    .extract_tuning_job_config = function(tuner){
      tuning_job_config = list(
        "Strategy"=tuner$strategy,
        "ResourceLimits"=list(
          "MaxNumberOfTrainingJobs"=tuner$max_jobs,
          "MaxParallelTrainingJobs"=tuner$max_parallel_jobs),
        "TrainingJobEarlyStoppingType"=tuner$early_stopping_type
      )

      if (!islistempty(tuner$objective_metric_name))
        tuning_job_config[["HyperParameterTuningJobObjective"]] = list(
          "Type"=tuner$objective_type,
          "MetricName"=tuner$objective_metric_name
        )

      parameter_ranges = tuner$hyperparameter_ranges()
      if (!islistempty(parameter_ranges))
        tuning_job_config[["ParameterRanges"]] = parameter_ranges

      return(tuning_job_config)
    },

    # Extract training job config from a HyperparameterTuner that uses the ``estimator`` field
    .extract_training_config_from_estimator = function(tuner,
                                                       inputs,
                                                       include_cls_metadata,
                                                       mini_batch_size){
      train_config = self$training_base_config(tuner$estimator, inputs, mini_batch_size)
      train_config[["HyperParameters"]] = NULL

      tuner$.__enclos_env__$.prepare_static_hyperparameters_for_tuning(
        include_cls_metadata=include_cls_metadata)
      train_config[["StaticHyperParameters"]] = tuner$static_hyperparameters

      if (!islistempty(tuner$metric_definitions))
        train_config[["AlgorithmSpecification"]][["MetricDefinitions"]] = tuner$metric_definitions

      s3_operations = train_config[["S3Operations"]]
      train_config[["S3Operations"]] = NULL

      return(list(train_config, s3_operations))
    },

    # Extracts a list of training job configs from a Hyperparameter Tuner.
    # It uses the ``estimator_dict`` field.
    .extract_training_config_list_from_estimator_dict = function(
      tuner, inputs, include_cls_metadata, mini_batch_size){
      estimator_names = sort(names(tuner$estimator_dict))
      tuner$.__enclos_env__$.validate_dict_argument(
        name="inputs", value=inputs, allowed_keys=estimator_names)
      tuner$.__enclos_env__$.validate_dict_argument(
        name="include_cls_metadata", value=include_cls_metadata, allowed_keys=estimator_names
      )
      tuner$.__enclos_env__$.validate_dict_argument(
        name="mini_batch_size", value=mini_batch_size, allowed_keys=estimator_names
      )

      train_config_dict = list()
      for (estimator_name in names(tuner$estimator_dict)){
        estimator = tuner$estimator_dict[[estimator_name]]
        train_config_dict[[estimator_name]] = self$training_base_config(
          estimator=estimator,
          inputs=if(!islistempty(inputs)) inputs[[estimator_name]] else NULL,
          mini_batch_size=if(!islistempty(mini_batch_size)) mini_batch_size[[estimator_name]] else NULL
        )
      }

      tuner$.__enclos_env__$.prepare_static_hyperparameters_for_tuning(
        include_cls_metadata=include_cls_metadata)

      train_config_list = list()
      s3_operations_list = list()

      for (estimator_name in sort(names(train_config_dict))){
        train_config = train_config_dict[[estimator_name]]
        train_config[["HyperParameters"]]=NULL
        train_config[["StaticHyperParameters"]] = tuner$static_hyperparameters_dict[[estimator_name]]

        train_config[["AlgorithmSpecification"]][[
          "MetricDefinitions"
        ]] = tuner$metric_definitions_dict[[estimator_name]]

        train_config[["DefinitionName"]] = estimator_name
        train_config[["TuningObjective"]] = list(
          "Type"=tuner$objective_type,
          "MetricName"=tuner$objective_metric_name_dict[[estimator_name]])
        train_config[["HyperParameterRanges"]] = tuner$hyperparameter_ranges_dict()[[estimator_name]]

        s3_operations_list = c(s3_operations_list, (train_config[["S3Operations"]] %||% list()))
        train_config[["S3Operations"]] = NULL

        train_config_list = c(train_config_list, train_config)
      }

      return(list(train_config_list, private$.merge_s3_operations(s3_operations_list)))
    },

    # Merge a list of S3 operation dictionaries into one
    .merge_s3_operations(s3_operations_list){
      s3_operations_merged =list()
      for (s3_operations in s3_operations_list){
        for (key in names(s3_operations)){
          operations = s3_operations[[key]]
          if (!(key %in% names(s3_operations_merged)))
            s3_operations_merged[[key]] = list()
          for (operation in operations){
            if (!(operation %in% names(s3_operations_merged[[key]])))
             s3_operations_merged[[key]] = c(s3_operations_merged[[key]], operation)
          }
        }
      }
      return(s3_operations_merged)
    }

  ),
  lock_object = F
)
