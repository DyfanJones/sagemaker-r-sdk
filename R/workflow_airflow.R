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

#' @title AirFlowWorkFlow helper class
#' @description Helper class to take sagemaker classes and format output for Airflow.
#' @export
AirFlowWorkFlow = R6Class("AirFlowWorkFlow",
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
    #' @param inputs : The training data.
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
    #' @param inputs : Information about the training data. Please refer to the ``fit()``
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
    #' @param inputs : Information about the training data. Please refer to the ``fit()``
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
    #' @param inputs : Information about the training data. Please refer to the ``fit()``
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
    #' @param include_cls_metadata : It can take one of the following two forms.
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
    #' @param mini_batch_size : It can take one of the following two forms.
    #'              * (int) - Specify this argument only when estimator is a built-in estimator of an
    #'              Amazon algorithm. For other estimators, batch size should be specified in the
    #'              estimator.
    #'              * (dict[str, int]) - This version should be used for tuners created via the factory
    #'              method ``HyperparameterTuner.create()``, to specify the value for individual
    #'              estimators provided in the ``estimator_dict`` argument of the method. The keys
    #'              would be the same estimator names as in ``estimator_dict``. If one estimator
    #'              doesn't need the value set, then no need to include it in the dictionary. If
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

    #' @description Export Airflow transform config from a SageMaker transformer
    #' @param transformer (sagemaker.transformer.Transformer): The SageMaker
    #'              transformer to export Airflow config from.
    #' @param data (str): Input data location in S3.
    #' @param data_type (str): What the S3 location defines (default: 'S3Prefix').
    #'              Valid values:
    #'              * 'S3Prefix' - the S3 URI defines a key name prefix. All objects with this prefix will
    #'              be used as inputs for the transform job.
    #'              * 'ManifestFile' - the S3 URI points to a single manifest file listing each S3 object
    #'              to use as an input for the transform job.
    #' @param content_type (str): MIME type of the input data (default: None).
    #' @param compression_type (str): Compression type of the input data, if
    #'              compressed (default: None). Valid values: 'Gzip', None.
    #' @param split_type (str): The record delimiter for the input object (default:
    #'              'None'). Valid values: 'None', 'Line', 'RecordIO', and 'TFRecord'.
    #' @param job_name (str): job name (default: None). If not specified, one will be
    #'              generated.
    #' @param input_filter (str): A JSONPath to select a portion of the input to
    #'              pass to the algorithm container for inference. If you omit the
    #'              field, it gets the value '$', representing the entire input.
    #'              For CSV data, each row is taken as a JSON array,
    #'              so only index-based JSONPaths can be applied, e.g. $[0], $[1:].
    #'              CSV data should follow the `RFC format <https://tools.ietf.org/html/rfc4180>`_.
    #'              See `Supported JSONPath Operators
    #'              <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html#data-processing-operators>`_
    #'              for a table of supported JSONPath operators.
    #'              For more information, see the SageMaker API documentation for
    #'              `CreateTransformJob
    #'              <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html>`_.
    #'              Some examples: "$[1:]", "$.features" (default: None).
    #' @param output_filter (str): A JSONPath to select a portion of the
    #'              joined/original output to return as the output.
    #'              For more information, see the SageMaker API documentation for
    #'              `CreateTransformJob
    #'              <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html>`_.
    #'              Some examples: "$[1:]", "$.prediction" (default: None).
    #' @param join_source (str): The source of data to be joined to the transform
    #'              output. It can be set to 'Input' meaning the entire input record
    #'              will be joined to the inference result. You can use OutputFilter
    #'              to select the useful portion before uploading to S3. (default:
    #'              None). Valid values: Input, None.
    #' @return dict: Transform config that can be directly used by
    #'              SageMakerTransformOperator in Airflow.
    transform_config = function(
      transformer,
      data,
      data_type="S3Prefix",
      content_type=NULL,
      compression_type=NULL,
      split_type=NULL,
      job_name=NULL,
      input_filter=NULL,
      output_filter=NULL,
      join_source=NULL){
      if (!is.null(job_name)) {
        transformer$.current_job_name = job_name
      } else {
        base_name = transformer$base_transform_job_name
        transformer$.current_job_name = (if (!is.null(base_name))
          name_from_base(base_name) else transformer$model_name)
      }
      if (is.null(transformer$output_path)){
        transformer$output_path = sprintf("s3://%s/%s",
          transformer$sagemaker_session$default_bucket(), transformer$.current_job_name
        )
      }
      job_config = transformer$.__enclose_env__$.load_config(
        data, data_type, content_type, compression_type, split_type, transformer
      )

      config = list(
        "TransformJobName"=transformer$.current_job_name,
        "ModelName"=transformer$model_name,
        "TransformInput"=job_config[["input_config"]],
        "TransformOutput"=job_config[["output_config"]],
        "TransformResources"=job_config[["resource_config"]])

      data_processing = sagemaker.transformer._TransformJob._prepare_data_processing(
        input_filter, output_filter, join_source
      )
      if (!is.null(data_processing))
        config[["DataProcessing"]] = data_processing

      if (!is.null(transformer$strategy))
        config[["BatchStrategy"]] = transformer$strategy

      if (!is.null(transformer$max_concurrent_transforms))
        config[["MaxConcurrentTransforms"]] = transformer$max_concurrent_transforms

      if (!is.null(transformer$max_payload))
        config[["MaxPayloadInMB"]] = transformer$max_payload

      if (!is.null(transformer$env))
        config[["Environment"]] = transformer$env

      if (!is.null(transformer$tags))
        config[["Tags"]] = transformer$tags

      return(config)
    },

    #' @description Export Airflow transform config from a SageMaker estimator
    #' @param estimator (sagemaker.model.EstimatorBase): The SageMaker estimator to
    #'              export Airflow config from. It has to be an estimator associated
    #'              with a training job.
    #' @param task_id (str): The task id of any
    #'              airflow.contrib.operators.SageMakerTrainingOperator or
    #'              airflow.contrib.operators.SageMakerTuningOperator that generates
    #'              training jobs in the DAG. The transform config is built based on the
    #'              training job generated in this operator.
    #' @param task_type (str): Whether the task is from SageMakerTrainingOperator or
    #'              SageMakerTuningOperator. Values can be 'training', 'tuning' or None
    #'              (which means training job is not from any task).
    #' @param instance_count (int): Number of EC2 instances to use.
    #' @param instance_type (str): Type of EC2 instance to use, for example,
    #'              'ml.c4.xlarge'.
    #' @param data (str): Input data location in S3.
    #' @param data_type (str): What the S3 location defines (default: 'S3Prefix').
    #'              Valid values:
    #'              * 'S3Prefix' - the S3 URI defines a key name prefix. All objects with this prefix will
    #'              be used as inputs for the transform job.
    #'              * 'ManifestFile' - the S3 URI points to a single manifest file listing each S3 object
    #'              to use as an input for the transform job.
    #' @param content_type (str): MIME type of the input data (default: None).
    #' @param compression_type (str): Compression type of the input data, if
    #'              compressed (default: None). Valid values: 'Gzip', None.
    #' @param split_type (str): The record delimiter for the input object (default:
    #'              'None'). Valid values: 'None', 'Line', 'RecordIO', and 'TFRecord'.
    #' @param job_name (str): transform job name (default: None). If not specified,
    #'              one will be generated.
    #' @param model_name (str): model name (default: None). If not specified, one will
    #'              be generated.
    #' @param strategy (str): The strategy used to decide how to batch records in a
    #'              single request (default: None). Valid values: 'MultiRecord' and
    #'              'SingleRecord'.
    #' @param assemble_with (str): How the output is assembled (default: None). Valid
    #'              values: 'Line' or 'None'.
    #' @param output_path (str): S3 location for saving the transform result. If not
    #'              specified, results are stored to a default bucket.
    #' @param output_kms_key (str): Optional. KMS key ID for encrypting the transform
    #'              output (default: None).
    #' @param accept (str): The accept header passed by the client to
    #'              the inference endpoint. If it is supported by the endpoint,
    #'              it will be the format of the batch transform output.
    #' @param env (dict): Environment variables to be set for use during the transform
    #'              job (default: None).
    #' @param max_concurrent_transforms (int): The maximum number of HTTP requests to
    #'              be made to each individual transform container at one time.
    #' @param max_payload (int): Maximum size of the payload in a single HTTP request
    #'              to the container in MB.
    #' @param tags (list[dict]): List of tags for labeling a transform job. If none
    #'              specified, then the tags used for the training job are used for the
    #'              transform job.
    #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
    #'              which is also used during transform jobs. If not specified, the role
    #'              from the Estimator will be used.
    #' @param volume_kms_key (str): Optional. KMS key ID for encrypting the volume
    #'              attached to the ML compute instance (default: None).
    #' @param model_server_workers (int): Optional. The number of worker processes
    #'              used by the inference server. If None, server will use one worker
    #'              per vCPU.
    #' @param image_uri (str): A Docker image URI to use for deploying the model
    #' @param vpc_config_override (dict[str, list[str]]): Override for VpcConfig set on
    #'              the model. Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param input_filter (str): A JSONPath to select a portion of the input to
    #'              pass to the algorithm container for inference. If you omit the
    #'              field, it gets the value '$', representing the entire input.
    #'              For CSV data, each row is taken as a JSON array,
    #'              so only index-based JSONPaths can be applied, e.g. $[0], $[1:].
    #'              CSV data should follow the `RFC format <https://tools.ietf.org/html/rfc4180>`_.
    #'              See `Supported JSONPath Operators
    #'              <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html#data-processing-operators>`_
    #'              for a table of supported JSONPath operators.
    #'              For more information, see the SageMaker API documentation for
    #'              `CreateTransformJob
    #'              <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html>`_.
    #'              Some examples: "$[1:]", "$.features" (default: None).
    #' @param output_filter (str): A JSONPath to select a portion of the
    #'              joined/original output to return as the output.
    #'              For more information, see the SageMaker API documentation for
    #'              `CreateTransformJob
    #'              <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html>`_.
    #'              Some examples: "$[1:]", "$.prediction" (default: None).
    #' @param join_source (str): The source of data to be joined to the transform
    #'              output. It can be set to 'Input' meaning the entire input record
    #'              will be joined to the inference result. You can use OutputFilter
    #'              to select the useful portion before uploading to S3. (default:
    #'              None). Valid values: Input, None.
    #' @return dict: Transform config that can be directly used by
    #'              SageMakerTransformOperator in Airflow.
    transform_config_from_estimator = function(
      estimator,
      task_id,
      task_type,
      instance_count,
      instance_type,
      data,
      data_type="S3Prefix",
      content_type=NULL,
      compression_type=NULL,
      split_type=NULL,
      job_name=NULL,
      model_name=NULL,
      strategy=NULL,
      assemble_with=NULL,
      output_path=NULL,
      output_kms_key=NULL,
      accept=NULL,
      env=NULL,
      max_concurrent_transforms=NULL,
      max_payload=NULL,
      tags=NULL,
      role=NULL,
      volume_kms_key=NULL,
      model_server_workers=NULL,
      image_uri=NULL,
      vpc_config_override=NULL,
      input_filter=NULL,
      output_filter=NULL,
      join_source=NULL
      ){
      model_base_config = self$model_config_from_estimator(
        estimator=estimator,
        task_id=task_id,
        task_type=task_type,
        instance_type=instance_type,
        role=role,
        image_uri=image_uri,
        name=model_name,
        model_server_workers=model_server_workers,
        vpc_config_override=vpc_config_override)

      if (inherits(estimator, "Framework")){
        transformer = estimator$transformer(
          instance_count,
          instance_type,
          strategy,
          assemble_with,
          output_path,
          output_kms_key,
          accept,
          env,
          max_concurrent_transforms,
          max_payload,
          tags,
          role,
          model_server_workers,
          volume_kms_key)
      } else {
        transformer = estimator$transformer(
          instance_count,
          instance_type,
          strategy,
          assemble_with,
          output_path,
          output_kms_key,
          accept,
          env,
          max_concurrent_transforms,
          max_payload,
          tags,
          role,
          volume_kms_key)
      }
      transformer.model_name = model_base_config[["ModelName"]]

      transform_base_config = self$transform_config(
        transformer,
        data,
        data_type,
        content_type,
        compression_type,
        split_type,
        job_name,
        input_filter,
        output_filter,
        join_source)

      config = list("Model"=model_base_config, "Transform"=transform_base_config)

      return(config)
    },

    #' @description Export Airflow deploy config from a SageMaker model
    #' @param model (sagemaker.model.Model): The SageMaker model to export the Airflow
    #'              config from.
    #' @param initial_instance_count (int): The initial number of instances to run in
    #'              the ``Endpoint`` created from this ``Model``.
    #' @param instance_type (str): The EC2 instance type to deploy this Model to. For
    #'              example, 'ml.p2.xlarge'.
    #' @param endpoint_name (str): The name of the endpoint to create (default: None).
    #'              If not specified, a unique endpoint name will be created.
    #' @param tags (list[dict]): List of tags for labeling a training job. For more,
    #'              see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @return dict: Deploy config that can be directly used by
    #'              SageMakerEndpointOperator in Airflow.
    deploy_config = function(model,
                             initial_instance_count,
                             instance_type,
                             endpoint_name=NULL,
                             tags=NULL){
      model_base_config = self$model_config(model, instance_type)

      production_variant = production_variant(
        model$name, instance_type, initial_instance_count
      )
      name = model$name
      config_options = list("EndpointConfigName"=name, "ProductionVariants"=list(production_variant))
      config_options[["Tags"]] = tags

      endpoint_name = endpoint_name %||% name
      endpoint_base_config = list("EndpointName"=endpoint_name, "EndpointConfigName"=name)

      config = list(
        "Model"=model_base_config,
        "EndpointConfig"=config_options,
        "Endpoint"=endpoint_base_config)

      # if there is s3 operations needed for model, move it to root level of config
      s3_operations = model_base_config[["S3Operations"]]
      model_base_config[["S3Operations"]] = NULL
      if (!is.null(s3_operations))
        config[["S3Operations"]] = s3_operations

      return(config)
    },

    #' @description Export Airflow deploy config from a SageMaker estimator
    #' @param estimator (sagemaker.model.EstimatorBase): The SageMaker estimator to
    #'              export Airflow config from. It has to be an estimator associated
    #'              with a training job.
    #' @param task_id (str): The task id of any
    #'              airflow.contrib.operators.SageMakerTrainingOperator or
    #'              airflow.contrib.operators.SageMakerTuningOperator that generates
    #'              training jobs in the DAG. The endpoint config is built based on the
    #'              training job generated in this operator.
    #' @param task_type (str): Whether the task is from SageMakerTrainingOperator or
    #'              SageMakerTuningOperator. Values can be 'training', 'tuning' or None
    #'              (which means training job is not from any task).
    #' @param initial_instance_count (int): Minimum number of EC2 instances to deploy
    #'              to an endpoint for prediction.
    #' @param instance_type (str): Type of EC2 instance to deploy to an endpoint for
    #'              prediction, for example, 'ml.c4.xlarge'.
    #' @param model_name (str): Name to use for creating an Amazon SageMaker model. If
    #'              not specified, one will be generated.
    #' @param endpoint_name (str): Name to use for creating an Amazon SageMaker
    #'              endpoint. If not specified, the name of the SageMaker model is used.
    #' @param tags (list[dict]): List of tags for labeling a training job. For more,
    #'              see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @param ... : Passed to invocation of ``create_model()``. Implementations
    #'              may customize ``create_model()`` to accept ``**kwargs`` to customize
    #'              model creation during deploy. For more, see the implementation docs.
    #' @return dict: Deploy config that can be directly used by
    #'              SageMakerEndpointOperator in Airflow.
    deploy_config_from_estimator = function(
      estimator,
      task_id,
      task_type,
      initial_instance_count,
      instance_type,
      model_name=NULL,
      endpoint_name=NULL,
      tags=NULL,
      ...){
      self$update_estimator_from_task(estimator, task_id, task_type)
      model = estimator$create_model(...)
      model.name = model_name
      config = self$deploy_config(model, initial_instance_count, instance_type, endpoint_name, tags)
      return(config)
    },

    #' @description Export Airflow processing config from a SageMaker processor
    #' @param processor (sagemaker.processor.Processor): The SageMaker
    #'              processor to export Airflow config from.
    #' @param inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
    #'              the processing job. These must be provided as
    #'              :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
    #' @param outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
    #'              the processing job. These can be specified as either path strings or
    #'              :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
    #' @param job_name (str): Processing job name. If not specified, the processor generates
    #'              a default job name, based on the base job name and current timestamp.
    #' @param experiment_config (dict[str, str]): Experiment management configuration.
    #'              Dictionary contains three optional keys:
    #'              'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
    #' @param container_arguments ([str]): The arguments for a container used to run a processing job.
    #' @param container_entrypoint ([str]): The entrypoint for a container used to run a processing job.
    #' @param kms_key_id (str): The AWS Key Management Service (AWS KMS) key that Amazon SageMaker
    #'              uses to encrypt the processing job output. KmsKeyId can be an ID of a KMS key,
    #'              ARN of a KMS key, alias of a KMS key, or alias of a KMS key.
    #'              The KmsKeyId is applied to all outputs.
    #' @return dict: Processing config that can be directly used by
    #'            SageMakerProcessingOperator in Airflow.
    processing_config = function(
      processor,
      inputs=NULL,
      outputs=NULL,
      job_name=NULL,
      experiment_config=NULL,
      container_arguments=NULL,
      container_entrypoint=NULL,
      kms_key_id=NULL){
      if (!is.null(job_name)){
        processor$.current_job_name = job_name
      } else {
        base_name = processor$base_job_name
      }
      processor$.current_job_name = (if (!is.null(base_name)) {
        name_from_base(base_name)
      } else base_name_from_image(processor$image_uri))

      config = list(
        "ProcessingJobName"=processor$.current_job_name,
        "ProcessingInputs"=self$input_output_list_converter(inputs))

      processing_output_config = ProcessingJob$public_methods$prepare_output_config(
        kms_key_id, self$input_output_list_converter(outputs))

      config[["ProcessingOutputConfig"]] = processing_output_config

      if (!is.null(experiment_config))
        config[["ExperimentConfig"]] = experiment_config

      app_specification = ProcessingJob$public_methods$prepare_app_specification(
        container_arguments, container_entrypoint, processor$image_uri)
      config[["AppSpecification"]] = app_specification

      config[["RoleArn"]] = processor$role

      config[["Environment"]]= processor.env

      if (!is.null(processor$network_config))
        config[["NetworkConfig"]] = processor$network_config$to_request_list()

      processing_resources = ProcessingJob$public_methods$prepare_processing_resources(
        instance_count=processor$instance_count,
        instance_type=processor$instance_type,
        volume_kms_key_id=processor$volume_kms_key,
        volume_size_in_gb=processor$volume_size_in_gb)
      config[["ProcessingResources"]] = processing_resources

      if (!is.null(processor$max_runtime_in_seconds))
        stopping_condition = ProcessingJob$public_methods$prepare_stopping_condition(
          processor$max_runtime_in_seconds)
      config[["StoppingCondition"]] = stopping_condition

      config[["Tags"]] = processor$tags

      return(config)
    },

    #' @description Converts a list of ProcessingInput or ProcessingOutput objects to a list of dicts
    #' @param object_list (list[ProcessingInput or ProcessingOutput]
    #' @return List of dicts
    input_output_list_converter = function(object_list){
      if (!islistempty(object_list))
        return(lapply(object_list, function(obj) obj$to_request_list()))
      return(object_list)
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
    .merge_s3_operations = function(s3_operations_list){
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
