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
        tune_config["TrainingJobDefinition"] = ll$TrainingJobDefinitions
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

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      print_class(self)
    }
  ),
  private = list(

  ),
  lock_object = F
)
