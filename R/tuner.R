# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tuner.py

#' @include job.R
#' @include estimator.R
#' @include session.R
#' @include utils.R
#' @include parameter.R

#' @import jsonlite
#' @import logger
#' @import R6

AMAZON_ESTIMATOR_MODULE <- "R6sagemaker"
AMAZON_ESTIMATOR_CLS_NAMES <- list(
  "factorization-machines"= "FactorizationMachines",
  "kmeans"= "KMeans",
  "lda"= "LDA",
  "linear-learner"= "LinearLearner",
  "ntm"= "NTM",
  "randomcutforest"= "RandomCutForest",
  "knn"= "KNN",
  "object2vec"= "Object2Vec")

HYPERPARAMETER_TUNING_JOB_NAME <- "HyperParameterTuningJobName"
PARENT_HYPERPARAMETER_TUNING_JOBS <- "ParentHyperParameterTuningJobs"
WARM_START_TYPE <- "WarmStartType"

#' @title WarmStartTypes Class
#' @description Warm Start Configuration type. There can be two types of warm start jobs:
#'              * IdenticalDataAndAlgorithm: Type of warm start that allows users to reuse
#'              training results from existing tuning jobs that have the same algorithm code
#'              and datasets.
#'              * TransferLearning: Type of warm start that allows users to
#'              reuse training results from existing tuning jobs that have similar algorithm
#'              code and datasets.
WarmStartTypes = R6Class("WarmStartTypes",
  public = list(
   #' @field IDENTICAL_DATA_AND_ALGORITHM
   #' Type of warm start that allows users to reuse training results from existing tuning jobs
   #' that have the same algorithm code
   IDENTICAL_DATA_AND_ALGORITHM = "IdenticalDataAndAlgorithm",

   #' @field TRANSFER_LEARNING
   #' Type of warm start that allows users to
   #' reuse training results from existing tuning jobs that have similar algorithm code
   TRANSFER_LEARNING = "TransferLearning"
  )
)

#' @title WarmStartConfig Class
#' @description Warm Start Configuration which defines the nature of the warm start
#'              ``HyperparameterTuner``, with type and parents for warm start.
#'              Examples:
#'              >>> warm_start_config = WarmStartConfig(
#'              >>>           type=WarmStartTypes.TransferLearning, parents={"p1","p2"})
#'              >>> warm_start_config.type
#'              "TransferLearning"
#'              >>> warm_start_config.parents
#'              {"p1","p2"}
#'@export
WarmStartConfig = R6Class("WarmStartConfig",
  public = list(

    #' @field type
    #' Supported warm start types
    type = NULL,

    #' @field parents
    #' Set of parent tuning jobs
    parents = NULL,

    #' @description Initializes the ``WarmStartConfig`` with the provided
    #'              ``WarmStartTypes`` and parents.
    #' @param warm_start_type (str): This should be one
    #'              of the supported warm start types ```("IdenticalDataAndAlgorithm", "TransferLearning")```
    #' @param parents (str/list): Set of parent tuning jobs which will be used to
    #'              warm start the new tuning job.
    initialize = function(warm_start_type = c("IdenticalDataAndAlgorithm", "TransferLearning"),
                          parents = NULL){
      stopifnot(is.character(parents) || is.list(parents) || is.null(parents))

      tryCatch({self$type = match.arg(warm_start_type)},
               error = function(e){
                 stop("Invalid type: `warm_start_config`, valid warm start types are: ",
                      "'IdenticalDataAndAlgorithm', 'TransferLearning'", call. = F)})
      self$parents =  if(inherits(parents, "list")) unique(parents) else as.list(unique(parents))
    },

    #' @description Creates an instance of ``WarmStartConfig`` class, from warm start
    #'              configuration response from DescribeTrainingJob.
    #'              Examples:
    #'              >>> warm_start_config = WarmStartConfig$new()$from_job_desc(warm_start_config=list(
    #'              >>>    "WarmStartType"="TransferLearning",
    #'              >>>    "ParentHyperParameterTuningJobs"= list(
    #'              >>>        list('HyperParameterTuningJobName'= "p1"),
    #'              >>>        list('HyperParameterTuningJobName'= "p2")
    #'              >>>    )
    #'              >>>))
    #'              >>> warm_start_config.type
    #'              "TransferLearning"
    #'              >>> warm_start_config.parents
    #'              ["p1","p2"]
    #' @param warm_start_config (dict): The expected format of the
    #'              ``warm_start_config`` contains two first-class
    #' @return sagemaker.tuner.WarmStartConfig: De-serialized instance of
    #'              WarmStartConfig containing the type and parents provided as part of
    #'              ``warm_start_config``.
    from_job_desc = function(warm_start_config){
      if (missing(warm_start_config)
        || !(WARM_START_TYPE %in% names(warm_start_config))
        || !(PARENT_HYPERPARAMETER_TUNING_JOBS %in% names(warm_start_config)))
        return(NULL)

      parents = lapply(warm_start_config[[PARENT_HYPERPARAMETER_TUNING_JOBS]],
                       function(parent) parent[[HYPERPARAMETER_TUNING_JOB_NAME]])

      if (is.null(parents))
        stop(sprintf("Invalid parents: %s, parents should not be NULL",parents))

      cls = self$clone()
      tryCatch({cls$type = match.arg(warm_start_config[[WARM_START_TYPE]], c("IdenticalDataAndAlgorithm", "TransferLearning"))},
               error = function(e){
                 stop("Invalid type: `WarmStartType`, valid warm start types are: ",
                      "'IdenticalDataAndAlgorithm', 'TransferLearning'", call. = F)})
      cls$parents =  if(inherits(parents, "list")) unique(parents) else as.list(unique(parents))

      return(cls)
    },

    #' @description Converts the ``self`` instance to the desired input request format.
    #'              Examples:
    #'              >>> warm_start_config = WarmStartConfig$new("TransferLearning",parents="p1,p2")
    #'              >>> warm_start_config$to_input_req()
    #'              list(
    #'                 "WarmStartType"="TransferLearning",
    #'                 "ParentHyperParameterTuningJobs"= list(
    #'                      list('HyperParameterTuningJobName': "p1"),
    #'                      list('HyperParameterTuningJobName': "p2")
    #'                   )
    #'              )
    #' @return list: Containing the "WarmStartType" and
    #'              "ParentHyperParameterTuningJobs" as the first class fields.
    to_input_req = function(){

      output = list(unname(self$type),
                    lapply(self$parents, function(parent) list(HyperParameterTuningJobName = parent)))

      names(output) = c(WARM_START_TYPE, PARENT_HYPERPARAMETER_TUNING_JOBS)
      return(output)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      cat("<WarmStartConfig>")
      invisible(self)
    }
  )
)

#' @title HyperparamerTuner
#' @description A class for creating and interacting with Amazon SageMaker hyperparameter
#'              tuning jobs, as well as deploying the resulting model(s).
#' @export
HyperparameterTuner = R6Class("HyperparameterTuner",
  public = list(
    #' @field TUNING_JOB_NAME_MAX_LENGTH
    #' Maximumn length of sagemaker job name
    TUNING_JOB_NAME_MAX_LENGTH = 32,

    #' @field SAGEMAKER_ESTIMATOR_MODULE
    #' Class metadata
    SAGEMAKER_ESTIMATOR_MODULE = "sagemaker_estimator_module",

    #' @field SAGEMAKER_ESTIMATOR_CLASS_NAME
    #' Class metadata
    SAGEMAKER_ESTIMATOR_CLASS_NAME = "sagemaker_estimator_class_name",

    #' @field DEFAULT_ESTIMATOR_MODULE
    #' Class metadata
    DEFAULT_ESTIMATOR_MODULE = "R6sagemaker",

    #' @field DEFAULT_ESTIMATOR_CLS_NAME
    #' Class metadata
    DEFAULT_ESTIMATOR_CLS_NAME = "Estimator",

    #' @description Initialize a ``HyperparameterTuner``. It takes an estimator to obtain
    #'              configuration information for training jobs that are created as the
    #'              result of a hyperparameter tuning job.
    #' @param estimator (sagemaker.estimator.EstimatorBase): An estimator object
    #'              that has been initialized with the desired configuration. There
    #'              does not need to be a training job associated with this
    #'              instance.
    #' @param objective_metric_name (str): Name of the metric for evaluating
    #'              training jobs.
    #' @param hyperparameter_ranges (dict[str, sagemaker.parameter.ParameterRange]): Dictionary of
    #'              parameter ranges. These parameter ranges can be one
    #'              of three types: Continuous, Integer, or Categorical. The keys of
    #'              the dictionary are the names of the hyperparameter, and the
    #'              values are the appropriate parameter range class to represent
    #'              the range.
    #' @param metric_definitions (list[dict]): A list of dictionaries that defines
    #'              the metric(s) used to evaluate the training jobs (default:
    #'              None). Each dictionary contains two keys: 'Name' for the name of
    #'              the metric, and 'Regex' for the regular expression used to
    #'              extract the metric from the logs. This should be defined only
    #'              for hyperparameter tuning jobs that don't use an Amazon
    #'              algorithm.
    #' @param strategy (str): Strategy to be used for hyperparameter estimations
    #'              (default: 'Bayesian').
    #' @param objective_type (str): The type of the objective metric for
    #'              evaluating training jobs. This value can be either 'Minimize' or
    #'              'Maximize' (default: 'Maximize').
    #' @param max_jobs (int): Maximum total number of training jobs to start for
    #'              the hyperparameter tuning job (default: 1).
    #' @param max_parallel_jobs (int): Maximum number of parallel training jobs to
    #'              start (default: 1).
    #' @param tags (list[dict]): List of tags for labeling the tuning job
    #'              (default: None). For more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @param base_tuning_job_name (str): Prefix for the hyperparameter tuning job
    #'              name when the :meth:`~sagemaker.tuner.HyperparameterTuner.fit`
    #'              method launches. If not specified, a default job name is
    #'              generated, based on the training image name and current
    #'              timestamp.
    #' @param warm_start_config (sagemaker.tuner.WarmStartConfig): A
    #'              ``WarmStartConfig`` object that has been initialized with the
    #'              configuration defining the nature of warm start tuning job.
    #' @param early_stopping_type (str): Specifies whether early stopping is
    #'              enabled for the job. Can be either 'Auto' or 'Off' (default:
    #'              'Off'). If set to 'Off', early stopping will not be attempted.
    #'              If set to 'Auto', early stopping of some training jobs may
    #'              happen, but is not guaranteed to.
    #' @param estimator_name (str): A unique name to identify an estimator within the
    #'              hyperparameter tuning job, when more than one estimator is used with
    #'              the same tuning job (default: None).
    initialize = function(estimator,
                          objective_metric_name,
                          hyperparameter_ranges,
                          metric_definitions=NULL,
                          strategy="Bayesian",
                          objective_type="Maximize",
                          max_jobs=1,
                          max_parallel_jobs=1,
                          tags=NULL,
                          base_tuning_job_name=NULL,
                          warm_start_config=NULL,
                          early_stopping_type=c("Off", "Auto"),
                          estimator_name=NULL){
      if (missing(hyperparameter_ranges) || length(hyperparameter_ranges) == 0)
        stop("Need to specify hyperparameter ranges", call. = F)

      if (!is.null(estimator_name)){
        self$estimator = NULL
        self$objective_metric_name = NULL
        self$.hyperparameter_ranges = NULL
        self$metric_definitions = NULL
        self$estimator_list = list(estimator)
        names(self$estimator_list) = estimator_name
        self$objective_metric_name_list = list(objective_metric_name)
        names(self$objective_metric_name_list) = estimator_name
        self$.hyperparameter_ranges_list = list(hyperparameter_ranges)
        names(self$.hyperparameter_ranges_list) = estimator_name
        if (!is.null(metric_definitions)) {
          self$metric_definitions_list = list(metric_definitions)
          names(self$metric_definitions_list) = estimator_name
        } else self$metric_definitions_list = list()
        self.static_hyperparameters = NULL
      } else {
        self$estimator = estimator
        self$objective_metric_name = objective_metric_name
        self$.hyperparameter_ranges = hyperparameter_ranges
        self$metric_definitions = metric_definitions
        self$estimator_list = NULL
        self$objective_metric_name_list = NULL
        self$.hyperparameter_ranges_list = NULL
        self$metric_definitions_list = NULL
        self$static_hyperparameters_list = NULL}

      private$.validate_parameter_ranges(estimator, hyperparameter_ranges)

      self$strategy = strategy
      self$objective_type = objective_type
      self$max_jobs = max_jobs
      self$max_parallel_jobs = max_parallel_jobs

      self$tags = tags
      self$base_tuning_job_name = base_tuning_job_name
      self$.current_job_name = NULL
      self$latest_tuning_job = NULL
      self$warm_start_config = warm_start_config
      self$early_stopping_type = match.arg(early_stopping_type)
    },

    #' @description Start a hyperparameter tuning job.
    #' @param inputs : Information about the training data. Please refer to the
    #'              ``fit()`` method of the associated estimator, as this can take
    #'              any of the following forms:
    #'              * (str) - The S3 location where training data is saved.
    #'              * (dict[str, str] or dict[str, sagemaker.session.s3_input]) -
    #'              If using multiple channels for training data, you can specify
    #'              a dict mapping channel names to strings or
    #'              :func:`~sagemaker.session.s3_input` objects.
    #'              * (sagemaker.session.s3_input) - Channel configuration for S3 data sources that can
    #'              provide additional information about the training dataset.
    #'              See :func:`sagemaker.session.s3_input` for full details.
    #'              * (sagemaker.session.FileSystemInput) - channel configuration for
    #'              a file system data source that can provide additional information as well as
    #'              the path to the training dataset.
    #'              * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
    #'              Amazon :class:~`Record` objects serialized and stored in S3.
    #'              For use with an estimator for an Amazon algorithm.
    #'              * (sagemaker.amazon.amazon_estimator.FileSystemRecordSet) -
    #'              Amazon SageMaker channel configuration for a file system data source for
    #'              Amazon algorithms.
    #'              * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
    #'              :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
    #'              where each instance is a different channel of training data.
    #'              * (list[sagemaker.amazon.amazon_estimator.FileSystemRecordSet]) - A list of
    #'              :class:~`sagemaker.amazon.amazon_estimator.FileSystemRecordSet` objects,
    #'              where each instance is a different channel of training data.
    #' @param job_name (str): Tuning job name. If not specified, the tuner
    #'              generates a default job name, based on the training image name
    #'              and current timestamp.
    #' @param include_cls_metadata : It can take one of the following two forms.
    #'              * (bool) - Whether or not the hyperparameter tuning job should include information
    #'              about the estimator class (default: False). This information is passed as a
    #'              hyperparameter, so if the algorithm you are using cannot handle unknown
    #'              hyperparameters (e.g. an Amazon SageMaker built-in algorithm that does not
    #'              have a custom estimator in the Python SDK), then set ``include_cls_metadata``
    #'              to ``False``.
    #'              * (dict[str, bool]) - This version should be used for tuners created via the
    #'              factory method create(), to specify the flag for each estimator provided in
    #'              the estimator_dict argument of the method. The keys would be the same
    #'              estimator names as in estimator_dict. If one estimator doesn't need the flag
    #'              set, then no need to include it in the dictionary.
    #' @param estimator_kwargs (dict[str, dict]): Dictionary for other arguments needed for
    #'              training. Should be used only for tuners created via the factory method create().
    #'              The keys are the estimator names for the estimator_dict argument of create()
    #'              method. Each value is a dictionary for the other arguments needed for training
    #'              of the corresponding estimator.
    #' @param  ... : Other arguments needed for training. Please refer to the
    #'              ``fit()`` method of the associated estimator to see what other
    #'              arguments are needed.
    fit = function(inputs=NULL,
                   job_name=NULL,
                   include_cls_metadata=FALSE,
                   estimator_kwargs=NULL,
                   ...){
      if (!is.null(self$estimator))
        private$.fit_with_estimator(inputs, job_name, include_cls_metadata, ...)
      else
        private$.fit_with_estimator_list(inputs, job_name, include_cls_metadata, estimator_kwargs)
    },

    #' @description Attach to an existing hyperparameter tuning job.
    #'              Create a HyperparameterTuner bound to an existing hyperparameter
    #'              tuning job. After attaching, if there exists a best training job (or any
    #'              other completed training job), that can be deployed to create an Amazon
    #'              SageMaker Endpoint and return a ``Predictor``.
    #'              The ``HyperparameterTuner`` instance could be created in one of the following two forms.
    #'              * If the 'TrainingJobDefinition' field is present in tuning job description, the tuner
    #'              will be created using the default constructor with a single estimator.
    #'              * If the 'TrainingJobDefinitions' field (list) is present in tuning job description,
    #'              the tuner will be created using the factory method ``create()`` with one or
    #'              several estimators. Each estimator corresponds to one item in the
    #'              'TrainingJobDefinitions' field, while the estimator names would come from the
    #'              'DefinitionName' field of items in the 'TrainingJobDefinitions' field. For more
    #'              details on how tuners are created from multiple estimators, see ``create()``
    #'              documentation.
    #'              For more details on 'TrainingJobDefinition' and 'TrainingJobDefinitions' fields in tuning
    #'              job description, see
    #'              https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_hyper_parameter_tuning_job
    #' @param tuning_job_name (str): The name of the hyperparameter tuning job to attach to.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which manages
    #'              interactions with Amazon SageMaker APIs and any other AWS services needed.
    #'              If not specified, one is created using the default AWS configuration chain.
    #' @param job_details (dict): The response to a ``DescribeHyperParameterTuningJob`` call.
    #'              If not specified, the ``HyperparameterTuner`` will perform one such call with
    #'              the provided hyperparameter tuning job name.
    #' @param estimator_cls : It can take one of the following two forms.
    #'              (str): The estimator class name associated with the training jobs, e.g.
    #'              'sagemaker.estimator.Estimator'. If not specified, the ``HyperparameterTuner``
    #'              will try to derive the correct estimator class from training job metadata,
    #'              defaulting to :class:~`Estimator` if it is unable to
    #'              determine a more specific class.
    #'              (dict[str, str]): This form should be used only when the 'TrainingJobDefinitions'
    #'              field (list) is present in tuning job description. In this scenario training
    #'              jobs could be created from different training job definitions in the
    #'              'TrainingJobDefinitions' field, each of which would be mapped to a different
    #'              estimator after the ``attach()`` call. The ``estimator_cls`` should then be a
    #'              dictionary to specify estimator class names for individual estimators as
    #'              needed. The keys should be the 'DefinitionName' value of items in
    #'              'TrainingJobDefinitions', which would be used as estimator names in the
    #'              resulting tuner instance.
    #'              # Example #1 - assuming we have the following tuning job description, which has the
    #'              # 'TrainingJobDefinition' field present using a SageMaker built-in algorithm (i.e. PCA),
    #'              # and ``attach()`` can derive the estimator class from the training image.
    #'              # So ``estimator_cls`` would not be needed.
    #'
    #'              # .. code:: R
    #'              list(
    #'                 'BestTrainingJob'= 'best_training_job_name',
    #'                 'TrainingJobDefinition' = list(
    #'                 'AlgorithmSpecification' = list(
    #'                 'TrainingImage'= '174872318107.dkr.ecr.us-west-2.amazonaws.com/pca:1
    #'                 )
    #'                )
    #'              )
    #'              #>>> my_tuner.fit()
    #'              #>>> job_name = my_tuner$latest_tuning_job$name
    #'              #Later on:
    #'              #>>> attached_tuner = HyperparameterTuner.attach(job_name)
    #'              #>>> attached_tuner.deploy()
    #'              #Example #2 - assuming we have the following tuning job description, which has a 2-item
    #'              #list for the 'TrainingJobDefinitions' field. In this case 'estimator_cls' is only
    #'              #needed for the 2nd item since the 1st item uses a SageMaker built-in algorithm
    #'              #(i.e. PCA).
    #'
    #'              #.. code:: R
    #'              list(
    #'                  'BestTrainingJob' = 'best_training_job_name',
    #'                  'TrainingJobDefinitions'= list(
    #'                     list(
    #'                       'DefinitionName'= 'estimator_pca',
    #'                       'AlgorithmSpecification'= list(
    #'                            'TrainingImage'= '174872318107.dkr.ecr.us-west-2.amazonaws.com/pca:1)
    #'                            ),
    #'                     list(
    #'                       'DefinitionName'= 'estimator_byoa',
    #'                       'AlgorithmSpecification' = list(
    #'                            'TrainingImage'= '123456789012.dkr.ecr.us-west-2.amazonaws.com/byoa:latest)
    #'                            )
    #'                        )
    #'                    )
    #'              >>> my_tuner.fit()
    #'              >>> job_name = my_tuner.latest_tuning_job.name
    #'              Later on:
    #'              >>> attached_tuner = HyperparameterTuner.attach(
    #'              >>>     job_name,
    #'              >>>     estimator_cls={
    #'              >>>         'estimator_byoa': 'org.byoa.Estimator'
    #'              >>>     })
    #'              >>> attached_tuner.deploy()
    #' @return sagemaker.tuner.HyperparameterTuner: A ``HyperparameterTuner``
    #'              instance with the attached hyperparameter tuning job.
    attach = function(tuning_job_name,
                      sagemaker_session=NULL,
                      job_details=NULL,
                      estimator_cls=NULL){
      sagemaker_session = sagemaker_session %||% Session$new()

      if (is.null(job_details))
        job_details = sagemaker_session$sagemaker_client$describe_hyper_parameter_tuning_job(
          HyperParameterTuningJobName=tuning_job_name)

      if ("TrainingJobDefinition" %in% names(job_details))
          return(private$.attach_with_training_details(
            tuning_job_name, sagemaker_session, estimator_cls, job_details))

      return(private$.attach_with_training_details_list(
          tuning_job_name, sagemaker_session, estimator_cls, job_details))
    },

    #' @description Deploy the best trained or user specified model to an Amazon
    #'              SageMaker endpoint and return a ``sagemaker.Predictor`` object.
    #'              For more information:
    #'              http://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html
    #' @param initial_instance_count (int): Minimum number of EC2 instances to
    #'              deploy to an endpoint for prediction.
    #' @param instance_type (str): Type of EC2 instance to deploy to an endpoint
    #'              for prediction, for example, 'ml.c4.xlarge'.
    #' @param accelerator_type (str): Type of Elastic Inference accelerator to
    #'              attach to an endpoint for model loading and inference, for
    #'              example, 'ml.eia1.medium'. If not specified, no Elastic
    #'              Inference accelerator will be attached to the endpoint. For more
    #'              information:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
    #' @param endpoint_name (str): Name to use for creating an Amazon SageMaker
    #'              endpoint. If not specified, the name of the training job is
    #'              used.
    #' @param wait (bool): Whether the call should wait until the deployment of
    #'              model completes (default: True).
    #' @param model_name (str): Name to use for creating an Amazon SageMaker
    #'              model. If not specified, the name of the training job is used.
    #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
    #'              data on the storage volume attached to the instance hosting the
    #'              endpoint.
    #' @param data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
    #'              configuration related to Endpoint data capture for use with
    #'              Amazon SageMaker Model Monitoring. Default: None.
    #' @param ... : Other arguments needed for deployment. Please refer to the
    #'              ``create_model()`` method of the associated estimator to see
    #'              what other arguments are needed.
    #' @return sagemaker.predictor.Predictor: A predictor that provides a ``predict()``
    #'              method, which can be used to send requests to the Amazon SageMaker endpoint
    #'              and obtain inferences.
    deploy = function(initial_instance_count,
                      instance_type,
                      accelerator_type=NULL,
                      endpoint_name=NULL,
                      wait=TRUE,
                      model_name=NULL,
                      kms_key=NULL,
                      data_capture_config=NULL,
                      ...){

      best_training_job = private$.get_best_training_job()
      best_estimator = self$best_estimator(best_training_job)

      params = list(initial_instance_count=initial_instance_count,
                    instance_type=instance_type,
                    accelerator_type=accelerator_type,
                    endpoint_name=endpoint_name %||% best_training_job$TrainingJobName,
                    wait=wait,
                    model_name=model_name,
                    kms_key=kms_key,
                    data_capture_config=data_capture_config,
                    ...)

      return (do.call(best_estimator$deploy, params))
    },


    #' @description Stop latest running hyperparameter tuning job.
    stop_tunning_job = function(){
      private$.ensure_last_tuning_job()
      private$stop()
    },

    #' @description Returns a response from the DescribeHyperParameterTuningJob API call.
    describe = function(){
      return(self$sagemaker_session$describe_tuning_job(self$.current_job_name))
    },

    #' @description Wait for latest hyperparameter tuning job to finish.
    wait = function(){
      private$.ensure_last_tuning_job()
      private$wait_tuningjob()
    },

    #' @description Return the estimator that has best training job attached. The trained model can then
    #'              be deployed to an Amazon SageMaker endpoint and return a ``sagemaker.Predictor``
    #'              object.
    #' @param best_training_job (dict): Dictionary containing "TrainingJobName" and
    #'              "TrainingJobDefinitionName".
    #'              Example:
    #'              .. code:: R
    #'              list(
    #'                 "TrainingJobName"= "my_training_job_name",
    #'                 "TrainingJobDefinitionName" "my_training_job_definition_name"
    #'              )
    #' @return sagemaker.estimator.EstimatorBase: The estimator that has the best training job
    #'              attached.
    best_estimator = function(best_training_job=NULL){
      if (islistempty(best_training_job))
        best_training_job = private$.get_best_training_job()

      if (!is.null(self$estimator))
        best_estimator = self$estimator
      else {
        best_estimator_name = best_training_job$TrainingJobDefinitionName
        best_estimator = self$estimator_list[[best_estimator_name]]}

      return(best_estimator$attach(
        training_job_name=best_training_job$TrainingJobName,
        sagemaker_session=self$sagemaker_session))
    },

    #' @description Return name of the best training job for the latest hyperparameter
    #'              tuning job.
    best_training_job = function(){
      return(private$.get_best_training_job()$TrainingJobName)
    },

    #' @description Delete an Amazon SageMaker endpoint.
    #'              If an endpoint name is not specified, this defaults to looking for an
    #'              endpoint that shares a name with the best training job for deletion.
    #' @param endpoint_name (str): Name of the endpoint to delete
    delete_endpoint = function(endpoint_name = NULL){
      log_warn(paste(
        "HyperparameterTuner.delete_endpoint() will be deprecated in SageMaker Python SDK v2.",
        "Please use the delete_endpoint() function on your predictor instead."))

      endpoint_name = endpoint_name %||% self$best_training_job()
      self$sagemaker_session$delete_endpoint(endpoint_name)
    },

    #' @description Return the hyperparameter ranges in a dictionary to be used as part
    #'              of a request for creating a hyperparameter tuning job.
    hyperparameter_ranges = function(){
      if(is.null(self$.hyperparameter_ranges))
        return(NULL)

      return(private$.prepare_parameter_ranges_for_tuning(
        self$.hyperparameter_ranges, self$estimator))
    },

    #' @description Return a dictionary of hyperparameter ranges for all estimators in ``estimator_dict``
    hyperparameter_ranges_list = function(){
      if (!islistempty(self$.hyperparameter_ranges_list))
        return(NULL)

      output = lapply(sort(names(self$estimator_list)),
                      function(estimator_name){
                        private$.prepare_parameter_ranges_for_tuning(
                          self$.hyperparameter_ranges_list$estimator_name,
                          self$estimator_list$estimator_name)})
      names(output) = sort(names(self$estimator_list))
      return(output)
    },

    #' @description An instance of HyperparameterTuningJobAnalytics for this latest
    #'              tuning job of this tuner. Analytics olbject gives you access to tuning
    #'              results summarized into a pandas dataframe.
    analytics = function(){
      return(HyperparameterTuningJobAnalytics$new(self$latest_tuning_job.name, self$sagemaker_session))
    },

    #' @description Creates a new ``HyperparameterTuner`` by copying the request fields
    #'              from the provided parent to the new instance of ``HyperparameterTuner``.
    #'              Followed by addition of warm start configuration with the type as
    #'              "TransferLearning" and parents as the union of provided list of
    #'              ``additional_parents`` and the ``self``. Also, training image in the new
    #'              tuner's estimator is updated with the provided ``training_image``.
    #'              Examples:
    #'              >>> parent_tuner = HyperparameterTuner.attach(tuning_job_name="parent-job-1")
    #'              >>> transfer_learning_tuner = parent_tuner.transfer_learning_tuner(
    #'              >>>                                             additional_parents={"parent-job-2"})
    #'              Later On:
    #'              >>> transfer_learning_tuner.fit(inputs={})
    #' @param additional_parents (set{str}): Set of additional parents along with
    #'              the self to be used in warm starting
    #' @param estimator (sagemaker.estimator.EstimatorBase): An estimator object
    #'              that has been initialized with the desired configuration. There
    #'              does not need to be a training job associated with this
    #'              instance.
    #' @return sagemaker.tuner.HyperparameterTuner: ``HyperparameterTuner``
    #'              instance which can be used to launch transfer learning tuning job.
    transfer_learning_tuner = function(additional_parents=NULL,
                                       estimator=NULL){
      return (private$.create_warm_start_tuner(
        additional_parents=additional_parents,
        warm_start_type=WarmStartTypes$new()$TRANSFER_LEARNING,
        estimator=estimator))
    },

    #' @description Creates a new ``HyperparameterTuner`` by copying the request fields
    #'              from the provided parent to the new instance of ``HyperparameterTuner``.
    #'              Followed by addition of warm start configuration with the type as
    #'              "IdenticalDataAndAlgorithm" and parents as the union of provided list of
    #'              ``additional_parents`` and the ``self``
    #'              Examples:
    #'              >>> parent_tuner = HyperparameterTuner.attach(tuning_job_name="parent-job-1")
    #'              >>> identical_dataset_algo_tuner = parent_tuner.identical_dataset_and_algorithm_tuner(
    #'              >>>                                                additional_parents={"parent-job-2"})
    #'              Later On:
    #'              >>> identical_dataset_algo_tuner.fit(inputs={})
    #' @param additional_parents (set{str}): Set of additional parents along with
    #'              the self to be used in warm starting
    #' @return sagemaker.tuner.HyperparameterTuner: HyperparameterTuner instance
    #'              which can be used to launch identical dataset and algorithm tuning
    #'              job.
    identical_dataset_and_algorithm_tuner = function(additional_parents=NULL){
      return(private$.create_warm_start_tuner(
        additional_parents=additional_parents,
        warm_start_type=WarmStartTypes$new()$IDENTICAL_DATA_AND_ALGORITHM))
    },

    #' @description  Factory method to create a ``HyperparameterTuner`` instance. It takes one or more
    #'               estimators to obtain configuration information for training jobs that are created as the
    #'               result of a hyperparameter tuning job. The estimators are provided through a dictionary
    #'               (i.e. ``estimator_dict``) with unique estimator names as the keys. For individual
    #'               estimators separate objective metric names and hyperparameter ranges should be provided in
    #'               two dictionaries, i.e. ``objective_metric_name_dict`` and ``hyperparameter_ranges_dict``,
    #'               with the same estimator names as the keys. Optional metrics definitions could also be
    #'               provided for individual estimators via another dictionary ``metric_definitions_dict``.
    #' @param estimator_list (dict[str, sagemaker.estimator.EstimatorBase]): Dictionary of estimator
    #'               instances that have been initialized with the desired configuration. There does not
    #'               need to be a training job associated with the estimator instances. The keys of the
    #'               dictionary would be referred to as "estimator names".
    #' @param objective_metric_name_list (dict[str, str]): Dictionary of names of the objective
    #'               metric for evaluating training jobs. The keys are the same set of estimator names
    #'               as in ``estimator_dict``, and there must be one entry for each estimator in
    #'               ``estimator_dict``.
    #' @param hyperparameter_ranges_list (dict[str, dict[str, sagemaker.parameter.ParameterRange]]):
    #'               Dictionary of tunable hyperparameter ranges. The keys are the same set of estimator
    #'               names as in estimator_dict, and there must be one entry for each estimator in
    #'               estimator_dict. Each value is a dictionary of sagemaker.parameter.ParameterRange
    #'               instance, which can be one of three types: Continuous, Integer, or Categorical.
    #'               The keys of each ParameterRange dictionaries are the names of the hyperparameter,
    #'               and the values are the appropriate parameter range class to represent the range.
    #' @param metric_definitions_list (dict(str, list[dict]])): Dictionary of metric definitions.
    #'               The keys are the same set or a subset of estimator names as in estimator_dict,
    #'               and there must be one entry for each estimator in estimator_dict. Each value is
    #'               a list of dictionaries that defines the metric(s) used to evaluate the training
    #'               jobs (default: None). Each of these dictionaries contains two keys: 'Name' for the
    #'               name of the metric, and 'Regex' for the regular expression used to extract the
    #'               metric from the logs. This should be defined only for hyperparameter tuning jobs
    #'               that don't use an Amazon algorithm.
    #' @param base_tuning_job_name (str): Prefix for the hyperparameter tuning job name when the
    #'               :meth:`~sagemaker.tuner.HyperparameterTuner.fit` method launches. If not specified,
    #'               a default job name is generated, based on the training image name and current
    #'               timestamp.
    #' @param strategy (str): Strategy to be used for hyperparameter estimations
    #'               (default: 'Bayesian').
    #' @param objective_type (str): The type of the objective metric for evaluating training jobs.
    #'               This value can be either 'Minimize' or 'Maximize' (default: 'Maximize').
    #' @param max_jobs (int): Maximum total number of training jobs to start for the hyperparameter
    #' @param tuning job (default: 1).
    #' @param max_parallel_jobs (int): Maximum number of parallel training jobs to start
    #'               (default: 1).
    #' @param tags (list[dict]): List of tags for labeling the tuning job (default: None). For more,
    #'               see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @param warm_start_config (sagemaker.tuner.WarmStartConfig): A ``WarmStartConfig`` object that
    #'               has been initialized with the configuration defining the nature of warm start
    #'               tuning job.
    #' @param early_stopping_type (str): Specifies whether early stopping is enabled for the job.
    #'               Can be either 'Auto' or 'Off' (default: 'Off'). If set to 'Off', early stopping
    #'               will not be attempted. If set to 'Auto', early stopping of some training jobs may
    #'               happen, but is not guaranteed to.
    #' @return sagemaker.tuner.HyperparameterTuner: a new ``HyperparameterTuner`` object that can
    #'               start a hyperparameter tuning job with one or more estimators.
    create = function(estimator_list,
                      objective_metric_name_list,
                      hyperparameter_ranges_list,
                      metric_definitions_list=NULL,
                      base_tuning_job_name=NULL,
                      strategy="Bayesian",
                      objective_type="Maximize",
                      max_jobs=1,
                      max_parallel_jobs=1,
                      tags=NULL,
                      warm_start_config=NULL,
                      early_stopping_type="Off"){
      private$.validate_create_tuner_inputs(
        estimator_list,
        objective_metric_name_list,
        hyperparameter_ranges_list,
        metric_definitions_list)

      estimator_names = sort(names(estimator_list))
      first_estimator_name = estimator_names[1]

      metric_definitions = (if (!islistempty(metric_definitions_list))
                                metric_definitions_list[[first_estimator_name]]
                            else NULL)

      tuner = self$clone()
      tuner$initialize(
        base_tuning_job_name=base_tuning_job_name,
        estimator_name=first_estimator_name,
        estimator=estimator_list[[first_estimator_name]],
        objective_metric_name=objective_metric_name_list[[first_estimator_name]],
        hyperparameter_ranges=hyperparameter_ranges_list[[first_estimator_name]],
        metric_definitions=metric_definitions,
        strategy=strategy,
        objective_type=objective_type,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        tags=tags,
        warm_start_config=warm_start_config,
        early_stopping_type=early_stopping_type)

      for (estimator_name in estimator_names[2:length(estimator_names)]){
        metric_definitions = (if(!islistempty(metric_definitions_list))
                                metric_definitions_list[[estimator_name]]
                              else NULL)
        tuner$.add_estimator(
          estimator_name=estimator_name,
          estimator=estimator_list[[estimator_name]],
          objective_metric_name=objective_metric_name_list[[estimator_name]],
          hyperparameter_ranges=hyperparameter_ranges_list[[estimator_name]],
          metric_definitions=metric_definitions)
      }
      return(tuner)
    },

    #' @description Add an estimator with corresponding objective metric name, parameter ranges and metric
    #'              definitions (if applicable). This method is called by other functions and isn't required
    #'              to be called directly
    #' @param estimator_name (str): A unique name to identify an estimator within the
    #'              hyperparameter tuning job, when more than one estimator is used with
    #'              the same tuning job (default: None).
    #' @param estimator (sagemaker.estimator.EstimatorBase): An estimator object
    #'              that has been initialized with the desired configuration. There
    #'              does not need to be a training job associated with this
    #'              instance.
    #' @param objective_metric_name (str): Name of the metric for evaluating
    #'              training jobs.
    #' @param hyperparameter_ranges (dict[str, sagemaker.parameter.ParameterRange]): Dictionary of
    #'              parameter ranges. These parameter ranges can be one
    #'              of three types: Continuous, Integer, or Categorical. The keys of
    #'              the dictionary are the names of the hyperparameter, and the
    #'              values are the appropriate parameter range class to represent
    #'              the range.
    #' @param metric_definitions (list[dict]): A list of dictionaries that defines
    #'              the metric(s) used to evaluate the training jobs (default:
    #'              None). Each dictionary contains two keys: 'Name' for the name of
    #'              the metric, and 'Regex' for the regular expression used to
    #'              extract the metric from the logs. This should be defined only
    #'              for hyperparameter tuning jobs that don't use an Amazon
    #'              algorithm.
    .attach_estimator = function(estimator_name,
                                 estimator,
                                 objective_metric_name,
                                 hyperparameter_ranges,
                                 metric_definitions=NULL){
      self$estimator_list[[estimator_name]] = estimator
      self$objective_metric_name_list[[estimator_name]] = objective_metric_name
      self$.hyperparameter_ranges_list[[estimator_name]] = hyperparameter_ranges
      if (!is.null(metric_definitions))
        self$metric_definitions_list[[estimator_name]] = metric_definitions
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      cat("<HyperparameterTuner>")
      invisible(self)
    }
  ),
  private = list(
    # Prepare the tuner instance for tuning (fit)
    .prepare_for_tuning = function(job_name=NULL,
                                    include_cls_metadata=FALSE){
      private$.prepare_job_name_for_tuning(job_name=job_name)
      private$.prepare_static_hyperparameters_for_tuning(include_cls_metadata=include_cls_metadata)
    },

    # Set current job name before starting tuning
    .prepare_job_name_for_tuning = function(job_name=NULL){
      if (!is.null(job_name))
        self$.current_job_name = job_name
      else {
        base_name = self$base_tuning_job_name
        if (is.null(base_name)){
          estimator = self$estimator %||%  self$estimator_list[sort(names(self$estimator_list))[1]]
          base_name = base_name_from_image(estimator$training_image_uri())}
        self$.current_job_name = name_from_base(base_name, max_length=self$TUNING_JOB_NAME_MAX_LENGTH, short=TRUE)
      }
    },

    # Prepare static hyperparameters for all estimators before tuning
    .prepare_static_hyperparameters_for_tuning = function(include_cls_metadata=FALSE){
      self$static_hyperparameters = NULL
      if (!is.null(self$estimator)){
        self$static_hyperparameters = private$.prepare_static_hyperparameters(
          self$estimator, self$.hyperparameter_ranges, include_cls_metadata)}

      self$static_hyperparameters_list = NULL
      if (!islistempty(self$estimator_list)){
        self$static_hyperparameters_list = lapply(
          seq_along(self$estimator_list), function(x){
            estimator_name = names(self$estimator_list)[x]
            estimator = self$estimator_list[[x]]
            private$.prepare_static_hyperparameters(
              estimator,
              self$.hyperparameter_ranges_list$estimator_name,
              if(!inherits(include_cls_metadata, "logical")) include_cls_metadata$estimator_name else include_cls_metadata)
            }
          )
        names(self$static_hyperparameters_list) = names(self$estimator_list)
        }
      },

    # Prepare static hyperparameters for one estimator before tuning
    .prepare_static_hyperparameters = function(estimator, hyperparameter_ranges, include_cls_metadata){
      # Remove any hyperparameter that will be tuned
      static_hyperparameters = lapply(estimator$hyperparameters(), as.character)
      for (hyperparameter_name in names(hyperparameter_ranges)){
        static_hyperparameters[hyperparameter_name] = NULL}

      # For attach() to know what estimator to use for frameworks
      # (other algorithms may not accept extra hyperparameters)
      if (isTRUE(include_cls_metadata) || inherits(estimator, "Framework")){
        static_hyperparameters[[self$SAGEMAKER_ESTIMATOR_CLASS_NAME]] = class(estimator)[[1]]
        # R doesn't have a means to reference module to what I am aware of
        static_hyperparameters[[self$SAGEMAKER_ESTIMATOR_MODULE]] = attritubes(estimator)$`__module__`
      }

      return(static_hyperparameters)
    },

    # Start tuning for tuner instances that have the ``estimator`` field set
    .fit_with_estimator = function(inputs,
                                   job_name,
                                   include_cls_metadata,
                                   ...){
      private$.prepare_estimator_for_tuning(self$estimator, inputs, job_name, ...)
      private$.prepare_for_tuning(job_name=job_name, include_cls_metadata=include_cls_metadata)
      self$latest_tuning_job = private$start_new(inputs)
    },

    # Start tuning for tuner instances that have the ``estimator_dict`` field set
    .fit_with_estimator_list = function(inputs,
                                        job_name,
                                        include_cls_metadata,
                                        estimator_kwargs){
      estimator_names = sort(names(self.estimator_list))
      private$.validate_list_argument(name="inputs", value=inputs, allowed_keys=estimator_names)
      private$.validate_list_argument(
        name="include_cls_metadata", value=include_cls_metadata, allowed_keys=estimator_names)
      private$.validate_list_argument(
        name="estimator_kwargs", value=estimator_kwargs, allowed_keys=estimator_names)

      for (i in self$estimator_list){
        estimator_name = names(self$estimator_list)[i]
        estimator = self$estimator_list[[i]]
        ins = if (!islistempty(inputs)) inputs$estimator_name else NULL
        args = if (!islistempty(estimator_kwargs)) estimator_kwargs$estimator_name %||% list() else list()
        args = list(estimator= estimator, inputs = ins, job_name = job_name, args)
        do.call(private$.prepare_estimator_for_tuning, args)
      }

      inc_cls_metadata = if (!islistempty(include_cls_metadata)) include_cls_metadata else list()
      private$.prepare_for_tuning(job_name=job_name, include_cls_metadata=inc_cls_metadata)

      self$latest_tuning_job = private$start_new(inputs)
    },

    # Attach an estimator from training job details
    .prepare_estimator = function(estimator_cls,
                                  training_details,
                                  parameter_ranges,
                                  sagemaker_session){
      estimator_cls = private$.prepare_estimator_cls(estimator_cls, training_details)
      return (private$.prepare_estimator_from_job_description(
        estimator_cls, training_details, parameter_ranges, sagemaker_session))
    },

    # Check for customer-specified estimator first
    .prepare_estimator_cls = function(estimator_cls,
                                      training_details){
      if (!is.null(estimator_cls)){
        return(eval(parse(text = estimator_cls)[1]))}

      # Then check for estimator class in hyperparameters
      hyperparameters = training_details$StaticHyperParameters
      if (self$SAGEMAKER_ESTIMATOR_CLASS_NAME %in% hyperparameters
          # TODO: need to set up metadata to map Python modules to R
          # && self$SAGEMAKER_ESTIMATOR_MODULE %in% hyperparameters
          ){
        module = hyperparameters[[self$SAGEMAKER_ESTIMATOR_MODULE]]
        cls_name = hyperparameters[[self$SAGEMAKER_ESTIMATOR_CLASS_NAME]]
        return(eval(parse(text = cls_name)[1]))}

      # Then try to derive the estimator from the image name for 1P algorithms
      image_name = training_details$AlgorithmSpecification$TrainingImage
      pos <- regexpr("/", image_name, perl=TRUE) + 1
      algorithm = substr(image_name, pos+1, nchar(image_name))
      if (algorithm %in% AMAZON_ESTIMATOR_CLS_NAMES){
        cls_name = AMAZON_ESTIMATOR_CLS_NAMES[[algorithm]]
        return(eval(parse(text = paste(AMAZON_ESTIMATOR_MODULE, cls_name, sep = "::")[1])))}

      # Default to the BYO estimator
      return(eval(parse(text = paste(self$DEFAULT_ESTIMATOR_MODULE,
                                     self$DEFAULT_ESTIMATOR_CLS_NAME, sep = "::")[1])))
    },

    .prepare_estimator_from_job_description = function(estimator_cls,
                                                       training_details,
                                                       parameter_ranges,
                                                       sagemaker_session){
      # Swap name for static hyperparameters to what an estimator would expect
      training_details$HyperParameters = training_details$StaticHyperParameters
      training_details$StaticHyperParameters = NULL

      # Remove hyperparameter reserved by SageMaker for tuning jobs
      training_details[["HyperParameters"]][["_tuning_objective_metric"]] = NULL

      # Add missing hyperparameters defined in the hyperparameter ranges,
      # as potentially required in the Amazon algorithm estimator's constructor
      # TODO: create AmazonAlgorithmEstimatorBase class
      if (IsSubR6Class(estimator_cls, "AmazonAlgorithmEstimatorBase")){
        additional_hyperparameters = private$.extract_hyperparameters_from_parameter_ranges(
          parameter_ranges)
        training_details[["HyperParameters"]] = c(training_details[["HyperParameters"]], additional_hyperparameters)}

      # Add items expected by the estimator (but aren't needed otherwise)
      training_details$TrainingJobName = ""
      if (!("KmsKeyId" %in% names(training_details$OutputDataConfig)))
        training_details$OutputDataConfig$KmsKeyId = ""

      estimator_init_params = estimator_cls$private_methods$.prepare_init_params_from_job_description(
        training_details)

      estimator_init_params = list(sagemaker_session=sagemaker_session,
                                   estimator_init_params)

      return(do.call(estimator_cls$new, estimator_init_params))
    },

    .prepare_init_params_from_job_description = function(job_details){
      tuning_config = job_details$HyperParameterTuningJobConfig

      params = list(
        "strategy"= tuning_config$Strategy,
        "max_jobs"= tuning_config$ResourceLimits$MaxNumberOfTrainingJobs,
        "max_parallel_jobs"= tuning_config$ResourceLimits$MaxParallelTrainingJobs,
        "warm_start_config"= WarmStartConfig$new()$from_job_desc(
          job_details$WarmStartConfig),
        "early_stopping_type"= tuning_config$TrainingJobEarlyStoppingType)

      params$objective_metric_name = tuning_config$HyperParameterTuningJobObjectiveMetricName
      params$objective_type = tuning_config$HyperParameterTuningJobObjective$Type

      if ("ParameterRanges" %in% names(tuning_config))
        params$hyperparameter_ranges = private$.prepare_parameter_ranges_from_job_description(
          tuning_config$ParameterRanges)

      params$metric_definitions = job_details$TrainingJobDefinition$AlgorithmSpecification$MetricDefinitions

      params$objective_type = job_details$TrainingJobDefinitions[[1]]$TuningObjective$Type

      return(params)
    },

    .prepare_parameter_ranges_from_job_description = function(parameter_ranges){
      ranges = list()

      for (parameter in parameter_ranges$CategoricalParameterRanges)
        ranges[[parameter$Name]] = CategoricalParameter$new(parameter$Values)

      for (parameter in parameter_ranges$ContinuousParameterRanges)
        ranges[[parameter$Name]] = ContinuousParameter$new(
          as.numeric(parameter$MinValue), as.numeric(parameter$MaxValue))

      for (parameter in parameter_ranges$IntegerParameterRanges)
        ranges[[parameter$Name]] = IntegerParameter$new(
          as.integer(parameter$MinValue), as.integer(parameter$MaxValue))

      return(ranges)
    },

    .extract_hyperparameters_from_parameter_ranges = function(parameter_ranges){
      hyperparameters = list()

      for (parameter in parameter_ranges$CategoricalParameterRanges)
        hyperparameters[[parameter$Name]] = parameter$Values[1]

      for (parameter in parameter_ranges$ContinuousParameterRanges)
        hyperparameters[[parameter$Name]] = as.numeric(parameter$MinValue)

      for (parameter in parameter_ranges$IntegerParameterRanges)
        hyperparameters[[parameter$Name]] = as.integer(parameter$MinValue)

      return(hyperparameters)
    },

    # Prepare hyperparameter ranges for tunin
    .prepare_parameter_ranges_for_tuning = function(parameter_ranges,
                                                    estimator){
      processed_parameter_ranges = list()
      for (range_type in ParameterRange$public_fields$.all_types){
        hp_ranges = list()
        for (i in seq_along(parameter_ranges)){
          parameter_name = names(parameter_ranges)[[i]]
          parameter = parameter_ranges[[i]]
          if (!islistempty(parameter) && parameter$.name == range_type){
            # Categorical parameters needed to be serialized as JSON for our framework
            # containers
            if(inherits(parameter, "CategoricalParameter")
              && inherits(estimator, "Framework"))
              tuning_range = parameter$as_json_range(parameter_name)
            else
              tuning_range = parameter$as_tuning_range(parameter_name)
            hp_ranges = c(hp_ranges, list(tuning_range))}
        processed_parameter_ranges[[paste0(range_type, "ParameterRanges")]] = hp_ranges
        }
      }
      return(processed_parameter_ranges)
    },

    # Prepare one estimator before starting tuning
    .prepare_estimator_for_tuning = function(estimator,
                                             inputs,
                                             job_name,
                                             ...){
      if (inherits(inputs, c("list", "RecordSet", "FileSystemRecordSet")))
        estimator$.prepare_for_training(inputs, ...)
      else
        estimator$.prepare_for_training(job_name)
    },

    # Create a HyperparameterTuner bound to an existing hyperparameter
    # tuning job that has the ``TrainingJobDefinition`` field set.
    .attach_with_training_details = function(tuning_job_name,
                                             sagemaker_session,
                                             estimator_cls,
                                             job_details){
      estimator = private$.prepare_estimator(
        estimator_cls=estimator_cls,
        training_details=job_details$TrainingJobDefinition,
        parameter_ranges=job_details$HyperParameterTuningJobConfig$ParameterRanges,
        sagemaker_session=sagemaker_session)
      init_params = private$.prepare_init_params_from_job_description(job_details)
      init_params = list(estimator = estimator, init_params)
      tunner = self$clone()
      do.call(tuner$initialize, init_params)
      tuner$latest_tuning_job = tuning_job_name
      return(tuner)
    },

    # Create a HyperparameterTuner bound to an existing hyperparameter
    # tuning job that has the ``TrainingJobDefinitions`` field set.
    .attach_with_training_details_list = function(tuning_job_name,
                                                  sagemaker_session,
                                                  estimator_cls,
                                                  job_details){
      estimator_names = sort(sapply(job_details$TrainingJobDefinitions,
                                    function(training_details) training_details$DefinitionName))

      private$.validate_list_argument(
        name="estimator_cls", value=estimator_cls, allowed_keys=estimator_names)

      estimator_dict = list()
      objective_metric_name_dict = list()
      hyperparameter_ranges_dict = list()
      metric_definitions_dict = list()

      for (training_details in job_details$TrainingJobDefinitions){
        estimator_name = training_details$DefinitionName

        estimator_dict[[estimator_name]] = private$.prepare_estimator(
          estimator_cls= (if(!islistempty(estimator_cls)) estimator_cls[[estimator_name]] else NULL),
          training_details=training_details,
          parameter_ranges=training_details$HyperParameterRanges,
          sagemaker_session=sagemaker_session)

        objective_metric_name_dict[[estimator_name]] = training_details$TuningObjective$MetricName

        hyperparameter_ranges_dict[[estimator_name]] = private$.prepare_parameter_ranges_from_job_description(
            training_details$HyperParameterRanges)

        metric_definitions = training_details$AlgorithmSpecification$MetricDefinitions

        if (!islistempty(metric_definitions))
          metric_definitions_dict[[estimator_name]] = metric_definitions
      }

      init_params = private$.prepare_init_params_from_job_description(job_details)

      init_params= list(estimator_list=estimator_dict,
                        objective_metric_name_list=objective_metric_name_dict,
                        hyperparameter_ranges_list=hyperparameter_ranges_dict,
                        metric_definitions_list=metric_definitions_dict,
                        init_params)

      tuner = do.call(self$create, init_params)

      tuner$latest_tuning_job = tuning_job_name

      return(tuner)
    },

    # Placeholder docstring
    .ensure_last_tuning_job = function(){
      if (is.null(self$latest_tuning_job))
        stop("No tuning job available", call. = F)
    },

    # Return the best training job for the latest hyperparameter
    # tuning job.
    .get_best_training_job = function(){
      private$.ensure_last_tuning_job()

      tuning_job_describe_result = self$sagemaker_session$sagemaker$describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=self$latest_tuning_job)

      best_job = tuning_job_describe_result$BestTrainingJob

      if(is.null(best_job)) {
        stop(sprintf("Best training job not available for tuning job: %s", self$latest_tuning_job),
             call. = F)
      } else {return(best_job)}
    },

    # Validate hyperparameter ranges for an estimator
    .validate_parameter_ranges = function(estimator,
                                          hyperparameter_ranges){
      for(kls in as.list(estimator)){
        if (inherits(kls, "Hyperparameter")){
          tryCatch({
            # The hyperparam names may not be the same as the class attribute that
            # holds them, for instance: local_lloyd_init_method is called
            # local_init_method. We need to map these and pass the correct name to
            # the constructor.
            parameter_range = hyperparameter_ranges[[kls$name]]
            if (inherits(parameter_range, "ParameterRange")){
              private$.validate_parameter_range(kls, parameter_range)}
            },
            error = function(e) NULL)
        }
      }
    },

    .validate_parameter_range = function(value_hp,
                                         parameter_range){
      # TODO: method to validate hyperparameters
      for (i in seq_along(parameter_range)){
        if (names(parameter_range[i]) == "scaling_type")
          next

        # Categorical ranges
        if (inherits(parameter_range[[i]], "list")){
          for (categorical_value in parameter_range[[i]])
            value_hp$validate(categorical_value)
          # Continuous, Integer ranges
        } else
          value_hp$validate(param_range)
      }
    },

    # Creates a new ``HyperparameterTuner`` with ``WarmStartConfig``, where
    # type will be equal to ``warm_start_type`` and``parents`` would be equal
    # to union of ``additional_parents`` and self.
    # Args:
    #   additional_parents (set{str}): Additional parents along with self,
    # to be used for warm starting.
    # warm_start_type (sagemaker.tuner.WarmStartTypes): Type of warm start
    # job.
    # estimator:
    #   Returns:
    #   sagemaker.tuner.HyperparameterTuner: Instance with the request
    # fields copied from self along with the warm start configuration
    .create_warm_start_tuner = function(additional_parents,
                                        warm_start_type,
                                        estimator=NULL){
      all_parents = list(self$latest_tuning_job)
      if (!islistempty(additional_parents))
        all_parents = union(all_parents, additional_parents)

      if (!is.null(self$estimator)){
        cls = self$clone()
        cls$intialize(
          estimator= estimator %||% self$estimator,
          objective_metric_name=self$objective_metric_name,
          hyperparameter_ranges=self$.hyperparameter_ranges,
          strategy=self$strategy,
          objective_type=self$objective_type,
          max_jobs=self.max_jobs,
          max_parallel_jobs=self$max_parallel_jobs,
          warm_start_config=WarmStartConfig$new(
            warm_start_type=warm_start_type, parents=all_parents),
          early_stopping_type=self$early_stopping_type)
        return(cls)
      }

      if (length(self$estimator_list) > 1)
        stop("Warm start is not supported currently for tuners with multiple estimators")

      if (is.null(estimator)){
        estimator_name = names(self$estimator_list)[[1]]
        estimator_list = list(estimator)
        names(estimator_list) = estimator_name
      } else{
        estimator_list = self$estimator_list}


      return (self$create(
        estimator_list=estimator_list,
        objective_metric_name_list=self$objective_metric_name_list,
        hyperparameter_ranges_list=self$.hyperparameter_ranges_list,
        metric_definitions_list=self$metric_definitions_list,
        strategy=self$strategy,
        objective_type=self$objective_type,
        max_jobs=self$max_jobs,
        max_parallel_jobs=self$max_parallel_jobs,
        warm_start_config=WarmStartConfig$new(warm_start_type=warm_start_type, parents=all_parents),
        early_stopping_type=self$early_stopping_type))
    },

    # Validate inputs for ``HyperparameterTuner.create()``
    .validate_create_tuner_inputs = function(estimator_list,
                                             objective_metric_name_list,
                                             hyperparameter_ranges_list,
                                             metric_definitions_list=NULL){
      private$.validate_estimator_list(estimator_list)

      estimator_names = sort(names(estimator_list))

      private$.validate_list_argument(
        name="objective_metric_name_dict",
        value=objective_metric_name_list,
        allowed_keys=estimator_names,
        require_same_keys=TRUE)
      private$.validate_list_argument(
        name="hyperparameter_ranges_dict",
        value=hyperparameter_ranges_list,
        allowed_keys=estimator_names,
        require_same_keys=TRUE)
      private$.validate_list_argument(
        name="metric_definitions_dict",
        value=metric_definitions_list,
        allowed_keys=estimator_names)
    },

    # Validate ``estimator_dict`` in inputs for ``HyperparameterTuner.create()``
    .validate_estimator_list = function(estimatorlist){
      if (islistempty(estimator_list))
        stop("At least one estimator should be provided", call.=F)
      if (NULL %in% names(estimator_list))
        stop("Estimator names cannot be None", call. = F)
    },

    # Check if an argument is an dictionary with correct key set
    .validate_list_argument = function(name,
                                       value,
                                       allowed_keys,
                                       require_same_keys=FALSE){
      if (missing(value))
        return(NULL)

      if (!inherits(value, list))
        stop(sprintf("Argument '%s' must be a dictionary using %s as keys", name, allowed_keys), call. = F)

      value_keys = sort(names(value))

      if (require_same_keys){
        if (value_keys != allowed_keys)
          stop(sprintf("The keys of argument '%s' must be the same as %s", name, allowed_keys),
               call. = F)
      } else {
        if (!any(value_keys %in% allowed_keys))
        stop(sprintf("The keys of argument '%s' must be a subset of %s", name, allowed_keys),
             call. = F)
      }
    },

    # ------------------------------------ Start _TuningJob ------------------------------------

    # Create a new Amazon SageMaker hyperparameter tuning job from the
    # HyperparameterTuner.
    # Args:
    #   tuner (sagemaker.tuner.HyperparameterTuner): HyperparameterTuner
    # object created by the user.
    # inputs (str): Parameters used when called
    # :meth:`~sagemaker.estimator.EstimatorBase.fit`.
    # Returns:
    #   sagemaker.tuner._TuningJob: Constructed object that captures all
    # information about the started job.
    start_new = function(inputs){

      log_info("_TuningJob.start_new!!!")

      warm_start_config_req = NULL
      if (!is.null(self$warm_start_config))
        warm_start_config_req = self$warm_start_config$to_input_req()

      tuning_config = list(
        "strategy"= self$strategy,
        "max_jobs"= self$max_jobs,
        "max_parallel_jobs"= self$max_parallel_jobs,
        "early_stopping_type"= self$early_stopping_type)

      if (!is.null(self$objective_metric_name)){
        tuning_config$objective_type = self$objective_type
        tuning_config$objective_metric_name = self$objective_metric_name}

      parameter_ranges = self$hyperparameter_ranges()
      if (!is.null(parameter_ranges))
        tuning_config$parameter_ranges = parameter_ranges

      tuner_args = list(
        "job_name"= self$.current_job_name,
        "tuning_config"= tuning_config,
        "tags"= self$tags,
        "warm_start_config"= warm_start_config_req)

      if (!is.null(self$estimator)){
        tuner_args$training_config = private$.prepare_training_config(
          inputs= inputs,
          estimator = self$estimator,
          static_hyperparameters = self$static_hyperparameters,
          metric_definitions = self$metric_definitions)}

      if (!islistempty(self$estimator_list))
        tuner_args$training_config_list=lapply(
          sort(names(self$estimator_list)),
               function(estimator_name) {
                 private$.prepare_training_config(
                   if (!islistempty(inputs)) inputs$estimator_name else NULL,
                   self$estimator_list[[estimator_name]],
                   self$static_hyperparameters_list[[estimator_name]],
                   self$metric_definitions_list$estimator_name,
                   estimator_name,
                   self$objective_type,
                   self$objective_metric_name_list[[estimator_name]],
                   self$hyperparameter_ranges_list()[[estimator_name]])
               })

      do.call(self$sagemaker_session$create_tuning_job, tuner_args)
      return(self$.current_job_name)
    },

    # Prepare training config for one estimator
    .prepare_training_config = function(inputs,
                                        estimator,
                                        static_hyperparameters,
                                        metric_definitions,
                                        estimator_name=NULL,
                                        objective_type=NULL,
                                        objective_metric_name=NULL,
                                        parameter_ranges=NULL){

      training_config = .Job$private_methods$.load_config(inputs, estimator)

      training_config$input_mode = estimator$input_mode
      training_config$metric_definitions = metric_definitions

      if (inherits(inputs, "s3_input")){
        if ("InputMode" %in% names(inputs$config)){
          log_debug(
            "Selecting s3_input's input_mode (%s) for TrainingInputMode.",
            toJSON(inputs$config$InputMode, pretty = T, auto_unbox = T))
          training_config$input_mode = inputs$config$InputMode}
      }

      if (inherits(estimator, "AlgorithmEstimator"))
        training_config$algorithm_arn = estimator$algorithm_arn
      else
        training_config$image = estimator$training_image_uri()

      training_config$enable_network_isolation = estimator$enable_network_isolation()
      training_config$encrypt_inter_container_traffic = estimator$encrypt_inter_container_traffic

      training_config$train_use_spot_instances = estimator$train_use_spot_instances
      training_config$checkpoint_s3_uri = estimator$checkpoint_s3_uri
      training_config$checkpoint_local_path = estimator$checkpoint_local_path

      training_config$static_hyperparameters = static_hyperparameters

      if (!is.null(estimator_name))
        training_config$estimator_name = estimator_name

      if (!is.null(objective_type))
        training_config$objective_type = objective_type

      if (!is.null(objective_metric_name))
        training_config$objective_metric_name = objective_metric_name

      if (!is.null(parameter_ranges))
        training_config$parameter_ranges = parameter_ranges

      return(training_config)
      },

    stop = function(){
      self$sagemaker_session$stop_tuning_job(name=self$latest_tuning_job)
    },

    # wait
    wait_tuningjob = function(){
      self$sagemaker_session$wait_for_tuning_job(self$latest_tuning_job)
    }
    # ------------------------------------------------------------------------------------------
  ),

  active = list(

    #' @field sagemaker_session
    #'        Convenience method for accessing the
    #'        :class:`~sagemaker.session.Session` object associated with the estimator
    #'        for the ``HyperparameterTuner``.
    sagemaker_session = function(){
      estimator = self$estimator
      if (is.null(estimator)){
        first_estimator_name = sort(names(self$estimator_list))[1]
        estimator = self$estimator_list[[first_estimator_name]]}
      return(estimator$sagemaker_session)
    }
  ),
  lock_objects = F
)

