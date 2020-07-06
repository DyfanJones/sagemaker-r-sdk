# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/analytics.py

#' @include session.R
#' @include utils.R

#' @import logger
#' @import R6
#' @import data.table


METRICS_PERIOD_DEFAULT = 60  # seconds

#' @title AnalyticsMetricsBase Class
#' @description Base class for tuning job or training job analytics classes. Understands
#'              common functionality like persistence and caching.
#' @export
AnalyticsMetricsBase = R6Class("AnalyticsMetricsBase",
                               public = list(

                                 #' @description Initialize a ``AnalyticsMetricsBase`` instance.
                                 initialize = function(){
                                   sefl$.dataframe = NULL
                                 },

                                 #' @description ersists the analytics dataframe to a file.
                                 #' @param filename (str): The name of the file to save to.
                                 export_csv = function(filename){
                                   fwrite(self$dataframe(), filename)
                                 },

                                 #' @description A dataframe with lots of interesting results about this
                                 #'              object. Created by calling SageMaker List and Describe APIs and
                                 #'              converting them into a convenient tabular summary.
                                 #' @param force_refresh (bool): Set to True to fetch the latest data from
                                 #'              SageMaker API.
                                 dataframe = function(force_refresh = FALSE){
                                   if(force_refresh)
                                     self$clear_cache()
                                   if (is.null(self$.dataframe))
                                     self$.dataframe = private$.fetch_dataframe()
                                     return(self$.dataframe)
                                 },

                                 #' @description Clear the object of all local caches of API methods, so that the next
                                 #'              time any properties are accessed they will be refreshed from the
                                 #'              service.
                                 clear_cache = function(){
                                   return(self$.dataframe = NULL)
                                 },

                                 #' @description
                                 #' Printer.
                                 #' @param ... (ignored).
                                 print = function(...){
                                   cat("<AnalyticsMetricsBase>")
                                   invisible(self)
                                 }
                               ),
                               private = list(
                                 # Sub-class must calculate the dataframe and return it.
                                 .fetch_dataframe = function(){stop("I'm an abstract interface method", call. = F)},
                               ),
                               lock_objects = F
)

#' @title HyperparameterTuningJobAnalytics Class
#' @description Fetch results about a hyperparameter tuning job and make them accessible
#'              for analytics.
#' @export
HyperparameterTuningJobAnalytics = R6Class("HyperparameterTuningJobAnalytics",
                                           inherit = AnalyticsMetricsBase,
                                           public = list(

                                             #' @description Initialize a ``HyperparameterTuningJobAnalytics`` instance.
                                             #' @param hyperparameter_tuning_job_name (str): name of the
                                             #'              HyperparameterTuningJob to analyze.
                                             #' @param sagemaker_session (sagemaker.session.Session): Session object which
                                             #'              manages interactions with Amazon SageMaker APIs and any other
                                             #'              AWS services needed. If not specified, one is created using the
                                             #'              default AWS configuration chain.
                                             initialize = function(hyperparameter_tuning_job_name,
                                                                   sagemaker_session=NULL){
                                               self$sagemaker_session = sagemaker_session %||% Session$new()
                                               self$.tuning_job_name = hyperparameter_tuning_job_name
                                               self$.tuning_job_describe_result = NULL
                                               self$.training_job_summaries = NULL
                                               super$initialize()
                                               self$clear_cache()
                                             },

                                             #' @description Clear the object of all local caches of API methods.
                                             clear_cache = function(){
                                               super$clear_cache()
                                               self$.tuning_job_describe_result = NULL
                                               self$.training_job_summaries = NULL
                                             },

                                             #' @description A dictionary describing the ranges of all tuned hyperparameters. The
                                             #'              keys are the names of the hyperparameter, and the values are the ranges.
                                             #'              The output can take one of two forms:
                                             #'              * If the 'TrainingJobDefinition' field is present in the job description, the output
                                             #'              is a dictionary constructed from 'ParameterRanges' in
                                             #'              'HyperParameterTuningJobConfig' of the job description. The keys are the
                                             #'              parameter names, while the values are the parameter ranges.
                                             #'              Example:
                                             #'              >>> {
                                             #'              >>>     "eta": {"MaxValue": "1", "MinValue": "0", "Name": "eta"},
                                             #'              >>>     "gamma": {"MaxValue": "10", "MinValue": "0", "Name": "gamma"},
                                             #'              >>>     "iterations": {"MaxValue": "100", "MinValue": "50", "Name": "iterations"},
                                             #'              >>>     "num_layers": {"MaxValue": "30", "MinValue": "5", "Name": "num_layers"},
                                             #'              >>> }
                                             #'              * If the 'TrainingJobDefinitions' field (list) is present in the job description,
                                             #'              the output is a dictionary with keys as the 'DefinitionName' values from
                                             #'              all items in 'TrainingJobDefinitions', and each value would be a dictionary
                                             #'              constructed from 'HyperParameterRanges' in each item in 'TrainingJobDefinitions'
                                             #'              in the same format as above
                                             #'              Example:
                                             #'              >>> {
                                             #'              >>>     "estimator_1": {
                                             #'              >>>         "eta": {"MaxValue": "1", "MinValue": "0", "Name": "eta"},
                                             #'              >>>         "gamma": {"MaxValue": "10", "MinValue": "0", "Name": "gamma"},
                                             #'              >>>     },
                                             #'              >>>     "estimator_2": {
                                             #'              >>>         "framework": {"Values": ["TF", "MXNet"], "Name": "framework"},
                                             #'              >>>         "gamma": {"MaxValue": "1.0", "MinValue": "0.2", "Name": "gamma"}
                                             #'              >>>     }
                                             #'              >>> }
                                             #'              For more details about the 'TrainingJobDefinition' and 'TrainingJobDefinitions' fields
                                             #'              in job description, see
                                             #'              https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_hyper_parameter_tuning_job
                                             tuning_ranges = function(){
                                               description = self$description()

                                               if(!islistempty(description$TrainingJobDefinition))
                                                 return(private$.prepare_parameter_ranges(
                                                   description$HyperParameterTuningJobConfig$ParameterRanges))

                                               output = lapply(description$TrainingJobDefinitions,
                                                               function(training_job_definition)
                                                                 private$.prepare_parameter_ranges(
                                                                   training_job_definition$HyperParameterRanges))
                                               names(output) = sapply(description$TrainingJobDefinitions, function(x) x$DefinitionName)
                                               return (output)
                                             },

                                             #' @description Call ``DescribeHyperParameterTuningJob`` for the hyperparameter
                                             #'              tuning job.
                                             #' @param force_refresh (bool): Set to True to fetch the latest data from
                                             #'              SageMaker API.
                                             #' @return dict: The Amazon SageMaker response for
                                             #'              ``DescribeHyperParameterTuningJob``.
                                             description = function(force_refresh=FALSE){
                                               if (force_refresh)
                                                 self$clear_cache()
                                               if(!islistempty(self$.tuning_job_describe_result))
                                                 self$.tuning_job_describe_result = self$sagemaker_session$sagemaker$describe_hyper_parameter_tuning_job(
                                                   HyperParameterTuningJobName=self$name)
                                               return (self._tuning_job_describe_result)
                                             },

                                             #' @description A (paginated) list of everything from
                                             #'              ``ListTrainingJobsForTuningJob``.
                                             #' @param force_refresh (bool): Set to True to fetch the latest data from
                                             #'              SageMaker API.
                                             #' @return dict: The Amazon SageMaker response for
                                             #'              ``ListTrainingJobsForTuningJob``.
                                             training_job_summaries = function(force_refresh=FALSE){
                                               if (force_refresh)
                                                 self$clear_cache()
                                               if (!is.null(self$.training_job_summaries))
                                                 return(self$.training_job_summaries)
                                               output = list()
                                               next_args = list()

                                               next_args = list(HyperParameterTuningJobName='xgboost-200706-1050', MaxResults=100)
                                               for (count in 1:100){
                                                 raw_result = self$sagemaker_session$sagemaker$list_training_jobs_for_hyper_parameter_tuning_job(
                                                   HyperParameterTuningJobName = next_args$HyperParameterTuningJobName,
                                                   MaxResults = next_args$MaxResults,
                                                   NextToken = next_args$NextToken)
                                                 new_output = raw_result$TrainingJobSummaries
                                                 output = c(output, new_output)
                                                 log_debug("Got %d more TrainingJobs. Total so far: %d",
                                                           length(new_output), length(output))
                                                 if (!length(raw_result$NextToken) == 0 && length(new_output) > 0)
                                                   next_args$NextToken = raw_result$NextToken
                                                 else
                                                   break
                                               }
                                               self$.training_job_summaries = output
                                               return (output)
                                             }
                                           ),
                                           private = list(
                                             ..repr..=function(){
                                                return (sprintf("<HyperparameterTuningJobAnalytics for %s>",self$name))
                                             },

                                             .fetch_dataframe = function(){
                                               # Run that helper over all the summaries.
                                               df = rbindlist(lapply(training_summary, function(training_summary){
                                                 end_time = training_summary$TrainingEndTime
                                                 start_time = training_summary$TrainingStartTime
                                                 diff_time = if(length(start_time) == 0 || length(start_time) == 0) NULL else as.numeric(end_time - start_time)
                                                 def_name = if(length(training_summary$TrainingJobDefinitionName) == 0) NA else training_summary$TrainingJobDefinitionName
                                                 output = data.table(TrainingJobName = training_summary$TrainingJobName,
                                                                     TrainingJobStatus = training_summary$TrainingJobStatus,
                                                                     FinalObjectiveValue = training_summary$FinalHyperParameterTuningJobObjectiveMetric$Value,
                                                                     TrainingStartTime = if(length(start_time) == 0) NULL else start_time,
                                                                     TrainingEndTime = if(length(start_time) == 0) NULL else start_time,
                                                                     TrainingElapsedTimeSeconds = diff_time,
                                                                     TrainingJobDefinitionName = def_name)
                                                 for (i in seq_along(training_summary$TunedHyperParameters)){
                                                   k = names(training_summary$TunedHyperParameters)[i]
                                                   v = as.numeric(training_summary$TunedHyperParameters[[i]])
                                                   output[[k]] = v}
                                                 output}), fill = TRUE)
                                               return(df)
                                             }
                                           ),
                                           active = list(
                                             #' @field name
                                             #' Name of the HyperparameterTuningJob being analyzed
                                             name = function(){
                                               return(self$.tuning_job_name)
                                             }
                                           ),
                                           lock_objects = F
)

