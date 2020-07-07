# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/analytics.py

#' @include session.R
#' @include utils.R

#' @import logger
#' @import R6
#' @import paws
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
                                   self$.dataframe = NULL
                                   return(self$.dataframe)
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
                                 .fetch_dataframe = function(){stop("I'm an abstract interface method", call. = F)}
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
                                               attr(self, "__repr__") = sprintf("<HyperparameterTuningJobAnalytics for %s>", self$name)

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


#' @title TrainingJobAnalytics Class
#' @description Fetch training curve data from CloudWatch Metrics for a specific training
#'              job.
#' @export
TrainingJobAnalytics = R6Class("TrainingJobAnalytics",
                               inherit = AnalyticsMetricsBase,
                               public = list(
                                 #' @field CLOUDWATCH_NAMESPACE
                                 #' class metadata
                                 CLOUDWATCH_NAMESPACE = "/aws/sagemaker/TrainingJobs",

                                 #' @description Initialize a ``TrainingJobAnalytics`` instance.
                                 #' @param training_job_name (str): name of the TrainingJob to analyze.
                                 #' @param metric_names (list, optional): string names of all the metrics to
                                 #'              collect for this training job. If not specified, then it will
                                 #'              use all metric names configured for this job.
                                 #' @param sagemaker_session (sagemaker.session.Session): Session object which
                                 #'              manages interactions with Amazon SageMaker APIs and any other
                                 #'              AWS services needed. If not specified, one is specified using
                                 #'              the default AWS configuration chain.
                                 #' @param start_time :
                                 #' @param end_time :
                                 #' @param period :
                                 initialize = function(training_job_name,
                                                       metric_names=NULL,
                                                       sagemaker_session=NULL,
                                                       start_time=NULL,
                                                       end_time=NULL,
                                                       period=NULL){
                                   self$sagemaker_session = sagemaker_session %||% Session$new()
                                   self$.cloudwatch = cloudwatch(config = self$sagemaker_session$paws_credentials$credentials)
                                   self$.training_job_name = training_job_name
                                   self$.start_time = start_time
                                   self$.end_time = end_time
                                   self$.period = period %||% METRICS_PERIOD_DEFAULT

                                   if (!is.null(metric_names))
                                     self$.metric_names = metric_names
                                   else
                                     self$.metric_names = private$.metric_names_for_training_job()

                                   super$initialize()
                                   self$clear_cache()

                                   attr(self, "__repr__") = sprintf("<sagemaker.TrainingJobAnalytics for %s>", self$name)
                                 },

                                 #' @description Clear the object of all local caches of API methods, so that the next
                                 #'              time any properties are accessed they will be refreshed from the
                                 #'              service.
                                 clear_cache = function(){
                                   super$clear_cache()
                                   self$.data = data.table()
                                   self$.time_interval = private$.determine_timeinterval()
                                 }
                               ),
                               private = list(
                                 # Return a dictionary with two datetime objects, start_time and
                                 # end_time, covering the interval of the training job
                                 .determine_timeinterval=function(){
                                   description = self$sagemaker_session$sagemaker$describe_training_job(TrainingJobName=self$name)
                                   start_time = self$.start_time %||% description$TrainingStartTime  # datetime object
                                   # Incrementing end time by 1 min since CloudWatch drops seconds before finding the logs.
                                   # This results in logs being searched in the time range in which the correct log line was
                                   # not present.
                                   # Example - Log time - 2018-10-22 08:25:55
                                   #       Here calculated end time would also be 2018-10-22 08:25:55 (without 1 min addition)
                                   #       CW will consider end time as 2018-10-22 08:25 and will not be able to search the
                                   #           correct log.
                                   end_time = self$.end_time %||% description$TrainingEndTime + 60 # add 1 minute

                                   return(list("start_time"= start_time, "end_time"= end_time))
                                 },

                                 .fetch_dataframe = function(){
                                   self$.data= lapply(self$.metric_names, private$.fetch_metric)
                                   return(rbindlist(self$.data, fill = T))
                                 },

                                 # Fetch all the values of a named metric, and add them to _data
                                 # Args:
                                 #   metric_name:
                                 .fetch_metric = function(metric_name){
                                   raw_cwm_data = self$.cloudwatch$get_metric_statistics(
                                        Namespace= self$CLOUDWATCH_NAMESPACE,
                                        MetricName= metric_name,
                                        Dimensions= list(list("Name"= "TrainingJobName", "Value"= self$name)),
                                        StartTime= self$.time_interval$start_time,
                                        EndTime= self$.time_interval$end_time,
                                        Period= self$.period,
                                        Statistics= list("Average"))$Datapoints
                                   if (islistempty(raw_cwm_data)){
                                     log_warn("Warning: No metrics called %s found", metric_name)
                                     return(NULL)}
                                   # Process data: normalize to starting time, and sort.
                                   base_time = min(raw_cwm_data$Timestamp)

                                   all_xy= rbindlist(lapply(raw_cwm_data, function(pt)
                                     data.table(timestamp = as.numeric(pt$Timestamp - base_time),
                                                metric_name	= metric_name,
                                                value = pt$Average)),
                                     fill = T)

                                   setorder(all_xy, Timestamp)
                                   return(all_xy)
                                 },

                                 # Helper method to discover the metrics defined for a training job.
                                 .metric_names_for_training_job = function(){
                                   raining_description = self$sagemaker_session$sagemaker$describe_training_job(
                                     TrainingJobName=self$.training_job_name)

                                   metric_definitions = training_description$AlgorithmSpecification$MetricDefinitions
                                   metric_names = lapply(metric_definitions, function(md) md$Name)

                                   return(metric_names)
                                 }

                               ),
                               active = list(

                                 #' @field name
                                 #' Name of the TrainingJob being analyzed
                                 name = function(){
                                  return(self$.training_job_name)
                                 }
                               ),
                               lock_objects = F
)


