# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/debugger/profiler_config.py

#' @include r_utils.R
#' @include debugger_framework_profile.R

#' @import R6
#' @import R6sagemaker.common

#' @title Configuration for collecting system and framework metrics of SageMaker training jobs.
#' @description SageMaker Debugger collects system and framework profiling
#'              information of training jobs and identify performance bottlenecks.
#' @export
ProfilerConfig = R6Class("ProfilerConfig",
  public = list(

    #' @description Initialize a ``ProfilerConfig`` instance.
    #'              Pass the output of this class
    #'              to the ``profiler_config`` parameter of the generic :class:`~sagemaker.estimator.Estimator`
    #'              class and SageMaker Framework estimators.
    #' @param s3_output_path (str): The location in Amazon S3 to store the output.
    #'              The default Debugger output path for profiling data is created under the
    #'              default output path of the :class:`~sagemaker.estimator.Estimator` class.
    #'              For example,
    #'              s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/profiler-output/.
    #' @param system_monitor_interval_millis (int): The time interval in milliseconds
    #'              to collect system metrics. Available values are 100, 200, 500, 1000 (1 second),
    #'              5000 (5 seconds), and 60000 (1 minute) milliseconds.
    #'              The default is 500 milliseconds.
    #' @param framework_profile_params (:class:`~sagemaker.debugger.FrameworkProfile`):
    #'              A parameter object for framework metrics profiling. Configure it using
    #'              the :class:`~sagemaker.debugger.FrameworkProfile` class.
    #'              To use the default framework profile parameters, pass ``FrameworkProfile()``.
    #'              For more information about the default values,
    #'              see :class:`~sagemaker.debugger.FrameworkProfile`.
    #' @examples
    #' # The following example shows the basic ``profiler_config``
    #' # parameter configuration, enabling system monitoring every 5000 milliseconds
    #' # and framework profiling with default parameter values.
    #' library(R6sagemaker)
    #'
    #' profiler_config = ProfilerConfig$new(
    #'       system_monitor_interval_millis = 5000,
    #'       framework_profile_params = FrameworkProfile$new()
    #'  )
    initialize = function(s3_output_path=NULL,
                          system_monitor_interval_millis=NULL,
                          framework_profile_params=NULL){
      if(!is.null(framework_profile_params) && !inherits(framework_profile_params, "FrameworkProfile"))
        ValueError$new("framework_profile_params must be of type FrameworkProfile if specified.")


      self$s3_output_path = s3_output_path
      self$system_monitor_interval_millis = system_monitor_interval_millis
      self$framework_profile_params = framework_profile_params
    },

    #' @description Generate a request dictionary using the parameters provided when initializing the object.
    #' @return dict: An portion of an API request as a dictionary.
    to_request_list = function(){
      profiler_config_request = list()
      profiler_config_request[["S3OutputPath"]] = self$s3_output_path
      profiler_config_request[[
        "ProfilingIntervalInMilliseconds"
        ]] = self$system_monitor_interval_millis
      profiler_config_request[[
          "ProfilingParameters"
          ]] = self$framework_profile_params.profiling_parameters
      return(profiler_config_request)
    },

    #' @description format class
    format = function(){
      return(format_class(self))
    }
  ),
  private = list(

    # Generate a request dictionary for updating the training job to disable profiler.
    # Returns:
    #   dict: An portion of an API request as a dictionary.
    .to_profiler_disabled_request_dict = function(){
      profiler_config_request = list("DisableProfiler"= TRUE)
      return(profiler_config_request)
    }
  ),
  lock_objects = F
)
