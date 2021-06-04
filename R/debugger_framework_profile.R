# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/debugger/framework_profile.py

#' @include r_utils.R
#' @include debugger_metrics_config.R
#' @include debugger_profiler_constants.R
#' @include debugger_utils.R

#' @import R6
#' @import R6sagemaker.common

ALL_METRIC_CONFIGS <- list(
  DetailedProfilingConfig,
  DataloaderProfilingConfig,
  PythonProfilingConfig,
  HorovodProfilingConfig,
  SMDataParallelProfilingConfig)

#' @title Sets up the profiling configuration for framework metrics.
#' @description Validates user inputs and fills in default values if no input is provided.
#'              There are three main profiling options to choose from:
#'              :class:`~sagemaker.debugger.metrics_config.DetailedProfilingConfig`,
#'              :class:`~sagemaker.debugger.metrics_config.DataloaderProfilingConfig`, and
#'              :class:`~sagemaker.debugger.metrics_config.PythonProfilingConfig`.
#'              The following list shows available scenarios of configuring the profiling options.
#'              \enumerate{
#'              \item None of the profiling configuration, step range, or time range is specified.
#'              SageMaker Debugger activates framework profiling based on the default settings
#'              of each profiling option.
#'              \item Target step or time range is specified to
#'              this :class:`~sagemaker.debugger.metrics_config.FrameworkProfile` class.
#'              The requested target step or time range setting propagates to all of
#'              the framework profiling options.
#'              For example, if you configure this class as following, all of the profiling options
#'              profiles the 6th step.
#'              \item Individual profiling configurations are specified through
#'              the ``*_profiling_config`` parameters.
#'              SageMaker Debugger profiles framework metrics only for the specified profiling configurations.
#'              For example, if the :class:`~sagemaker.debugger.metrics_config.DetailedProfilingConfig` class
#'              is configured but not the other profiling options, Debugger only profiles based on the settings
#'              specified to the
#'              :class:`~sagemaker.debugger.metrics_config.DetailedProfilingConfig` class.
#'              For example, the following example shows a profiling configuration to perform
#'              detailed profiling at step 10, data loader profiling at step 9 and 10,
#'              and Python profiling at step 12.
#'              If the individual profiling configurations are specified in addition to
#'              the step or time range,
#'              SageMaker Debugger prioritizes the individual profiling configurations and ignores
#'              the step or time range. For example, in the following code,
#'              the ``start_step=1`` and ``num_steps=10`` will be ignored.
#'              }
#' @examples
#' library(R6sagemaker)
#' profiler_config=ProfilerConfig$new(
#'        framework_profile_params=FrameworkProfile$new(
#'        start_step=1,
#'        num_steps=10,
#'        detailed_profiling_config=DetailedProfilingConfig$new(start_step=10, num_steps=1),
#'        dataloader_profiling_config=DataloaderProfilingConfig$new(start_step=9, num_steps=2),
#'        python_profiling_config=PythonProfilingConfig$new(start_step=12, num_steps=1)
#'      )
#'    )
#' @export
FrameworkProfile = R6Class("FrameworkProfile",
  public = list(

    #' @description Initialize the FrameworkProfile class object.
    #' @param local_path (str):
    #' @param file_max_size (int):
    #' @param file_close_interval (int):
    #' @param file_open_fail_threshold (int):
    #' @param detailed_profiling_config (DetailedProfilingConfig): The configuration for detailed
    #'              profiling. Configure it using the
    #'              :class:`~sagemaker.debugger.metrics_config.DetailedProfilingConfig` class.
    #'              Pass ``DetailedProfilingConfig()`` to use the default configuration.
    #' @param dataloader_profiling_config (DataloaderProfilingConfig): The configuration for
    #'              dataloader metrics profiling. Configure it using the
    #'              :class:`~sagemaker.debugger.metrics_config.DataloaderProfilingConfig` class.
    #'              Pass ``DataloaderProfilingConfig()`` to use the default configuration.
    #' @param python_profiling_config (PythonProfilingConfig): The configuration for stats
    #'              collected by the Python profiler (cProfile or Pyinstrument).
    #'              Configure it using the
    #'              :class:`~sagemaker.debugger.metrics_config.PythonProfilingConfig` class.
    #'              Pass ``PythonProfilingConfig()`` to use the default configuration.
    #' @param horovod_profiling_config :
    #' @param smdataparallel_profiling_config :
    #' @param start_step (int): The step at which to start profiling.
    #' @param num_steps (int): The number of steps to profile.
    #' @param start_unix_time (int): The Unix time at which to start profiling.
    #' @param duration (float): The duration in seconds to profile.
    intialize = function(local_path=BASE_FOLDER_DEFAULT,
                         file_max_size=MAX_FILE_SIZE_DEFAULT,
                         file_close_interval=CLOSE_FILE_INTERVAL_DEFAULT,
                         file_open_fail_threshold=FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
                         detailed_profiling_config=NULL,
                         dataloader_profiling_config=NULL,
                         python_profiling_config=NULL,
                         horovod_profiling_config=NULL,
                         smdataparallel_profiling_config=NULL,
                         start_step=NULL,
                         num_steps=NULL,
                         start_unix_time=NULL,
                         duration=NULL){
      self$profiling_parameters = list()
      private$.use_default_metrics_configs = FALSE
      private$.use_one_config_for_all_metrics = FALSE
      private$.use_custom_metrics_configs = FALSE

      private$.process_trace_file_parameters(
        local_path, file_max_size, file_close_interval, file_open_fail_threshold
      )
      use_custom_metrics_configs = private$.process_metrics_configs(
        detailed_profiling_config,
        dataloader_profiling_config,
        python_profiling_config,
        horovod_profiling_config,
        smdataparallel_profiling_config
      )

      use_one_config_for_all_metrics = (
        if (is.null(use_custom_metrics_configs))
          private$.process_range_fields(start_step, num_steps, start_unix_time, duration)
        else
          FALSE
      )

      if (is.null(use_custom_metrics_configs) && is.null(use_one_config_for_all_metrics))
        private$.create_default_metrics_configs()
    },

    #' @description format class
    #' @param ... (ignored).
    format = function(...){
      return(format_class(self))
    }
  ),
  private = list(
    .use_default_metrics_configs = NULL,
    .use_one_config_for_all_metrics = NULL,
    .use_custom_metrics_configs = NULL,

    # Helper function to validate and set the provided trace file parameters.
    # Args:
    #   local_path (str): The path where profiler events have to be saved.
    # file_max_size (int): Max size a trace file can be, before being rotated.
    # file_close_interval (float): Interval in seconds from the last close, before being
    # rotated.
    # file_open_fail_threshold (int): Number of times to attempt to open a trace fail before
    # marking the writer as unhealthy.
    .process_trace_file_parameters = function(local_path,
                                              file_max_size,
                                              file_close_interval,
                                              file_open_fail_threshold){
      if(!is.character(local_path))
        stop(ErrorMessages$public_fields$INVALID_LOCAL_PATH, call. = F)

      if(!is.integer(file_max_size))
        stop(ErrorMessages$public_fields$INVALID_FILE_MAX_SIZE, call. = F)

      if(!inherits(file_max_size, c("double", "integer")) && file_close_interval > 0)
        stop(ErrorMessages$public_fields$INVALID_FILE_CLOSE_INTERVAL, call. = F)

      if(!inherits(file_max_size, "integer") && file_open_fail_threshold > 0)
        stop(ErrorMessages$public_fields$INVALID_FILE_OPEN_FAIL_THRESHOLD, call. = F)

      self$profiling_parameters[["LocalPath"]] = local_path
      self$profiling_parameters[["RotateMaxFileSizeInBytes"]] = as.character(file_max_size)
      self$profiling_parameters[["RotateFileCloseIntervalInSeconds"]] = as.character(file_close_interval)
      self$profiling_parameters[["FileOpenFailThreshold"]] = as.character(file_open_fail_threshold)
    },

    # Helper function to validate and set the provided metrics_configs.
    # In this case,
    # the user specifies configurations for the metrics they want to profile.
    # Profiling does not occur
    # for metrics if the configurations are not specified for them.
    # Args:
    #   metrics_configs: The list of metrics configs specified by the user.
    # Returns:
    #   bool: Indicates whether custom metrics configs will be used for profiling.
    .process_metrics_configs = function(...){
      metrics_configs = list(...)

      metrics_configs = Filter(Negate(is.null), metrics_configs)
      if (islistempty(metrics_configs))
        return(FALSE)

      for(config in metrics_configs){
        config_name = config$name
        config_json = config$to_json_string()
        self$profiling_parameters[[config_name]] = config_json
      }
      return(TRUE)
    },

    # Helper function to validate and set the provided range fields.
    # Profiling occurs
    # for all of the metrics using these fields as the specified range and default parameters
    # for the rest of the configuration fields (if necessary).
    # Args:
    #   start_step (int): The step at which to start profiling.
    # num_steps (int): The number of steps to profile.
    # start_unix_time (int): The UNIX time at which to start profiling.
    # duration (float): The duration in seconds to profile.
    # Returns:
    #   bool: Indicates whether a custom step or time range will be used for profiling.
    .process_range_fields = function(start_step = NULL,
                                     num_steps = NULL,
                                     start_unix_time = NULL,
                                     duration = NULL){

      if(is.null(start_step) && is.null(num_steps) && is.null(start_unix_time) && is.null(duration)){
        return(FALSE)
      }

      for (config_class in ALL_METRIC_CONFIGS){
        config = config_class$new(
          start_step=start_step,
          num_steps=num_steps,
          start_unix_time=start_unix_time,
          duration=duration)
        config_name = config$name
        config_json = config$to_json_string()
        self$profiling_parameters[[config_name]] = config_json
      }
      return(TRUE)
    },

    # Helper function for creating the default configs for each set of metrics.
    .create_default_metrics_configs = function(){
      for (config_class in ALL_METRIC_CONFIGS){
        config = config_class$new(profile_default_steps=True)
        config_name = config$name
        config_json = config$to_json_string()
        self$profiling_parameters[[config_name]] = config_json
      }
    }
  ),
  lock_objects = F
)
