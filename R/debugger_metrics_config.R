# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/debugger/metrics_config.py

#' @include r_utils.R
#' @include debugger_profiler_constants.R
#' @include debugger_utils.R

#' @import R6
#' @import R6sagemaker.common
#' @importFrom jsonlite toJSON

#' @title Configuration for the range of steps to profile.
#' @description It returns the target steps in dictionary format that you can pass to the
#'              :class:`~sagemaker.debugger.FrameworkProfile` class.
#' @export
StepRange = R6Class("StepRange",
  public = list(

    #' @description Set the start step and num steps.
    #'              If the start step is not specified,
    #'              Debugger starts profiling
    #'              at step 0. If num steps is not specified, profile for 1 step.
    #' @param start_step (int): The step to start profiling.
    #' @param num_steps (int): The number of steps to profile.
    initialize = function(start_step,
                          num_steps){
      if (is.null(start_step)){
        start_step = START_STEP_DEFAULT
      } else if (is.null(num_steps)){
        num_steps = PROFILING_NUM_STEPS_DEFAULT
      }
      self$start_step = start_step
      self$num_steps = num_steps
    },

    #' @description Convert the step range into a dictionary.
    #' @return list: The step range as a dictionary.
    to_json = function(){
      return(list("StartStep" = self$start_step, "NumSteps"=self$num_steps))
    },

    #' @description format class
    format = function(){
      return(format_class(self))
    }
  ),
  lock_objects = F
)


#' @title Configuration for the range of Unix time to profile.
#' @description It returns the target time duration in dictionary format that you can pass to the
#'              :class:`~sagemaker.debugger.FrameworkProfile` class.
#' @export
TimeRange = R6Class("TimeRange",
  public = list(

    #' @description Set the start Unix time and duration.
    #'              If the start Unix time is not specified,
    #'              profile starting at step 0. If the duration is not specified, profile for 1 step.
    #' @param start_unix_time (int): The Unix time to start profiling.
    #' @param duration (float): The duration in seconds to profile.
    initialize = function(start_unix_time = NULL,
                          duration = NULL){
      self$start_unix_time = start_unix_time
      self$duration = duration
    },

    #' @description Convert the time range into a dictionary.
    #' @return dict: The time range as a dictionary.
    to_json = function(){
      time_range_json = list()
      time_range_json[["StartTimeInSecSinceEpoch"]] = self$start_unix_time
      time_range_json[["Duration"]] = self$duration
      return(time_range_json)
    },

    #' @description format class
    #' @param ... (ignored).
    format = function(){
      return(format_class(self))
    }
  ),
  lock_object = F
)

#' @title The base class for the metrics configuration.
#' @description It determines the step or time range that needs to be
#'              profiled and validates the input value pairs. Available profiling range parameter pairs are
#'              (\code{start_step} and \code{num_steps}) and (\code{start_unix_time} and \code{duration}).
#'              The two parameter pairs are mutually exclusive, and this class validates
#'              if one of the two pairs is used. If both pairs are specified, a
#'              FOUND_BOTH_STEP_AND_TIME_FIELDS error occurs.
#' @export
MetricsConfigBase = R6Class("MetricsConfigBase",
  public = list(

    #' @description Validate the provided range fields and set the range to be profiled accordingly.
    #' @param name (str): The name of the metrics config.
    #' @param start_step (int): The step to start profiling.
    #' @param num_steps (int): The number of steps to profile.
    #' @param start_unix_time (int): The Unix time to start profiling.
    #' @param duration (float): The duration in seconds to profile.
    initialize = function(name,
                          start_step=NULL,
                          num_steps=NULL,
                          start_unix_time=NULL,
                          duration=NULL){
      self$name = name
      start_unix_time = if(!is.null(start_unix_time)) as.integer(start_unix_time) else start_unix_time

      if(!is.null(start_step)){
        if(!is.integer(start_step) && start_step < 0)
          stop(ErrorMessages$INVALID_START_STEP, call. = F)
      }
      if(!is.null(start_step)){
        if(!is.integer(start_step) && num_steps <= 0)
          stop(ErrorMessages$INVALID_NUM_STEPS, call. = F)
      }
      if(!is.null(start_unix_time) && !inherits(start_unix_time, c("integer","numeric")))
        stop(ErrorMessages$INVALID_START_UNIX_TIME, call. = F)

      if(!is.null(duration)){
        if(!inherits(duration, c("numeric", "integer")) || duration < 0)
          stop(ErrorMessages$INVALID_DURATION, call. = F)
      }
      has_step_range = (!is.null(start_step) || !is.null(num_steps))
      has_time_range = (!is.null(start_unix_time) || !is.null(duration))

      if(has_step_range && has_time_range)
        stop(ErrorMessages$FOUND_BOTH_STEP_AND_TIME_FIELDS, call. = F)

      self$range = (
        if(has_step_range)
          StepRange$new(start_step, num_steps)
        else TimeRange$new(start_unix_time, duration)
      )
    },

    #' @description Convert this metrics configuration to dictionary formatted as a string.
    #'              Calling eval on the
    #'              return value is the same as calling _to_json directly.
    #' @return str: This metrics configuration as a dictionary and formatted as a string.
    to_json_string = function(){
      return(toJSON(private$.to_json(), auto_unbox = T))
    },

    #' @description format class
    format = function(){
      return(format_class(self))
    }
  ),
  private = list(

    # Convert the metrics configuration to a dictionary.
    # Convert the range object into a
    # dictionary.
    # Returns:
    #   dict: This metrics config as a dictionary.
    .to_json = function(){
      return (self$range$to_json())
    }
  ),
  lock_objects = F
)

#' @title DetailedProfilingConfig Class
#' @description The configuration for framework metrics to be collected for detailed profiling.
#' @export
DetailedProfilingConfig = R6Class("DetailedProfilingConfig",
  inherit = MetricsConfigBase,
  public = list(

    #' @description Specify target steps or a target duration to profile.
    #'              By default, it profiles step 5
    #'              of training.
    #'              If \code{profile_default_steps} is set to `True` and none of the other
    #'              range parameters is specified,
    #'              the class uses the default configuration for detailed profiling.
    #' @param start_step (int): The step to start profiling. The default is step 5.
    #' @param num_steps (int): The number of steps to profile. The default is for 1 step.
    #' @param start_unix_time (int): The Unix time to start profiling.
    #' @param duration (float): The duration in seconds to profile.
    #' @param profile_default_steps (bool): Indicates whether the default config should be used.
    initialize = function(start_step=NULL,
                          num_steps=NULL,
                          start_unix_time=NULL,
                          duration=NULL,
                          profile_default_steps=FALSE){
      if(!inherits(profile_default_steps, "logical"))
        stop(ErrorMessages$INVALID_PROFILE_DEFAULT_STEPS, call. = F)

      if (profile_default_steps || (is.null(start_step) && is.null(num_steps) && is.null(start_unix_time) && is.null(duration))){
        start_step = DETAILED_PROFILING_START_STEP_DEFAULT
        num_steps = PROFILING_NUM_STEPS_DEFAULT
      }
      super$initialize(
        DETAILED_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
      )
    }
  ),
  lock_objects = F
)

#' @title DataloaderProfilingConfig Class
#' @description The configuration for framework metrics to be collected for data loader profiling.
#' @export
DataloaderProfilingConfig = R6Class("DataloaderProfilingConfig",
  inherit = MetricsConfigBase,
  public = list(

    #' @description Specify target steps or a target duration to profile.
    #'              By default, it profiles step 7 of
    #'              training. If \code{profile_default_steps} is set to `True` and none of the other
    #'              range parameters is specified,
    #'              the class uses the default config for dataloader profiling.
    #' @param start_step (int): The step to start profiling. The default is step 7.
    #' @param num_steps (int): The number of steps to profile. The default is for 1 step.
    #' @param start_unix_time (int): The Unix time to start profiling. The default is for 1 step.
    #' @param duration (float): The duration in seconds to profile.
    #' @param profile_default_steps (bool): Indicates whether the default config should be used.
    #' @param metrics_regex (str): Regex pattern
    initialize = function(start_step=NULL,
                          num_steps=NULL,
                          start_unix_time=NULL,
                          duration=NULL,
                          profile_default_steps=FALSE,
                          metrics_regex=".*"){

      if(!inherits(profile_default_steps, "logical"))
        stop(ErrorMessages$INVALID_PROFILE_DEFAULT_STEPS, call. = F)

      if (profile_default_steps || (is.null(start_step) && is.null(num_steps) && is.null(start_unix_time) && is.null(duration))){
        start_step = DETAILED_PROFILING_START_STEP_DEFAULT
        num_steps = PROFILING_NUM_STEPS_DEFAULT
      }

      super$initialize(
        DATALOADER_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
      )

      if(!is_valid_regex(metrics_regex)) stop(ErrorMessages$INVALID_METRICS_REGEX, call. = F)

      self$metrics_regex = metrics_regex
    }
  ),

  private = list(

    # Convert the dataloader profiling config to a dictionary.
    # Build off of the base metrics
    # configuration dictionary to add the metrics regex.
    # Returns:
    #   dict: The dataloader that profiles the configuration as a dictionary.
    .to_json = function(){
      dataloader_profiling_config = super$.to_json()
      dataloader_profiling_config[["MetricsRegex"]] = self$metrics_regex
      return(dataloader_profiling_config)
    }
  ),
  lock_objects = F
)

#' @title PythonProfilingConfig Class
#' @description The configuration for framework metrics to be collected for Python profiling.
#' @export
PythonProfilingConfig = R6Class("PythonProfilingConfig",
  inherit = MetricsConfigBase,
  public = list(

    #' @description Choose a Python profiler: cProfile or Pyinstrument.
    #'              Specify target steps or a target duration to profile.
    #'              If no parameter is specified,
    #'              it profiles based on profiling configurations
    #'              preset by the \code{profile_default_steps} parameter,
    #'              which is set to `True` by default.
    #'              If you specify the following parameters,
    #'              then the \code{profile_default_steps} parameter
    #'              will be ignored.
    #' @param start_step (int): The step to start profiling. The default is step 9.
    #' @param num_steps (int): The number of steps to profile. The default is for 3 steps.
    #' @param start_unix_time (int): The Unix time to start profiling.
    #' @param duration (float): The duration in seconds to profile.
    #' @param profile_default_steps (bool): Indicates whether the default configuration
    #'              should be used. If set to `True`, Python profiling will be done
    #'              at step 9, 10, and 11 of training, using cProfiler
    #'              and collecting metrics based on the total time, cpu time,
    #'              and off cpu time for these three steps respectively.
    #'              The default is ``True``.
    #' @param python_profiler (PythonProfiler): The Python profiler to use to collect
    #'              python profiling stats. Available options are ``"cProfile"``
    #'              and ``"Pyinstrument"``. The default is ``"cProfile"``.
    #'              Instead of passing the string values, you can also use the enumerator util,
    #'              :class:`~sagemaker.debugger.utils.PythonProfiler`,
    #'              to choose one of the available options.
    #' @param cprofile_timer (cProfileTimer): The timer to be used by cProfile when collecting
    #'              python profiling stats. Available options are ``"total_time"``, ``"cpu_time"``,
    #'              and ``"off_cpu_time"``. The default is ``"total_time"``.
    #'              If you choose Pyinstrument, this parameter is ignored.
    #'              Instead of passing the string values, you can also use the enumerator util,
    #'              :class:`~sagemaker.debugger.utils.cProfileTimer`,
    #'              to choose one of the available options.
    initialize = function(start_step=NULL,
                         num_steps=NULL,
                         start_unix_time=NULL,
                         duration=NULL,
                         profile_default_steps=FALSE,
                         python_profiler=PythonProfiler$CPROFILE,
                         cprofile_timer=cProfileTimer$TOTAL_TIME){

      if(!inherits(profile_default_steps, "logical"))
        stop(ErrorMessages$INVALID_PROFILE_DEFAULT_STEPS, call. = F)

      if (profile_default_steps || (is.null(start_step) && is.null(num_steps) && is.null(start_unix_time) && is.null(duration))){
        start_step = DETAILED_PROFILING_START_STEP_DEFAULT
        num_steps = PROFILING_NUM_STEPS_DEFAULT
      }

      if (profile_default_steps)
        cprofile_timer = cProfileTimer$DEFAULT

      super$initialize(
        PYTHON_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
        )

      if(!(python_profiler %in% unlist(as.list(PythonProfiler))))
        stop(ErrorMessages$INVALID_PYTHON_PROFILER, call. = F)

      if(!(cprofile_timer %in% unlist(as.list(cProfileTimer))))
        stop(ErrorMessages$INVALID_CPROFILE_TIMER, call. = F)

      self$python_profiler = python_profiler

      # The cprofile timer can only be used when the python profiler is cProfile.
      if (python_profiler == PythonProfiler$PYINSTRUMENT)
        self$cprofile_timer = NULL
      else
        self$cprofile_timer = cprofile_timer
    }
  ),
  private = list(

    # Convert the Python profiling config to a dictionary.
    # Build off of the base metrics configuration
    # dictionary to add the Python profiler and cProfile timer.
    # Returns:
    #   dict: The python profiling config as a dictionary.
    .to_json = function(){
      python_profiling_config = super$.to_json()
      python_profiling_config[["ProfilerName"]] = which(self$python_profiler %in% names(PythonProfiler))
      if(!is.null(self$cprofile_timer))
        python_profiling_config[["cProfileTimer"]] = which(self$cprofile_timer %in% names(cProfileTimer))
      return(python_profiling_config)
    }
  ),
  lock_objects = F
)

#' @title HorovodProfilingConfig Class
#' @description The configuration for framework metrics from Horovod distributed training.
#' @export
HorovodProfilingConfig = R6Class("HorovodProfilingConfig",
  inherit = MetricsConfigBase,
  public = list(

    #' @description Specify target steps or a target duration to profile.
    #'              By default, it profiles step 13 of training.
    #'              If \code{profile_default_steps} is set to `True` and none of the other range
    #'              parameters is specified,
    #'              the class uses the default config for horovod profiling.
    #' @param start_step (int): The step to start profiling. The default is step 13.
    #' @param num_steps (int): The number of steps to profile. The default is for 1 steps.
    #' @param start_unix_time (int): The Unix time to start profiling.
    #' @param duration (float): The duration in seconds to profile.
    #' @param profile_default_steps (bool): Indicates whether the default config should be used.
    initialize = function(start_step=NULL,
                          num_steps=NULL,
                          start_unix_time=NULL,
                          duration=NULL,
                          profile_default_steps=FALSE){
      if(!inherits(profile_default_steps, "logical"))
        stop(ErrorMessages$INVALID_PROFILE_DEFAULT_STEPS, call. = F)

      if (profile_default_steps ||
          (is.null(start_step) && is.null(num_steps) && is.null(start_unix_time) && is.null(duration))){
        start_step = DETAILED_PROFILING_START_STEP_DEFAULT
        num_steps = PROFILING_NUM_STEPS_DEFAULT
      }

      super$intialize(
        HOROVOD_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
      )
    }
  ),
  lock_objects = F
)

#' @title SMDataParallelProfilingConfig Class
#' @description Configuration for framework metrics collected from a SageMaker Distributed training job.
#' @export
SMDataParallelProfilingConfig = R6Class("SMDataParallelProfilingConfig",
  inherit = MetricsConfigBase,
  public = list(

    #' @description Specify target steps or a target duration to profile.
    #'              By default, it profiles step 15 of training.
    #'              If \code{profile_default_steps} is set to `True` and none of the other
    #'              range parameters is specified,
    #'              the class uses the default configuration for SageMaker Distributed profiling.
    #' @param start_step (int): The step to start profiling. The default is step 15.
    #' @param num_steps (int): The number of steps to profile. The default is for 1 steps.
    #' @param start_unix_time (int): The Unix time to start profiling.
    #' @param duration (float): The duration in seconds to profile.
    #' @param profile_default_steps (bool): Indicates whether the default configuration
    #'              should be used.
    initialize = function(start_step=NULL,
                          num_steps=NULL,
                          start_unix_time=NULL,
                          duration=NULL,
                          profile_default_steps=FALSE){
      if(!inherits(profile_default_steps, "logical"))
        stop(ErrorMessages$INVALID_PROFILE_DEFAULT_STEPS, call. = F)

      if (profile_default_steps || (is.null(start_step) && is.null(num_steps) && is.null(start_unix_time) && is.null(duration))){
        start_step = DETAILED_PROFILING_START_STEP_DEFAULT
        num_steps = PROFILING_NUM_STEPS_DEFAULT
      }
      super$initialize(
        SMDATAPARALLEL_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
      )
    }
  ),
  lock_objects = F
)
