# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/debugger/utils.py

#' @import R6sagemaker.common

# Store all possible messages during failures in validation of user arguments.
ErrorMessages = Enum(
    INVALID_LOCAL_PATH = "local_path must be a string!",
    INVALID_FILE_MAX_SIZE = "file_max_size must be an integer greater than 0!",
    INVALID_FILE_CLOSE_INTERVAL = "file_close_interval must be a float/integer greater than 0!",
    INVALID_FILE_OPEN_FAIL_THRESHOLD = "file_open_fail threshold must be an integer greater than 0!",
    INVALID_PROFILE_DEFAULT_STEPS = "profile_default_steps must be a boolean!",
    INVALID_START_STEP = "start_step must be integer greater or equal to 0!",
    INVALID_NUM_STEPS = "num_steps must be integer greater than 0!",
    INVALID_START_UNIX_TIME = "start_unix_time must be valid integer unix time!",
    INVALID_DURATION = "duration must be float greater than 0!",
    FOUND_BOTH_STEP_AND_TIME_FIELDS = "Both step and time fields cannot be specified in the metrics config!",
    INVALID_METRICS_REGEX = "metrics_regex is invalid!",
    INVALID_PYTHON_PROFILER = "python_profiler must be of type PythonProfiler!",
    INVALID_CPROFILE_TIMER = "cprofile_timer must be of type cProfileTimer"
)

#' @title PythonProfiler enum environment list
#' @description List the Python profiler options for Python profiling.
#' @return environment containing [CPROFILE, PYINSTRUMENT]
#' @export
PythonProfiler = Enum(
  CPROFILE = "cprofile",
  PYINSTRUMENT = "pyinstrument",
  .class = "PythonProfiler"
)

#' @title cProfileTimer enum environment list
#' @description List the possible cProfile timers for Python profiling.
#' @return environment containing [TOTAL_TIME, CPU_TIME, OFF_CPU_TIME, DEFAULT]
#' @export
cProfileTimer = Enum(
  TOTAL_TIME = "total_time",
  CPU_TIME = "cpu_time",
  OFF_CPU_TIME = "off_cpu_time",
  DEFAULT = "default",
  .class = "cProfileTimer"
)

#' @title Helper function to determine whether the provided regex is valid.
#' @param regex (str): The user provided regex.
#' @return bool: Indicates whether the provided regex was valid or not.
#' @return
is_valid_regex = function(regex){
  tryCatch({
    suppressWarnings(regexpr(regex,"regex", perl = TRUE))
    return(TRUE)
    },
    error = function(e)
      return(FALSE))
}

