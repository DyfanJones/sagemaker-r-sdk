
# Store all possible messages during failures in validation of user arguments.
ErrorMessages = R6Class("ErrorMessages",
  public = list(
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
)

# List the Python profiler options for Python profiling.
# .. py:attribute:: CPROFILE
# Use to choose ``"cProfile"``.
# .. py:attribute:: PYINSTRUMENT
# Use to choose ``"Pyinstrument"``.
PythonProfiler = R6Class("PythonProfiler",
  public = list(
    CPROFILE = "cprofile",
    PYINSTRUMENT = "pyinstrument",
    print = function(...){
      return(print_class(self))
    }
  )
)

# List the possible cProfile timers for Python profiling.
# .. py:attribute:: TOTAL_TIME
# Use to choose ``"total_time"``.
# .. py:attribute:: CPU_TIME
# Use to choose ``"cpu_time"``.
# .. py:attribute:: OFF_CPU_TIME
# Use to choose ``"off_cpu_time"``.
cProfileTimer = R6Class("cProfileTimer",
  public =list(
    TOTAL_TIME = "total_time",
    CPU_TIME = "cpu_time",
    OFF_CPU_TIME = "off_cpu_time",
    DEFAULT = "default",
    print = function(...){
      return(print_class(self))
    }
  )
)
