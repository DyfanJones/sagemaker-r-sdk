# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/logs.py

#' @import lgr

# set format for SageMaker
sagemaker_log_layout <- lgr::LayoutFormat$new(
  fmt = "%L[%t] %m",
  timestamp_fmt = "%Y-%m-%d %H:%M:%OS",
  colors = getOption("lgr.colors", list())
)

# format logs from cloud watch
sagemaker_colour_wrapper <- function(logs){
  ifelse(grepl("^\\[.*FATAL.*\\].*|^FATAL:.*",logs),
         sprintf("\033[38;5;%sm%s\033[39m", 196, logs),
  ifelse(grepl("^\\[.*ERROR.*\\].*|^ERROR:.*",logs),
         sprintf("\033[38;5;%sm%s\033[39m", 124, logs),
  ifelse(grepl("^\\[.*WARNING.*\\].*|^.*WARNING:.*",logs),
         sprintf("\033[38;5;%sm%s\033[39m", 214, logs),
  ifelse(grepl("^\\[.*SUCCESS.*\\].*|^SUCCESS:.*",logs),
         sprintf("\033[38;5;%sm%s\033[39m", 34, logs),
  ifelse(grepl("^\\[.*INFO.*\\].*|^INFO:.*",logs),
         sprintf("\x1b[%sm%s\x1b[0m", 34, logs),
  ifelse(grepl("^\\[.*DEBUG.*\\].*|^DEBUG:.*",logs),
         sprintf("\033[38;5;%sm%s\033[39m", 31, logs),
  ifelse(grepl("^\\[.*TRACE.*\\].*|^TRACE:.*",logs),
         sprintf("\033[38;5;%sm%s\033[39m", 25, logs),
         sprintf("\033[38;5;246m%s\033[39m", logs))))))))
}

multi_stream_iter <- function(cloudwatchlogs, log_group, streams, positions= NULL){
  if(is.null(positions)){
    positions = rep(list(list(timestamp = 0, skip = 1)), length(streams))
    names(positions) = streams

    # update sm_env positions
    sm_env$positions = positions}

  # Get all logs and remove 1 list level
  events = lapply(streams, function(s)
    log_stream(
      cloudwatchlogs,
      log_group,
      s,
      positions[[s]]$timestamp,
      positions[[s]]$skip))
  return(events)
}


log_stream <- function(cloudwatchlogs,
                      log_group,
                      stream_name,
                      start_time=0,
                      skip=1){
  next_token = NULL
  event_count = 1
  events = list()
  while (event_count > 0){
    # if(length(next_token) == 0) next_token = NULL

    response = retry_api_call(cloudwatchlogs$get_log_events(
      logGroupName=log_group,
      logStreamName=stream_name,
      startTime=start_time,
      startFromHead=TRUE,
      nextToken = next_token))

    event_count = length(response$events)
    if(event_count){
      next_token = response$nextForwardToken
      events = c(events, response$events)}

    if(event_count == 0) break

    if(event_count > skip){
      events = events[skip:event_count]
      skip = 1
    } else {
      skip = skip - event_count
      events = list()
    }
  }
  return(events)
}

