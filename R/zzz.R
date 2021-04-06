#' @include logs.R

.onLoad <- function(libname, pkgname) {
  # set package logs and don't propagate root logs
  .logger = lgr::get_logger(name = pkgname)$set_propagate(FALSE)

  # set logging layout
  .logger$add_appender(
    lgr::AppenderConsole$new(
      layout=sagemaker_log_layout()
      )
    )

  # set package logger
  assign(
    "LOGGER",
    .logger,
    envir = parent.env(environment())
  )
}

