#' r6 sagemaker: this is just a placeholder
#'
#' @import lgr
#' @import R6sagemaker.common
#' @import R6sagemaker.mlcore
#' @import R6sagemaker.local
#' @import R6sagemaker.workflow
#' @import R6sagemaker.mlamazon
"_PACKAGE"

.onLoad <- function(libname, pkgname) {
  # set package logs and don't propagate root logs
  .logger = lgr::get_logger(name = pkgname)$set_propagate(FALSE)

  # set package logger
  assign(
    "LOGGER",
    .logger,
    envir = parent.env(environment())
  )
}
