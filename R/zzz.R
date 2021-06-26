#' r6 sagemaker: this is just a placeholder
#'
#' @import R6
#' @import paws
#' @import jsonlite
#' @import lgr
#' @import R6sagemaker.common
#' @import R6sagemaker.mlcore
#' @importFrom urltools url_parse
#' @import data.table
#' @import uuid
#' @import tools
#' @importFrom stats runif
#' @importFrom methods is as
"_PACKAGE"

.onLoad <- function(libname, pkgname) {
  # set package logs and don't propagate root logs
  .logger = lgr::get_logger(name = pkgname)

  # set package logger
  assign(
    "LOGGER",
    .logger,
    envir = parent.env(environment())
  )
}
