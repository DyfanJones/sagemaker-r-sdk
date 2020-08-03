.onLoad <- function(libname, pkgname) {
  sagemaker_logging_format()
  record.proto <<- system.file("proto", "record.proto", package= "R6sagemaker", lib.loc=libname)
  assign("record.proto",record.proto ,envir = parent.env(environment()))
}
