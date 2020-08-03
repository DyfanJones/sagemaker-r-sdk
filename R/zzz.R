.onLoad <- function(libname, pkgname) {
  sagemaker_logging_format()

  RProtoBuf::readProtoFiles(.recordProto(), package = "R6sagmaker")
}
