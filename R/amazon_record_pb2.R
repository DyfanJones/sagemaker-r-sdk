
.recordProto <- function() system.file("proto", "record.proto", package= "R6sagemaker")

# create descriptors
.FLOAT32TENSOR <- function() RProtoBuf::P("aialgs.data.Float32Tensor", file = .recordProto())
.FLOAT64TENSOR <- function() RProtoBuf::P("aialgs.data.Float64Tensor", file = .recordProto())
.INT32TENSOR <- function() RProtoBuf::P("aialgs.data.Int32Tensor", file = .recordProto())
.BYTES <- function() RProtoBuf::P("aialgs.data.Bytes", file =.recordProto())
.VALUE <- function() RProtoBuf::P("aialgs.data.Value", file =.recordProto())
.RECORD_FEATURESENTRY <- function() RProtoBuf::P("aialgs.data.Record.FeaturesEntry", file = .recordProto())$new()
.RECORD_LABELENTRY <- function() RProtoBuf::P("aialgs.data.Record.LabelEntry", file = .recordProto())$new()
.RECORD <- function() RProtoBuf::P("aialgs.data.Record", file = .recordProto())

# Create message
Float32Tensor <- function() .FLOAT32TENSOR()$new()
Float64Tensor <- function() .FLOAT64TENSOR()$new()
Int32Tensor <- function() .INT32TENSOR()$new()
Bytes <- function() .BYTES()$new()
Value <- function() .VALUE()$new()
Record <- function() .RECORD()$new()

# initialise ProtoBuf when required
initProtoBuf <- function(){
  if(!exists("aialgs.data.Record")){
    if(requireNamespace('RProtoBuf', quietly=TRUE)){
      if(!file.exist(.recordProto())) stop("ProtoBuf File failed to install.")
      RProtoBuf::readProtoFiles(.recordProto(), package = "R6sagmaker")
    } else {
      stop('Please install RProtoBuf package and try again', call. = F)
    }
  }
}
