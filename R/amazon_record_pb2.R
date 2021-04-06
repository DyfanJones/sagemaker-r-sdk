
# Directory to aws sagemaker proto buf file
.recordProto <- function() system.file("proto", package= "R6sagemaker")

# create descriptors
.FLOAT32TENSOR <- function() RProtoBuf::P("aialgs.data.Float32Tensor")
.FLOAT64TENSOR <- function() RProtoBuf::P("aialgs.data.Float64Tensor")
.INT32TENSOR <- function() RProtoBuf::P("aialgs.data.Int32Tensor")
.BYTES <- function() RProtoBuf::P("aialgs.data.Bytes")
.VALUE <- function() RProtoBuf::P("aialgs.data.Value")
.RECORD_FEATURESENTRY <- function() RProtoBuf::P("aialgs.data.Record.FeaturesEntry")$new()
.RECORD_LABELENTRY <- function() RProtoBuf::P("aialgs.data.Record.LabelEntry")$new()
.RECORD <- function() RProtoBuf::P("aialgs.data.Record")

# Create message
Float32Tensor_pb <- function() .FLOAT32TENSOR()$new()
Float64Tensor_pb <- function() .FLOAT64TENSOR()$new()
Int32Tensor_pb <- function() .INT32TENSOR()$new()
Bytes_pb <- function() .BYTES()$new()
Value_pb <- function() .VALUE()$new()
Record_pb <- function() .RECORD()$new()

# initialise ProtoBuf when required
initProtoBuf <- function(){
  if(!exists("aialgs.data.Record")){
    if(requireNamespace('RProtoBuf', quietly=TRUE)){
      RProtoBuf::readProtoFiles2(protoPath=.recordProto())
    } else {
      stop('Please install RProtoBuf package and try again', call. = F)
    }
  }
}
