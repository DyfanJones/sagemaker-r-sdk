#' @importFrom RProtoBuf P

.recordProto <- function() system.file("proto", "record.proto", package= "R6sagemaker")

# create descriptors
.FLOAT32TENSOR <- RProtoBuf::P("aialgs.data.Float32Tensor", file = .recordProto())
.FLOAT64TENSOR <- RProtoBuf::P("aialgs.data.Float64Tensor", file = .recordProto())
.INT32TENSOR <- RProtoBuf::P("aialgs.data.Int32Tensor", file = .recordProto())
.BYTES <- RProtoBuf::P("aialgs.data.Bytes", file =.recordProto())
.VALUE <- RProtoBuf::P("aialgs.data.Value", file =.recordProto())
.RECORD_FEATURESENTRY <- RProtoBuf::P("aialgs.data.Record.FeaturesEntry", file = .recordProto())
.RECORD_LABELENTRY <- RProtoBuf::P("aialgs.data.Record.LabelEntry", file = .recordProto())
.RECORD <- RProtoBuf::P("aialgs.data.Record", file = .recordProto())

# Create message
Float32Tensor <- .FLOAT32TENSOR$new()
Float64Tensor <- .FLOAT64TENSOR$new()
Int32Tensor <- .INT32TENSOR$new()
Bytes <- .BYTES$new()
Value <- .VALUE$new()
Record <- .RECORD$new(features = .RECORD_FEATURESENTRY$new(), label = .RECORD_LABELENTRY$new())
