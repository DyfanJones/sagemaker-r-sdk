
#' @import RProtoBuf

# get record.proto
record.proto = system.file("proto", "record.proto", package= "R6sagemaker")

# create descriptors
.FLOAT32TENSOR <- P("aialgs.data.Float32Tensor", file = record.proto)
.FLOAT64TENSOR <- P("aialgs.data.Float64Tensor", file = record.proto)
.INT32TENSOR <- P("aialgs.data.Int32Tensor", file = record.proto)
.BYTES <- P("aialgs.data.Bytes", file =record.proto)
.VALUE <- P("aialgs.data.Value", file =record.proto)
.RECORD_FEATURESENTRY <- P("aialgs.data.Record.FeaturesEntry", file = record.proto)
.RECORD_LABELENTRY = P("aialgs.data.Record.LabelEntry", file = record.proto)
.RECORD <- P("aialgs.data.Record", file = record.proto)

# Create message
Float32Tensor = .FLOAT32TENSOR$new()
Float64Tensor = .FLOAT64TENSOR$new()
Int32Tensor = .INT32TENSOR$new()
Bytes = .BYTES$new()
Value = .VALUE$new()
Record = .RECORD$new(features = .RECORD_FEATURESENTRY$new(), label = .RECORD_LABELENTRY$new())
