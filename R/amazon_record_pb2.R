#' @importFrom RProtoBuf P

.recordProto <- function() system.file("proto", "record.proto", package= "R6sagemaker")

# create descriptors
.FLOAT32TENSOR <- function() P("aialgs.data.Float32Tensor", file = .recordProto())
.FLOAT64TENSOR <- function() P("aialgs.data.Float64Tensor", file = .recordProto())
.INT32TENSOR <- function() P("aialgs.data.Int32Tensor", file = .recordProto())
.BYTES <- function() P("aialgs.data.Bytes", file =.recordProto())
.VALUE <- function() P("aialgs.data.Value", file =.recordProto())
.RECORD_FEATURESENTRY <- function() P("aialgs.data.Record.FeaturesEntry", file = .recordProto())$new()
.RECORD_LABELENTRY <- function() P("aialgs.data.Record.LabelEntry", file = .recordProto())$new()
.RECORD <- function() P("aialgs.data.Record", file = .recordProto())

# Create message
Float32Tensor <- function() .FLOAT32TENSOR()$new()
Float64Tensor <- function() .FLOAT64TENSOR()$new()
Int32Tensor <- function() .INT32TENSOR()$new()
Bytes <- function() .BYTES()$new()
Value <- function() .VALUE()$new()
Record <- function() .RECORD()$new()
