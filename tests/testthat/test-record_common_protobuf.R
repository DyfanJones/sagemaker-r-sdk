context("Protobuf Serialize/Deserialize")

test_that("test matrix",{
  set.seed(42)
  num = rnorm(9)
  mat = matrix(num, ncol= 3)

  # serialize matrix
  serialize_proto = RecordSerializer$new()
  obj = serialize_proto$serialize(mat)

  # deserialize matrix using protobuf
  deserialize_proto = RecordDeserializer$new()
  proto_mat = deserialize_proto$deserializer(obj, serialize_proto$CONTENT_TYPE)

  proto_mat = t(sapply(proto_mat, function(x) x$features[[1]]$value$float64_tensor$values))

  expect_equal(mat, proto_mat)
})

test_that("test integer vector",{
  num = 1:9

  # serialize vector
  serialize_proto = RecordSerializer$new()
  obj = serialize_proto$serialize(num)

  # deserialize matrix using protobuf
  deserialize_proto = RecordDeserializer$new()
  proto_num = deserialize_proto$deserializer(obj, serialize_proto$CONTENT_TYPE)

  expect_equal(num, proto_num[[1]]$features[[1]]$value$int32_tensor$values)
})

test_that("test character vector",{
  Vector = LETTERS

  serialize_proto = RecordSerializer$new()

  expect_error(serialize_proto$serialize(Vector))
})
