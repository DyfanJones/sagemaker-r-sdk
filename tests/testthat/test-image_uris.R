context("ImageUris Sagemaker uris construction")

image_uris = ImageUris$new()

test_that("test for wrong framework", {
  expect_error(
    image_uris$retrieve(
      framework = "useless-string",
      version = "1.0.0",
      py_version = "py3",
      instance_type = "ml.c4.xlarge",
      region = "us-west-2",
      image_scope = "training"
    )
  )
})

test_that("test for wrong version", {
  expect_error(
    image_uris$retrieve(
      framework = "xgboost",
      version = "useless-version",
      py_version = "py3",
      instance_type = "ml.c4.xlarge",
      region = "us-west-2",
      image_scope = "training"
    )
  )
})

test_that("test xgboost uri creation", {
  test_uri = image_uris$retrieve(
    framework = "xgboost",
    version = "1",
    region = "us-west-2",
    image_scope = "training"
  )
  exp_uri = "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:1"
  expect_equal(test_uri, exp_uri)
})

test_that("test unsupported instance type", {
  expect_error(
    image_uris$retrieve(
      framework = "xgboost",
      version = "0.90-1",
      region = "us-west-2",
      image_scope = "training",
      instance_type = "ml.g4dn.xlarge"
    )
  )
})

test_that("test supported instance type", {
  # gpu uri creation
  test_uri_1 = image_uris$retrieve(
    framework = "ray-pytorch",
    version = "0.8.5",
    py_version = "py36",
    region = "us-west-2",
    image_scope = "training",
    instance_type = "ml.g4dn.xlarge"
    )

  # cpu uri creation
  test_uri_2 = image_uris$retrieve(
    framework = "ray-pytorch",
    version = "0.8.5",
    py_version = "py36",
    region = "us-west-2",
    image_scope = "training",
    instance_type = "ml.c4.xlarge"
  )

  exp_uri_1 = "462105765813.dkr.ecr.us-west-2.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.5-torch-gpu-py36"
  exp_uri_2 = "462105765813.dkr.ecr.us-west-2.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.5-torch-cpu-py36"

  expect_equal(test_uri_1, exp_uri_1)
  expect_equal(test_uri_2, exp_uri_2)
})


test_that("test unsupported image_scope", {
  expect_error(
    image_uris$retrieve(
      framework = "xgboost",
      version = "0.90-1",
      region = "us-west-2",
      image_scope = "useless-string"
    )
  )
})

test_that("test retrieve default image_scope", {
  test_uri = image_uris$retrieve(
    framework = "ray-pytorch",
    version = "0.8.5",
    region = "us-west-2",
    py_version = "py36",
    instance_type = "ml.c4.xlarge"
    )

  exp_uri = "462105765813.dkr.ecr.us-west-2.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.5-torch-cpu-py36"
  expect_equal(test_uri, exp_uri)
})

test_that("test retrieve eia image_scope", {
  test_uri = image_uris$retrieve(
    framework = "tensorflow",
    version = "1.10.0",
    region = "us-west-2",
    py_version = "py2",
    image_scope = "eia",
    instance_type = "ml.c4.xlarge"
  )
  exp_uri = "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-eia:1.10.0-cpu-py2"
  expect_equal(test_uri, exp_uri)
})

test_that("test retrieve accelerator", {
  test_uri = image_uris$retrieve(
    framework = "tensorflow",
    version = "2.0",
    region = "us-west-2",
    instance_type= "ml.c4.xlarge",
    accelerator_type = "ml.eia2.medium"
  )

  exp_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference-eia:2.0-cpu"
  expect_equal(test_uri, exp_uri)
})

test_that("test retrieve invalid accelerator", {
  expect_error(
    image_uris$retrieve(
      framework = "tensorflow",
      version = "2.0",
      region = "us-west-2",
      instance_type= "ml.c4.xlarge",
      accelerator_type = "useless-string"
      )
  )
})
