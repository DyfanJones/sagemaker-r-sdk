context("fw_utils helper function")

test_that("test framework name_from_image_tf_scriptmode", {
  image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-scriptmode:1.12-cpu-py3"
  test_fw1 = framework_name_from_image(image_uri)
  exp_fw1 = list("tensorflow", "py3", "1.12-cpu-py3", "scriptmode")

  image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.13-cpu-py3"
  test_fw2 = framework_name_from_image(image_uri)
  exp_fw2 = list("tensorflow", "py3", "1.13-cpu-py3", "training")

  expect_equal(test_fw1, exp_fw1)
  expect_equal(test_fw2, exp_fw2)
})

test_that("test framework_name_from_image rl", {
  image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-rl-mxnet:toolkit1.1-gpu-py3"
  test_fw1 = framework_name_from_image(image_uri)
  exp_fw1 = list("mxnet", "py3", "toolkit1.1-gpu-py3", NULL)

  expect_equal(test_fw1, exp_fw1)
})

test_that("test framework_name_from_image python versions", {
  image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.2-cpu-py37"
  test_fw1 = framework_name_from_image(image_uri)
  exp_fw1 = list("tensorflow", "py37", "2.2-cpu-py37", "training")

  image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.15.2-cpu-py36"
  exp_fw2 = list("tensorflow", "py36", "1.15.2-cpu-py36", "training")
  test_fw2 = framework_name_from_image(image_uri)

  expect_equal(test_fw1, exp_fw1)
  expect_equal(test_fw2, exp_fw2)
})

test_that("test legacy name_from_framework_image", {
  image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py3-gpu:2.5.6-gpu-py2"
  test_fw1 = framework_name_from_image(image_uri)
  exp_fw1 = list("mxnet", "py3", "2.5.6-gpu-py2", NULL)

  expect_equal(test_fw1, exp_fw1)
})

test_that("test legacy name_from_framework_image wrong python", {
  image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py4-gpu:1"
  test_fw1 = framework_name_from_image(image_uri)
  exp_fw1 = list(NULL, NULL, NULL, NULL)

  expect_equal(test_fw1, exp_fw1)
})

test_that("test legacy name_from_framework_image wrong device", {
  image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py4-gpu:1"
  test_fw1 = framework_name_from_image(image_uri)
  exp_fw1 = list(NULL, NULL, NULL, NULL)

  expect_equal(test_fw1, exp_fw1)
})

test_that("test framework_version_from_tag", {
  tags = list(
    "1.5rc-keras-cpu-py2",
    "1.5rc-keras-gpu-py2",
    "1.5rc-keras-cpu-py3",
    "1.5rc-keras-gpu-py36",
    "1.5rc-keras-gpu-py37"
  )
  exp_tag = "1.5rc-keras"

  for(tag in tags){
    test_tag = framework_version_from_tag(tag)
    expect_equal(test_tag, exp_tag)
  }
})

test_that("test framework_version_from_tag other", {
  test_tag = framework_version_from_tag("weird-tag-py2")
  exp_tag = NULL

  expect_equal(test_tag, exp_tag)
})
