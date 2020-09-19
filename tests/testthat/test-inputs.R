# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/64f600d677872fe8656cdf25d68fc4950b2cd28f/tests/unit/test_inputs.py
context("inputs Classes")

test_that("test training input all defaults", {
  prefix = "pre"
  actual = TrainingInput$new(s3_data=prefix)

  expected = list(
    "DataSource"= list(
      "S3DataSource" = list(
        "S3DataType"= "S3Prefix",
        "S3Uri"= prefix,
        "S3DataDistributionType"= "FullyReplicated"
      )
    )
  )

  expect_equal(actual$config, expected)
})

test_that("test training input all arguments", {
  prefix = "pre"
  distribution = "FullyReplicated"
  compression = "Gzip"
  content_type = "text/csv"
  record_wrapping = "RecordIO"
  s3_data_type = "Manifestfile"
  input_mode = "Pipe"
  result = TrainingInput$new(
    s3_data=prefix,
    distribution=distribution,
    compression=compression,
    input_mode=input_mode,
    content_type=content_type,
    record_wrapping=record_wrapping,
    s3_data_type=s3_data_type
  )
  expected = list(
    "DataSource" = list(
      "S3DataSource" = list(
        "S3DataType"= s3_data_type,
        "S3Uri"= prefix,
        "S3DataDistributionType"= distribution
        )
      ),
    "CompressionType"= compression,
    "ContentType"= content_type,
    "RecordWrapperType"= record_wrapping,
    "InputMode"= input_mode
  )
  expect_equal(result$config, expected)
})

test_that("test file system input default access mode", {
  file_system_id = "fs-0a48d2a1"
  file_system_type = "EFS"
  directory_path = "tensorflow"
  actual = FileSystemInput$new(
    file_system_id=file_system_id,
    file_system_type=file_system_type,
    directory_path=directory_path
  )
  expected = list(
    "DataSource" = list(
      "FileSystemDataSource" = list(
        "FileSystemId"= file_system_id,
        "FileSystemType"= file_system_type,
        "DirectoryPath"= directory_path,
        "FileSystemAccessMode"= "ro"
      )
    )
  )
  expect_equal(actual$config, expected)
})

test_that("test file system input all argument", {
  file_system_id = "fs-0a48d2a1"
  file_system_type = "FSxLustre"
  directory_path = "tensorflow"
  file_system_access_mode = "rw"
  actual = FileSystemInput$new(
    file_system_id=file_system_id,
    file_system_type=file_system_type,
    directory_path=directory_path,
    file_system_access_mode=file_system_access_mode,
  )
  expected = list(
    "DataSource" = list(
      "FileSystemDataSource" = list(
        "FileSystemId"= file_system_id,
        "FileSystemType"= file_system_type,
        "DirectoryPath"= directory_path,
        "FileSystemAccessMode"= "rw"
      )
    )
  )
  expect_equal(actual$config, expected)
})

test_that("test file system input content type", {
  file_system_id = "fs-0a48d2a1"
  file_system_type = "FSxLustre"
  directory_path = "tensorflow"
  file_system_access_mode = "rw"
  content_type = "application/json"
  actual = FileSystemInput$new(
    file_system_id=file_system_id,
    file_system_type=file_system_type,
    directory_path=directory_path,
    file_system_access_mode=file_system_access_mode,
    content_type=content_type
  )
  expected = list(
    "DataSource" = list(
      "FileSystemDataSource" = list(
        "FileSystemId"= file_system_id,
        "FileSystemType"= file_system_type,
        "DirectoryPath"= directory_path,
        "FileSystemAccessMode"= "rw"
      )
    ),
    "ContentType"= content_type
  )
  expect_equal(actual$config, expected)
})

test_that("test file system input type invalid", {
  file_system_id = "fs-0a48d2a1"
  file_system_type = "ABC"
  directory_path = "tensorflow"

  expect_error(
    FileSystemInput$new(
      file_system_id=file_system_id,
      file_system_type=file_system_type,
      directory_path=directory_path
    )
  )
})

test_that("test file system input mode invalid", {
  file_system_id = "fs-0a48d2a1"
  file_system_type = "EFS"
  directory_path = "tensorflow"
  file_system_access_mode = "p"

  expect_error(
    FileSystemInput$new(
      file_system_id=file_system_id,
      file_system_type=file_system_type,
      directory_path=directory_path,
      file_system_access_mode=file_system_access_mode
    )
  )
})
