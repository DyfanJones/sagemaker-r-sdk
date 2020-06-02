#' @export
s3_input <- function(s3_data,
                     distribution=NULL,
                     compression=NULL,
                     content_type=NULL,
                     record_wrapping=NULL,
                     s3_data_type="S3Prefix",
                     input_mode=NULL,
                     attribute_names=NULL,
                     target_attribute_name=NULL,
                     shuffle_config=NULL){
  log_warn("'s3_input' class will be renamed to 'TrainingInput' to align with SageMaker Python SDK v2.")

  config <- list(DataSource =
                  list(S3DataSource =
                         list(S3DataType = s3_data_type, S3Uri = s3_data)))

  if (is.null(target_attribute_name) || is.null(distribution)) distribution = "FullyReplicated"

  config[["DataSource"]][["S3DataSource"]][["S3DataDistributionType"]] = distribution
  config[["CompressionType"]] = compression
  config[["ContentType"]] = content_type
  config[["RecordWrapperType"]] = record_wrapping
  config[["InputMode"]] = input_mode
  config[["DataSource"]][["S3DataSource"]][["AttributeNames"]] = attribute_names
  config[["TargetAttributeName"]] = target_attribute_name
  if(!is.null(shuffle_config)) shuffleconfig[["ShuffleConfig"]] = list(Seed = shuffle_config)

  return(config)
}


#' @export
FileSystemInput <- function(file_system_id,
                            file_system_type = c("FSxLustre", "EFS"),
                            directory_path,
                            file_system_access_mode= c("ro", "rw"),
                            content_type=NULL){

  file_system_type = match.arg(file_system_type)
  file_system_access_mode = match.arg(file_system_access_mode)

  config = list(
    "DataSource"= list(
      "FileSystemDataSource"= list(
        "FileSystemId"= file_system_id,
        "FileSystemType"= file_system_type,
        "DirectoryPath"= directory_path,
        "FileSystemAccessMode"= file_system_access_mode
      )
    )
  )
  config[["ContentType"]] = content_type

  return(config)
}
