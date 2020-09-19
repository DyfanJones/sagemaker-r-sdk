# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/inputs.py

#' @import R6

#' @title Create a definition for input data used by an SageMaker training job.
#' @description Amazon SageMaker channel configurations for S3 data sources.
#' @export
TrainingInput = R6Class("TrainingInput",
  public = list(

   #' @field config
   #' A SageMaker ``DataSource`` referencing a SageMaker ``S3DataSource``.
   config = NULL,

   #' @description See AWS documentation on the ``CreateTrainingJob`` API for more details on the parameters.
   #' @param s3_data (str): Defines the location of s3 data to train on.
   #' @param distribution (str): Valid values: 'FullyReplicated', 'ShardedByS3Key'
   #'              (default: 'FullyReplicated').
   #' @param compression (str): Valid values: 'Gzip', None (default: None). This is used only in
   #'              Pipe input mode.
   #' @param content_type (str): MIME type of the input data (default: None).
   #' @param record_wrapping (str): Valid values: 'RecordIO' (default: None).
   #' @param s3_data_type (str): Valid values: 'S3Prefix', 'ManifestFile', 'AugmentedManifestFile'.
   #'              If 'S3Prefix', ``s3_data`` defines a prefix of s3 objects to train on.
   #'              All objects with s3 keys beginning with ``s3_data`` will be used to train.
   #'              If 'ManifestFile' or 'AugmentedManifestFile', then ``s3_data`` defines a
   #'              single S3 manifest file or augmented manifest file (respectively),
   #'              listing the S3 data to train on. Both the ManifestFile and
   #'              AugmentedManifestFile formats are described in the SageMaker API documentation:
   #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_S3DataSource.html
   #' @param input_mode (str): Optional override for this channel's input mode (default: None).
   #'              By default, channels will use the input mode defined on
   #'              ``sagemaker.estimator.EstimatorBase.input_mode``, but they will ignore
   #'              that setting if this parameter is set.
   #'                  * None - Amazon SageMaker will use the input mode specified in the ``Estimator``
   #'                  * 'File' - Amazon SageMaker copies the training dataset from the S3 location to
   #'                      a local directory.
   #'                  * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via
   #'                      a Unix-named pipe.
   #' @param attribute_names (list[str]): A list of one or more attribute names to use that are
   #'              found in a specified AugmentedManifestFile.
   #' @param target_attribute_name (str): The name of the attribute will be predicted (classified)
   #'              in a SageMaker AutoML job. It is required if the input is for SageMaker AutoML job.
   #' @param shuffle_config (ShuffleConfig): If specified this configuration enables shuffling on
   #'              this channel. See the SageMaker API documentation for more info:
   #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_ShuffleConfig.html
   initialize = function(s3_data,
                         distribution=NULL,
                         compression=NULL,
                         content_type=NULL,
                         record_wrapping=NULL,
                         s3_data_type="S3Prefix",
                         input_mode=NULL,
                         attribute_names=NULL,
                         target_attribute_name=NULL,
                         shuffle_config=NULL){
     self$config <- list(DataSource = list(S3DataSource = list(S3DataType = s3_data_type, S3Uri = s3_data)))

     if (is.null(self$target_attribute_name) || is.null(self$distribution)) distribution = "FullyReplicated"

     self$config$DataSource$S3DataSource$S3DataDistributionType = distribution
     self$config$CompressionType = compression
     self$config$ContentType = content_type
     self$config$RecordWrapperType = record_wrapping
     self$config$InputMode = input_mode
     self$config$DataSource$S3DataSource$AttributeNames = attribute_names
     self$config$TargetAttributeName = target_attribute_name
     if(!is.null(self$shuffle_config))
        self$config$ShuffleConfig = list(Seed = shuffle_config$seed)
   },

   #' @description
   #' Printer.
   #' @param ... (ignored).
   print = function(...){
      return(print_class(self))
   }
  )
)

#' @title ShuffleConfig Class
#' @description For configuring channel shuffling using a seed
#'              For more detail, see the AWS documentation:
#'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_ShuffleConfig.html
#' @export
ShuffleConfig = R6Class("ShuffleConfig",
   public = list(

      #' @field seed
      #' value used to seed the shuffled sequence.
      seed = NULL,

      #' @description Create a ShuffleConfig.
      #' @param seed (numeric): value used to seed the shuffled sequence.
      initialize = function(seed){
         self$seed = seed
      },

      #' @description
      #' Printer.
      #' @param ... (ignored).
      print = function(...){
         return(print_class(self))
      }
   )
)

#' @title Amazon SageMaker channel configurations for file system data sources.
#' @export
FileSystemInput = R6Class("FileSystemInput",
  public = list(
    #' @field config (dict[str, dict])\cr
    #' A Sagemaker File System ``DataSource``.
    config = NULL,

    #' @description Create a new file system input used by an SageMaker training job.
    #' @param file_system_id (str): An Amazon file system ID starting with 'fs-'.
    #' @param file_system_type (str): The type of file system used for the input.
    #'              Valid values: 'EFS', 'FSxLustre'.
    #' @param directory_path (str): Absolute or normalized path to the root directory (mount point) in
    #'              the file system.
    #'              Reference: https://docs.aws.amazon.com/efs/latest/ug/mounting-fs.html and
    #'              https://docs.aws.amazon.com/fsx/latest/LustreGuide/mount-fs-auto-mount-onreboot.html
    #' @param file_system_access_mode (str): Permissions for read and write.
    #'              Valid values: 'ro' or 'rw'. Defaults to 'ro'.
    #' @param content_type :
    initialize = function(file_system_id,
                          file_system_type = c("FSxLustre", "EFS"),
                          directory_path,
                          file_system_access_mode= c("ro", "rw"),
                          content_type=NULL){

      file_system_type = match.arg(file_system_type)
      file_system_access_mode = match.arg(file_system_access_mode)

      self$config = list(
        "DataSource"= list(
          "FileSystemDataSource"= list(
            "FileSystemId"= file_system_id,
            "FileSystemType"= file_system_type,
            "DirectoryPath"= directory_path,
            "FileSystemAccessMode"= file_system_access_mode
          )
        )
      )
      self$config$ContentType = content_type
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
       return(print_class(self))
    }
  )
)
