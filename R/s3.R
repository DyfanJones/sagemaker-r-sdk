# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/s3.py

#' @include session.R
#' @include utils.R
#'
#' @import R6

# validation check of s3 uri
is.s3_uri <- function(x) {
  if(is.null(x) || !is.character(x)) return(FALSE)
  regex <- '^s3://[a-z0-9][a-z0-9\\.-]+[a-z0-9](/(.*)?)?$'
  grepl(regex, x)
}

# split s3 uri
split_s3_uri <- function(uri) {
  stopifnot(is.s3_uri(uri))
  path <- gsub('^s3://', '', uri)
  list(
    bucket = gsub('/.*$', '', path),
    key = gsub('^[a-z0-9][a-z0-9\\.-]+[a-z0-9]/', '', path)
  )
}


S3Uploader = R6Class("S3Uploader",
                     public = list(
                       # Static method that uploads a given file or directory to S3.
                       # Args:
                       #   local_path (str): Path (absolute or relative) of local file or directory to upload.
                       # desired_s3_uri (str): The desired S3 location to upload to. It is the prefix to
                       # which the local filename will be added.
                       # kms_key (str): The KMS key to use to encrypt the files.
                       # session (sagemaker.session.Session): Session object which
                       # manages interactions with Amazon SageMaker APIs and any other
                       # AWS services needed. If not specified, the estimator creates one
                       # using the default AWS configuration chain.
                       # Returns:
                       #   The S3 uri of the uploaded file(s).
                       upload = function(local_path = NULL,
                                         desired_s3_uri = NULL,
                                         kms_key=NULL,
                                         sagemaker_session=NULL){

                       sagemaker_session = sagemaker_session %||% Session$new()
                       s3_parts = split_s3_uri(desired_s3_uri)
                       if (!is.null(kms_key))
                         return(sagemaker_session$upload_data(
                           path=local_path, bucket=s3_parts$bucket,
                           key_prefix=s3_parts$key, SSEKMSKeyId = kms_key))

                       return(sagemaker_session$upload_data(
                                  path=local_path, bucket=s3_parts$bucket,
                                  key_prefix=s3_parts$key))
                       },

                       # Static method that uploads a given file or directory to S3.
                       # Args:
                       #   body (str): String representing the body of the file.
                       # desired_s3_uri (str): The desired S3 uri to upload to.
                       # kms_key (str): The KMS key to use to encrypt the files.
                       # session (sagemaker.session.Session): AWS session to use. Automatically
                       # generates one if not provided.
                       # Returns:
                       #   str: The S3 uri of the uploaded file(s).
                       upload_string_as_file_body = function(body,
                                                             desired_s3_uri=NULL,
                                                             kms_key=NULL,
                                                             sagemaker_session=NULL){
                         sagemaker_session = sagemaker_session %||% Session$new()
                         s3_parts = split_s3_uri(desired_s3_uri)

                         sagemaker_session$upload_string_as_file_body(
                           body=body, bucket=s3_parts$bucket,
                           key_prefix=s3_parts$key, kms_key=kms_key)

                         return(desired_s3_uri)
                       }
                     )
)


S3Downloader = R6Class("S3Downloader",
                       public =list(
                         # Static method that downloads a given S3 uri to the local machine.
                         # Args:
                         #   s3_uri (str): An S3 uri to download from.
                         # local_path (str): A local path to download the file(s) to.
                         # kms_key (str): The KMS key to use to decrypt the files.
                         # session (sagemaker.session.Session): Session object which
                         # manages interactions with Amazon SageMaker APIs and any other
                         # AWS services needed. If not specified, the estimator creates one
                         # using the default AWS configuration chain.
                         download = function(s3_uri,
                                             local_path,
                                             kms_key=NULL,
                                             sagemaker_session=NULL){

                           sagemaker_session = sagemaker_session %||% Session$new()
                           s3_parts = split_s3_uri(desired_s3_uri)
                           if (!is.null(kms_key))
                             return(sagemaker_session$download_data(
                               path=local_path, bucket=s3_parts$bucket,
                               key_prefix=s3_parts$key, SSEKMSKeyId = kms_key))

                           return(sagemaker_session$download_data(
                             path=local_path, bucket=s3_parts$bucket,
                             key_prefix=s3_parts$key))
                         },

                         # Static method that returns the contents of an s3 uri file body as a string.
                         # Args:
                         #   s3_uri (str): An S3 uri that refers to a single file.
                         # session (sagemaker.session.Session): AWS session to use. Automatically
                         # generates one if not provided.
                         # Returns:
                         #   str: The body of the file.
                         read_file = function(s3_uri,
                                              sagemaker_session=NULL){

                           sagemaker_session = sagemaker_session %||% Session$new()
                           s3_parts = split_s3_uri(desired_s3_uri)

                          return(sagemaker_session$read_s3_file(bucket=s3_parts$bucket, key_prefix=s3_parts$key))
                         },

                         # Static method that lists the contents of an S3 uri.
                         # Args:
                         #   s3_uri (str): The S3 base uri to list objects in.
                         # session (sagemaker.session.Session): AWS session to use. Automatically
                         # generates one if not provided.
                         # Returns:
                         #   [str]: The list of S3 URIs in the given S3 base uri.
                         list_s3_uri = function(s3_uri,
                                                sagemaker_session = NULL){
                           sagemaker_session = sagemaker_session %||% Session()
                           s3_parts = split_s3_uri(desired_s3_uri)

                           file_keys = sagemaker_session$list_s3_files(bucket=s3_parts$bucket, key_prefix=s3_parts$key)
                           return(lapply(file_keys, function(file_key) sprintf("s3://%s/%s", s3_parts$bucket, file_key)))
                         }
                       )
)
