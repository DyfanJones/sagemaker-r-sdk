# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/model_monitor/dataset_format.py

#' @import R6
#' @import R6sagemaker.common

#' @title DatasetFormat Class
#' @description Represents a Dataset Format that is used when calling a DefaultModelMonitor.
#' @export
DatasetFormat = R6Class("DatasetFormat",
  public = list(

    #' @description Returns a DatasetFormat JSON string for use with a DefaultModelMonitor.
    #' @param header (bool): Whether the csv dataset to baseline and monitor has a header.
    #'             Default: True.
    #' @param output_columns_position (str): The position of the output columns.
    #'             Must be one of ("START", "END"). Default: "START".
    #' @return dict: JSON string containing DatasetFormat to be used by DefaultModelMonitor.
    csv = function(header=TRUE, output_columns_position=c("START", "END")){
      return(
        list(
          "csv"=list(
            "header"=header,
            "output_columns_position"=match.arg(output_columns_position))
          )
      )
    },

    #' @description Returns a DatasetFormat JSON string for use with a DefaultModelMonitor.
    #' @param lines (bool): Whether the file should be read as a json object per line. Default: True.
    #' @return dict: JSON string containing DatasetFormat to be used by DefaultModelMonitor.
    josn = function(lines=TRUE){
      return(list("json"=list("lines"=lines)))
    },

    #' @description Returns a DatasetFormat SageMaker Capture Json string for use with a DefaultModelMonitor.
    #' @return dict: JSON string containing DatasetFormat to be used by DefaultModelMonitor.
    sagemaker_capture_json = function(){
      return(list("sagemakerCaptureJson"=list()))
    },

    #' @description format class
    format = function(){
      return(format_class(self))
    }
  )
)
