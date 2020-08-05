# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/serializers.py

#' @import R6
#' @import jsonlite
#' @import data.table

#' @title Default BaseSerializer Class
#' @description  All serializer are children of this class. If a custom
#'               serializer is desired, inherit this class.
#' @export
BaseSerializer = R6Class("Serializer",
  public = list(
   #' @field content_type
   #' Method in how data is going to be seperated
   content_type = NULL,

   #' @description  Take data of various data formats and serialize them
   #' @param data (object): Data to be serialized.
   initialize = function(){},

   #' @description Take data of various data formats and serialize them into CSV.
   #' @param data (object): Data to be serialized
   serialize = function(data) {stop("I'm an abstract interface method", call. = F)},

   #' @description
   #' Printer.
   #' @param ... (ignored).
   print = function(...){
     cat("<Serializer>")
     invisible(self)
   }
  )
)

#' @title CsvSerializer Class
#' @description Make Raw data using text/csv format
#' @export
CsvSerializer = R6Class("CsvSerializer",
  inherit = Serializer,
  public = list(
    #' @description Initialize Serializer Class
    initialize = function(){
      self$content_type = "text/csv"
    },
    #' @description Take data of various data formats and serialize them into CSV.
    #' @param data (object): Data to be serialized. Any list of same length vectors; e.g. data.frame and data.table.
    #'               If matrix, it gets internally coerced to data.table preserving col names but not row names
    serialize = function(data) {
      TempFile = tempfile()
      fwrite(data, TempFile, col.names = FALSE)
      obj = readBin(TempFile, "raw", n = file.size(TempFile))
      unlink(TempFile)
      return(obj)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      cat("<CsvSerializer>")
      invisible(self)
    }
  )
)

#' @title S3 method to call CsvSerializer class
#' @export
csv_serializer <- CsvSerializer$new()

#' @title JsonSerializer Class
#' @description Make Raw data using json format
#' @export
JsonSerializer = R6Class("JsonSerializer",
  inherit = Serializer,
  public = list(
   #' @description Initialize Csv Serializer
   initialize = function(){
     self$content_type = "application/json"
   },
   #' @description Take data of various data formats and serialize them into CSV.
   #' @param data (object): Data to be serialized.
   serialize = function(data) {

     con = rawConnection(raw(0), "r+")
     on.exit(close(con))
     write_json(data, con, dataframe = "columns", auto_unbox = T)

     return(rawConnectionValue(con))
   },

   #' @description
   #' Printer.
   #' @param ... (ignored).
   print = function(...){
     cat("<JsonSerializer>")
     invisible(self)
   }
  )
)

#' @title S3 method to call JsonSerializer class
#' @export
json_serializer <- JsonSerializer$new()
