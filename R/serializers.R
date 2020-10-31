# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/serializers.py

#' @import R6
#' @import jsonlite
#' @import data.table

#' @title Default BaseSerializer Class
#' @description  All serializer are children of this class. If a custom
#'               serializer is desired, inherit this class.
#' @export
BaseSerializer = R6Class("BaseSerializer",
  public = list(
   #' @field CONTENT_TYPE
   #' Method in how data is going to be seperated
   CONTENT_TYPE = NULL,

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
      print_class(self)
   }
  )
)

#' @title CSVSerializer Class
#' @description Make Raw data using text/csv format
#' @export
CSVSerializer = R6Class("CSVSerializer",
  inherit = BaseSerializer,
  public = list(
    #' @description Initialize CSVSerializer Class
    initialize = function(){
      self$CONTENT_TYPE = "text/csv"
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
    }
  )
)

#' @title S3 method to call CSVSerializer class
#' @export
csv_serializer <- CSVSerializer$new()

#' @title JSONSerializer Class
#' @description Make Raw data using json format
#' @export
JSONSerializer = R6Class("JSONSerializer",
  inherit = BaseSerializer,
  public = list(
   #' @description Initialize JSONSerializer
   initialize = function(){
     self$CONTENT_TYPE = "application/json"
   },
   #' @description Take data of various data formats and serialize them into CSV.
   #' @param data (object): Data to be serialized.
   serialize = function(data) {

     con = rawConnection(raw(0), "r+")
     on.exit(close(con))
     write_json(data, con, dataframe = "columns", auto_unbox = T)

     return(rawConnectionValue(con))
   }
  )
)

#' @title S3 method to call JsonSerializer class
#' @export
json_serializer <- JSONSerializer$new()

#' @title LibSVMSerializer Class
#' @description Serialize data of various formats to a LibSVM-formatted string.
#'              The data must already be in LIBSVM file format:
#'              <label> <index1>:<value1> <index2>:<value2> ...
#'              It is suitable for sparse datasets since it does not store zero-valued
#'              features.
#' @export
LibSVMSerializer = R6Class("LibSVMSerializer",
   inherit = BaseSerializer,
   public = list(
      #' @description Initialize LibSVMSerializer class
      initialize = function(){
         self$CONTENT_TYPE = "text/libsvm"
         if(!requireNamespace('sparsio', quietly=TRUE))
            stop('Please install sparsio package and try again', call. = F)
      },
      #' @description Serialize data of various formats to a LibSVM-formatted string.
      #' @param data (object): Data to be serialized. Can be a string or a
      #'              file-like object.
      #' @return str: The data serialized as a LibSVM-formatted string.
      serialize = function(data) {
         f = tempfile(fileext = ".svmlight")
         on.exit(unlink(f))
         sparsio::write_svmlight(data, file = f)
         obj = readBin(f, what = "raw", n = file.size(f))
         return(obj)
      }
   )
)

#' @title NumpySerializer Class
#' @description Serialize data of various formats to a numpy npy file format.
#'              This serializer class uses python numpy package to serialize,
#'              R objects through the use of the `reticualte` package.
#' @export
NumpySerializer = R6Class("NumpySerializer",
   inherit = BaseSerializer,
   public = list(

      #' @field np
      #' Python Numpy package
      np = NULL,

      #' @description Initialize NumpySerializer class
      initialize = function(){
        self$CONTENT_TYPE = "application/x-npy"
        if(!requireNamespace('reticulate', quietly=TRUE))
           stop('Please install reticulate package and try again', call. = F)
        self$np = reticulate::import("numpy")
      },

      #' @description Serialize data of various formats to a Numpy file.
      #' @param data (object): Data to be serialized. Can be a string or a
      #'              file-like object.
      #' @return str: The data serialized as a LibSVM-formatted string.
      serialize = function(data) {

         if(inherits(data, "array")){
            if(length(data) == 0)
               stop("Cannot serialize empty array.", call. = F)
         }

         if(inherits(data, "data.frame")){
            if(nrow(data) == 0)
               stop("Cannot serialize empty data.frame.", call. = F)
         }

         if(inherits(data, "character")){
            if(!file.exists(data))
               stop(sprintf("File '%s' doesn't exist.", data), call. = F)
            f = data
         } else {
            f = tempfile(fileext = ".npy")
            on.exit(unlink(f))
            self$np$save(f, data)
         }

         obj = readBin(f, what = "raw", n = file.size(f))
         return(obj)
      }
   )
)

# TODO: Serializers:
# - IdentitySerializer
# - JSONLinesSerializer
# - SparseMatrixSerializer (issue write .npz format: possibly look into (https://github.com/scipy/scipy/blob/e777eb9e4a4cd9844629a3c37b3e94902328ad0b/scipy/sparse/_matrix_io.py) in combination with the use of RcppCNPy)

