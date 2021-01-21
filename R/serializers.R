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

   #' @description Take data of various data formats and serialize them into CSV.
   #' @param data (object): Data to be serialized
   #' @return object: Serialized data used for a request.
   serialize = function(data) {stop("I'm an abstract interface method", call. = F)},

   #' @description
   #' Printer.
   #' @param ... (ignored).
   print = function(...){
      print_class(self)
   }
  ),
  active = list(
     #' @field CONTENT_TYPE
     #' The MIME type of the data sent to the inference endpoint.
     CONTENT_TYPE = function(){}
  )
)

#' @title Abstract base class for creation of new serializers.
#' @description This class extends the API of :class:~`sagemaker.serializers.BaseSerializer` with more
#'              user-friendly options for setting the Content-Type header, in situations where it can be
#'              provided at init and freely updated.
#' @export
SimpleBaseSerializer = R6Class("SimpleBaseSerializer",
   inherit = BaseSerializer,
   public = list(

      #' @field content_type
      #' The data MIME type
      content_type = NULL,

      #' @description Initialize a ``SimpleBaseSerializer`` instance.
      #' @param content_type (str): The MIME type to signal to the inference endpoint when sending
      #'              request data (default: "application/json").
      initialize = function(content_type = "application/json"){
         if (!is.character(content_type)){
            stop("content_type must be a string specifying the MIME type of the data sent in ",
                 sprintf("requests: e.g. 'application/json', 'text/csv', etc. Got %s", content_type),
                 call. = F)
         }
         self$content_type = content_type
      },

      #' @description Take data of various data formats and serialize them into CSV.
      #' @param data (object): Data to be serialized
      serialize = function(data) {stop("I'm an abstract interface method", call. = F)}
   ),
   active = list(
   #' @field CONTENT_TYPE
   #' The data MIME type set in the Content-Type header on prediction endpoint requests.
   CONTENT_TYPE = function(){
      return(self$content_type)
      }
   )
)

#' @title CSVSerializer Class
#' @description Make Raw data using text/csv format
#' @export
CSVSerializer = R6Class("CSVSerializer",
  inherit = SimpleBaseSerializer,
  public = list(
    #' @description Initialize a ``CSVSerializer`` instance.
    #' @param content_type (str): The MIME type to signal to the inference endpoint when sending
    #'              request data (default: "text/csv").
    initialize = function(content_type="text/csv"){
       super$initialize(content_type=content_type)
    },
    #' @description Take data of various data formats and serialize them into CSV.
    #' @param data (object): Data to be serialized. Any list of same length vectors; e.g. data.frame and data.table.
    #'               If matrix, it gets internally coerced to data.table preserving col names but not row names
    serialize = function(data) {
      TempFile = tempfile()
      fwrite(data, TempFile, col.names = FALSE, showProgress = FALSE)
      obj = readBin(TempFile, "raw", n = file.size(TempFile))
      unlink(TempFile)
      return(obj)
    }
  )
)

#' @title S3 method to call CSVSerializer class
#' @export
csv_serializer <- CSVSerializer$new()

#' @title NumpySerializer Class
#' @description Serialize data of various formats to a numpy npy file format.
#'              This serializer class uses python numpy package to serialize,
#'              R objects through the use of the `reticualte` package.
#' @export
NumpySerializer = R6Class("NumpySerializer",
   inherit = SimpleBaseSerializer,
   public = list(

     #' @field np
     #' Python Numpy package
     np = NULL,

     #' @field dtype
     #' The dtype of the data
     dtype = NULL,

     #' @description Initialize a ``NumpySerializer`` instance.
     #' @param content_type (str): The MIME type to signal to the inference endpoint when sending
     #'              request data (default: "application/x-npy").
     #' @param dtype (str): The dtype of the data.
     initialize = function(dtype=NULL,
                           content_type="application/x-npy"){
        super$initialize(content_type = content_type)
        self$dtype = dtype
        if(!requireNamespace('reticulate', quietly=TRUE))
           stop('Please install reticulate package and try again', call. = F)
        self$np = reticulate::import("numpy")
     },

     #' @description Serialize data to a buffer using the .npy format.
     #' @param data (object): Data to be serialized. Can be a NumPy array, list,
     #'              file, or buffer.
     #' @return (raw): A buffer containing data serialzied in the .npy format.
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

#' @title JSONSerializer Class
#' @description Serialize data to a JSON formatted string.
#' @export
JSONSerializer = R6Class("JSONSerializer",
  inherit = SimpleBaseSerializer,
  public = list(

     #' @description Serialize data of various formats to a JSON formatted string.
     #' @param data (object): Data to be serialized.
     #' @return (raw): The data serialized as a JSON string.
     serialize = function(data){
        con = rawConnection(raw(0), "r+")
        on.exit(close(con))
        write_json(data, con, dataframe = "columns", auto_unbox = F)
        return(rawConnectionValue(con))
      }
   )
)

#' @title S3 method to call JsonSerializer class
#' @export
json_serializer <- JSONSerializer$new()

#' @title Serialize data by returning data without modification.
#' @description This serializer may be useful if, for example, you're sending raw bytes such as from an image
#'              file's .read() method.
#' @export
IdentitySerializer = R6Class("IdentitySerializer",
  inherit = SimpleBaseSerializer,
  public = list(

    #' @description Initialize an ``IdentitySerializer`` instance.
    #' @param content_type (str): The MIME type to signal to the inference endpoint when sending
    #'              request data (default: "application/octet-stream").
    initialize = function(content_type="application/octet-stream"){
      super$intialize(content_type = content_type)
    },

    #' @description Return data without modification.
    #' @param data (object): Data to be serialized.
    #' @return object: The unmodified data.
    serialize = function(data){
      return(data)
    }
  )
)

#' @title JSONLinesSerializer Class
#' @description Serialize data to a JSON Lines formatted string.
#' @export
JSONLinesSerializer = R6Class("IdentitySerializer",
  inherit = SimpleBaseSerializer,
  public = list(

    #' @description Initialize a ``JSONLinesSerializer`` instance.
    #' @param content_type (str): The MIME type to signal to the inference endpoint when sending
    #'              request data (default: "application/jsonlines").
    initialize = function(content_type="application/jsonlines"){
      super$initialize(content_type = content_type)
    },

    #' @description Serialize data of various formats to a JSON Lines formatted string.
    #' @param data (object): Data to be serialized. The data can be a string,
    #'              iterable of JSON serializable objects, or a file-like object.
    #' @return str: The data serialized as a string containing newline-separated
    #'              JSON values.
    serialize = function(data){
      con = rawConnection(raw(0), "r+")
      on.exit(close(con))
      stream_out(data, con = con, verbose = FALSE)
      return(rawConnectionValue(con))
    }
  )
)

#' @title LibSVMSerializer Class
#' @description Serialize data of various formats to a LibSVM-formatted string.
#'              The data must already be in LIBSVM file format:
#'              <label> <index1>:<value1> <index2>:<value2> ...
#'              It is suitable for sparse datasets since it does not store zero-valued
#'              features.
#' @export
LibSVMSerializer = R6Class("LibSVMSerializer",
  inherit = SimpleBaseSerializer,
  public = list(

    #' @description Initialize a ``LibSVMSerializer`` instance.
    #' @param content_type (str): The MIME type to signal to the inference endpoint when sending
    #'              request data (default: "text/libsvm").
    initialize = function(content_type="text/libsvm"){
      super$initialize(content_type = content_type)
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

# TODO: Serializers:
# - SparseMatrixSerializer (issue write .npz format:
#       possibly look into (https://github.com/scipy/scipy/blob/e777eb9e4a4cd9844629a3c37b3e94902328ad0b/scipy/sparse/_matrix_io.py) in combination with the use of RcppCNPy)
