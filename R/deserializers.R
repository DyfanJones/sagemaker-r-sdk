# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/99773e590aa3739b80319ca6200b743855766b39/src/sagemaker/deserializers.py

#' @import R6
#' @import jsonlite
#' @import data.table

#' @title Default BaseDeserializer Class
#' @description  All BaseDeserializer are children of this class. If a custom
#'               BaseDeserializer is desired, inherit this class.
#' @export
BaseDeserializer = R6Class("BaseDeserializer",
  public = list(

    #' @field ACCEPT
    #' format ACCEPTed by BaseDeserializer
    ACCEPT=NULL,

    #' @description Initialize Serializer Class
    initialize = function(){},

    #' @description  Takes raw data stream and deserializes it.
    #' @param stream raw data to be deserialize
    #' @param content_type (str): The MIME type of the data.
    deserialize = function(stream, content_type) {stop("I'm an abstract interface method", call. = F)},

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      print_class(self)
    }
  )
)

#' @title StringBaseDeserializer Class
#' @description  Deserialize raw data stream into a character string
#' @export
StringDeserializer = R6Class("StringBaseDeserializer",
  inherit = BaseDeserializer,
  public = list(

   #' @field encoding
   #' string encoding to be used
   encoding = NULL,
   #' @description Initialize StringBaseDeserializer Class
   initialize = function(){
     self$ACCEPT = "application/json"
   },

   #' @description  Takes raw data stream and deserializes it.
   #' @param stream raw data to be deserialize
   #' @param content_type (str): The MIME type of the data.
   deserialize = function(stream, content_type) {
     obj = rawToChar(stream)
     return(obj)
   }
  )
)

#' @title S3 method to call StringBaseDeserializer class
#' @export
string_deserializer <- StringDeserializer$new()

#' @title CSVDeserializer Class
#' @description  Use csv format to deserialize raw data stream
#' @export
CSVDeserializer = R6Class("CSVDeserializer",
  inherit = BaseDeserializer,
  public = list(
    #' @description Initialize CsvSerializer Class
    initialize = function(){
      self$ACCEPT = c("text/csv")
    },

    #' @description  Takes raw data stream and deserializes it.
    #' @param stream raw data to be deserialize
    #' @param content_type (str): The MIME type of the data.
    deserialize = function(stream, content_type) {
      if(inherits(stream, "raw")){
        TempFile = tempfile()
        write_bin(stream, TempFile)
        dt = fread(TempFile)
        unlink(TempFile)
        return(melt(dt, measure = 1:ncol(dt), value.name ="prediction")[,-"variable"])
      }
      fread(stream, col.names = "prediction")
    }
  )
)

#' @title S3 method to call CSVDeserializer class
#' @export
csv_deserializer <- CSVDeserializer$new()


#' @title JSONDeserializer Class
#' @description  Use json format to deserialize raw data stream
#' @export
JSONDeserializer = R6Class("JSONDeserializer",
  inherit = BaseDeserializer,
  public = list(

   #' @field encoding
   #' string encoding to be used
   encoding = NULL,
   #' @description Initialize StringBaseDeserializer Class
   initialize = function(){
     self$ACCEPT = "application/json"
   },

   #' @description  Takes raw data stream and deserializes it.
   #' @param stream raw data to be deserialize
   #' @param content_type (str): The MIME type of the data.
   deserialize = function(stream, content_type) {
     con = rawConnection(stream)
     on.exit(close(con))
     data = as.data.table(parse_json(con))
     return(data)
   }
  )
)

#' @title S3 method to call JSONDeserializer class
#' @export
json_deserializer <- JSONDeserializer$new()

#' @title NumpySerializer Class
#' @description Deserialize a stream of data in the .npy format.
#'              This serializer class uses python numpy package to deserialize,
#'              R objects through the use of the `reticualte` package.
#' @export
NumpyDeserializer = R6Class("NumpyDeserializer",
  inherit = BaseDeserializer,
  public = list(

    #' @field np
    #' Python Numpy package
    np = NULL,

    #' @description Initialize the dtype and allow_pickle arguments.
    #' @param dtype (str): The dtype of the data (default: None).
    #' @param accept (str): The MIME type that is expected from the inference
    #'              endpoint (default: "application/x-npy").
    #' @param allow_pickle (bool): Allow loading pickled object arrays (default: True).
    initialize = function(dtype=NULL,
                          accept="application/x-npy",
                          allow_pickle=TRUE){
      if(!requireNamespace('reticulate', quietly=TRUE))
        stop('Please install reticulate package and try again', call. = F)
      self$npy = reticulate::import("numpy")

      self$dtype = dtype
      self$accept = accept
      self$allow_pickle = allow_pickle
    },

    #' @description Deserialize data from an inference endpoint into a NumPy array.
    #' @param stream (botocore.response.StreamingBody): Data to be deserialized.
    #' @param content_type (str): The MIME type of the data.
    #' @return matrix: The data deserialized into a NumPy array.
    deserialize = function(stream, content_type){
      tryCatch({
        if(content_type == "application/x-npy"){
          TempFile = tempfile()
          write_bin(stream, TempFile)
          return(self$np$load(TempFile, allow_pickle = self$allow_pickle))
          }
        },
        finally = function(f) unlink(TempFile)
      )
      stop(sprintf("%s cannot read content type %s.",
                   class(self)[1L], content_type),
           call. = F)
    }
  ),
  active = list(

    #' @field ACCEPT
    #' The content types that are expected from the inference endpoint.
    #'        To maintain backwards compatability with legacy images, the
    #'        NumpyDeserializer supports sending only one content type in the Accept
    #'        header.
    ACCEPT = function(){
      return (self$accept)
    }
  )
)


# TODO: DeSerialize classes:
# - BytesDeserializer
# - StreamDeserializer
# - PandasDeserializer (Should R have dplyr / data.table Deserializers?)
# - JSONLinesDeserializer
