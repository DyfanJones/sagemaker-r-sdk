

#' @import R6
#' @import jsonlite
#' @import data.table


#' @title Default BaseDeserializer Class
#' @description  All BaseDeserializer are children of this class. If a custom
#'               BaseDeserializer is desired, inherit this class.
#' @export
BaseDeserializer = R6Class("BaseDeserializer",
  public = list(

   #' @field accept
   #' format accepted by BaseDeserializer
   accept = NULL,

   #' @description Initialize Serializer Class
   initialize = function(){},

   #' @description  Takes raw data stream and deserializes it.
   #' @param stream raw data to be deserialize
   deserialize = function(stream) {stop("I'm an abstract interface method", call. = F)},

   #' @description
   #' Printer.
   #' @param ... (ignored).
   print = function(...){
     cat("<BaseDeserializer>")
     invisible(self)
   }
  )
)

#' @title CsvBaseDeserializer Class
#' @description  Use csv format to deserialize raw data stream
#' @export
CsvDeserializer = R6Class("CsvBaseDeserializer",
  inherit = BaseDeserializer,
  public = list(
    #' @description Initialize CsvSerializer Class
    initialize = function(){
      self$accept = "text/csv"
    },

    #' @description  Takes raw data stream and deserializes it.
    #' @param stream raw data to be deserialize
    deserialize = function(stream) {
      if(inherits(stream, "raw")){
        TempFile = tempfile()
        write_bin(stream, TempFile)
        dt = fread(TempFile)
        unlink(TempFile)
        return(melt(dt, measure = 1:ncol(dt), value.name ="prediction")[,-"variable"])
      }
      fread(stream, col.names = "prediction")
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      cat("<CsvBaseDeserializer>")
      invisible(self)
    }
  )
)

#' @title S3 method to call CsvBaseDeserializer class
#' @export
csv_deserializer <- CsvDeserializer$new()

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
     self$accept = "text"
   },

   #' @description  Takes raw data stream and deserializes it.
   #' @param stream raw data to be deserialize
   deserialize = function(stream) {
     obj = rawToChar(stream)
     return(obj)
   },

   #' @description
   #' Printer.
   #' @param ... (ignored).
   print = function(...){
     cat("<StringBaseDeserializer>")
     invisible(self)
   }
  )
)

#' @title S3 method to call StringBaseDeserializer class
#' @export
string_deserializer <- StringDeserializer$new()

#' @title JsonBaseDeserializer Class
#' @description  Use json format to deserialize raw data stream
#' @export
JsonDeserializer = R6Class("JsonBaseDeserializer",
  inherit = BaseDeserializer,
  public = list(

   #' @field encoding
   #' string encoding to be used
   encoding = NULL,
   #' @description Initialize StringBaseDeserializer Class
   initialize = function(){
     self$accept = "application/json"
   },

   #' @description  Takes raw data stream and deserializes it.
   #' @param stream raw data to be deserialize
   deserialize = function(stream) {
     con = rawConnection(stream)
     on.exit(close(con))
     data = as.data.table(parse_json(con))
     return(data)
   },

   #' @description
   #' Printer.
   #' @param ... (ignored).
   print = function(...){
     cat("<JsonBaseDeserializer>")
     invisible(self)
   }
  )
)

#' @title S3 method to call StringBaseDeserializer class
#' @export
json_deserializer <- JsonDeserializer$new()
