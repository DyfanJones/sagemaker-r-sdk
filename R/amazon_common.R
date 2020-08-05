# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/common.py

#' @import R6
#' @importFrom RProtoBuf read

#' @include amazon_record_pb2.R
#' @include serializers.R
#' @include deserializers.R

#' @title RecordSerializer Class
#' @description Serialize a matrices and array for an inference request.
#' @export
RecordSerializer = R6Class("RecordSerializer",
  inherit = BaseSerializer,
  public = list(

    #' @description intialize RecordSerializer class
    initialize = function(){
      self$CONTENT_TYPE = "application/x-recordio-protobuf"
    },

    #' @description Serialize a NumPy array into a buffer containing RecordIO records.
    #' @param data (numpy.ndarray): The data to serialize.
    #' @return raw: A buffer containing the data serialized as records.
    serialize = function(data){
      if(is.vector(data))
        data = as.array(data)

      if(length(dim(data)) == 1)
        data = matrix(data, 1, dim(data)[1])

      obj = raw(0)
      buf = rawConnection(obj, open = "wb")
      on.exit(close(buf))

      write_matrix_to_dense_tensor(buf, data)

      return(rawConnectionValue(buf))
    }
  )
)

#' @title RecordDeserializer Class
#' @description Deserialize RecordIO Protobuf data from an inference endpoint.
#' @export
RecordDeserializer = R6Class("RecordDeserializer",
  inherit = BaseDeserializer,
  public = list(

    #' @description intialize RecordDeserializer class
    initialize = function(){
      self$ACCEPT = "application/x-recordio-protobuf"
    },

    #' @description Deserialize RecordIO Protobuf data from an inference endpoint.
    #' @param data (object): The protobuf message to deserialize.
    #' @param content_type (str): The MIME type of the data.
    #' @return list: A list of records.
    deserializer = function(data, content_type){
      tryCatch(read_records_io(data),
               finally = function(f) close(data))
    }
  )
)

.write_feature_tensor <- function(resolved_type, record, vector){
  if (resolved_type == "Int32")
    record$features[[1]]$value$int32_tensor$values <- vector
  if (resolved_type == "Float64")
    record$features[[1]]$value$float64_tensor$values <- vector
  if (resolved_type == "Float32")
    record$features[[1]]$value$float32_tensor$values <- vector
}

.write_label_tensor <- function(resolved_type, record, scalar){
  if (resolved_type == "Int32")
    record$label[[1]]$values$int32_tensor$values <- c(scalar)
  if (resolved_type == "Float64")
    record$label[[1]]$values$float64_tensor$values <- c(scalar)
  if (resolved_type == "Float32")
    record$label[[1]]$values$float32_tensor$values <- c(scalar)
}

.write_keys_tensor <- function(resolved_type, record, vector){
  if (resolved_type == "Int32")
    record$features[[1]]$value$int32_tensor$keys <- vector
  if (resolved_type == "Float64")
    record$features[[1]]$value$float64_tensor$keys <- vector
  if (resolved_type == "Float32")
    record.features[[1]]$value$float32_tensor$keys <- vector
}

.write_shape <- function(resolved_type, record, scalar){
  if(resolved_type == "Int32")
    record$features[[1]]$value$int32_tensor$shape <- c(scalar)
  if (resolved_type == "Float64")
    record$features[[1]]$values$float64_tensor$shape <- c(scalar)
  if (resolved_type == "Float32")
    record$features[[1]]$value$float32_tensor$shape <- c(scalar)
}

write_matrix_to_dense_tensor <- function(file, array, labels = NULL){
  # Validate shape of array and labels, resolve array and label types
  if (!length(dim(array)) ==2)
    stop("Array must be a Matrix", call. = F)

  if(!is.null(labels)){
    if (!length(dim(labels)) == 1)
      stop("Labels must be a Vector", call. = F)
    if (!(dim(labels)[1] %in% dim(array)))
        stop(sprintf("Label shape (%s) not compatible with array shape (%s)",
                     paste(dim(labels), collapse = ", "),
                     paste(dim(array), collapse = ", ")),
             call. = F)
    # As matrix/sparse consist of all elements being the same clase can check first element
    resolved_label_type = .resolve_type(labels[1])
  }
  resolved_type = .resolve_type(array[1])
  record = Record()
  # Write each vector in array into a Record in the file object
  for(index in 1:nrow(array)){
    vector = array[index,]
    record$clear()
    record$update(features = .RECORD_FEATURESENTRY(), label = .RECORD_LABELENTRY())
    .write_feature_tensor(resolved_type, record, vector)
    if (!is.null(labels))
      .write_label_tensor(resolved_label_type, record, labels[index])

    .write_recordio(file, record$serialize(NULL))
  }
}

write_spmatrix_to_sparse_tensor <- function(file, array, labels=NULL){
  if (!Matrix::is(array, "sparseMatrix"))
    stop("Array must be sparse", call. = F)

  # Validate shape of array and labels, resolve array and label types
  if (!length(dim(array)) == 2)
    stop("Array must be a Matrix", call.=F)
  if (!is.null(labels)){
    if (!is.vector(labels))
      stop("Labels must be a Vector", call. = F)
    if (!(length(labels) %in% dim(array)))
      stop(sprintf("Label shape (%s) not compatible with array shape (%s)",
                   paste(dim(labels), collapse = ", "),
                   paste(dim(array), collapse = ", ")),
           call. = F)
    resolved_label_type = .resolve_type(labels[1])
  }
  resolved_type = .resolve_type(labels[1])

  for (row_idx in 1:nrow(array)){
    record = Record()
    row = array[row_idx,]

    # Write values
    .write_feature_tensor(resolved_type, record, row.data)

    # Write keys
    .write_keys_tensor(resolved_type, record, row.indices.astype(np.uint64))

    # Write labels
    if (!is.null(labels))
      .write_label_tensor(resolved_label_type, record, labels[row_idx])

    # Write shape
    .write_shape(resolved_type, record, n_cols)

    .write_recordio(file, record$serialize(NULL))
  }
}

# Eagerly read a collection of amazon Record protobuf objects from file
read_records_io = function(file){

  # create raw connection
  f = rawConnection(file,  "rb")
  on.exit(close(f))

  records = list()
  i = 1

  while(TRUE){
    read_kmagic = readBin(f, "numeric", n = 1)
    (check = read_kmagic == .kmagic)

    # break loop
    if(!check || length(check) == 0)
      return(records)

    len_record = readBin(f, "int", n = 1)

    data = readBin(f, "raw", n = len_record)

    pad = bitwShiftL(bitwShiftR((len_record + 3), 2), 2) - len_record

    record = read(aialgs.data.Record, data)
    records[[i]] = record
    i = i +1

    if(pad > 0)
      readBin(f, "raw", pad)
  }
}

# MXNet requires recordio records have length in bytes that's a multiple of 4
# This sets up padding bytes to append to the end of the record, for diferent
# amounts of padding required.
padding = list()
for (amount in 0:3){
  padding[[amount+1]] = writeBin(rep(0x00, amount), raw())
}

.kmagic = 0xCED7230A

.write_recordio = function(f, data){
  len = length(data)
  writeBin(.kmagic, f)
  writeBin(len, f)
  pad = 1 + bitwShiftL(bitwShiftR((len + 3), 2), 2) - len # added +1 to map to R indexing
  writeBin(data, f)
  writeBin(padding[[pad]], f)
}

.resolve_type=function(dtype){
  switch(typeof(dtype),
         integer =   "Int32",
         integer64 = "Int32",
         numeric =   "Float64",
         double = "Float64",
         stop(sprintf("Unsupported dtype %s on array",dtype), call. = FALSE)
  )
}
