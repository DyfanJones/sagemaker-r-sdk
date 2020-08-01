
#' @import R6
#' @import RProtobuf

matrix_to_record_serializer = R6Class("matrix_to_record_serializer",
  public = list(
    initialize = function(content_type="application/x-recordio-protobuf"){
      self$content_type = content_type
    },

    serialize = function(array){
      if(length(dim(array)) == 1)
        # reshape array

      obj = raw(0)
      buf = rawConnection(obj, open = "wb")
      write_matrix_to_dense_tensor(buf, array)
      close(buf)

      return(obj)
    }
  )
)

record_deserializer = R6Class("record_deserializer",
  public = list(
    initialize = function(accept="application/x-recordio-protobuf"){
      self$accept = accept
    },

    deserializer = function(stream,
                            content_type){

      # TODO: create read_records function
      tryCatch(read_records(stream),
               finally = function(f) close(stream))
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
  if (!length(dim(mat)) ==2)
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
  # Write each vector in array into a Record in the file object
  for(index in 1:nrow(array)){
    vector = array[i,]
    record = Record$clone()
    .write_feature_tensor(resolved_type, record, vector)
    if (!is.null(labels))
      .write_label_tensor(resolved_label_type, record, labels[index])

    .write_recordio(file, record)
  }
}

write_spmatrix_to_sparse_tensor <- function(file, array, labels=NULL){
  if (!Matrix::is(array, "sparseMatrix"))
    stop("Array must be sparse", call. = F)

  # Validate shape of array and labels, resolve array and label types
  if (!length(dim(array)) == 2)
    stop("Array must be a Matrix", call.=F)
  if (!is.null(labels)){
    # TODO: need to double check this method works in R
    if (length(dim(labels)) != 1)
      stop("Labels must be a Vector", call. = F)
    if (!(dim(labels)[1] %in% dim(array)))
      stop(sprintf("Label shape (%s) not compatible with array shape (%s)",
                   paste(dim(labels), collapse = ", "),
                   paste(dim(array), collapse = ", ")),
           call. = F)
    resolved_label_type = .resolve_type(labels[1])
  }
  resolved_type = .resolve_type(labels[1])

  # csr_array = array.tocsr()
  array_dim = dim(array)


  for (row_idx in 1:nrow(array)){
    record = Record$clone()
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

    # TODO: create .write_recordio function
    # TODO: replace record serializeToString with RProtobuf method
    .write_recordio(file, record.SerializeToString())
  }
}

# Eagerly read a collection of amazon Record protobuf objects from file
read_records = function(file){
  records = list
  # TODO: read_recordio
  for (record_data in read_recordio(file)){
    record = Record()
    record.ParseFromString(record_data)
    records.append(record)
  }
  return(records)
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
  len = length(record$serialize(NULL))
  writeBin(.kmagic, f)
  writeBin(len, f)
  pad = 1 + bitwShiftL(bitwShiftR((len + 3), 2), 2) - len # added +1 to map to R indexing
  serialize_pb(record, f)
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
