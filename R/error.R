#' @import R6

#' @title SagemakerError
#' @description Create Sagemaker error format.
#' @keywords internal
#' @name SagemakerError
SagemakerError = R6Class("SagemakerError",
  public = list(
    #' @description Initialize SagemakerError class
    #' @param ... (character): message to be outputted in error.
    initialize = function(...){
      private$.construct_msg_cls(...)
      stop(private$.construct_error_str(
        private$.error_msg,
        private$.error_cls))
    }
  ),
  private = list(
    .error_cls = NULL,
    .error_msg = NULL,
    .error_str = NULL,
    .construct_msg_cls = function(...){
      msg_list = list(...)
      msg = if(length(msg_list) == 0) NULL else paste(msg_list, collapse = "")
      private$.error_cls = c(class(self)[-length(class(self))], "error", "condition")
      private$.error_msg = paste(Filter(Negate(is.null), c(private$.error_cls[1], msg)), collapse = ". ")
    },
    .construct_error_str = function(msg, class, attributes = NULL){
      .Data = list(message = msg)
      for(i in names(attributes)) .Data[[i]] = attributes[[i]]
      private$.error_str = structure(.Data, class = class)
      return(private$.error_str)
    }
  )
)

#' @rdname SagemakerError
NotImplementedError = R6Class("NotImplementedError", inherit = SagemakerError)

#' @rdname SagemakerError
ValueError = R6Class("ValueError", inherit = SagemakerError)

#' @rdname SagemakerError
TypeError = R6Class("TypeError", inherit = SagemakerError)

# Raised when resource status is not expected and thus not allowed for further execution
#' @rdname SagemakerError
UnexpectedStatusError = R6Class("UnexpectedStatusError", inherit = ValueError,
  public = list(

    #' @description Initialize UnexpectedStatusError class
    #' @param ... (character): message to be outputted in error.
    #' @param allowed_statuses (character): allowed status from sagemaker
    #' @param actual_status (character): returning status from aws sagemaker
    initialize = function(..., allowed_statuses, actual_status){
      private$.construct_msg_cls(...)
      stop(private$.construct_error_str(
        private$.error_msg,
        private$.error_cls,
        as.list(environment())))
    }
  )
)

#' @rdname SagemakerError
RuntimeError = R6Class("RuntimeError", inherit = SagemakerError)

#' @rdname SagemakerError
AttributeError = R6Class("AttributeError", inherit = SagemakerError)
