# Class to mock R6 classes
# used for testing only
Mock <- R6::R6Class("Mock",
  public = list(
    initialize = function(name, ...){
      if(!missing(name)) class(self) <- append(name, class(self))
      args = list(...)
      for(arg in names(args)){
        self[[arg]] = args[[arg]]
      }
    },
    return_value = function(value){
      if(is.function(value))
        stop("`value` is a function, please use `side_effect`.", call. = FALSE)
      private$.value = value
      return(private$.return)
    },

    side_effect = function(effect){
      private$.effect = effect
      return(private$.effect)
    },

    add_function = function(name, meth) {
      self[[name]] <- meth
      environment(self[[name]]) <- environment(self$add_function)
    }
  ),
  private = list(
    .value = NULL,
    .effect = NULL,
    .return = function(...){
      return(private$.value)
    }
  ),
  lock_objects = F
)

call_args = function(...){
  private$.args = list(...)
  return(private$.args)
}
