# Class to mock R6 classes
# used for testing only

Mock <- R6Class("Mock",
  public = list(
    initialize = function(name, ... ){
      if(!missing(name)) class(self) <- append(name, class(self))
      args = list(...)
      for(arg in names(args)){
        self[[arg]] = args[[arg]]
      }
    },
    return_value = function(value){
      self$value = value
      return(private$.return)
    }
  ),
  private = list(
    .return = function(...){
      return(self$value)
    }
  ),
  lock_objects = F
)
