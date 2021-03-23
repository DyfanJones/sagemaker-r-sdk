# Class to mock R6 classes
# used for testing only
Mock <- R6::R6Class("Mock",
  public = list(
    initialize = function(name, ...){
      if(!missing(name)) class(self) <- append(name, class(self))
      args = list(...)
      # dynamically assign public methods
      sapply(names(args), function(i) self[[i]] = args[[i]])
    },
    return_value = function(value, .min_var = 1){
      if(is.function(value))
        stop("`value` is a function, please use `side_effect`.", call. = FALSE)
      private$.value = value
      private$.min_var = .min_var
      return(private$.return)
    },

    side_effect = function(effect){
      private$.effect = effect
      return(private$.effect)
    },

    call_args = function(name){
      self[[name]] = function(...){
        args = list(...)
        if(length(args) != 0)
          private[[paste0(".",name)]] = args
        return(private[[paste0(".",name)]])
      }
    }
  ),
  private = list(
    .value = NULL,
    .effect = NULL,
    .min_var = NULL,
    .return = function(...){
      args = list(...)

      # capture arguments
      if(length(args) > 0)
        self$.call_args = args

      if(private$.min_var >= 1 && length(args) == 0)
        return(self)

      return(private$.value)
    }
  ),
  lock_objects = F
)
