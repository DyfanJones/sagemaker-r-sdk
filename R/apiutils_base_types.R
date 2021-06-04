# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/apiutils/_base_types.py

#' @include r_utils.R
#' @include apiutils_paws_functions.R

#' @import R6

# A Python class representation of a boto API object.
# Converts boto dicts of 'UpperCamelCase' names to dicts into/from a Python object with standard
# python members. Clients invoke to_boto on an instance of ApiObject to transform the ApiObject
# into a boto representation. Clients invoke from_boto on a sub-class of ApiObject to
# instantiate an instance of that class from a boto representation.
ApiObject = R6Class("ApiObject",
  public = list(

    # Init ApiObject.
    initialize = function(...){
      args = list(...)
      private$.args = names(args)
      sapply(private$.args, function(i) self[[i]] = args[[i]])
    },

    # Construct an instance of this ApiObject from a boto response.
    # paws_dict (dict): A dictionary of a boto response.
    # ... : Arbitrary keyword arguments
    from_paws = function(paws_dict,
                         ...){
      if (!islistempty(paws_dict))
        return(NULL)

      for(ign in private$.paws_ignore()) {
        paws_dict[[ign]] <- NULL
      }

      if(!islistempty(private$.custom_paws_names)){
        custom_paws_names_to_member_names = names(private$.custom_paws_names)
        names(custom_paws_names_to_member_names) = unname(private$.custom_paws_names)
      } else {
        custom_paws_names_to_member_names = list()
      }

      cls_kwargs = PawsFunctions$new()$from_paws(
        paws_dict, custom_paws_names_to_member_names, private$.custom_paws_types
      )
      cls_kwargs = c(cls_kwargs, kwargs)

      cls = self$clone()
      return(do.call(cls$new, cls_kwargs))
    },

    # Convert an object to a boto representation.
    # Args:
    #   obj (dict): The object to convert to boto.
    to_paws = function(obj){
      if (!is.list(obj))
        var_dict = as.list(obj)
      else
        var_dict = obj
      return (PawsFunctions$new()$to_paws(var_dict, private$.custom_paws_names, private$.custom_paws_types))
    },

    # Return a string representation of this ApiObject.
    # ... (ignored).
    print = function(...){
      ll = as.list(self)[private$.args]
      cat(sprintf("%s(%s)", class(self)[[1]], paste(names(ll), ll, sep = "=",  collapse = ",")))
      invisible(self)
    }
  ),
  private = list(
    # store methods created by class
    .args = NULL,

    # A map from boto 'UpperCamelCase' name to member name. If a boto name does not appear in
    # this dict then it is converted to lower_snake_case.
    .custom_paws_names = list(),

    # A map from name to an ApiObject subclass. Allows ApiObjects to contain ApiObject members.
    .custom_paws_types = list(),

    # Response fields to ignore by default.
    .paws_ignore = function(){
      return(list("ResponseMetadata"))
    }
  ),
  lock_objects = F
)

# -----------------
# Return True if this ApiObject equals other.
`==.ApiObject` <- function(self, other){
  if(inherits(other, class(self)[[1]]))
    return(identical(
      as.list(self)[self$.__enclos_env__$private$.args],
      as.list(other)[other$.__enclos_env__$private$.args])
    )
  return(FALSE)
}

`!=.ApiObject` <- function(self, other){
  if(inherits(other, class(self)[[1]]))
    return(!identical(
      as.list(self)[self$.__enclos_env__$private$.args],
      as.list(other)[self$.__enclos_env__$private$.args])
    )
  return(TRUE)
}
# -----------------


#' @title Record class
#' @description A boto based Active Record class based on convention of CRUD operations.
#' @export
Record = R6Class("Record",
  inherit = ApiObject,
  public = list(

    #' @description Init Record.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    #' @param ... parameters passed to `R6` class `ApiObject`
    initialize = function(sagemaker_session=NULL, ...){
      self$sagemaker_session = sagemaker_session
      super$initialize(...)
    },

    #' @description Update this ApiObject with a paws response.
    #' @param paws_list (dict): A dictionary of a paws response.
    with_paws = function(paws_list){
      custom_boto_names_to_member_names = as.list(names(private$.custom_paws_names))
      names(custom_boto_names_to_member_names) = unname(private$.custom_paws_names)

      p_l = PawsFunctions$new()$from_paws(
        paws_list, custom_boto_names_to_member_names, private$.custom_paws_types)
      sapply(names(p_l), function(i) self[[i]] = p_l[[i]])
      return(self)
    }
  ),
  private = list(

    # update / delete / list method names
    .paws_update_method = NULL,
    .paws_delete_method = NULL,
    .paws_list_method = NULL,

    # List of member names to convert to paws representations and pass to the update method.
    .paws_update_members = list(),

    # List of member names to convert to paws representations and pass to the delete method.
    .paws_delete_members = list(),

    .list = function(paws_list_method,
                     list_item_factory,
                     paws_list_items_name,
                     paws_next_token_name="NextToken",
                     sagemaker_session=NULL,
                     ...){
      kwargs = list(...)
      sagemaker_session = sagemaker_session %||% Session$new()
      sagemaker_client = sagemaker_session$sagemaker

      list_method = sagemaker_client[[paws_list_method]] %||% stop("Method not identified.", call. = F)

      list_request_kwargs = PawsFunctions$new()$to_paws(
        kwargs, private$.custom_boto_names, private$.custom_paws_types)

      next_token = NULL
      resp_ll <- list()
      tryCatch({
        while(!identical(next_token, character(0))){
          list_request_kwargs[[paws_next_token_name]] = next_token
          response_chunk = do.call(list_method, list_request_kwargs)
          list_items = response_chunk[[paws_list_items_name]] %||% list()
          next_token = response_chunk[[paws_next_token_name]]

          resp_ll = c(resp_ll, list_items)}
        },
        error = function(e) NULL
      )
      return(lapply(resp_ll, function(ll) lapply(ll, list_item_factory)))
    },

    # Search for objects with the SageMaker API.
    .search = function(search_resource,
                       search_item_factory,
                       paws_next_token_name="NextToken",
                       sagemaker_session=NULL,
                       ...){
      sagemaker_session = sagemaker_session %||% Session$new()
      sagemaker_client = sagemaker_session$sagemaker

      search_method = sagemaker_client[["search"]]
      search_request_kwargs = PawsFunctions$new()$to_paws(
        kwargs, private$.custom_paws_names,
        private$.custom_paws_types)
      search_request_kwargs[["Resource"]] = search_resource

      next_token = NULL
      resp_ll <- list()

      tryCatch({

        while(!identical(next_token, character(0))){
          search_request_kwargs[[paws_next_token_name]] = next_token
          search_method_response = do.call(search_method, search_request_kwargs)
          search_items = search_method_response[["Results"]] %||% list()
          next_token = search_method_response[[paws_next_token_name]]

          resp_ll = c(resp_ll, search_items)}
        },
        error = function(e) NULL
      )
      return(
        lapply(resp_ll, function(ll) {
          lapply(ll, function(item) search_item_factory(item[[class(self)[[1]]]]))})
      )
    },

    # Create and invoke a SageMaker API call request.
    .construct = function(paws_method_name,
                          sagemaker_session=NULL,
                          ...){
      sagemaker_session = sagemaker_session %||% Session$new()
      cls = self$clone()
      instance = cls$initialize(sagemaker_session, ...)
      return (instance$.__enclos_env__$private$.invoke_api(paws_method_name, ...))
    },

    # Set tags on this ApiObject.
    # Args:
    #   resource_arn (str): The arn of the Record
    # tags (dict): An array of Tag objects that set to Record
    # Returns:
    #   A list of key, value pair objects. i.e. [{"key":"value"}]
    .set_tags = function(resource_arn=NULL, tags=NULL){
      tag_list = self$sagemaker_session$sagemaker$add_tags(
        ResourceArn=resource_arn, Tags=tags)[["Tags"]]
      return(tag_list)
    },

    # Invoke a SageMaker API.
    .invoke_api = function(paws_method,
                           paws_method_members){
      api_values = as.list(self)[private$.args[paws_method_members]]
      api_kwargs = self$to_paws(api_values)
      api_method = self$sagemaker_session$sagemaker[[paws_method]]
      api_paws_response = do.call(api_method, api_kwargs)
      return(self$with_paws(api_paws_response))
    }
  ),
  lock_objects = F
)
