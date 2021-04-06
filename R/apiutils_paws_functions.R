# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/apiutils/_boto_functions.py

#' @include utils.R

#' @importFrom tools toTitleCase

PawsFunctions = R6Class("PawsFunctions",
  public = list(

    # Convert a snake case string to camel case.
    # Args:
    #   snake_case (str): String to convert to camel case.
    # Returns:
    #   str: String converted to camel case.
    to_camel_case = function(snake_case){
      return(paste(toTitleCase(split_str(snake_case, "_")), collapse = ""))
    },

    # Convert a camel case string to snake case.
    # Args:
    #   name (str): String to convert to snake case.
    # Returns:
    #   str: String converted to snake case.
    to_snake_case = function(name){
      s1 = gsub("(.)([A-Z][a-z]+)", "\\1_\\2", name)
      return(tolower(gsub("([a-z0-9])([A-Z])", "\\1_\\2", s1)))
    },

    # Convert an UpperCamelCase boto response to a snake case representation.
    # Args:
    #   boto_dict (dict[str, ?]): A boto response dictionary.
    # boto_name_to_member_name (dict[str, str]):  A map from boto name to snake_case name.
    # If a given boto name is not in the map then a default mapping is applied.
    # member_name_to_type (dict[str, (_base_types.ApiObject, boolean)]): A map from snake case
    # name to a type description tuple. The first element of the tuple, a subclass of
    # ApiObject, is the type of the mapped object. The second element indicates whether the
    # mapped element is a collection or singleton.
    # Returns:
    #   dict: Boto response in snake case.
    from_paws = function(paws_dict,
                         paws_name_to_member_name,
                         member_name_to_type){
      from_paws_values = list()
      for (i in paws_dict){
        paws_name = names(paws_dict)[i]
        paws_value = paws_dict[[i]]
        # Convert the paws_name to a snake-case name by preferentially looking up the boto name in
        # boto_name_to_member_name before defaulting to the snake case representation
        member_name = paws_name_to_member_name[[paws_name]] %||% self$to_snake_case(paws_name)

        # If the member name maps to a subclass of _base_types.ApiObject
        # (i.e. it's in member_name_to_type), then transform its boto dictionary using that type:
        if (member_name %in% names(member_name_to_type)){
          sub_ll <- member_name_to_type[[member_name]]
          names(sub_ll) <- c("api_type", "is_collection")
          if (sub_ll$is_collection){
            if (is.list(paws_value))
              member_value = sub_ll$api_type$from_paws(boto_value)
            else
              member_value = lapply(paws_value, function(item) sub_ll$api_type$from_paws(item))
          } else {
            member_value = sub_ll$api_type$from_paws(paws_value)
            # If the member name does not have a custom type definition then simply assign it the
            # boto value.  Appropriate if the type is simple and requires not further conversion (e.g.
            # a number or list of strings).
          }
        } else
          member_value = paws_value

        from_paws_values[[member_name]] = member_value
      }
      return(from_paws_values)
    },

    #Convert a dict of of snake case names to values into a boto UpperCamelCase representation.
    # Args:
    #   member_vars dict[str, ?]: A map from snake case name to value.
    # member_name_to_boto_name dict[str, ?]: A map from snake_case name to boto name.
    # Returns:
    #   dict: boto dict converted to snake case
    to_paws = function(member_vars,
                       member_name_to_paws_name,
                       member_name_to_type){
      to_paws_values = list()
      # Strip out all entries in member_vars that have a None value. None values are treated as
      # not having a value
      # set, required as API operations can have optional parameters that may not take a null value.
      member_vars = Filter(Negate(is.null), member_vars)

      # Iterate over each snake_case name and its value and map to a camel case name. If the value
      # is an ApiObject subclass then recursively map its entries.
      for (i in member_vars){
        member_name = names(member_vars)[[i]]
        member_value = member_vars[[i]]
        paws_name = member_name_to_paws_name[[member_name]] %||% self$to_camel_case(member_name)
        sub_ll =  member_name_to_type[[member_name]] %||% list(NULL, NULL)
        names(sub_ll) = c("api_type", "is_api_collection_type")
        if (sub_ll$is_api_collection_type && is.list(member_value)){
          paws_value = lapply(member_value, function(v) if(!is.null(sub_ll$api_type)) sub_ll$api_type$to_paws(v) else v)
          if(!is.null(names(member_value)))
            names(paws_value) = names(member_value)
        } else {
          paws_value = if (!is.null(sub_ll$api_type)) sub_ll$api_type$to_paws(member_value) else member_value
        }
        to_paws_values[[paws_name]] = paws_value
      }
      return(to_paws_values)
    }
  )
)
