# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/6a8bb6df1a0c34990a52fbfe4bbc6ec840d1bdcd/src/sagemaker/vpc_utils.py


# """Checks that an instance of VpcConfig has the expected keys and values,
#   removes unexpected keys, and raises ValueErrors if any expectations are
#   violated
#   Args:
#       vpc_config (dict): a VpcConfig dict containing 'Subnets' and
#           'SecurityGroupIds'
#   Returns:
#       A valid VpcConfig dict containing only 'Subnets' and 'SecurityGroupIds'
#       from the vpc_config parameter If vpc_config parameter is None, returns
#       None
#   Raises:
#       ValueError if any expectations are violated:
#           * vpc_config must be a non-empty dict
#           * vpc_config must have key `Subnets` and the value must be a non-empty list
#           * vpc_config must have key `SecurityGroupIds` and the value must be a non-empty list
#   """
vpc_sannitize = function(vpc_config = NULL){
  if (is.null(vpc_config)) return(vpc_config)
  if (!inherits(vpc_config, "list")) stop("vpc_config is not a `list()`: ", vpc_config, call. = F)

  if (length(vpc_config) == 0) stop("vpc_config is empty", call. = F)

  subnets = vpc_config$Subnets

  if (is.null(subnets)) stop("vpc_config is missing key: Subnets", call. = F)
  if (!inherits(subnets, "list")) stop("vpc_config value for Subnets is not a list: ", subnets, call. = F)

  if (length(subnets) == 0) stop("vpc_config value for Subnets is empty", call. = F)

  security_group_ids = vpc_config$SecurityGroupIds
  if (length(security_group_ids) == 0) stop("vpc_config is missing key: SecurityGroupIds", call. = F)

  if (!inherits(subnets, "list")) stop("vpc_config value for SecurityGroupIds is not a list: ", security_group_ids, call. = F)

  if (length(security_group_ids) == 0) stop("vpc_config value for SecurityGroupIds is empty", call. = F)

  return(list(Subnets = subnets, SecurityGroupIds = security_group_ids))
}

# Extracts subnets and security group ids as lists from a VpcConfig dict
# Args:
#   vpc_config (dict): a VpcConfig dict containing 'Subnets' and
# 'SecurityGroupIds'
# do_sanitize (bool): whether to sanitize the VpcConfig dict before
# extracting values
# Returns:
#   Tuple of lists as (subnets, security_group_ids) If vpc_config parameter
# is None, returns (None, None)
# Raises:
#   * ValueError if sanitize enabled and vpc_config is invalid
#   * KeyError if sanitize disabled and vpc_config is missing key(s)
vpc_from_list = function(vpc_config,
                         do_sanitize=FALSE){
  if (do_sanitize)
    vpc_config = vpc_sannitize(vpc_config)
  if (islistempty(vpc_config))
    return(list(Subnets= NULL, SecurityGroupIds= NULL))
  return (list(Subnets = vpc_config$Subnets, SecurityGroupIds= vpc_config$SecurityGroupIds))
}


# Prepares a VpcConfig dict containing keys 'Subnets' and
# 'SecurityGroupIds' This is the dict format expected by SageMaker
# CreateTrainingJob and CreateModel APIs See
# https://docs.aws.amazon.com/sagemaker/latest/dg/API_VpcConfig.html
# Args:
#   subnets (list): list of subnet IDs to use in VpcConfig
# security_group_ids (list): list of security group IDs to use in
# VpcConfig
# Returns:
#   A VpcConfig dict containing keys 'Subnets' and 'SecurityGroupIds' If
# either or both parameters are None, returns None
vpc_to_list = function(subnets,
                       security_group_ids){
  if (islistempty(subnets) || islistempty(security_group_ids))
    return(NULL)
  return(list(Subnets= subnets, SecurityGroupIds= security_group_ids))
}
