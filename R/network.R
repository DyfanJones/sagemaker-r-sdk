# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/network.py

#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common

#' @title NetworkConfig class
#' @description Accepts network configuration parameters and provides a method to turn these parameters
#'              into a dictionary.
#' @export
NetworkConfig = R6Class("NetworkConfig",
  public = list(

    #' @description Initialize a ``NetworkConfig`` instance. NetworkConfig accepts network configuration
    #'              parameters and provides a method to turn these parameters into a dictionary.
    #' @param enable_network_isolation (bool): Boolean that determines whether to enable
    #'              network isolation.
    #' @param security_group_ids ([str]): A list of strings representing security group IDs.
    #' @param subnets ([str]): A list of strings representing subnets.
    #' @param encrypt_inter_container_traffic (bool): Boolean that determines whether to
    #'              encrypt inter-container traffic. Default value is None.
    initialize = function(enable_network_isolation=FALSE,
                          security_group_ids=NULL,
                          subnets=NULL,
                          encrypt_inter_container_traffic=NULL){
      self$enable_network_isolation = enable_network_isolation
      self$security_group_ids = security_group_ids
      self$subnets = subnets
      self$encrypt_inter_container_traffic = encrypt_inter_container_traffic
    },

    #' @description Generates a request dictionary using the parameters provided to the class.
    to_request_list = function(){
      network_config_request = list("EnableNetworkIsolation"= self$enable_network_isolation)

      if (!is.null(self$encrypt_inter_container_traffic))
        network_config_request$EnableInterContainerTrafficEncryption = self$encrypt_inter_container_traffic

      if (!is.null(self.security_group_ids) %||% !is.null(self$subnets))
        network_config_request$VpcConfig = list()

      if (!is.null(self$security_group_ids))
        network_config_request$VpcConfig$SecurityGroupIds = self$security_group_ids

      if (!is.null(self$subnets))
        network_config_request$VpcConfig$Subnets = self$subnets

      return(network_config_request)
    },

    #' @description format class
    format = function(){
      format_class(self)
    }
  ),
  lock_objects = F
)
