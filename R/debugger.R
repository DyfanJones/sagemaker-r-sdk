#' @import paws
#' @import jsonlite
#' @import R6
#' @import logger
#' @import utils
#' @export
DebuggerHookConfig <- R6Class("DebuggerHookConfig",
                          public = list(
                            s3_output_path = NULL,
                            container_local_output_path = NULL,
                            hook_parameters = NULL,
                            collection_configs = NULL,
                            initialize = function(s3_output_path = NULL,
                                                  container_local_output_path = NULL,
                                                  hook_parameters = NULL,
                                                  collection_configs = NULL){

                              self$s3_output_path = s3_output_path
                              self$container_local_output_path = container_local_output_path
                              self$hook_parameters = hook_parameters
                              self$collection_configs = collection_configs
                              },

                              to_list = function(){
                                debugger_hook_config_request = list("S3OutputPath"= self$s3_output_path)
                                debugger_hook_config_request[["LocalPath"]] = self$container_local_output_path
                                debugger_hook_config_request[["HookParameters"]] = self$hook_parameters
                                if(!is.null(self$collection_configs)) {
                                  collection_config_request = list("CollectionName"= as.list(self$collection_configs$name))
                                  if(!is.null(self$collection_configs$parameters)){
                                    collection_config_request[["CollectionParameters"]] = as.list(self$collection_configs$parameters)}
                                  debugger_hook_config_request[["CollectionConfigurations"]] = collection_config_request}
                                return(debugger_hook_config_request)
                              }
                            )
)
