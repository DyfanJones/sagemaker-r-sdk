#' @import paws
#' @import jsonlite
#' @import R6
#' @import logger
#' @import utils
#' @export
PawsCredentials = R6Class("PawsCredentials",
                      public = list(
                        aws_access_key_id = NULL,
                        aws_secret_access_key = NULL ,
                        aws_session_token = NULL,
                        region_name = NULL,
                        profile_name = NULL,
                        credentials = list(),
                        initialize = function(aws_access_key_id = NULL,
                                              aws_secret_access_key = NULL,
                                              aws_session_token = NULL,
                                              region_name = NULL,
                                              profile_name = NULL){
                          self$aws_access_key_id <- aws_access_key_id %||% get_aws_env("AWS_ACCESS_KEY_ID")
                          self$aws_secret_access_key <- aws_secret_access_key %||% get_aws_env("AWS_SECRET_ACCESS_KEY")
                          self$aws_session_token <- aws_session_token %||% get_aws_env("AWS_SESSION_TOKEN")
                          self$region_name <- region_name %||% get_region(profile_name)
                          self$profile_name <- if(!(is.null(self$aws_access_key_id) || is.null(self$aws_secret_access_key) || is.null(self$aws_session_token))) NULL else get_profile_name(profile_name)
                          self$credentials <<- private$cred_set(self$aws_access_key_id, self$aws_secret_access_key, self$aws_session_token, self$profile_name, self$region_name)
                        }
                      ),
                      private = list(
                        # set credentials
                        cred_set = function(aws_access_key_id,
                                             aws_secret_access_key,
                                             aws_session_token,
                                             profile_name,
                                             region_name){
                          add_list <-function(x) if(length(x) == 0) NULL else x
                          config <- list()
                          credentials <- list()
                          cred <- list()

                          cred$access_key_id = aws_access_key_id
                          cred$secret_access_key = aws_secret_access_key
                          cred$session_token = aws_session_token

                          credentials$creds <- add_list(cred)
                          credentials$profile <- profile_name
                          config$credentials <- add_list(credentials)
                          config$region <- region_name

                          config
                          }
                        ),
                      lock_objects = FALSE
                      )

#' @export
paws_cred <- function(aws_access_key_id = NULL,
                      aws_secret_access_key = NULL,
                      aws_session_token = NULL,
                      region_name = NULL,
                      profile_name = NULL){
  PawsCredentials$new(aws_access_key_id, aws_secret_access_key, aws_session_token, region_name, profile_name)
}


