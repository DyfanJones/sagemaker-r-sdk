# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/_studio.py

#' @include r_utils.R

#' @import R6sagemaker.common
#' @import lgr

STUDIO_PROJECT_CONFIG = ".sagemaker-code-config"

# Appends the project tag to the list of tags, if it exists.
# Args:
#   working_dir: the working directory to start looking.
# tags: the list of tags to append to.
# Returns:
#   A possibly extended list of tags that includes the project id
.append_project_tags <- function(tags = NULL, working_dir = NULL){
  path = .find_config(working_dir)
  if (is.null(path))
    return(tags)

  config = .load_config(path)
  if (is.null(config))
    return(tags)

  additional_tags = .parse_tags(config)
  if (is.null(additional_tags))
    return (tags)

  all_tags = tags %||% list()
  all_tags = c(all_tags, additional_tags)

  return(all_tags)
}

# Gets project config on SageMaker Studio platforms, if it exists.
# Args:
#   working_dir: the working directory to start looking.
# Returns:
#   The project config path, if it exists. Otherwise None.
.find_config <- function(working_dir = NULL){
  tryCatch({
    wd = if(!is.null(working_dir)) work_dir else getwd()
    path = NULL
    while(is.null(path) && !grepl("/", wd)){
      candidate = file.path(wd, STUDIO_PROJECT_CONFIG)
      if (file.exists(candidate))
        path = candidate
      wd = dirname(candidate)
    }
    return(path)
  },
  error = function(e){
    LOGGER$debug("Could not find the studio project config. %s", e)}
  )
}

# Parse out the projectId attribute if it exists at path.
# Args:
#   path: path to project config
# Returns:
#   Project config Json, or None if it does not exist.
.load_config <- function(path){
  if(!file.exists(path)){
    LOGGER$debug("Could not load project config. No such file or directory: %s", path)
    return(NULL)}
  tryCatch({
    config = jsonlite::read_json(path)
    return(config)},
    error = function(e){
      LOGGER$debug("Could not load project config. %s", e)
    }
  )
}

# Parse out appropriate attributes and formats as tags.
# Args:
#   config: project config dict
# Returns:
#   List of tags
.parse_tags <- function(config){
  if(!islistempty(config$sagemakerProjectId) || !islistempty(config$sagemakerProjectName))
    return(list(
      list("Key"= "sagemaker:project-id", "Value"= config$sagemakerProjectId),
      list("Key"= "sagemaker:project-name", "Value"= config$sagemakerProjectName))
    )
  else {
    LOGGER$debug("Could not parse project config.")
    return(NULL)
  }
}
