# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/image_uris.py

#' @include utils.R

#' @import jsonlite
#' @import R6
#' @import lgr
#' @import paws
#' @import data.table

#' @title ImageUris Class
#' @description Class to create and format sagemaker docker images stored in ECR
#' @export
ImageUris = R6Class("ImageUris",
  public = list(

    #' @field sagemaker_session
    #' Session object which manages interactions with Amazon SageMaker APIs
    sagemaker_session = NULL,

    #' @description Initializes the ``ImageUris`` class
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    initialize = function(sagemaker_session = NULL){
      self$sagemaker_session = sagemaker_session %||% Session$new()

    },

    #' @description Retrieves the ECR URI for the Docker image matching the given arguments of inbuilt AWS Sagemaker models.
    #' @param framework (str): The name of the framework or algorithm.
    #' @param region (str): The AWS region.
    #' @param version (str): The framework or algorithm version. This is required if there is
    #'              more than one supported version for the given framework or algorithm.
    #' @param py_version (str): The Python version. This is required if there is
    #'              more than one supported Python version for the given framework version.
    #' @param instance_type (str): The SageMaker instance type. For supported types, see
    #'              https://aws.amazon.com/sagemaker/pricing/instance-types. This is required if
    #'              there are different images for different processor types.
    #' @param accelerator_type (str): Elastic Inference accelerator type. For more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html.
    #' @param image_scope (str): The image type, i.e. what it is used for.
    #'              Valid values: "training", "inference", "eia". If ``accelerator_type`` is set,
    #'              ``image_scope`` is ignored.
    #' @return str: the ECR URI for the corresponding SageMaker Docker image.
    retrieve = function(framework,
                        region=NULL,
                        version=NULL,
                        py_version=NULL,
                        instance_type=NULL,
                        accelerator_type=NULL,
                        image_scope=NULL){
      config = private$.config_for_framework_and_scope(framework, image_scope, accelerator_type)

      version = private$.validate_version_and_set_if_needed(version, config, framework)

      version_config = config$versions[[private$.version_for_config(version, config)]]

      py_version = private$.validate_py_version_and_set_if_needed(py_version, version_config)
      version_config = if(is.null(py_version)) version_config else {version_config[[py_version]] %||% version_config}

      region = region %||% self$sagemaker_session$paws_region_name

      registry = private$.registry_from_region(region, version_config$registries)

      hostname = private$.hostname(region)

      repo = version_config$repository

      processor = private$.processor(
        instance_type, config$processors %||% version_config$processors
      )

      version = if(!islistempty(version_config$tag_prefix)) version_config$tag_prefix else version

      tag = private$.format_tag(version, processor, py_version)

      repo = sprintf("%s:%s", repo, tag)

      return(sprintf(private$ECR_URI_TEMPLATE, registry, hostname, repo))
    },

    #' @description lists URIs of custom built models stored in ECR
    #' @param force_refresh (bool): Set to True to fetch the latest data from
    #'              SageMaker API.
    #' @return (data.frame): containing all custom built models
    list_ecr_uris = function(force_refresh = FALSE){
      if (force_refresh)
        private$clear_cache()
      if (is.null(private$.list_ecr_uris)){

        ecr = paws::ecr(config = self$sagemaker_session$paws_credentials$credentials)

        nextToken = NULL
        repos = list()
        # get list of repositories in ecr
        while (!identical(nextToken, character(0))){
          repo_chunk = ecr$describe_repositories()
          repo_dt = rbindlist(repo_chunk$repositories)
          repos = list(repos, repo_dt)
          nextToken = repo_chunk$nextToken
        }
        private$.list_ecr_uris = rbindlist(repos)
      }
      return(private$.list_ecr_uris)
    },

    #' @description Clear the object of all local caches of API methods, so that the next
    #'              time any properties are accessed they will be refreshed from the
    #'              service.
    clear_cache = function() {
      private$.list_ecr_uris = NULL
    },

    #' @description Gets tags of framework stored in ECR
    #' @param repo_name (str): The name of the repository name stored in ECR.
    #' @return (data.frame): listing all tagged version of repository names ordered by time stamp.
    ecr_tags = function(repo_name){
      stopifnot(is.character(repo_name))

      # ensure list of custom ecr uris are retrieved
      self$list_ecr_uris()

      # check if repo_name exists in registered ecr repositories
      if(nrow(private$.list_ecr_uris[repositoryName == repo_name]) == 0)
        stop(sprintf("Custom repository %s doesn't exist in AWS ECR", repo_name))

      # after repo_name check only use repo_name
      repos = private$.list_ecr_uris[repositoryName == repo_name]

      ecr = paws::ecr(config = self$sagemaker_session$paws_credentials$credentials)
      nextToken = NULL
      image_meta = list()
      # get all tags from repository
      while(!identical(nextToken, character(0))){
        image_chunk = ecr$describe_images(repos[, registryId], repos[, repositoryName])
        nextToken = image_chunk$nextToken
        image_chunk = lapply(image_chunk$imageDetails, function(x)
          data.table(imageTags = x$imageTags, imagePushedAt= x$imagePushedAt))
        image_meta = c(image_meta, image_chunk)
      }

      image_meta = rbindlist(image_meta)
      image_meta[["repositoryName"]] = repo_name

      setcolorder(image_meta, "repositoryName")

      return(image_meta[order(-imagePushedAt)])
    },

    #' @description Retrieves the ECR URI for the Docker image matching the given arguments for custome inbuilt models
    #' @param repo_name (str): The name of the repository that stores framework or algorithm in ecr
    #' @param tag (str): The framework or algorithm version. This is required if there is
    #'              more than one supported version for the given framework or algorithm.
    retrieve_ecr_uri = function(repo_name,
                                tag = NULL){
      stopifnot(is.character(repo_name),
                is.character(tag) || is.null(tag))

      # list all ecr uris
      repos = self$list_ecr_uris()

      # get all tags for custom framework
      image_meta = self$ecr_tags(repo_name)

      # check if repo tag matches given tag parameter
      if(!is.null(tag) && nrow(image_meta[imageTags == tag]) == 0)
        stop(sprintf("Repository version %s doesn't exist", tag))

      if(is.null(tag)) {
        # get latest tag if not provided
        tag = image_meta[1, imageTags]
        LOGGER$info("Defaulting to latest version of framework: %s.", repo_name)
      }

      paste(repos$repositoryUri, tag, sep = ":")
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      return(print_class(self))
    }
  ),

  private = list(
    ECR_URI_TEMPLATE = "%s.dkr.%s/%s",

    .list_ecr_uris= NULL,

    # Loads the JSON config for the given framework and image scope.
    .config_for_framework_and_scope = function(framework,
                                               image_scope = NULL,
                                               accelerator_type=NULL){

      config = config_for_framework(framework)

      if (!is.null(accelerator_type)){
        private$.validate_accelerator_type(accelerator_type)

        if (!(is.null(image_scope) || image_scope %in% c("eia", "inference")))
          LOGGER$warn(
            "Elastic inference is for inference only. Ignoring image scope: %s.", image_scope
            )
        image_scope = "eia"
      }

      available_scopes = if(!islistempty(config$scope)) config$scope else names(config)
      if (length(available_scopes) == 1){
        if (!islistempty(image_scope) && image_scope != available_scopes[[1]])
          LOGGER$warn(
            "Defaulting to only supported image scope: %s. Ignoring image scope: %s.",
            available_scopes[1],
            image_scope
            )
        image_scope = available_scopes[[1]]
      }

      if(islistempty(image_scope) && "scope" %in% names(config) && any(unique(available_scopes) %in% list("training", "inference"))){
        LOGGER$info(
          "Same images used for training and inference. Defaulting to image scope: %s.",
          available_scopes[[1]]
        )
        image_scope = available_scopes[[1]]
      }

      private$.validate_arg(image_scope, available_scopes, "image scope")
      return(if("scope" %in% names(config)) config else config[[image_scope]])
      },

    # Raises a ``ValueError`` if ``accelerator_type`` is invalid.
    .validate_accelerator_type = function(accelerator_type){
      if (!startsWith(accelerator_type, "ml.eia") && accelerator_type != "local_sagemaker_notebook")
        stop(sprintf("Invalid SageMaker Elastic Inference accelerator type: %s. ",accelerator_type),
          "See https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html", call. = F
          )
    },

    # Checks if the framework/algorithm version is one of the supported versions.
    .validate_version_and_set_if_needed = function(version = NULL,
                                                   config,
                                                   framework){
      available_versions = names(config$versions)
      aliased_versions = names(config$version_aliases) %||% list()
      version = version %||% NA

      if (length(available_versions) == 1 && !(version %in% aliased_versions)){
        log_message = sprintf("Defaulting to the only supported framework/algorithm version: %s.", available_versions[[1]])
        if (!is.na(version) && version != available_versions[[1]])
          LOGGER$warn("%s Ignoring framework/algorithm version: %s.", log_message, version)
        if (is.na(version))
          LOGGER$info(log_message)

        return(available_versions[[1]])
      }

      private$.validate_arg(if(is.na(version)) NULL else version, c(available_versions, aliased_versions), sprintf("%s version",framework))
      return(version)
    },

    # Returns the version string for retrieving a framework version's specific config.
    .version_for_config = function(version,
                                   config){
      if ("version_aliases" %in% names(config)){
        if (version %in% names(config$version_aliases))
        return(config$version_aliases[[version]])
      }
      return(version)
    },

    # Returns the ECR registry (AWS account number) for the given region.
    .registry_from_region = function(region,
                                     registry_dict){
      private$.validate_arg(region, names(registry_dict), "region")
      return(registry_dict[[region]])
    },

    # Returns the processor type for the given instance type.
    .processor = function(instance_type = NULL,
                          available_processors = NULL){
      if (is.null(available_processors)){
        LOGGER$info("Ignoring unnecessary instance type: %s.", instance_type)
        return(NULL)
      }

      if (length(available_processors) == 1 && is.null(instance_type)){
        LOGGER$info("Defaulting to only supported image scope: %s.", available_processors[[1]])
        return(available_processors[[1]])
      }

      if (islistempty(instance_type)){
        stop("Empty SageMaker instance type. For options, see: ",
             "https://aws.amazon.com/sagemaker/pricing/instance-types",
             call. = F)
      }

      if (startsWith(instance_type,"local")){
        processor = if(instance_type == "local") "cpu" else "gpu"
      } else {
        # looks for either "ml.<family>.<size>" or "ml_<family>"
        match = regmatches(instance_type,regexec("^ml[\\._]([a-z0-9]+)\\.?\\w*$",instance_type))[[1]][2]
        if (!is.na(match)){
          family = match

          # For some frameworks, we have optimized images for specific families, e.g c5 or p3.
          # In those cases, we use the family name in the image tag. In other cases, we use
          # 'cpu' or 'gpu'.
          if (family %in% available_processors) {
            processor = family}
          if (startsWith(family, "inf")){
            processor = "inf"}
          if (substr(family, 1,1) %in% c("g", "p"))
            processor = "gpu"
          else
            processor = "cpu"
        } else {
            stop(sprintf("Invalid SageMaker instance type: %s. For options, see: ", instance_type),
              "https://aws.amazon.com/sagemaker/pricing/instance-types", call. = F
            )
        }
      }
      private$.validate_arg(processor, available_processors, "processor")
      return(processor)
    },

    # # Checks if the Python version is one of the supported versions.
    .validate_py_version_and_set_if_needed = function(py_version,
                                                      version_config){
      if ("repository" %in% names(version_config)){
        available_versions = unlist(version_config$py_versions)
      } else {
        available_versions = names(version_config)
      }

      if (islistempty(available_versions)){
        if(!is.null(py_version))
          LOGGER$info("Ignoring unnecessary Python version: %s.", py_version)
        return(NULL)
      }

      if (is.null(py_version) && length(available_versions) == 1){
        LOGGER$info("Defaulting to only available Python version: %s", available_versions[[1]])
        return(available_versions[[1]])
      }

      private$.validate_arg(py_version, available_versions, "Python version")
      return(py_version)
    },

    # Checks if the arg is in the available options, and raises a ``ValueError`` if not.
    .validate_arg = function(arg, available_options, arg_name){
      if (!(arg %in% available_options) || is.null(arg))
        stop(sprintf(paste(
          "Unsupported %s: %s. You may need to upgrade your SDK version",
          "(remotes::install_github('dyfanjones/R6sagemaker')) for newer %ss.",
          "\nSupported %s(s): {%s}."), arg_name, arg %||% "NULL", arg_name, arg_name,
          paste(available_options, collapse = ", ")
          ),
          call. = F
        )
    },

    .hostname = function(region){
      sprintf("ecr.%s.amazonaws.com", region)
    },

    # Creates a tag for the image URI.
    .format_tag = function(tag_prefix,
                          processor,
                          py_version){
      tag_list = list(tag_prefix, processor, py_version)
      tag_list = Filter(Negate(is.null), tag_list)
      return (paste(tag_list, collapse = "-"))
    }
  )
)

# Loads the JSON config for the given framework.
config_for_framework = function(framework){
  fname= system.file("image_uri_config", sprintf("%s.json", framework), package= "R6sagemaker")

  # check if framework json file exists first
  if(!file.exists(fname))
    stop(sprintf(paste(
      "Unsupported framework: %s. You may need to upgrade your SDK version",
      "(remotes::install_github('dyfanjones/R6sagemaker')) for newer frameworks."),
      framework),
    call. = F)

  return(read_json(fname))
}
