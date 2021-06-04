# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/rl/estimator.py

#' @include estimator.R
#' @include model.R
#' @include tensorflow_estimator.R
#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common
#' @import lgr

SAGEMAKER_ESTIMATOR <- "sagemaker_estimator"
SAGEMAKER_ESTIMATOR_VALUE <- "RLEstimator"
RL_PYTHON_VERSION <- "py3"
TOOLKIT_FRAMEWORK_VERSION_MAP <- list(
  "coach"=list(
    "0.10.1"=list("tensorflow"="1.11"),
    "0.10"=list("tensorflow"="1.11"),
    "0.11.0"=list("tensorflow"="1.11", "mxnet"="1.3"),
    "0.11.1"=list("tensorflow"="1.12"),
    "0.11"=list("tensorflow"="1.12", "mxnet"="1.3"),
    "1.0.0"=list("tensorflow"="1.12")
  ),
  "ray"=list(
    "0.5.3"=list("tensorflow"="1.11"),
    "0.5"=list("tensorflow"="1.11"),
    "0.6.5"=list("tensorflow"="1.12"),
    "0.6"=list("tensorflow"="1.12"),
    "0.8.2"=list("tensorflow"="2.1"),
    "0.8.5"=list("tensorflow"="2.1", "pytorch"="1.5")
  )
)

#' @title RLToolkit enum environment list
#' @description RL toolkit you want to use for
#'              executing your model training code.
#' @return environment containing [COACH, RAY]
#' @export
RLToolkit = Enum(COACH = "coach", RAY = "ray")

#' @title RLFramework enum environment list
#' @description Framework (MXNet, TensorFlow or PyTorch) you want to be used
#'              as a toolkit backed for
#'              reinforcement learning training.
#' @return environment containing [TENSORFLOW, MXNET, PYTORCH]
#' @export
RLFramework = Enum(
  TENSORFLOW = "tensorflow",
  MXNET = "mxnet",
  PYTORCH = "pytorch"
)

#' @title RLEstimator Class
#' @description Handle end-to-end training and deployment of custom RLEstimator code.
#' @export
RLEstimator = R6Class("RLEstimator",
  inherit = Framework,
  public = list(

    #' @field COACH_LATEST_VERSION_TF
    #' latest version of toolkit coach for tensorflow
    COACH_LATEST_VERSION_TF = "0.11.1",

    #' @field COACH_LATEST_VERSION_MXNET
    #' latest version of toolkit coach for mxnet
    COACH_LATEST_VERSION_MXNET = "0.11.0",

    #' @field RAY_LATEST_VERSION
    #' latest version of toolkit ray
    RAY_LATEST_VERSION = "0.8.5",

    #' @description Creates an RLEstimator for managed Reinforcement Learning (RL).
    #'              It will execute an RLEstimator script within a SageMaker Training Job. The managed RL
    #'              environment is an Amazon-built Docker container that executes functions defined in the
    #'              supplied ``entry_point`` Python script.
    #'              Training is started by calling
    #'              :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
    #'              After training is complete, calling
    #'              :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
    #'              SageMaker endpoint and based on the specified framework returns an
    #'              :class:`~sagemaker.amazon.mxnet.model.MXNetPredictor` or
    #'              :class:`~sagemaker.amazon.tensorflow.model.TensorFlowPredictor` instance that
    #'              can be used to perform inference against the hosted model.
    #'              Technical documentation on preparing RLEstimator scripts for
    #'              SageMaker training and using the RLEstimator is available on the project
    #'              homepage: https://github.com/aws/sagemaker-python-sdk
    #' @param entry_point (str): Path (absolute or relative) to the Python source
    #'              file which should be executed as the entry point to training.
    #'              If ``source_dir`` is specified, then ``entry_point``
    #'              must point to a file located at the root of ``source_dir``.
    #' @param toolkit (sagemaker.rl.RLToolkit): RL toolkit you want to use for
    #'              executing your model training code.
    #' @param toolkit_version (str): RL toolkit version you want to be use for
    #'              executing your model training code.
    #' @param framework (sagemaker.rl.RLFramework): Framework (MXNet or
    #'              TensorFlow) you want to be used as a toolkit backed for
    #'              reinforcement learning training.
    #' @param source_dir (str): Path (absolute, relative or an S3 URI) to a directory
    #'              with any other training source code dependencies aside from the entry
    #'              point file (default: NULL). If ``source_dir`` is an S3 URI, it must
    #'              point to a tar.gz file. Structure within this directory are preserved
    #'              when training on Amazon SageMaker.
    #' @param hyperparameters (dict): Hyperparameters that will be used for
    #'              training (default: NULL). The hyperparameters are made
    #'              accessible as a dict[str, str] to the training code on
    #'              SageMaker. For convenience, this accepts other types for keys
    #'              and values.
    #' @param image_uri (str): An ECR url. If specified, the estimator will use
    #'              this image for training and hosting, instead of selecting the
    #'              appropriate SageMaker official image based on framework_version
    #'              and py_version. Example:
    #'              123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
    #' @param metric_definitions (list[dict]): A list of dictionaries that defines
    #'              the metric(s) used to evaluate the training jobs. Each
    #'              dictionary contains two keys: 'Name' for the name of the metric,
    #'              and 'Regex' for the regular expression used to extract the
    #'              metric from the logs. This should be defined only for jobs that
    #'              don't use an Amazon algorithm.
    #' @param ... : Additional kwargs passed to the
    #'              :class:`~sagemaker.estimator.Framework` constructor.
    #'              .. tip::
    #'              You can find additional parameters for initializing this class at
    #'              :class:`~sagemaker.estimator.Framework` and
    #'              :class:`~sagemaker.estimator.EstimatorBase`.
    initialize = function(entry_point,
                          toolkit=NULL,
                          toolkit_version=NULL,
                          framework=NULL,
                          source_dir=NULL,
                          hyperparameters=NULL,
                          image_uri=NULL,
                          metric_definitions=NULL,
                          ...){
      private$.validate_images_args(toolkit, toolkit_version, framework, image_uri)

      if (is.null(image_uri)){
        private$.validate_toolkit_support(toolkit, toolkit_version, framework)
        self$toolkit = toolkit
        self$toolkit_version = toolkit_version
        self$framework = framework
        self$framework_version = TOOLKIT_FRAMEWORK_VERSION_MAP[[self$toolkit]][[
          self$toolkit_version
        ]][[self$framework]]

        # set default metric_definitions based on the toolkit
        if (is.null(metric_definitions))
          metric_definitions = self$default_metric_definitions(toolkit)
      }
      super$initialize(
        entry_point,
        source_dir,
        hyperparameters,
        image_uri=image_uri,
        metric_definitions=metric_definitions,
        ...
      )
    },

    #' @description Create a SageMaker ``RLEstimatorModel`` object that can be deployed to an Endpoint.
    #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
    #'              which is also used during transform jobs. If not specified, the
    #'              role from the Estimator will be used.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
    #'              the model. Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param entry_point (str): Path (absolute or relative) to the Python source
    #'              file which should be executed as the entry point for MXNet
    #'              hosting (default: self.entry_point). If ``source_dir`` is specified,
    #'              then ``entry_point`` must point to a file located at the root of ``source_dir``.
    #' @param source_dir (str): Path (absolute or relative) to a directory with
    #'              any other training source code dependencies aside from the entry
    #'              point file (default: self.source_dir). Structure within this
    #'              directory are preserved when hosting on Amazon SageMaker.
    #' @param dependencies (list[str]): A list of paths to directories (absolute
    #'              or relative) with any additional libraries that will be exported
    #'              to the container (default: self.dependencies). The library
    #'              folders will be copied to SageMaker in the same folder where the
    #'              entry_point is copied. If the ```source_dir``` points to S3,
    #'              code will be uploaded and the S3 location will be used instead.
    #'              This is not supported with "local code" in Local Mode.
    #' @param ... : Additional kwargs passed to the :class:`~sagemaker.model.FrameworkModel`
    #'              constructor.
    #' @return sagemaker.model.FrameworkModel: Depending on input parameters returns
    #'              one of the following:
    #'              * :class:`~sagemaker.model.FrameworkModel` - if ``image_uri`` is specified
    #'              on the estimator;
    #'              * :class:`~sagemaker.mxnet.MXNetModel` - if ``image_uri`` isn't specified and
    #'                                  MXNet is used as the RL backend;
    #'              * :class:`~sagemaker.tensorflow.model.TensorFlowModel` - if ``image_uri`` isn't
    #'              specified and TensorFlow is used as the RL backend.
    create_model = function(role=NULL,
                            vpc_config_override="VPC_CONFIG_DEFAULT",
                            entry_point=NULL,
                            source_dir=NULL,
                            dependencies=NULL,
                            ...){
      kwargs = list(...)
      base_args = list(
        model_data=self$model_data,
        role=role %||% self$role,
        image_uri=kwargs$image_uri %||% self$image_uri,
        container_log_level=self$container_log_level,
        sagemaker_session=self$sagemaker_session,
        vpc_config=self$get_vpc_config(vpc_config_override))

      base_args[["name"]] = private$.get_or_create_name(kwargs[["name"]])

      if (is.null(entry_point) && (!is.null(source_dir) || !is.null(dependencies)))
        AttributeError$new("Please provide an `entry_point`.")

      entry_point = entry_point %||% private$.model_entry_point()
      source_dir = source_dir %||% private$.model_source_dir()
      dependencies = dependencies %||% self$dependencies

      extended_args = list(
        entry_point=entry_point,
        source_dir=source_dir,
        code_location=self$code_location,
        dependencies=dependencies
      )
      extended_args = modifyList(base_args, extended_args)

      if (!is.null(self$image_uri))
        return(do.call(FrameworkModel$new, extended_args))

      if (self$toolkit == RLToolkit$RAY)
        NotImplementedError$new(
          "Automatic deployment of Ray models is not currently available.",
          " Train policy parameters are available in model checkpoints",
          " in the TrainingJob output.")

      if (self$framework == RLFramework$TENSORFLOW){
        extended_args = c(framework_version=self$framework_version, extended_args)
        return(do.call(TensorFlowModel$new, extended_args))}
      if (self$framework == RLFramework$MXNET){
        extended_args = c(framework_version=self$framework_version, py_version=RL_PYTHON_VERSION, extended_args)
        return(do.call(MXNetModel$new, extended_args))
      }
      ValueError$new(sprintf(
        "An unknown RLFramework enum was passed in. framework: %s", self$framework)
      )
    },

    #' @description Return the Docker image to use for training.
    #'              The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does
    #'              the model training, calls this method to find the image to use for model
    #'              training.
    #' @return str: The URI of the Docker image.
    training_image_uri = function(){
      if (!is.null(self$image_uri))
        return(self$image_uri)
      return(ImageUris$new()$retrieve(
        private$.image_framework(),
        self$sagemaker_session$paws_region_name,
        version=self$toolkit_version,
        instance_type=self$instance_type)
      )
    },

    #' @description Return hyperparameters used by your custom TensorFlow code during model training.
    hyperparameters = function(){
      hyperparameters = super$hyperparameters()

      additional_hyperparameters = list(
        self$output_path,
        # TODO: can be applied to all other estimators
        SAGEMAKER_ESTIMATOR_VALUE)
      names(additional_hyperparameters) <- c(SAGEMAKER_OUTPUT_LOCATION , SAGEMAKER_ESTIMATOR)

      # hyperparameters.update(Framework._json_encode_hyperparameters(additional_hyperparameters))
      hyperparameters = modifyList(hyperparameters, additional_hyperparameters)

      return(hyperparameters)
    },

    #' @description Provides default metric definitions based on provided toolkit.
    #' @param toolkit (sagemaker.rl.RLToolkit): RL Toolkit to be used for
    #'              training.
    #' @return list: metric definitions
    default_metric_definitions = function(toolkit){
      if (toolkit == RLToolkit$COACH){
        return(list(
          list("Name"="reward-training", "Regex"="^Training>.*Total reward=(.*?),"),
          list("Name"="reward-testing", "Regex"="^Testing>.*Total reward=(.*?),"))
        )
      }
      if (toolkit == RLToolkit$RAY){
        float_regex = "[-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?"  # noqa: W605, E501
        return(list(
          list("Name"="episode_reward_mean", "Regex"=sprintf("episode_reward_mean: (%s)" , float_regex)),
          list("Name"="episode_reward_max", "Regex"=sprintf("episode_reward_max: (%s)",float_regex)))
        )
      }
      ValueError$new(sprintf("An unknown RLToolkit enum was passed in. toolkit: %s", toolkit))
    }
  ),

  private = list(

    # Convert the job description to init params.
    # This is done so that the init params can be handled by the class constructor.
    # Args:
    #   job_details: the returned job details from a describe_training_job
    # API call.
    # model_channel_name (str): Name of the channel where pre-trained
    # model data will be downloaded.
    # Returns:
    #   dictionary: The transformed init_params
    .prepare_init_params_from_job_description = function(job_details,
                                                       model_channel_name=NULL){
      init_params = super$.prepare_init_params_from_job_description(
        job_details, model_channel_name)

      image_uri = init_params$image_uri
      init_params$image_uri = NULL
      img_split = framework_name_from_image(image_uri)
      names(img_split) = c("framework", "py_version", "tag", "scriptmode")

      if (is.null(img_split$framework)) {
        # If we were unable to parse the framework name from the image it is not one of our
        # officially supported images, in this case just add the image to the init params.
        init_params[["image_uri"]] = image_uri
        return(init_params)
      }
      ll_tag = private$.toolkit_and_version_from_tag(img_split$tag)
      names(ll_tag) <- c("toolkit", "toolkit_version")

      if (!private$.is_combination_supported(ll_tag$toolkit, ll_tag$toolkit_version, img_split$framework))
        ValueError$new(sprintf(
          "Training job: %s didn't use image for requested framework",
            job_details[["TrainingJobName"]])
          )

      init_params[["toolkit"]] = ll_tag$toolkit
      init_params[["toolkit_version"]] = ll_tag$toolkit_version
      init_params[["framework"]] = img_split$framework

      return(init_params)
    },

    .toolkit_and_version_from_tag = function(image_tag){
      tag_pattern = "^([A-Z]*|[a-z]*)(\\d.*)-(cpu|gpu)-(py2|py3)$"
      m = regexec(tag_pattern, image_tag)
      tag_match = unlist(regmatches(image_tag, m))
      if (length(tag_match) > 0)
        return(list(tag_match[[2]], tag_match[[3]]))
      return(list(NULL, NULL))
    },

    .validate_framework_format = function(framework){
      rl_framework = unname(as.list(RLFramework))
      if (!is.null(framework) && !(framework %in% rl_framework))
        ValueError$new(sprintf(
          "Invalid type: %s, valid RL frameworks types are: %s",
          framework, paste(rl_framework, collapse = ", "))
        )
    },

    .validate_toolkit_format = function(toolkit){
      rl_toolkit = unname(as.list(RLToolkit))
      if (!is.null(toolkit) && !(toolkit %in% rl_toolkit))
        ValueError$new(sprintf(
          "Invalid type: %s, valid RL toolkits types are: %s",
          toolkit, paste(rl_toolkit, collapse = ", "))
        )
    },

    .validate_images_args = function(toolkit=NULL,
                                     toolkit_version=NULL,
                                     framework=NULL,
                                     image_uri=NULL){
      private$.validate_toolkit_format(toolkit)
      private$.validate_framework_format(framework)

      if (is.null(image_uri)){
        not_found_args = list()
        if (is.null(toolkit))
          not_found_args = c(not_found_args, "toolkit")
        if (is.null(toolkit_version))
          not_found_args = c(not_found_args, "toolkit_version")
        if (is.null(framework))
            not_found_args = c(not_found_args, "framework")
        if (!islistempty(not_found_args))
          AttributeError$new(sprintf(
            "Please provide `%s` or `image_uri` parameter.",
              paste(not_found_args, collapse = "`, `"))
            )
      } else {
        found_args = list()
      if (!is.null(toolkit))
        found_args = c(found_args, "toolkit")
      if (!is.null(toolkit_version))
        found_args = c(found_args, "toolkit_version")
      if (!is.null(framework))
        found_args = c(found_args, "framework")
      if (!islistempty(found_args))
        LOGGER$warn(paste(
          "Parameter `image_uri` is specified,",
          "`%s` are going to be ignored when choosing the image."),
          paste(found_args, collapse = "`, `"))
      }
    },

    .is_combination_supported = function(toolkit,
                                         toolkit_version,
                                         framework){
      supported_versions = if(is.null(toolkit)) NULL else TOOLKIT_FRAMEWORK_VERSION_MAP[[toolkit]]
      if (!is.null(supported_versions)){
        supported_frameworks = supported_versions[[toolkit_version]]
        if (!is.null(supported_frameworks) && !is.null(supported_frameworks[[framework]]))
          return(TRUE)
      }
      return(FALSE)
    },

    .validate_toolkit_support = function(toolkit,
                                         toolkit_version,
                                         framework){
      if (!private$.is_combination_supported(toolkit, toolkit_version, framework))
        AttributeError$new(sprintf(
          "Provided `%s-%s` and `%s` combination is not supported.",
            toolkit, toolkit_version, framework)
        )
    },

    # Toolkit name and framework name for retrieving Docker image URI config.
    .image_framework = function(){
      return(paste(self$toolkit, self$framework, sep = "-", collapse = "-"))
    }
  ),
  lock_objects = F
)
