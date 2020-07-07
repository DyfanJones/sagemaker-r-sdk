# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/762b509f711daf4d0d7b759626f614fcf618b74e/src/sagemaker/model.py

#' @include session.R
#' @include logs.R
#' @include vpc_utils.R
#' @include set_credentials.R
#' @include fw_utils.R
#' @include transformer.R
#' @include git_utils.R


#' @import paws
#' @import jsonlite
#' @import R6
#' @import logger


NEO_ALLOWED_FRAMEWORKS <- list("mxnet", "tensorflow", "keras", "pytorch", "onnx", "xgboost", "tflite")

NEO_IMAGE_ACCOUNT <- list(
  "us-west-1"= "710691900526",
  "us-west-2"= "301217895009",
  "us-east-1"= "785573368785",
  "us-east-2"= "007439368137",
  "eu-west-1"= "802834080501",
  "eu-west-2"= "205493899709",
  "eu-west-3"= "254080097072",
  "eu-central-1"= "746233611703",
  "eu-north-1"= "601324751636",
  "ap-northeast-1"= "941853720454",
  "ap-northeast-2"= "151534178276",
  "ap-east-1"= "110948597952",
  "ap-southeast-1"= "324986816169",
  "ap-southeast-2"= "355873309152",
  "ap-south-1"= "763008648453",
  "sa-east-1"= "756306329178",
  "ca-central-1"= "464438896020",
  "me-south-1"= "836785723513",
  "cn-north-1"= "472730292857",
  "cn-northwest-1"= "474822919863",
  "us-gov-west-1"= "263933020539")

INFERENTIA_INSTANCE_PREFIX <- "ml_inf"

#' @title Model Class
#' @description Initialize an SageMaker ``Model``.
#' @export
Model = R6Class("Model",
  public = list(

    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @param model_data (str): The S3 location of a SageMaker model data
    #'              ``.tar.gz`` file.
    #' @param image_uri (str): A Docker image URI.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role if it needs to access some AWS resources.
    #'              It can be null if this is being used to create a Model to pass
    #'              to a ``PipelineModel`` which has its own Role field. (Default:
    #'              NULL)
    #' @param predictor_cls (callable[string, :Session]): A
    #'              function to call to create a predictor (default: None). If not
    #'              None, ``deploy`` will return the result of invoking this
    #'              function on the created endpoint name.
    #' @param env (dict[str, str]): Environment variables to run with ``image``
    #'              when hosted in SageMaker (Default: NULL).
    #' @param name (str): The model name. If None, a default model name will be
    #'              selected on each ``deploy``.
    #' @param vpc_config (dict[str, list[str]]): The VpcConfig set on the model
    #'              (Default: NULL)
    #'              \itemize{
    #'                \item{\strong{'Subnets' (list[str])} List of subnet ids.}
    #'                \item{\strong{'SecurityGroupIds' (list[str]):} List of security group ids.}}
    #' @param sagemaker_session (:Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    #' @param enable_network_isolation (Boolean): Default False. if True, enables
    #'              network isolation in the endpoint, isolating the model
    #'              container. No inbound or outbound network calls can be made to
    #'              or from the model container.
    #' @param model_kms_key (str): KMS key ARN used to encrypt the repacked
    #'              model archive file if the model is repacked
    initialize = function(model_data,
                          image_uri,
                          role=NULL,
                          predictor_cls=NULL,
                          env=NULL,
                          name=NULL,
                          vpc_config=NULL,
                          sagemaker_session=NULL,
                          enable_network_isolation=FALSE,
                          model_kms_key=NULL){

      self$model_data = model_data
      self$image = image_uri
      self$role = role
      self$predictor_cls = predictor_cls
      self$env = env %||% list()
      self$name = name
      self$vpc_config = vpc_config
      self$sagemaker_session = sagemaker_session
      self$.model_name = NULL
      self$endpoint_name = NULL
      self$.is_compiled_model = FALSE
      self$.enable_network_isolation = enable_network_isolation
      self$model_kms_key = model_kms_key
    },

    #' @description Return a dict created by ``sagemaker.container_def()`` for deploying
    #'              this model to a specified instance type.
    #'              Subclasses can override this to provide custom container definitions
    #'              for deployment to a specific instance type. Called by ``deploy()``.
    #' @param instance_type (str): The EC2 instance type to deploy this Model to.
    #'              For example, 'ml.p2.xlarge'.
    #' @param accelerator_type (str): The Elastic Inference accelerator type to
    #'              deploy to the instance for loading and making inferences to the
    #'              model. For example, 'ml.eia1.medium'.
    #' @return dict: A container definition object usable with the CreateModel API.
    prepare_container_def = function(instance_type,
                                     accelerator_type=NULL){
      return (container_def(self$image, self$model_data, self$env))
    },

    #' @description Whether to enable network isolation when creating this Model
    #' @return bool: If network isolation should be enabled or not.
    enable_network_isolation = function(){
      return (self$.enable_network_isolation)
    },

    #' @description Check if this ``Model`` in the available region where neo support.
    #' @param region (str): Specifies the region where want to execute compilation
    #' @return bool: boolean value whether if neo is available in the specified
    #'              region
    check_neo_region = function(region){
      if(region %in% names(NEO_IMAGE_ACCOUNT)) return(TRUE)
      return(FALSE)
    },

    #' @description Compile this ``Model`` with SageMaker Neo.
    #' @param target_instance_family (str): Identifies the device that you want to
    #'              run your model after compilation, for example: ml_c5. For allowed
    #'              strings see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
    #' @param input_shape (list): Specifies the name and shape of the expected
    #'              inputs for your trained model in json dictionary form, for
    #'              example: \code{list('data'= list(1,3,1024,1024)), or list('var1'= list(1,1,28,28),
    #'              'var2'= list(1,1,28,28))}
    #' @param output_path (str): Specifies where to store the compiled model
    #' @param role (str): Execution role
    #' @param tags (list[dict]): List of tags for labeling a compilation job. For
    #'              more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @param job_name (str): The name of the compilation job
    #' @param compile_max_run (int): Timeout in seconds for compilation (default:
    #'              3 * 60). After this amount of time Amazon SageMaker Neo
    #'              terminates the compilation job regardless of its current status.
    #' @param framework (str): The framework that is used to train the original
    #'              model. Allowed values: 'mxnet', 'tensorflow', 'keras', 'pytorch',
    #'              'onnx', 'xgboost'
    #' @param framework_version (str):
    #' @return sagemaker.model.Model: A SageMaker ``Model`` object. See
    #'              :func:`~sagemaker.model.Model` for full details.
    compile = function(target_instance_family,
                       input_shape,
                       output_path,
                       role,
                       tags=NULL,
                       job_name=NULL,
                       compile_max_run=5 * 60,
                       framework=NULL,
                       framework_version=NULL){

      if (is.null(framework))
        stop(sprintf("You must specify framework, allowed values %s",
                     paste0(NEO_ALLOWED_FRAMEWORKS, collapse = ", ")), call. = F)

      if (!(framework %in% NEO_ALLOWED_FRAMEWORKS))
        stop(sprintf("You must provide valid framework, allowed values %s",
          paste0(NEO_ALLOWED_FRAMEWORKS, collapse = ", ")), call. = F)

      if(is.null(job_name))
        stop("You must provide a compilation job name", call. = F)

      framework = toupper(framework)
      private$.init_sagemaker_session_if_does_not_exist(target_instance_family)
      config = private$.compilation_job_config(
                  target_instance_family,
                  input_shape,
                  output_path,
                  role,
                  compile_max_run,
                  job_name,
                  framework,
                  tags)

      do.call(self$sagemaker_session$compile_model, config)
      job_status = self$sagemaker_session$wait_for_compilation_job(job_name)
      self$model_data = job_status$ModelArtifacts$S3ModelArtifacts
      if (startsWith(target_instance_family,"ml_")){
        self$image = private$.neo_image(
              self$sagemaker_session$paws_region_name,
              target_instance_family,
              framework,
              framework_version)
        self$.is_compiled_model = TRUE
      } else if(startsWith(target_instance_family,INFERENTIA_INSTANCE_PREFIX)){
        self$image = private$.inferentia_image(
          self$sagemaker_session$paws_region_name,
          target_instance_family,
          framework,
          framework_version)
        self$.is_compiled_model = TRUE
      } else{
          log_warn("The instance type %s is not supported to deploy via SageMaker, please deploy the model manually.",
                   target_instance_family)
      }

      return(self)
    },

    #' @description Deploy this ``Model`` to an ``Endpoint`` and optionally return a
    #'              ``Predictor``.
    #'              Create a SageMaker ``Model`` and ``EndpointConfig``, and deploy an
    #'              ``Endpoint`` from this ``Model``. If ``self.predictor_cls`` is not None,
    #'              this method returns a the result of invoking ``self.predictor_cls`` on
    #'              the created endpoint name.
    #'              The name of the created model is accessible in the ``name`` field of
    #'              this ``Model`` after deploy returns
    #'              The name of the created endpoint is accessible in the
    #'              ``endpoint_name`` field of this ``Model`` after deploy returns.
    #' @param initial_instance_count (int): The initial number of instances to run
    #'              in the ``Endpoint`` created from this ``Model``.
    #' @param instance_type (str): The EC2 instance type to deploy this Model to.
    #'              For example, 'ml.p2.xlarge', or 'local' for local mode.
    #' @param accelerator_type (str): Type of Elastic Inference accelerator to
    #'              deploy this model for model loading and inference, for example,
    #'              'ml.eia1.medium'. If not specified, no Elastic Inference
    #'              accelerator will be attached to the endpoint. For more
    #'              information:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
    #' @param endpoint_name (str): The name of the endpoint to create (Default:
    #'              NULL). If not specified, a unique endpoint name will be created.
    #' @param update_endpoint (bool): Flag to update the model in an existing
    #'              Amazon SageMaker endpoint. If True, this will deploy a new
    #'              EndpointConfig to an already existing endpoint and delete
    #'              resources corresponding to the previous EndpointConfig. If
    #'              False, a new endpoint will be created. Default: False
    #' @param tags (List[dict[str, str]]): The list of tags to attach to this
    #'              specific endpoint.
    #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
    #'              data on the storage volume attached to the instance hosting the
    #'              endpoint.
    #' @param wait (bool): Whether the call should wait until the deployment of
    #'              this model completes (default: True).
    #' @param data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
    #'              configuration related to Endpoint data capture for use with
    #'              Amazon SageMaker Model Monitoring. Default: None.
    #' @return callable[string, sagemaker.session.Session] or None: Invocation of
    #'              ``self.predictor_cls`` on the created endpoint name, if ``self.predictor_cls``
    #'              is not None. Otherwise, return None.
    deploy = function(initial_instance_count,
                      instance_type,
                      accelerator_type=NULL,
                      endpoint_name=NULL,
                      update_endpoint=FALSE,
                      tags=NULL,
                      kms_key=NULL,
                      wait=TRUE,
                      data_capture_config=NULL){
      private$.init_sagemaker_session_if_does_not_exist(instance_type)

      if(is.null(self$role))
        stop("Role can not be null for deploying a model", call. = F)

      if (startsWith(instance_type,"ml.inf") && !self.is_compiled_model)
        log_warn("Your model is not compiled. Please compile your model before using Inferentia.")

      compiled_model_suffix = paste0(split_str(instance_type, "\\."), collapse = "-")
      if (self$.is_compiled_model){
        name_prefix = self$name %||% name_from_image(self$image)
        self$name = sprintf("%s%s",name_prefix, compiled_model_suffix)
      }

      self$.create_sagemaker_model(instance_type, accelerator_type, tags)
      production_variant = production_variant(self$name,
                                              instance_type,
                                              initial_instance_count,
                                              accelerator_type=accelerator_type)

      if (!is.null(endpoint_name)) {
        self$endpoint_name = endpoint_name
      } else{
        self$endpoint_name = self$name
        if (self$.is_compiled_model && !endsWith(self$endpoint_name, compiled_model_suffix))
          self$endpoint_name = c(self$endpoint_name, compiled_model_suffix)
      }

      data_capture_config_list = NULL
      if (!is.null(data_capture_config))
        data_capture_config_list = data_capture_config$to_request_list

      if (update_endpoint){
        endpoint_config_name = self$sagemaker_session$create_endpoint_config(
          name=self$name,
          model_name=self$name,
          initial_instance_count=initial_instance_count,
          instance_type=instance_type,
          accelerator_type=accelerator_type,
          tags=tags,
          kms_key=kms_key,
          data_capture_config_list=data_capture_config_list)
        self$sagemaker_session$update_endpoint(self$endpoint_name, endpoint_config_name, wait=wait)
      } else{
        self$sagemaker_session$endpoint_from_production_variants(
          name=self$endpoint_name,
          production_variants=list(production_variant),
          tags=tags,
          kms_key=kms_key,
          wait=wait,
          data_capture_config_list=data_capture_config_list)
      }

      if (!is.null(self$predictor_cls))
        return(self$predictor_cls(self$endpoint_name, self$sagemaker_session))
      return(NULL)
    },

    #' @description Return a ``Transformer`` that uses this Model.
    #' @param instance_count (int): Number of EC2 instances to use.
    #' @param instance_type (str): Type of EC2 instance to use, for example,
    #'              'ml.c4.xlarge'.
    #' @param strategy (str): The strategy used to decide how to batch records in
    #'              a single request (default: None). Valid values: 'MultiRecord'
    #'              and 'SingleRecord'.
    #' @param assemble_with (str): How the output is assembled (default: None).
    #'              Valid values: 'Line' or 'None'.
    #' @param output_path (str): S3 location for saving the transform result. If
    #'              not specified, results are stored to a default bucket.
    #' @param output_kms_key (str): Optional. KMS key ID for encrypting the
    #'              transform output (default: None).
    #' @param accept (str): The accept header passed by the client to
    #'              the inference endpoint. If it is supported by the endpoint,
    #'              it will be the format of the batch transform output.
    #' @param env (dict): Environment variables to be set for use during the
    #'              transform job (default: None).
    #' @param max_concurrent_transforms (int): The maximum number of HTTP requests
    #'              to be made to each individual transform container at one time.
    #' @param max_payload (int): Maximum size of the payload in a single HTTP
    #'              request to the container in MB.
    #' @param tags (list[dict]): List of tags for labeling a transform job. If
    #'              none specified, then the tags used for the training job are used
    #'              for the transform job.
    #' @param volume_kms_key (str): Optional. KMS key ID for encrypting the volume
    #'              attached to the ML compute instance (default: None).
    transformer = function(instance_count,
                           instance_type,
                           strategy=NULL,
                           assemble_with=NULL,
                           output_path=NULL,
                           output_kms_key=NULL,
                           accept=NULL,
                           env=NULL,
                           max_concurrent_transforms=NULL,
                           max_payload=NULL,
                           tags=NULL,
                           volume_kms_key=NULL){

      private$.init_sagemaker_session_if_does_not_exist(instance_type)

      self$.create_sagemaker_model(instance_type, tags=tags)
      if (self$enable_network_isolation())
        env = NULL

      return(Transformer$new(
                  self$name,
                  instance_count,
                  instance_type,
                  strategy=strategy,
                  assemble_with=assemble_with,
                  output_path=output_path,
                  output_kms_key=output_kms_key,
                  accept=accept,
                  max_concurrent_transforms=max_concurrent_transforms,
                  max_payload=max_payload,
                  env=env,
                  tags=tags,
                  base_transform_job_name=self.name,
                  volume_kms_key=volume_kms_key,
                  sagemaker_session=self$sagemaker_session))

    },

    #' @description Delete an Amazon SageMaker Model.
    delete_model = function(){
      if(is.null(self$name))
        stop("The SageMaker model must be created first before attempting to delete.", call. = F)

      self$sagemaker_session$delete_model(self$name)

    },

    #' @description Create a SageMaker Model Entity
    #' @param instance_type (str): The EC2 instance type that this Model will be
    #'              used for, this is only used to determine if the image needs GPU
    #'              support or not.
    #' @param accelerator_type (str): Type of Elastic Inference accelerator to
    #'              attach to an endpoint for model loading and inference, for
    #'              example, 'ml.eia1.medium'. If not specified, no Elastic
    #'              Inference accelerator will be attached to the endpoint.
    #' @param tags (List[dict[str, str]]): Optional. The list of tags to add to
    #'              the model. Example: >>> tags = [{'Key': 'tagname', 'Value':
    #'              'tagvalue'}] For more information about tags, see
    #'              https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags
    .create_sagemaker_model = function(instance_type,
                                       accelerator_type=NULL,
                                       tags=NULL){
      container_def = self$prepare_container_def(instance_type, accelerator_type=accelerator_type)
      self$name = self$name %||% name_from_image(container_def$Image)
      enable_network_isolation = self$enable_network_isolation()

      private$.init_sagemaker_session_if_does_not_exist(instance_type)
      self$sagemaker_session$create_model(
        self$name,
        self$role,
        container_def,
        vpc_config=self$vpc_config,
        enable_network_isolation=enable_network_isolation,
        tags=tags)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      cat("<Model>")
      invisible(self)
    }
  ),
  private = list(
    .init_sagemaker_session_if_does_not_exist = function(instance_type){
      # Set ``self.sagemaker_session`` to be a ``LocalSession`` or
      # ``Session`` if it is not already. The type of session object is
      # determined by the instance type.
      if (!is.null(self$sagemaker_session)) return(invisible(NULL))
      if (instance_type %in% c("local", "local_gpu")){
        # TODO: local sagemaker session
        log_error("Currently LocalSession has not been implemented")
        stop(sprintf("instance_type %s is currently not supported", call. =F))
        self$sagemaker_session = LocalSession$new()
      } else {
          self$sagemaker_session = Session$new()}
      },

    .framework = function(obj){
      return(attr(obj, "__framework_name__"))
      },

    .get_framework_version = function(obj){
      return(attr(obj, "framework_version"))
      },

    .compilation_job_config = function(target_instance_type,
                                       input_shape,
                                       output_path,
                                       role,
                                       compile_max_run,
                                       job_name,
                                       framework,
                                       tag){
      input_model_config = list(
        "S3Uri" = self$model_data,
        "DataInputConfig" = input_shape,
        "Framework" = framework)
      role = self$sagemaker_session$expand_role(role)
      output_model_config = list(
        "TargetDevice" = target_instance_type,
        "S3OutputLocation" = output_path)

      return(list(
        "input_model_config"= input_model_config,
        "output_model_config"= output_model_config,
        "role"= role,
        "stop_condition"= list("MaxRuntimeInSeconds"= compile_max_run),
        "tags"= tags,
        "job_name"= job_name))
    },

    .neo_image = function(region,
                          target_instance_type,
                          framework,
                          framework_version){
      return(create_image_uri(
                region,
                sprintf("neo-%s", tolower(framework)),
                gsub("_", "\\.", target_instance_type),
                framework_version,
                py_version="py3",
                account=private$.neo_image_account(region)))
      },

    .inferentia_image = function(region,
                                 target_instance_type,
                                 framework, framework_version){
      return(create_image_uri(
                region,
                sprintf("neo-%s", tolower(framework)),
                gsub("_", "\\.", target_instance_type),
                framework_version,
                py_version="py3",
                account=private$.neo_image_account(region)))
      },

    .neo_image_account = function(region){
      if (!(region %in% names(NEO_IMAGE_ACCOUNT)))
        stop(sprintf("Neo is not currently supported in %s, valid regions:\n%s",
                     region, paste0(names(NEO_IMAGE_ACCOUNT), collapse = ", \n")), call. = F)
      return(NEO_IMAGE_ACCOUNT[[region]])
      }
    ),
  lock_objects = F
)


SCRIPT_PARAM_NAME <- "sagemaker_program"
DIR_PARAM_NAME <- "sagemaker_submit_directory"
CLOUDWATCH_METRICS_PARAM_NAME <- "sagemaker_enable_cloudwatch_metrics"
CONTAINER_LOG_LEVEL_PARAM_NAME <- "sagemaker_container_log_level"
JOB_NAME_PARAM_NAME <- "sagemaker_job_name"
MODEL_SERVER_WORKERS_PARAM_NAME <- "sagemaker_model_server_workers"
SAGEMAKER_REGION_PARAM_NAME <- "sagemaker_region"
SAGEMAKER_OUTPUT_LOCATION <- "sagemaker_s3_output"

#' @title FrameworkModel Class
#' @description A Model for working with an SageMaker ``Framework``.
#'              This class hosts user-defined code in S3 and sets code location and
#'              configuration in model environment variables.
#' @export
FrameworkModel = R6Class("FrameWorkModel",
  inherit = Model,
  public = list(

   #' @description Initialize a ``FrameworkModel``.
   #' @param model_data (str): The S3 location of a SageMaker model data
   #'              ``.tar.gz`` file.
   #' @param image (str): A Docker image URI.
   #' @param role (str): An IAM role name or ARN for SageMaker to access AWS
   #'              resources on your behalf.
   #' @param entry_point (str): Path (absolute or relative) to the Python source
   #'              file which should be executed as the entry point to model
   #'              hosting. This should be compatible with either Python 2.7 or
   #'              Python 3.5. If 'git_config' is provided, 'entry_point' should be
   #'              a relative location to the Python source file in the Git repo.
   #'              Example
   #'              With the following GitHub repo directory structure:
   #'              >>> |----- README.md
   #'              >>> |----- src
   #'              >>>         |----- inference.py
   #'              >>>         |----- test.py
   #'              You can assign entry_point='src/inference.py'.
   #' @param source_dir (str): Path (absolute, relative or an S3 URI) to a directory
   #'              with any other training source code dependencies aside from the entry
   #'              point file (default: None). If ``source_dir`` is an S3 URI, it must
   #'              point to a tar.gz file. Structure within this directory are preserved
   #'              when training on Amazon SageMaker. If 'git_config' is provided,
   #'              'source_dir' should be a relative location to a directory in the Git repo.
   #'              If the directory points to S3, no code will be uploaded and the S3 location
   #'              will be used instead.
   #'              .. admonition:: Example
   #'              With the following GitHub repo directory structure:
   #'              >>> |----- README.md
   #'              >>> |----- src
   #'              >>>         |----- inference.py
   #'              >>>         |----- test.py
   #'              You can assign entry_point='inference.py', source_dir='src'.
   #' @param predictor_cls (callable[string, sagemaker.session.Session]): A
   #'              function to call to create a predictor (default: None). If not
   #'              None, ``deploy`` will return the result of invoking this
   #'              function on the created endpoint name.
   #' @param env (dict[str, str]): Environment variables to run with ``image``
   #'              when hosted in SageMaker (default: None).
   #' @param name (str): The model name. If None, a default model name will be
   #'              selected on each ``deploy``.
   #' @param enable_cloudwatch_metrics (bool): Whether training and hosting
   #'              containers will generate CloudWatch metrics under the
   #'              AWS/SageMakerContainer namespace (default: False).
   #' @param container_log_level (str): Log level to use within the container
   #'              (default: "INFO").
   #' @param code_location (str): Name of the S3 bucket where custom code is
   #'              uploaded (default: None). If not specified, default bucket
   #'              created by ``sagemaker.session.Session`` is used.
   #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
   #'              object, used for SageMaker interactions (default: None). If not
   #'              specified, one is created using the default AWS configuration
   #'              chain.
   #' @param dependencies (list[str]): A list of paths to directories (absolute
   #'              or relative) with any additional libraries that will be exported
   #'              to the container (default: []). The library folders will be
   #'              copied to SageMaker in the same folder where the entrypoint is
   #'              copied. If 'git_config' is provided, 'dependencies' should be a
   #'              list of relative locations to directories with any additional
   #'              libraries needed in the Git repo. If the ```source_dir``` points
   #'              to S3, code will be uploaded and the S3 location will be used
   #'              instead. .. admonition:: Example
   #'              The following call >>> Estimator(entry_point='inference.py',
   #'              dependencies=['my/libs/common', 'virtual-env']) results in
   #'              the following inside the container:
   #'              >>> $ ls
   #'              >>> opt/ml/code
   #'              >>>     |------ inference.py
   #'              >>>     |------ common
   #'              >>>     |------ virtual-env
   #' @param git_config (dict[str, str]): Git configurations used for cloning
   #'              files, including ``repo``, ``branch``, ``commit``,
   #'              ``2FA_enabled``, ``username``, ``password`` and ``token``. The
   #'              ``repo`` field is required. All other fields are optional.
   #'              ``repo`` specifies the Git repository where your training script
   #'              is stored. If you don't provide ``branch``, the default value
   #'              'master' is used. If you don't provide ``commit``, the latest
   #'              commit in the specified branch is used. .. admonition:: Example
   #'              The following config:
   #'              >>> git_config = {'repo': 'https://github.com/aws/sagemaker-python-sdk.git',
   #'              >>>               'branch': 'test-branch-git-config',
   #'              >>>               'commit': '329bfcf884482002c05ff7f44f62599ebc9f445a'}
   #'              results in cloning the repo specified in 'repo', then
   #'              checkout the 'master' branch, and checkout the specified
   #'              commit.
   #'              ``2FA_enabled``, ``username``, ``password`` and ``token`` are
   #'              used for authentication. For GitHub (or other Git) accounts, set
   #'              ``2FA_enabled`` to 'True' if two-factor authentication is
   #'              enabled for the account, otherwise set it to 'False'. If you do
   #'              not provide a value for ``2FA_enabled``, a default value of
   #'              'False' is used. CodeCommit does not support two-factor
   #'              authentication, so do not provide "2FA_enabled" with CodeCommit
   #'              repositories.
   #'              For GitHub and other Git repos, when SSH URLs are provided, it
   #'              doesn't matter whether 2FA is enabled or disabled; you should
   #'              either have no passphrase for the SSH key pairs, or have the
   #'              ssh-agent configured so that you will not be prompted for SSH
   #'              passphrase when you do 'git clone' command with SSH URLs. When
   #'              HTTPS URLs are provided: if 2FA is disabled, then either token
   #'              or username+password will be used for authentication if provided
   #'              (token prioritized); if 2FA is enabled, only token will be used
   #'              for authentication if provided. If required authentication info
   #'              is not provided, python SDK will try to use local credentials
   #'              storage to authenticate. If that fails either, an error message
   #'              will be thrown.
   #'              For CodeCommit repos, 2FA is not supported, so '2FA_enabled'
   #'              should not be provided. There is no token in CodeCommit, so
   #'              'token' should not be provided too. When 'repo' is an SSH URL,
   #'              the requirements are the same as GitHub-like repos. When 'repo'
   #'              is an HTTPS URL, username+password will be used for
   #'              authentication if they are provided; otherwise, python SDK will
   #'              try to use either CodeCommit credential helper or local
   #'              credential storage for authentication.
   #' @param  ... : Keyword arguments passed to the ``Model`` initializer.
   initialize = function(model_data,
                         image,
                         role,
                         entry_point,
                         source_dir=NULL,
                         predictor_cls=NULL,
                         env=NULL,
                         name=NULL,
                         enable_cloudwatch_metrics=FALSE,
                         container_log_level=c("INFO", "WARN", "ERROR", "FATAL", "CRITICAL"),
                         code_location=NULL,
                         sagemaker_session=NULL,
                         dependencies=NULL,
                         git_config=NULL,
                         ...){
     super$initialize(model_data,
                      image,
                      role,
                      predictor_cls=predictor_cls,
                      env=env,
                      name=name,
                      sagemaker_session=sagemaker_session,
                      ...)
     self$entry_point = entry_point
     self$source_dir = source_dir
     self$dependencies = dependencies %||% list()
     self$git_config = git_config
     self$enable_cloudwatch_metrics = enable_cloudwatch_metrics

     # Align logging level with python logging
     container_log_level = match.arg(toupper(container_log_level))
     container_log_level = switch(container_log_level,
                                  "INFO" = "20",
                                  "WARN" = "30",
                                  "ERROR" = "40",
                                  "FATAL" = "50",
                                  "CRITICAL" = "50")
     self$container_log_level = container_log_level
     if (!is.null(code_location)){
       s3_parts = split_s3_uri(code_location)
       self$bucket =s3_parts$bucket
       self$key_prefix = s3_parts$key
     } else {
       self$bucket = NULL
       self$key_prefix = NULL
     }

     if (!islistempty(self$git_config)){
       updates = git_clone_repo(self$git_config, self$entry_point, self$source_dir, self$dependencies)
       self$entry_point = updates$entry_point
       self$source_dir = updates$source_dir
       self$dependencies = updates$dependencies}
     self$uploaded_code = NULL
     self$repacked_model_data = NULL
   },

   #' @description Return a container definition with framework configuration set in
   #'              model environment variables.
   #'              This also uploads user-supplied code to S3.
   #' @param instance_type (str): The EC2 instance type to deploy this Model to.
   #'              For example, 'ml.p2.xlarge'.
   #' @param accelerator_type (str): The Elastic Inference accelerator type to
   #'              deploy to the instance for loading and making inferences to the
   #'              model. For example, 'ml.eia1.medium'.
   #' @return dict[str, str]: A container definition object usable with the
   #'              CreateModel API.
   prepaper_container = function(instance_type=NULL,
                                 accelerator_type=NULL){
     deploy_key_prefix = model_code_key_prefix(self.key_prefix, self.name, self.image)
     private$.upload_code(deploy_key_prefix)
     deploy_env = list(self$env)
     deploy_env = c(deploy_env,private$.framework_env_vars())
     return (container_def(self$image, self$model_data, deploy_env))
   },

   #' @description
   #' Printer.
   #' @param ... (ignored).
   print = function(...){
     cat("<FrameworkModel>")
     invisible(self)
   }
  ),
  private = list(
   .upload_code = function(key_prefix, repack=FALSE){
      local_code = get_config_value("local.local_code", self$sagemaker_session$config)
      if (self$sagemaker_session$local_mode && local_code)
        self$uploaded_code = NULL
      else if (!repack){
        bucket = self$bucket %||% self$sagemaker_session$default_bucket()
        self$uploaded_code = tar_and_upload_dir(
          sagemaker_session=self$sagemaker_session,
          bucket=bucket,
          s3_key_prefix=key_prefix,
          script=self$entry_point,
          directory=self$source_dir,
          dependencies=self$dependencies)
      }

      if (repack){
        bucket = self$bucket %||% self$sagemaker_session$default_bucket()
        repacked_model_data = paste0("s3://", paste(c(bucket, key_prefix, "model.tar.gz"), collapse = "/"))

        repack_model(
          inference_script=self$entry_point,
          source_directory=self$source_dir,
          dependencies=self$dependencies,
          model_uri=self$model_data,
          repacked_model_uri=repacked_model_data,
          sagemaker_session=self$sagemaker_session,
          kms_key=self$model_kms_key)

        self$repacked_model_data = repacked_model_data

        UploadedCode$UserCode$s3_prefix=self$repacked_model_data
        UploadedCode$UserCode$script_name=basename(self$entry_point)

        self$uploaded_code = UploadedCode
      }
   },

   .framework_env_vars = function(){
     if (!is.null(self$uploaded_code)){
       script_name = self$uploaded_code$UserCode$script_name
       if (self$enable_network_isolation())
         dir_name = "/opt/ml/model/code"
       else
         dir_name = self$uploaded_code$UserCode$s3_prefix
     } else if (!islistempty(self$entry_point)){
        script_name = self$entry_point
        dir_name = paste0("file://", self$source_dir)
     } else {
       script_name = NULL
       dir_name = NULL}

     output = list(script_name,
                   dir_name,
                   tolower(as.character(self$enable_cloudwatch_metrics)),
                   self$container_log_level,
                   self$sagemaker_session$paws_region_name)

     names(output) = c(toupper(SCRIPT_PARAM_NAME),
                       toupper(DIR_PARAM_NAME),
                       toupper(CLOUDWATCH_METRICS_PARAM_NAME),
                       toupper(CONTAINER_LOG_LEVEL_PARAM_NAME),
                       toupper(SAGEMAKER_REGION_PARAM_NAME))

     return(output)
   }
  ),
  lock_objects = F
)

#' @title ModelPackage class
#' @description A SageMaker ``Model`` that can be deployed to an ``Endpoint``.
#' @export
ModelPackage = R6Class("ModelPackage",
  inherit = Model,
  public = list(

   #' @description Initialize a SageMaker ModelPackage.
   #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
   #'              SageMaker training jobs and APIs that create Amazon SageMaker
   #'              endpoints use this role to access training data and model
   #'              artifacts. After the endpoint is created, the inference code
   #'              might use the IAM role, if it needs to access an AWS resource.
   #' @param model_data (str): The S3 location of a SageMaker model data
   #'              ``.tar.gz`` file. Must be provided if algorithm_arn is provided.
   #' @param algorithm_arn (str): algorithm arn used to train the model, can be
   #'              just the name if your account owns the algorithm. Must also
   #'              provide ``model_data``.
   #' @param model_package_arn (str): An existing SageMaker Model Package arn,
   #'              can be just the name if your account owns the Model Package.
   #'              ``model_data`` is not required.
   #' @param ... : Additional kwargs passed to the Model constructor.
   initialize = function(role,
                         model_data=NULL,
                         algorithm_arn=NULL,
                         model_package_arn=NULL,
                         ...){
     super$initialize(role = role, model_data = model_data, image_uri = NULL, ...)

     if(!is.null(model_package_arn) && !is.null(algorithm_arn))
       stop("model_package_arn and algorithm_arn are mutually exclusive.",
            sprintf("Both were provided: model_package_arn: %s algorithm_arn: %s", model_package_arn, algorithm_arn),
            call. = F)

     if (is.null(model_package_arn) && is.null(algorithm_arn))
       stop("either model_package_arn or algorithm_arn is required. NULL was provided.",
            call. = F)

     self$algorithm_arn = algorithm_arn
     if (!is.null(self$algorithm_arn)){
       if (is.null(model_data))
         stop("model_data must be provided with algorithm_arn", call. = F)
       self$model_data = model_data}

     self$model_package_arn = model_package_arn
     self$.created_model_package_name = NULL
   },

   #' @description Whether to enable network isolation when creating a model out of this
   #'              ModelPackage
   #' @return bool: If network isolation should be enabled or not.
   enable_network_isolation = function(){
     return(private$.is_marketplace())
   },

   #' @description Create a SageMaker Model Entity
   #' @param ... : Positional arguments coming from the caller. This class does not require
   #'              any so they are ignored.
   .create_sagemaker_model = function(...){
     if (!is.null(self$algorithm_arn)){
       # When ModelPackage is created using an algorithm_arn we need to first
       # create a ModelPackage. If we had already created one then its fine to re-use it.
       if (is.null(self$.created_model_package_name)){
         model_package_name = private$.create_sagemaker_model_package()
         self$sagemaker_session$wait_for_model_package(model_package_name)
         self$.created_model_package_name = model_package_name}
       model_package_name = self$.created_model_package_name
     } else {
       # When a ModelPackageArn is provided we just create the Model
       model_package_name = self$model_package_arn}

     container_def = list("ModelPackageName"= model_package_name)

     if (self$env != list())
       container_def$Environment = self$env

     model_package_short_name = model_package_name.split("/")[-1]
     enable_network_isolation = self$enable_network_isolation()
     self$name = self$name %||% name_from_base(model_package_short_name)
     self$sagemaker_session$create_model(
       self$name,
       self$role,
       container_def,
       vpc_config=self$vpc_config,
       enable_network_isolation=enable_network_isolation)
   },

   #' @description Printer.
   #' @param ... (ignored).
   print = function(...){
     cat("<ModelPackage>")
     invisible(self)
   }
  ),
  private = list(
   .is_marketplace = function(){
     model_package_name = self$model_package_arn %||% self$.created_model_package_name
     if (is.null(model_package_name))
       return(TRUE)

     # Models can lazy-init sagemaker_session until deploy() is called to support
     # LocalMode so we must make sure we have an actual session to describe the model package.
     sagemaker_session = self$sagemaker_session %||% Session$new()

     model_package_desc = sagemaker_session$sagemaker_client$describe_model_package(
       ModelPackageName=model_package_name)

     for (container in model_package_desc$InferenceSpecification$Containers){
       if ("ProductId" %in% names(container))
         return(TRUE)}
     return(FALSE)
   },

   .create_sagemaker_model_package = function(){
     if (is.null(self$algorithm_arn))
       stop("No algorithm_arn was provided to create a SageMaker Model Package", call.= F)

     name = self$name %||% name_from_base(split_str(self$algorithm_arn, "/")[length(split_str(self$algorithm_arn, "/"))])
     description = sprintf("Model Package created from training with %s", self$algorithm_arn)
     self$sagemaker_session$create_model_package_from_algorithm(
       name, description, self$algorithm_arn, self$model_data)
     return(name)
   }

  ),
  lock_objects =  F
)

