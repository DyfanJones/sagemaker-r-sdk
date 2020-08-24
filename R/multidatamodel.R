# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/multidatamodel.py

#' @include session.R
#' @include s3.R

#' @import R6
#' @import paws
#' @importFrom urltools url_parse

MULTI_MODEL_CONTAINER_MODE <- "MultiModel"

#' @title MultiDataModel Class
#' @description A SageMaker ``MultiDataModel`` that can be used to deploy multiple models to the same
#'              SageMaker ``Endpoint``, and also deploy additional models to an existing SageMaker
#'              multi-model ``Endpoint``
#' @export
MultiDataModel = R6Class("MultiDataModel",
  inherit = Model,
  public = list(

    #' @description Initialize a ``MultiDataModel``. In addition to these arguments, it supports all
    #'              arguments supported by ``Model`` constructor
    #' @param name (str): The model name.
    #' @param model_data_prefix (str): The S3 prefix where all the models artifacts (.tar.gz)
    #'              in a Multi-Model endpoint are located
    #' @param model (sagemaker.Model): The Model object that would define the
    #'              SageMaker model attributes like vpc_config, predictors, etc.
    #'              If this is present, the attributes from this model are used when
    #'              deploying the ``MultiDataModel``.  Parameters 'image_uri', 'role' and 'kwargs'
    #'              are not permitted when model parameter is set.
    #' @param image_uri (str): A Docker image_uri URI. It can be null if the 'model' parameter
    #'              is passed to during ``MultiDataModel`` initialization (default: None)
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role if it needs to access some AWS resources.
    #'              It can be null if this is being used to create a Model to pass
    #'              to a ``PipelineModel`` which has its own Role field or if the 'model' parameter
    #'              is passed to during ``MultiDataModel`` initialization (default: None)
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    #' @param ... : Keyword arguments passed to the
    #'              :class:`~sagemaker.model.Model` initializer.
    #'              .. tip::
    #'              You can find additional parameters for initializing this class at
    #'              :class:`~sagemaker.model.Model`.
    initialize = function(name,
                          model_data_prefix,
                          model=NULL,
                          image_uri=NULL,
                          role=NULL,
                          sagemaker_session=NULL,
                          ...){
      # Validate path
      if (!startsWith(model_data_prefix,"s3://"))
        stop(sprintf('Expecting S3 model prefix beginning with "s3://". Received: "%s"',
            model_data_prefix), call. = F)

      if (!is.null(model) && (!is.null(image_uri) || !is.null(role) || !islistemptylist(...)))
        stop("Parameters image_uri, role or kwargs are not permitted when model parameter is passed.",
             call. = F)

      self$name = name
      self$model_data_prefix = model_data_prefix
      self$model = model
      self$container_mode = MULTI_MODEL_CONTAINER_MODE
      self$sagemaker_session = sagemaker_session %||% Session$new()

      self$s3_client = paws::s3(config = self$paws_credentials$credentials)

      # Set the ``Model`` parameters if the model parameter is not specified
      if (is.null(self.model))
        super$initialize(
          image_uri,
          self$model_data_prefix,
          role,
          name=self$name,
          sagemaker_session=self$sagemaker_session,
          ...)
    },

    #' @description Return a container definition set with MultiModel mode,
    #'              model data and other parameters from the model (if available).
    #'              Subclasses can override this to provide custom container definitions
    #'              for deployment to a specific instance type. Called by ``deploy()``.
    #' @param instance_type (str): The EC2 instance type to deploy this Model to.
    #'              For example, 'ml.p2.xlarge'.
    #' @param accelerator_type (str): The Elastic Inference accelerator type to
    #'              deploy to the instance for loading and making inferences to the
    #'              model. For example, 'ml.eia1.medium'.
    #' @return dict[str, str]: A complete container definition object usable with the CreateModel API
    prepare_container_def = function(instance_type=NULL,
                                     accelerator_type=NULL){
    # Copy the trained model's image_uri and environment variables if they exist. Models trained
    # with FrameworkEstimator set framework specific environment variables which need to be
    # copied over
      if (!is.null(self$model)){
        container_definition = self$model$prepare_container_def(instance_type, accelerator_type)
        image_uri = container_definition$Image
        environment = container_definition$Environment
      } else {
        image_uri = self$image_uri
        environment = self$env
        return (container_def(
          image_uri,
          env=environment,
          model_data_url=self$model_data_prefix,
          container_mode=self$container_mode)
        )
      }
    },

    #' @description Deploy this ``Model`` to an ``Endpoint`` and optionally return a ``Predictor``.
    #'              Create a SageMaker ``Model`` and ``EndpointConfig``, and deploy an
    #'              ``Endpoint`` from this ``Model``. If self.model is not None, then the ``Endpoint``
    #'              will be deployed with parameters in self.model (like vpc_config,
    #'              enable_network_isolation, etc).  If self.model is None, then use the parameters
    #'              in ``MultiDataModel`` constructor will be used. If ``self.predictor_cls`` is not
    #'              None, this method returns a the result of invoking ``self.predictor_cls`` on
    #'              the created endpoint name.
    #'              The name of the created model is accessible in the ``name`` field of
    #'              this ``Model`` after deploy returns
    #'              The name of the created endpoint is accessible in the
    #'              ``endpoint_name`` field of this ``Model`` after deploy returns.
    #' @param initial_instance_count (int): The initial number of instances to run
    #'              in the ``Endpoint`` created from this ``Model``.
    #' @param instance_type (str): The EC2 instance type to deploy this Model to.
    #'              For example, 'ml.p2.xlarge', or 'local' for local mode.
    #' @param serializer (:class:`~sagemaker.serializers.BaseSerializer`): A
    #'              serializer object, used to encode data for an inference endpoint
    #'              (default: None). If ``serializer`` is not None, then
    #'              ``serializer`` will override the default serializer. The
    #'              default serializer is set by the ``predictor_cls``.
    #' @param deserializer (:class:`~sagemaker.deserializers.BaseDeserializer`): A
    #'              deserializer object, used to decode data from an inference
    #'              endpoint (default: None). If ``deserializer`` is not None, then
    #'              ``deserializer`` will override the default deserializer. The
    #'              default deserializer is set by the ``predictor_cls``.
    #' @param accelerator_type (str): Type of Elastic Inference accelerator to
    #'              deploy this model for model loading and inference, for example,
    #'              'ml.eia1.medium'. If not specified, no Elastic Inference
    #'              accelerator will be attached to the endpoint. For more
    #'              information:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
    #' @param endpoint_name (str): The name of the endpoint to create (default:
    #'              None). If not specified, a unique endpoint name will be created.
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
    #'              ``self.predictor_cls`` on the created endpoint name,
    #'              if ``self.predictor_cls``
    #'              is not None. Otherwise, return None.
    deploy = function(initial_instance_count,
                      instance_type,
                      serializer=NULL,
                      deserializer=NULL,
                      accelerator_type=NULL,
                      endpoint_name=NULL,
                      tags=NULL,
                      kms_key=NULL,
                      wait=TRUE,
                      data_capture_config=NULL){
      # Set model specific parameters
      if (!is.null(self$model)){
        enable_network_isolation = self$model$enable_network_isolation()
        role = self$model$role
        vpc_config = self$model$vpc_config
        predictor_cls = self$model$predictor_cls
      } else {
        enable_network_isolation = self$enable_network_isolation()
        role = self$role
        vpc_config = self$vpc_config
        predictor_cls = self$predictor_cls
      }

      if (is.null(role))
        stop("Role can not be null for deploying a model", call. = F)

      # TODO: create LocalSession
      if (instance_type == "local" && !inherits(self$sagemaker_session, "LocalSession"))
        self$sagemaker_session = LocalSession$new()

      container_def = self$prepare_container_def(instance_type, accelerator_type=accelerator_type)
      self$sagemaker_session$create_model(
        self$name,
        role,
        container_def,
        vpc_config=vpc_config,
        enable_network_isolation=enable_network_isolation,
        tags=tags)

      production_variant = production_variant(
        self$name, instance_type, initial_instance_count, accelerator_type=accelerator_type)

      if (!is.null(endpoint_name))
        self$endpoint_name = endpoint_name
      else
        self$endpoint_name = self$name

      data_capture_config_dict = NULL
      if (!islistempty(data_capture_config))
        data_capture_config_dict = data_capture_config$to_request_list()

      self$sagemaker_session$endpoint_from_production_variants(
        name=self$endpoint_name,
        production_variants=list(production_variant),
        tags=tags,
        kms_key=kms_key,
        wait=wait,
        data_capture_config_list=data_capture_config_dict)

      if (!is.null(predictor_cls)){
        predictor = predictor_cls$new(self$endpoint_name, self$sagemaker_session)
        if (!is.null(serializer))
          predictor$serializer = serializer
        if (!is.null(deserializer))
          predictor$deserializer = deserializer
          return(predictor)
      }
      return(NULL)
    },

    #' @description Adds a model to the ``MultiDataModel`` by uploading or copying the model_data_source
    #'              artifact to the given S3 path model_data_path relative to model_data_prefix
    #' @param model_data_source : Valid local file path or S3 path of the trained model artifact
    #' @param model_data_path : S3 path where the trained model artifact
    #'              should be uploaded relative to ``self.model_data_prefix`` path. (default: None).
    #'              If None, then the model artifact is uploaded to a path relative to model_data_prefix
    #' @return str: S3 uri to uploaded model artifact
    add_model = function(model_data_source,
                         model_data_path=NULL){
      parse_result = url_parse(model_data_source)

      # If the model source is an S3 path, copy the model artifact to the destination S3 path
      if (parse_result$scheme == "s3"){
        s3_parts = split_s3_uri(model_data_source)
        copy_source = list("Bucket"= s3_parts$bucket, "Key"= s3_parts$key)

        if (is.null(model_data_path))
          model_data_path = source_model_data_path

        # Construct the destination path
        dst_url = file.path(self$model_data_prefix, model_data_path)
        dst_parts = split_s3_uri(dst_url)

        # Copy the model artifact
        self$s3_client$copy_object(Bucket = dst_parts$bucket,
                                   CopySource = paste(copy_source, collapse = "/"),
                                   Key =  dst_parts$key)
        return(file.path("s3:/", dst_parts$bucket, dst_parts$key))
      }

      # If the model source is a local path, upload the local model artifact to the destination
      #  s3 path
      if (file.exists(model_data_source)){
        dst_parts = split_s3_uri(self$model_data_prefix)
        if (!is.null(model_data_path))
          dst_s3_uri = file.path(dst_parts$key, model_data_path)
        else
          dst_s3_uri = file.path(dst_parts$key, basename(model_data_source))
        # Upload file to s3
        obj = readBin(model_data_source, "raw", n = file.size(model_data_source))
        s3$put_object(Body = obj, Bucket = dst_parts$bucket, Key = dst_s3_uri)
        # return upload_path
        return(file.path("s3:/", dst_parts$bucket, dst_s3_uri))
      }

      # Raise error if the model source is of an unexpected type
      stop(sprintf(
        "model_source must either be a valid local file path or s3 uri. Received: %s",
        model_data_source), call. = F)
    },

    #' @return Generates and returns relative paths to model archives stored at model_data_prefix
    #'         S3 location.
    list_models = function(){
      uri_parts = split_s3_uri(self$model_data_prefix)
      file_keys = self$sagemaker_session$list_s3_files(bucket=uri_parts$bucket, key_prefix=uri_parts$key)
      # Return the model paths relative to the model_data_prefix
      # Ex: "a/b/c.tar.gz" -> "b/c.tar.gz" where url_prefix = "a/"
      return(gsub(uri_parts$key, "", file_keys))
    }
  ),
  lock_objects = F
)
