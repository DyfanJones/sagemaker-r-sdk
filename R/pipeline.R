# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/pipeline.py

#' @include session.R
#' @include utils.R
#' @include transformer.R

#' @import R6

#' @title A pipeline of SageMaker `Model` instances.
#' @description This pipeline can be deployed as an `Endpoint` on SageMaker.
#' @export
PipeLineModel = R6Class("PipeLineModel",
  public = list(

    #' @description Initialize an SageMaker ``Model`` which can be used to build an
    #'              Inference Pipeline comprising of multiple model containers.
    #' @param models (list[sagemaker.Model]): For using multiple containers to
    #'              build an inference pipeline, you can pass a list of ``sagemaker.Model`` objects
    #'              in the order you want the inference to happen.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if it needs to access an AWS resource.
    #' @param predictor_cls (callable[string, sagemaker.session.Session]): A
    #'              function to call to create a predictor (default: None). If not
    #'              None, ``deploy`` will return the result of invoking this
    #'              function on the created endpoint name.
    #' @param name (str): The model name. If None, a default model name will be
    #'              selected on each ``deploy``.
    #' @param vpc_config (dict[str, list[str]]): The VpcConfig set on the model
    #'              (default: None)
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    #' @param enable_network_isolation (bool): Default False. if True, enables
    #'              network isolation in the endpoint, isolating the model
    #'              container. No inbound or outbound network calls can be made to
    #'              or from the model container.Boolean
    initialize = function(models,
                          role,
                          predictor_cls=NULL,
                          name=NULL,
                          vpc_config=NULL,
                          sagemaker_session=NULL,
                          enable_network_isolation=FALSE){
      self$models = models
      self$role = role
      self$predictor_cls = predictor_cls
      self$name = name
      self$vpc_config = vpc_config
      self$sagemaker_session = sagemaker_session
      self$enable_network_isolation = enable_network_isolation
      self$endpoint_name = NULL
    },

    #' @description Return a dict created by ``sagemaker.pipeline_container_def()`` for
    #'              deploying this model to a specified instance type.
    #'              Subclasses can override this to provide custom container definitions
    #'              for deployment to a specific instance type. Called by ``deploy()``.
    #' @param instance_type (str): The EC2 instance type to deploy this Model to.
    #'              For example, 'ml.p2.xlarge'.
    #' @return list[dict[str, str]]: A list of container definition objects usable
    #'              with the CreateModel API in the scenario of multiple containers
    #'              (Inference Pipeline).
    pipeline_container_def = function(instance_type){
      return(pipeline_container_def(self$models, instance_type))
    },

    #' @description Deploy the ``Model`` to an ``Endpoint``.
    #'              It optionally return a ``Predictor``.
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
    #'              For example, 'ml.p2.xlarge'.
    #' @param serializer (:class:`~sagemaker.serializers.BaseSerializer`): A
    #'              serializer object, used to encode data for an inference endpoint
    #'              (default: None). If ``serializer`` is not None, then
    #'              ``serializer`` will override the default serializer. The
    #'              default serializer is set by the ``predictor_cls``.
    #' @param deserializer (:class:`~sagemaker.deserializers.BaseDeserializer`): A
    #'              deserializer object, used to decode data from an inference
    #' @param endpoint (default: None). If ``deserializer`` is not None, then
    #'              ``deserializer`` will override the default deserializer. The
    #'              default deserializer is set by the ``predictor_cls``.
    #' @param endpoint_name (str): The name of the endpoint to create (default:
    #'              None). If not specified, a unique endpoint name will be created.
    #' @param tags (List[dict[str, str]]): The list of tags to attach to this
    #'              specific endpoint.
    #' @param wait (bool): Whether the call should wait until the deployment of
    #'              model completes (default: True).
    #' @param update_endpoint (bool): Flag to update the model in an existing
    #'              Amazon SageMaker endpoint. If True, this will deploy a new
    #'              EndpointConfig to an already existing endpoint and delete
    #'              resources corresponding to the previous EndpointConfig. If
    #'              False, a new endpoint will be created. Default: False
    #' @param data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
    #'              configuration related to Endpoint data capture for use with
    #'              Amazon SageMaker Model Monitoring. Default: None.
    #' @return callable[string, sagemaker.session.Session] or None: Invocation of
    #'              ``self.predictor_cls`` on the created endpoint name, if ``self.predictor_cls``
    #'              is not None. Otherwise, return None.
    deploy = function(initial_instance_count,
                      instance_type,
                      serializer=NULL,
                      deserializer=NULL,
                      endpoint_name=NULL,
                      tags=NULL,
                      wait=TRUE,
                      update_endpoint=FALSE,
                      data_capture_config=NULL){
      if (is.null(self$sagemaker_session))
        self$sagemaker_session = Session$new()

        containers = self$pipeline_container_def(instance_type)

        self$name = self$name %||% name_from_image(containers[[1]]$Image)
        self$sagemaker_session$create_model(
          self$name,
          self$role,
          containers,
          vpc_config=self$vpc_config,
          enable_network_isolation=self$enable_network_isolation
        )

        production_variant = production_variant(
          self$name, instance_type, initial_instance_count
        )
        self$endpoint_name = endpoint_name %||% self$name

        data_capture_config_dict = NULL
        if (!is.null(data_capture_config))
          data_capture_config_dict = data_capture_config$to_request_list()

        if (update_endpoint){
          endpoint_config_name = self$sagemaker_session$create_endpoint_config(
            name=self$name,
            model_name=self$name,
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            tags=tags,
            data_capture_config_dict=data_capture_config_dict)
          self$sagemaker_session$update_endpoint(
            self$endpoint_name, endpoint_config_name, wait=wait)
        } else {
          self$sagemaker_session$endpoint_from_production_variants(
            name=self$endpoint_name,
            production_variants=list(production_variant),
            tags=tags,
            wait=wait,
            data_capture_config_dict=data_capture_config_dict)
        }

        if (!islistempty(self$predictor_cls)){
          predictor = self$predictor_cls(self$endpoint_name, self$sagemaker_session)
          if (!islistempty(serializer))
            predictor$serializer = serializer
          if (!islistempty(deserializer))
            predictor$deserializer = deserializer
          return(predictor)}
        return(NULL)
    },

    #' @description Return a ``Transformer`` that uses this Model.
    #' @param instance_count (int): Number of EC2 instances to use.
    #' @param instance_type (str): Type of EC2 instance to use, for example,
    #'              ml.c4.xlarge'.
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
      private$.create_sagemaker_pipeline_model(instance_type)

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
        sagemaker_session=self$sagemaker_session)
      )
    },

    #' @description Delete the SageMaker model backing this pipeline model. This does not
    #'              delete the list of SageMaker models used in multiple containers to build
    #'              the inference pipeline.
    delete_model = function(){
      if (is.null(self$name))
        stop("The SageMaker model must be created before attempting to delete.", call. = F)

      self$sagemaker_session$delete_model(self$name)
    }
  ),
  private = list(

    # Create a SageMaker Model Entity
    # Args:
    #   instance_type (str): The EC2 instance type that this Model will be
    # used for, this is only used to determine if the image needs GPU
    # support or not.
    .create_sagemaker_pipeline_model = function(instance_type){
      if (!is.null(self$sagemaker_session))
        self$sagemaker_session = Session$new()

      containers = self$pipeline_container_def(instance_type)

      self$name = self.$name %||% name_from_image(containers[[1]]$Image)
      self$sagemaker_session$create_model(
        self$name,
        self$role,
        containers,
        vpc_config=self$vpc_config,
        enable_network_isolation=self$enable_network_isolation
      )
    }
  ),
  lock_objects = F
)
