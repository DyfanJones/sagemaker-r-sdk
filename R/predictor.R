# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/predictor.py

#' @include utils.R
#' @include fw_utils.R
#' @include model.R
#' @include fw_utils.R
#' @include session.R
#' @include vpc_utils.R
#' @include analytics.R
#' @include model_monitor_model_monitoring.R
#' @include serializers.R
#' @include deserializers.R
#' @include lineage_context.R
#' @include error.R

#' @import paws
#' @import R6
#' @import utils

#' @title Predictor Class
#' @description Make prediction requests to an Amazon SageMaker endpoint.
#' @export
Predictor = R6Class("Predictor",
  public = list(

    #' @field endpoint_name
    #'        Name of the Amazon SageMaker endpoint
    endpoint_name = NULL,

    #' @field sagemaker_session
    #'        A SageMaker Session object
    sagemaker_session = NULL,

    #' @field serializer
    #'        Class to convert data into raw to send to endpoint
    serializer = NULL,

    #' @field deserializer
    #'        Class to convert raw data back from the endpoint
    deserializer = NULL,

    #' @description Initialize a ``Predictor``.
    #'              Behavior for serialization of input data and deserialization of
    #'              result data can be configured through initializer arguments. If not
    #'              specified, a sequence of bytes is expected and the API sends it in the
    #'              request body without modifications. In response, the API returns the
    #'              sequence of bytes from the prediction result without any modifications.
    #' @param endpoint_name (str): Name of the Amazon SageMaker endpoint_name to which
    #'              requests are sent.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: NULL). If not
    #'              specified, one is created using the default AWS configuration
    #'              chain.
    #' @param serializer (callable): Accepts a single argument, the input data,
    #'              and returns a sequence of bytes. It may provide a
    #'              ``content_type`` attribute that defines the endpoint request
    #'              content type. If not specified, a sequence of bytes is expected
    #'              for the data. (default: ``IdentitySerializer``)
    #' @param deserializer (callable): Accepts two arguments, the result data and
    #'              the response content type, and returns a sequence of bytes. It
    #'              may provide a ``content_type`` attribute that defines the
    #'              endpoint response's "Accept" content type. If not specified, a
    #'              sequence of bytes is expected for the data. (default: ``BytesDeserializer``)
    #' @param ... Any other parameters, including and deprecate parameters from sagemaker v-1.
    initialize = function(endpoint_name,
                          sagemaker_session=NULL,
                          serializer=IdentitySerializer$new(),
                          deserializer=BytesDeserializer$new(),
                          ...){
      kwargs = list(...)
      removed_kwargs("content_type", kwargs)
      removed_kwargs("accept", kwargs)

      endpoint_name = renamed_kwargs("endpoint" ,"endpoint_name", endpoint_name, kwargs)
      self$endpoint_name = endpoint_name
      self$sagemaker_session = sagemaker_session %||% Session$new()
      self$serializer = if(inherits(serializer, "BaseSerializer")) serializer else ValueError$new("Please use a R6 BaseSerializer Class.")
      self$deserializer = if(inherits(deserializer, "BaseDeserializer")) deserializer else ValueError$new("Please use a R6 Deserializer Class.")

      private$.endpoint_config_name = private$.get_endpoint_config_name()
      private$.model_names = private$.get_model_names()
      private$.context = NULL
    },

    #' @description Return the inference from the specified endpoint.
    #' @param data (object): Input data for which you want the model to provide
    #'              inference. If a serializer was specified when creating the
    #'              Predictor, the result of the serializer is sent as input
    #'              data. Otherwise the data must be sequence of bytes, and the
    #'              predict method then sends the bytes in the request body as is.
    #' @param initial_args (list[str,str]): Optional. Default arguments for boto3
    #'              ``invoke_endpoint`` call. Default is NULL (no default
    #'              arguments).
    #' @param target_model (str): S3 model artifact path to run an inference request on,
    #'              in case of a multi model endpoint. Does not apply to endpoints hosting
    #'              single model (Default: NULL)
    #' @param target_variant (str): The name of the production variant to run an inference
    #'              request on (Default: NULL). Note that the ProductionVariant identifies the model
    #'              you want to host and the resources you want to deploy for hosting it.
    #' @param inference_id (str): If you provide a value, it is added to the captured data
    #'              when you enable data capture on the endpoint (Default: None).
    #' @return object: Inference for the given input. If a deserializer was specified when creating
    #'              the Predictor, the result of the deserializer is
    #'              returned. Otherwise the response returns the sequence of bytes
    #'              as is.
    predict = function(data,
                       initial_args=NULL,
                       target_model=NULL,
                       target_variant=NULL,
                       inference_id=NULL){

      request_args = private$.create_request_args(
        data, initial_args, target_model, target_variant, inference_id)

      response = do.call(self$sagemaker_session$sagemaker_runtime$invoke_endpoint,
                         request_args)

      return(private$.handle_response(response))
    },

    #' @description Update the existing endpoint with the provided attributes.
    #'              This creates a new EndpointConfig in the process. If ``initial_instance_count``,
    #'              ``instance_type``, ``accelerator_type``, or ``model_name`` is specified, then a new
    #'              ProductionVariant configuration is created; values from the existing configuration
    #'              are not preserved if any of those parameters are specified.
    #' @param initial_instance_count (int): The initial number of instances to run in the endpoint.
    #'              This is required if ``instance_type``, ``accelerator_type``, or ``model_name`` is
    #' specified. Otherwise, the values from the existing endpoint configuration's
    #'              ProductionVariants are used.
    #' @param instance_type (str): The EC2 instance type to deploy the endpoint to.
    #'              This is required if ``initial_instance_count`` or ``accelerator_type`` is specified.
    #'              Otherwise, the values from the existing endpoint configuration's
    #'              ``ProductionVariants`` are used.
    #' @param accelerator_type (str): The type of Elastic Inference accelerator to attach to
    #'              the endpoint, e.g. "ml.eia1.medium". If not specified, and
    #'              ``initial_instance_count``, ``instance_type``, and ``model_name`` are also ``None``,
    #' the values from the existing endpoint configuration's ``ProductionVariants`` are
    #'              used. Otherwise, no Elastic Inference accelerator is attached to the endpoint.
    #' @param model_name (str): The name of the model to be associated with the endpoint.
    #'              This is required if ``initial_instance_count``, ``instance_type``, or
    #'              ``accelerator_type`` is specified and if there is more than one model associated
    #'              with the endpoint. Otherwise, the existing model for the endpoint is used.
    #' @param tags (list[dict[str, str]]): The list of tags to add to the endpoint
    #'              config. If not specified, the tags of the existing endpoint configuration are used.
    #'              If any of the existing tags are reserved AWS ones (i.e. begin with "aws"),
    #'              they are not carried over to the new endpoint configuration.
    #' @param kms_key (str): The KMS key that is used to encrypt the data on the storage volume
    #'              attached to the instance hosting the endpoint If not specified,
    #'              the KMS key of the existing endpoint configuration is used.
    #' @param data_capture_config_dict (dict): The endpoint data capture configuration
    #'              for use with Amazon SageMaker Model Monitoring. If not specified,
    #'              the data capture configuration of the existing endpoint configuration is used.
    #' @param wait (boolean): Waits for endpoint to update.
    update_endpoint = function(initial_instance_count=NULL,
                               instance_type=NULL,
                               accelerator_type=NULL,
                               model_name=NULL,
                               tags=NULL,
                               kms_key=NULL,
                               data_capture_config_dict=NULL,
                               wait=TRUE){
      production_variants = NULL

      if (!is.null(initial_instance_count) || !is.null(instance_type) || !is.null(accelerator_type) || !is.null(model_name)){
        if (is.null(instance_type) || is.null(initial_instance_count)){
          ValueError$new(
            "Missing initial_instance_count and/or instance_type. Provided values: ",
            sprintf("initial_instance_count=%s, instance_type=%s, accelerator_type=%s, model_name=%s.",
                    initial_instance_count, instance_type, accelerator_type, model_name))}
        if (is.null(model_name)){
          if (length(private$.model_names) > 1){
            ValueError$new(
              "Unable to choose a default model for a new EndpointConfig because ",
              sprintf("the endpoint has multiple models: %s",
                      paste(private$.model_names, collapse = ", ")))}
          model_name = private$.model_names[[1]]
        } else {
          private$.model_names = list(model_name)
        }

        production_variant_config = production_variant(
          model_name,
          instance_type,
          initial_instance_count=initial_instance_count,
          accelerator_type=accelerator_type)

        production_variants = list(production_variant_config)
      }

      new_endpoint_config_name = name_from_base(private$.endpoint_config_name)
      self$sagemaker_session$create_endpoint_config_from_existing(
        private$.endpoint_config_name,
        new_endpoint_config_name,
        new_tags=tags,
        new_kms_key=kms_key,
        new_data_capture_config_dict=data_capture_config_dict,
        new_production_variants=production_variants)

      self$sagemaker_session$update_endpoint(
        self$endpoint_name, new_endpoint_config_name, wait=wait
      )

      private$.endpoint_config_name = new_endpoint_config_name
    },

    #' @description Delete the Amazon SageMaker endpoint backing this predictor. Also
    #'              delete the endpoint configuration attached to it if
    #'              delete_endpoint_config is True.
    #' @param delete_endpoint_config (bool, optional): Flag to indicate whether to
    #'              delete endpoint configuration together with endpoint. Defaults
    #'              to True. If True, both endpoint and endpoint configuration will
    #'              be deleted. If False, only endpoint will be deleted.
    delete_endpoint = function(delete_endpoint_config=TRUE){
      if(delete_endpoint_config)
        private$.delete_endpoint_config()

      self$sagemaker_session$delete_endpoint(self$endpoint_name)
    },

    #' @description Deletes the Amazon SageMaker models backing this predictor.
    delete_model = function(){
      request_failed = FALSE
      failed_models = list()
      for (model_name in private$.model_names){
        tryCatch({
          self$sagemaker_session$delete_model(model_name)
        },
        error = function(e) {
          request_failed = TRUE
          failed_models[[model_name]] = request_failed})
        }
      if (any(unlist(request_failed)))
        SagemakerError$new(
          "One or more models cannot be deleted, please retry. \n",
          sprintf("Failed models: %s", paste0(failed_models[[failed_models == FALSE]], collapse = ", ")))
    },

    #' @description Updates the DataCaptureConfig for the Predictor's associated Amazon SageMaker Endpoint
    #'              to enable data capture. For a more customized experience, refer to
    #'              update_data_capture_config, instead.
    enable_data_capture = function(){
      self$update_data_capture_config(
        data_capture_config=DataCaptureConfig$new(
          enable_capture=TRUE, sagemaker_session=self$sagemaker_session)
        )
    },

    #' @description Updates the DataCaptureConfig for the Predictor's associated Amazon SageMaker Endpoint
    #'              to disable data capture. For a more customized experience, refer to
    #'              update_data_capture_config, instead.
    disable_data_capture = function(){
      self$update_data_capture_config(
        data_capture_config=DataCaptureConfig$new(
          enable_capture=FALSE, sagemaker_session=self$sagemaker_session))
    },

    #' @description Updates the DataCaptureConfig for the Predictor's associated Amazon SageMaker Endpoint
    #'              with the provided DataCaptureConfig.
    #' @param data_capture_config (sagemaker.model_monitor.DataCaptureConfig): The
    #'              DataCaptureConfig to update the predictor's endpoint to use.
    update_data_capture_config = function(data_capture_config = NULL){
      endpoint_desc = self$sagemaker_session$sagemaker$describe_endpoint(
        EndpointName=self$endpoint_name)

      new_config_name = name_from_base(base=self$endpoint_name)

      data_capture_config_dict = NULL
      if (!is.null(data_capture_config))
        data_capture_config_dict = data_capture_config$to_request_list()

      self$sagemaker_session$create_endpoint_config_from_existing(
        existing_config_name=endpoint_desc$EndpointConfigName,
        new_config_name=new_config_name,
        new_data_capture_config_dict=data_capture_config_dict)

      self$sagemaker_session$update_endpoint(
        endpoint_name=self$endpoint_name, endpoint_config_name=new_config_name)
    },

    #' @description Generates ModelMonitor objects (or DefaultModelMonitors).
    #'              Objects are generated based on the schedule(s) associated with the endpoint
    #'              that this predictor refers to.
    #' @return [sagemaker.model_monitor.model_monitoring.ModelMonitor]: A list of
    #'              ModelMonitor (or DefaultModelMonitor) objects.
    list_monitors = function(){
      monitoring_schedules_list = self$sagemaker_session$list_monitoring_schedules(
        endpoint_name=self$endpoint_name)
      if (islistempty(monitoring_schedules_list$MonitoringScheduleSummaries)){
        writeLines(sprintf("No monitors found for endpoint. endpoint: %s",self$endpoint_name))
        return(list())}

      monitors = list()
      for (schedule_dict in monitoring_schedules_list$MonitoringScheduleSummaries){
        schedule_name = schedule_dict$MonitoringScheduleName
        monitoring_type = schedule_dict$MonitoringType
        clazz = private$.get_model_monitor_class(schedule_name, monitoring_type)
        monitors = c(monitors, clazz$new()$attach(
          monitor_schedule_name=schedule_name,
          sagemaker_session=self$sagemaker_session))
      }

      return(monitors)
    },

    #' @description Retrieves the lineage context object representing the endpoint.
    #' @return ContextEndpoint: The context for the endpoint.
    endpoint_context = function(){
      if (!is.null(private$.context))
        return(private$.context)

      # retrieve endpoint by name to get arn
      response = self$sagemaker_session$sagemaker$describe_endpoint(
        EndpointName=self$endpoint_name)

      endpoint_arn = response[["EndpointArn"]]

      # list context by source uri using arn
      contexts = list(
        EndpointContext$new()$list(sagemaker_session=self$sagemaker_session, source_uri=endpoint_arn))

      if (length(contexts) != 0){
        # create endpoint context object
        private$.context = EndpointContext$new()$load(
          sagemaker_session=self$sagemaker_session, context_name=contexts[[1]]$context_name)
      }
      return (private$.context)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      return(print_class(self))
    }
  ),
  private = list(
    .endpoint_config_name = NULL,
    .model_names = NULL,
    .context = NULL,

    # Placeholder docstring
    .handle_response = function(response){
      response_body = response[["Body"]]
      content_type = response[["ContentType"]] %||% "application/octet-stream"
      return(self$deserializer$deserialize(response_body, content_type))
    },

    # Placeholder docstring
    .create_request_args = function(data,
                                    initial_args=NULL,
                                    target_model=NULL,
                                    target_variant=NULL,
                                    inference_id=NULL){
      args = if (!islistempty(initial_args)) initial_args else list()

      if (!("EndpointName" %in% names(args)))
        args[["EndpointName"]] = self$endpoint_name

      if (!("ContentType" %in% names(args)))
        args[["ContentType"]] = self$content_type

      if (!("Accept" %in% names(args)))
        args[["Accept"]] = paste(self$accept, collapse = ",")

      args[["TargetModel"]] = target_model
      args[["TargetVariant"]] = target_variant
      args[["InferenceId"]] = inference_id

      args[["Body"]] = self$serializer$serialize(data)
      return(args)
    },

    # Delete the Amazon SageMaker endpoint configuration
    .delete_endpoint_config = function(){
      self$sagemaker_session$delete_endpoint_config(private$.endpoint_config_name)
    },

    # Retrieves the lineage context object representing the endpoint.
    # Examples:
    #   .. code-block:: python
    # predictor = Predictor()
    # ...
    # context = predictor.endpoint_context()
    # models = context.models()
    # Returns:
    #   ContextEndpoint: The context for the endpoint.
    .get_endpoint_config_name = function(){
      endpoint_desc = self$sagemaker_session$sagemaker$describe_endpoint(
            EndpointName=self$endpoint_name)
      endpoint_config_name = endpoint_desc$EndpointConfigName
      return(endpoint_config_name)
    },

    # Placeholder docstring
    .get_model_names = function() {
      endpoint_config = self$sagemaker_session$sagemaker$describe_endpoint_config(
            EndpointConfigName=private$.endpoint_config_name)
      production_variants = endpoint_config$ProductionVariants
      return (sapply(production_variants, function(x) x$ModelName))
    },

    # Decide which ModelMonitor class the given schedule should attach to
    # Args:
    #   schedule_name (str): The schedule to be attached.
    # monitoring_type (str): The monitoring type of the schedule
    # Returns:
    #   sagemaker.model_monitor.ModelMonitor: ModelMonitor or a subclass of ModelMonitor.
    # Raises:
    #   TypeError: If the class could not be decided (due to unknown monitoring type).
    .get_model_monitor_class = function(schedule_name,
                                        monitoring_type){
      if (monitoring_type == "ModelBias"){
        clazz = ModelBiasMonitor
      } else if (monitoring_type == "ModelExplainability"){
          clazz = ModelExplainabilityMonitor
      } else {
          schedule = self$sagemaker_session$describe_monitoring_schedule(
            monitoring_schedule_name=schedule_name)
          embedded_job_definition = schedule[["MonitoringScheduleConfig"]][[
            "MonitoringJobDefinition"]]
      }
      if (!is.null(embedded_job_definition)){  # legacy v1 schedule
        image_uri = embedded_job_definition[["MonitoringAppSpecification"]][["ImageUri"]]
        if (endsWith(image_uri, DEFAULT_REPOSITORY_NAME)) {
          clazz = DefaultModelMonitor
        } else {
          clazz = ModelMonitor}
      } else if (monitoring_type == "DataQuality"){
        clazz = DefaultModelMonitor
      } else if (monitoring_type == "ModelQuality"){
        clazz = ModelQualityMonitor
      } else {
        SagemakerError$new(sprintf("Unknown monitoring type: %s",monitoring_type))}
      return(clazz)
    }
  ),
  active = list(

    #' @field content_type
    #' The MIME type of the data sent to the inference endpoint.
    content_type = function(){
      return(self$serializer$CONTENT_TYPE)
    },

    #' @field accept
    #' The content type(s) that are expected from the inference endpoint.
    accept = function(){
      return(self$deserializer$ACCEPT)
    },

    #' @field endpoint
    #' Deprecated attribute. Please use endpoint_name.
    endpoint = function(){
      renamed_warning("The endpoint attribute")
      return(self$endpoint_name)
    }
  )
)

#' @title S3 method that wraps Predictor Class
#' @description Predicted values returned from endpoint
#' @param object a sagemaker model
#' @param newdata data for model to predict
#' @param serializer method class to serializer data to sagemaker model. Requires to be
#'              a class inherited from \link{BaseSerializer}. (Default: \link{csv_serializer})
#' @param deserializer method class to deserializer return data streams from sagemaker model.
#'              Requires to be a class inherited from \link{BaseDeserializer}.
#'              (Default: \link{csv_deserializer})
#' @param ... arguments passed to ``Predictor$predict``
#' @export
predict.Predictor <- function(object, newdata, serializer = CSVSerializer$new(), deserializer = CSVDeserializer$new(), ...){
  stopifnot(is.null(serializer) || inherits(serializer,"BaseSerializer"),
            is.null(deserializer) || inherits(deserializer,"BaseDeserializer"))

  obj = object$clone()
  obj$serializer = serializer
  obj$deserializer = deserializer
  obj$predict(newdata, ...)
}

# TODO: R s3 methods for all aws modelling methods
