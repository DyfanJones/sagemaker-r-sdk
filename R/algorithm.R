# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/algorithm.py

#' @include parameter.R
#' @include vpc_utils.R
#' @include utils.R
#' @include estimator.R
#' @include transformer.R
#' @include predictor.R

#' @import R6

#' @title AlgorithmEstimator Class
#' @description A generic Estimator to train using any algorithm object (with an
#'              ``algorithm_arn``). The Algorithm can be your own, or any Algorithm from AWS
#'              Marketplace that you have a valid subscription for. This class will perform
#'              client-side validation on all the inputs.
#' @export
AlgorithmEstimator = R6Class("AlgorithmEstimator",
   inherit = EstimatorBase,
   public = list(
     #' @field .hyperpameters_with_range
     #' These Hyperparameter Types have a range definition.
     .hyperpameters_with_range = c("Integer", "Continuous", "Categorical"),

     #' @description Initialize an ``AlgorithmEstimator`` instance.
     #' @param algorithm_arn (str): algorithm arn used for training. Can be just the name if your
     #'              account owns the algorithm.
     #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker
     #'              training jobs and APIs that create Amazon SageMaker endpoints use this role to
     #'              access training data and model artifacts. After the endpoint
     #'              is created, the inference code might use the IAM role, if it
     #'              needs to access an AWS resource.
     #' @param train_instance_count (int): Number of Amazon EC2 instances to
     #'              use for training.
     #' @param train_instance_type (str): Type of EC2
     #'              instance to use for training, for example, 'ml.c4.xlarge'.
     #' @param train_volume_size (int): Size in GB of the EBS volume to use for
     #'              storing input data during training (default: 30). Must be large enough to store
     #'              training data if File Mode is used (which is the default).
     #' @param train_volume_kms_key (str): Optional. KMS key ID for encrypting EBS volume attached
     #'              to the training instance (default: NULL).
     #' @param train_max_run (int): Timeout in seconds for training (default: 24 * 60 * 60).
     #'              After this amount of time Amazon SageMaker terminates the
     #'              job regardless of its current status.
     #' @param input_mode (str): The input mode that the algorithm supports
     #'              (default: 'File'). Valid modes:
     #'              * 'File' - Amazon SageMaker copies the training dataset from
     #'              the S3 location to a local directory.
     #'              * 'Pipe' - Amazon SageMaker streams data directly from S3 to
     #'              the container via a Unix-named pipe.
     #'              This argument can be overriden on a per-channel basis using
     #'              ``sagemaker.session.s3_input.input_mode``.
     #' @param output_path (str): S3 location for saving the training result (model artifacts and
     #'              output files). If not specified, results are stored to a default bucket. If
     #'              the bucket with the specific name does not exist, the
     #'              estimator creates the bucket during the
     #'              :meth:`~sagemaker.estimator.EstimatorBase.fit` method
     #'              execution.
     #' @param output_kms_key (str): Optional. KMS key ID for encrypting the
     #'              training output (default: NULL).
     #' @param base_job_name (str): Prefix for
     #'              training job name when the
     #'              :meth:`~sagemaker.estimator.EstimatorBase.fit`
     #'              method launches. If not specified, the estimator generates a
     #'              default job name, based on the training image name and
     #'              current timestamp.
     #' @param hyperparameters (dict): Dictionary containing the hyperparameters to
     #'              initialize this estimator with.
     #' @param sagemaker_session (sagemaker.session.Session): Session object which manages
     #'              interactions with Amazon SageMaker APIs and any other AWS services needed. If
     #'              not specified, the estimator creates one using the default
     #'              AWS configuration chain.
     #' @param tags (list[dict]): List of tags for labeling a training job. For more, see
     #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
     #' @param subnets (list[str]): List of subnet ids. If not specified
     #'              training job will be created without VPC config.
     #' @param security_group_ids (list[str]): List of security group ids. If
     #'              not specified training job will be created without VPC config.
     #' @param model_uri (str): URI where a pre-trained model is stored, either locally or in S3
     #'              (default: NULL). If specified, the estimator will create a channel pointing to
     #'              the model so the training job can download it. This model
     #'              can be a 'model.tar.gz' from a previous training job, or
     #'              other artifacts coming from a different source.
     #'              More information:
     #'              https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html#td-deserialization
     #' @param model_channel_name (str): Name of the channel where 'model_uri'
     #'              will be downloaded (default: 'model').
     #' @param metric_definitions (list[dict]): A list of dictionaries that defines the metric(s)
     #'              used to evaluate the training jobs. Each dictionary contains two keys: 'Name' for
     #'              the name of the metric, and 'Regex' for the regular
     #'              expression used to extract the metric from the logs.
     #' @param encrypt_inter_container_traffic (bool): Specifies whether traffic between training
     #'              containers is encrypted for the training job (default: ``False``).
     #' @param ... : Additional kwargs. This is unused. It's only added for AlgorithmEstimator
     #'              to ignore the irrelevant arguments.
     initialize = function(algorithm_arn,
                           role,
                           train_instance_count,
                           train_instance_type,
                           train_volume_size=30,
                           train_volume_kms_key=NULL,
                           train_max_run=24 * 60 * 60,
                           input_mode="File",
                           output_path=NULL,
                           output_kms_key=NULL,
                           base_job_name=NULL,
                           sagemaker_session=NULL,
                           hyperparameters=NULL,
                           tags=NULL,
                           subnets=NULL,
                           security_group_ids=NULL,
                           model_uri=NULL,
                           model_channel_name="model",
                           metric_definitions=NULL,
                           encrypt_inter_container_traffic=FALSE,
                           ...){
       self$algorithm_arn = algorithm_arn
       super$initialize(
         role,
         train_instance_count,
         train_instance_type,
         train_volume_size,
         train_volume_kms_key,
         train_max_run,
         input_mode,
         output_path,
         output_kms_key,
         base_job_name,
         sagemaker_session,
         tags,
         subnets,
         security_group_ids,
         model_uri=model_uri,
         model_channel_name=model_channel_name,
         metric_definitions=metric_definitions,
         encrypt_inter_container_traffic=encrypt_inter_container_traffic)

       self$algorithm_spec = self$sagemaker_session$sagemaker$describe_algorithm(
         AlgorithmName=algorithm_arn)
       self$validate_train_spec()
       self$hyperparameter_definitions = private$.parse_hyperparameters()

       self$hyperparam_list = list()
       if (!is.null(hyperparameters))
         do.call(self$set_hyperparameters, hyperparameters)
     },

     #' @description Placeholder docstring
     validate_train_spec = function(){
       train_spec = self$algorithm_spec[["TrainingSpecification"]]
       algorithm_name = self$algorithm_spec[["AlgorithmName"]]

       # Check that the input mode provided is compatible with the training input modes for the
       # algorithm.
       train_input_modes = private$.algorithm_training_input_modes(train_spec["TrainingChannels"])
       if (!(self$input_mode %in% train_input_modes))
         stop(sprintf("Invalid input mode: %s. %s only supports: %s", self$input_mode, algorithm_name, train_input_modes),
              call. = F)

       # Check that the training instance type is compatible with the algorithm.
       supported_instances = train_spec[["SupportedTrainingInstanceTypes"]]
       if (!(self$train_instance_type %in% supported_instances)){
         stop(sprint("Invalid train_instance_type: %s. %s supports the following instance types: %s",
                     self$train_instance_type, algorithm_name, supported_instances),
              call. = F)}

       # Verify if distributed training is supported by the algorithm
       if (self$train_instance_count > 1
         && "SupportsDistributedTraining" %in% train_spec
         && !is.null(train_spec[["SupportsDistributedTraining"]]))
         stop(sprintf("Distributed training is not supported by %s. Please set train_instance_count=1", algorithm_name),
              call. = F)
     },

     #' @description formats hyperparameters for model tunning
     #' @param ... model hyperparameters
     set_hyperparameter = function(...){
       args = list(...)
       for(x in names(args)){
         value = private$.validate_and_cast_hyperparameter(x, args[[x]])
         self$hyperparam_list[[x]] = value}
       private$.validate_and_set_default_hyperparameters()
     },

     #' @description Returns the hyperparameters as a dictionary to use for training.
     #'              The fit() method, that does the model training, calls this method to
     #'              find the hyperparameters you specified.
     hyperparameters = function(){
       return(self$hyperparam_list)
     },

     #' @description Returns the docker image to use for training.
     #'              The fit() method, that does the model training, calls this method to
     #'              find the image to use for model training.
     training_image_uri = function(){
       stop("training_image_uri is never meant to be called on Algorithm Estimators", call. = F)
     },

     #' @description Return True if this Estimator will need network isolation to run.
     #'              On Algorithm Estimators this depends on the algorithm being used. If
     #'              this is algorithm owned by your account it will be False. If this is an
     #'              an algorithm consumed from Marketplace it will be True.
     #' @return bool: Whether this Estimator needs network isolation or not.
     enable_network_isolation = function(){
       return(private$.is_marketplace())
     },

     #' @description Create a model to deploy.
     #'              The serializer, deserializer, content_type, and accept arguments are
     #'              only used to define a default RealTimePredictor. They are ignored if an
     #'              explicit predictor class is passed in. Other arguments are passed
     #'              through to the Model class.
     #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
     #'              which is also used during transform jobs. If not specified, the
     #'              role from the Estimator will be used.
     #' @param predictor_cls (RealTimePredictor): The predictor class to use when
     #'              deploying the model.
     #' @param serializer (callable): Should accept a single argument, the input
     #'              data, and return a sequence of bytes. May provide a content_type
     #'              attribute that defines the endpoint request content type
     #' @param deserializer (callable): Should accept two arguments, the result
     #'              data and the response content type, and return a sequence of
     #'              bytes. May provide a content_type attribute that defines the
     #'              endpoint response Accept content type.
     #' @param content_type (str): The invocation ContentType, overriding any
     #'              content_type from the serializer
     #' @param accept (str): The invocation Accept, overriding any accept from the
     #'              deserializer.
     #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
     #'              the model. Default: use subnets and security groups from this Estimator.
     #'              * 'Subnets' (list[str]): List of subnet ids.
     #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
     #' @param ... : Additional arguments for creating a :class:`~sagemaker.model.ModelPackage`.
     #'              .. tip::
     #'              You can find additional parameters for using this method at
     #'              :class:`~sagemaker.model.ModelPackage` and
     #'              :class:`~sagemaker.model.Model`.
     #' @return a Model ready for deployment.
     create_model = function(role=NULL,
                             predictor_cls=NULL,
                             serializer=NULL,
                             deserializer=NULL,
                             content_type=NULL,
                             accept=NULL,
                             vpc_config_override="VPC_CONFIG_DEFAULT",
                             ...){
       if (is.null(predictor_cls)) {
         predict_wrapper = function(endpoint, session){
         return (RealTimePredictor$new(
           endpoint, session, serializer, deserializer, content_type, accept))}

        predictor_cls = predict_wrapper}

       role = role %||% self$role

       param = list(role=role,
                    algorithm_arn=self$algorithm_arn,
                    model_data=self$model_data,
                    vpc_config=self$get_vpc_config(vpc_config_override),
                    sagemaker_session=self$sagemaker_session,
                    predictor_cls=predictor_cls,
                    ...)

       return(do.call(ModelPackage$new, param))
     },

     #' @description Return a ``Transformer`` that uses a SageMaker Model based on the
     #'              training job. It reuses the SageMaker Session and base job name used by
     #'              the Estimator.
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
     #' @param role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
     #'              which is also used during transform jobs. If not specified, the
     #'              role from the Estimator will be used.
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
                            role=NULL,
                            volume_kms_key=NULL){
       role = role %||% self.role

       if (!is.null(self$latest_training_job)){
         model = self$create_model(role=role)
         model$.create_sagemaker_model()
         model_name = model$name
         transform_env = list()
         if(!islistempty(env)){
           transform_env = model$env
           transform_env = modifyList(transform_env, env, keep.null =  T)}
         if(private$.is_marketplace()){
           transform_env = NULL}

         tags = tags %||% self$tags
        } else {
           stop("No finished training job found associated with this estimator", call. = F)}

       return(Transformer$new(
         model_name,
         instance_count,
         instance_type,
         strategy=strategy,
         assemble_with=assemble_with,
         output_path=output_path,
         output_kms_key=output_kms_key,
         accept=accept,
         max_concurrent_transforms=max_concurrent_transforms,
         max_payload=max_payload,
         env=transform_env,
         tags=tags,
         base_transform_job_name=self$base_job_name,
         volume_kms_key=volume_kms_key,
         sagemaker_session=self$sagemaker_session))
     },

     #' @description Train a model using the input training dataset.
     #'              The API calls the Amazon SageMaker CreateTrainingJob API to start
     #'              model training. The API uses configuration you provided to create the
     #'              estimator and the specified input training data to send the
     #'              CreatingTrainingJob request to Amazon SageMaker.
     #'              This is a synchronous operation. After the model training
     #'              successfully completes, you can call the ``deploy()`` method to host the
     #'              model using the Amazon SageMaker hosting services.
     #' @param inputs (str or dict or sagemaker.session.s3_input): Information
     #'              about the training data. This can be one of three types:
     #'              \itemize{
     #'                \item{\strong{(str)} the S3 location where training data is saved, or a file:// path in
     #'              local mode.}
     #'                \item{\strong{(dict[str, str]} or dict[str, sagemaker.session.s3_input]) If using multiple
     #'              channels for training data, you can specify a dict mapping channel names to
     #'              strings or :func:`~sagemaker.session.s3_input` objects.}
     #'                \item{\strong{(sagemaker.session.s3_input)} - channel configuration for S3 data sources that can
     #'              provide additional information as well as the path to the training dataset.
     #'              See :func:`sagemaker.session.s3_input` for full details.}
     #'                \item{\strong{(sagemaker.session.FileSystemInput)} - channel configuration for
     #'              a file system data source that can provide additional information as well as
     #'              the path to the training dataset.}}
     #' @param wait (bool): Whether the call should wait until the job completes (default: True).
     #' @param logs ([str]): A list of strings specifying which logs to print. Acceptable
     #'              strings are "All", "NULL", "Training", or "Rules". To maintain backwards
     #'              compatibility, boolean values are also accepted and converted to strings.
     #'              Only meaningful when wait is True.
     #' @param job_name (str): Training job name. If not specified, the estimator generates
     #'              a default job name, based on the training image name and current timestamp.
     #' @param experiment_config (dict[str, str]): Experiment management configuration.
     #'              Dictionary contains three optional keys,
     #'              'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
     fit = function(inputs=NULL,
                    wait=TRUE,
                    logs=TRUE,
                    job_name=NULL){
       if (!is.null(inputs))
         private$.validate_input_channels(inputs)

       super$fit(inputs, wait, logs, job_name)
     },

     #' @description
     #' Printer.
     #' @param ... (ignored).
     print = function(...){
        cat("<AlgorithmEstimator>")
        invisible(self)
     }
   ),
   private = list(
     .is_marketplace= function(){
       return ("ProductId" %in% self$algorithm_spec)
     },

     .prepare_for_training = function(job_name = NULL){
       # Validate hyperparameters
       # an explicit call to set_hyperparameters() will also validate the hyperparameters
       # but it is possible that the user never called it.
       private$.validate_and_set_default_hyperparameters()
       super$.prepare_for_training(job_name)
     },

     .validate_input_channels = function(channels){
       train_spec = self$algorithm_spec$TrainingSpecification
       algorithm_name = self$algorithm_spec$AlgorithmName
       training_channels = lapply(train_spec$TrainingChannels, function(c) c)
       names(training_channels) = sapply(train_spec$TrainingChannels, function(c) c$Name)

       # check for unknown channels that the algorithm does not support
       for (c in channels){
         if(!(c %in% training_channels))
           stop(sprintf("Unknown input channel: %s is not supported by: %s", c, algorithm_name), call. = F)
       }

       # check for required channels that were not provided
       for (i in seq_along(training_channels)){
         name = names(training_channels)[i]
         channel = training_channels[[i]]
         if (!(name %in% names(channels)) && "IsRequired" %in% names(channel) && !islistempty(channel$IsRequired))
           stop(sprintf("Required input channel: %s Was not provided.", name), call. = F)
       }
     },

     .vatlidate_and_cast_hyperparameter = function(name, v){
       algorithm_name = self$algorithm_spec$AlgorithmName

       if (!(name %in% names(self$hyperparameter_definitions)))
         stop(sprintf("Invalid hyperparameter: %s is not supported by %s", name, algorithm_name),
              call. = F)

       definition = self$hyperparameter_definitions[[name]]
       if ("class" %in% names(definition))
         value = definition$class$public_methods$cast_to_type(v)
       else
         value = v

       if ("range" %in% names(definition) && !definition$range$is_valid(value)){
         valid_range = definition$range$as_tuning_range(name)
         stop(sprintf("Invalid value: %s Supported range: %s", value, valid_range), call. = F)}
       return(value)
     },

     .validate_and_set_default_hyperparameters = function(){
       # Check if all the required hyperparameters are set. If there is a default value
       # for one, set it.
       for (i in seq_along(self$hyperparameter_definitions)){
         name = names(self$hyperparameter_definitions)[i]
         definition = self$hyperparameter_definitions[[i]]
         if (!(name %in% self$hyperparam_list)){
           spec = definition$spec
           if ("DefaultValue" %in% names(spec))
             self$hyperparam_list[[name]] = spec$DefaultValue
           else if ("IsRequired" %in% names(spec) && !islistempty(spec$IsRequired))
             stop(sprintf("Required hyperparameter: %s is not set", name), call. = F)}
       }
     },

     .parse_hyperparameters = function(){
       definitions = list()

       training_spec = self$algorithm_spec$TrainingSpecification
       if ("SupportedHyperParameters" %in% names(training_spec)){
         hyperparameters = training_spec$SupportedHyperParameters
         for (h in hyperparameters){
           parameter_type = h$Type
           name = h$Name
           param = private$.hyperparameter_range_and_class(
             parameter_type, h)

           definitions[[name]] = list("spec"= h)
           if (!islistempty(param$parameter_range))
             definitions[[name]]$range = parameter_range
           if (!islistempty(param$parameter_class))
             definitions[[name]]$class = parameter_class
         }
       }

       return(definitions)
     },

     .hyperparameter_range_and_class = function(parameter_type, hyperparameter){
       if (parameter_type %in% self$.hyperpameters_with_range)
         range_name = paste0(parameter_type , "ParameterRangeSpecification")

       parameter_class = NULL
       parameter_range = NULL

       if (parameter_type %in% c("Integer", "Continuous")){
         # Integer and Continuous are handled the same way. We get the min and max values
         # and just create an Instance of Parameter. Note that the range is optional for all
         # the Parameter Types.
         if (parameter_type == "Integer")
           parameter_class = IntegerParameter
         else
           parameter_class = ContinuousParameter

         if ("Range" %in% names(hyperparameter)){
           min_value = parameter_class$public_methods$cast_to_type(
             hyperparameter$Range[[range_name]][["MinValue"]])
           max_value = parameter_class$public_methods$cast_to_type(
             hyperparameter$Range[[range_name]][["MaxValue"]])
           parameter_range = parameter_class$new(min_value, max_value)
           }
         } else if(parameter_type == "Categorical") {
             parameter_class = CategoricalParameter
             if("Range" %in% names(hyperparameter)){
               values = hyperparameter$Range[[range_name]][["Values"]]
               parameter_range = CategoricalParameter$new(values)}
          } else if(parameter_type == "FreeText") {
            NULL
          } else
            stop(sprintf("Invalid Hyperparameter type: %s. Valid ones are: (Integer, Continuous, Categorical, FreeText)", parameter_type),
                 call. = F)

       return(list(parameter_class = parameter_class, parameter_range = parameter_range))
     },

     .algorithm_training_input_modes = function(training_channels){
       current_input_modes = c("File", "Pipe")
       for (channel in training_channels){
         supported_input_modes = unique(channel$SupportedInputModes)
         current_input_modes = c(current_input_modes, supported_input_modes)}

      return(current_input_modes)
     },

     # Convert the job description to init params that can be handled by the
     # class constructor
     # Args:
     #   job_details (dict): the returned job details from a DescribeTrainingJob
     # API call.
     # model_channel_name (str): Name of the channel where pre-trained
     # model data will be downloaded.
     # Returns:
     #   dict: The transformed init_params
     .prepare_init_params_from_job_description= function(job_details, model_channel_name=None){
     init_params = super$.prepare_init_params_from_job_description(
       job_details, model_channel_name)

     # This hyperparameter is added by Amazon SageMaker Automatic Model Tuning.
     # It cannot be set through instantiating an estimator.
     if ("_tuning_objective_metric" %in% names(init_params$hyperparameters)){
       init_params[["hyperparameters"]][["_tuning_objective_metric"]] = NULL}
     return(init_params)
     }

   ),

   lock_objects = FALSE
)
