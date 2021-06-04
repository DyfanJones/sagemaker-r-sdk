# NOTE: This code has been modified from AWS Sagemaker Python:
#bhttps://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/automl/automl.py

#' @include model.R
#' @include pipeline.R
#' @include automl_candidate_estimator.R
#' @include job.R
#' @include session.R
#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common
#' @import lgr

#' @title AutoML Class
#' @description A class for creating and interacting with SageMaker AutoML jobs.
#' @export
AutoML = R6Class("AutoML",
  public = list(

    #' @description Initialize AutoML class
    #'              Place holder doc string
    #' @param role :
    #' @param target_attribute_name :
    #' @param output_kms_key :
    #' @param output_path :
    #' @param base_job_name :
    #' @param compression_type :
    #' @param sagemaker_session :
    #' @param volume_kms_key :
    #' @param encrypt_inter_container_traffic :
    #' @param vpc_config :
    #' @param problem_type :
    #' @param max_candidates :
    #' @param max_runtime_per_training_job_in_seconds :
    #' @param total_job_runtime_in_seconds :
    #' @param job_objective :
    #' @param generate_candidate_definitions_only :
    #' @param tags :
    initialize = function(role,
                          target_attribute_name,
                          output_kms_key=NULL,
                          output_path=NULL,
                          base_job_name=NULL,
                          compression_type=NULL,
                          sagemaker_session=NULL,
                          volume_kms_key=NULL,
                          encrypt_inter_container_traffic=FALSE,
                          vpc_config=NULL,
                          problem_type=NULL,
                          max_candidates=NULL,
                          max_runtime_per_training_job_in_seconds=NULL,
                          total_job_runtime_in_seconds=NULL,
                          job_objective=NULL,
                          generate_candidate_definitions_only=FALSE,
                          tags=NULL){
      self$role = role
      self$output_kms_key = output_kms_key
      self$output_path = output_path
      self$base_job_name = base_job_name
      self$compression_type = compression_type
      self$volume_kms_key = volume_kms_key
      self$encrypt_inter_container_traffic = encrypt_inter_container_traffic
      self$vpc_config = vpc_config
      self$problem_type = problem_type
      self$max_candidate = max_candidates
      self$max_runtime_per_training_job_in_seconds = max_runtime_per_training_job_in_seconds
      self$total_job_runtime_in_seconds = total_job_runtime_in_seconds
      self$target_attribute_name = target_attribute_name
      self$job_objective = job_objective
      self$generate_candidate_definitions_only = generate_candidate_definitions_only
      self$tags = tags

      self$current_job_name = NULL
      self$.auto_ml_job_desc = NULL
      self$.best_candidate = NULL
      self$sagemaker_session = sagemaker_session %||% Session$new()

      private$.check_problem_type_and_job_objective(self$problem_type, self$job_objective)
    },

    #' @description Create an AutoML Job with the input dataset.
    #' @param inputs (str or list[str] or AutoMLInput): Local path or S3 Uri where the training data
    #'              is stored. Or an AutoMLInput object. If a local path is provided, the dataset will
    #'              be uploaded to an S3 location.
    #' @param wait (bool): Whether the call should wait until the job completes (default: True).
    #' @param logs (bool): Whether to show the logs produced by the job. Only meaningful when wait
    #'              is True (default: True). if ``wait`` is False, ``logs`` will be set to False as
    #'              well.
    #' @param job_name (str): Training job name. If not specified, the estimator generates
    #'              a default job name, based on the training image name and current timestamp.
    fit = function(inputs=NULL,
                   wait=TRUE,
                   logs=TRUE,
                   job_name=NULL){
      if (!wait && logs){
        logs = FALSE
        LOGGER$warn("Setting `logs` to FALSE. `logs` is only meaningful when `wait` is TRUE.")}

      # upload data for users if provided local path
      # validations are done in .Job._format_inputs_to_input_config
      if (inherits(inputs, "character")){
        if (!startsWith(inputs, "s3://"))
          inputs = self$sagemaker_session$upload_data(inputs, key_prefix="auto-ml-input-data")
      }
      private$.prepare_for_auto_ml_job(job_name=job_name)

      self$latest_auto_ml_job = AutoMLJob$new(self$sagemaker_session)$start_new(self, inputs)  # pylint: disable=W0201
      if (wait)
        self$latest_auto_ml_job$wait(logs=logs)
    },

    #' @description Attach to an existing AutoML job.
    #'              Creates and returns a AutoML bound to an existing automl job.
    #' @param auto_ml_job_name (str): AutoML job name
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, the one originally associated with the ``AutoML`` instance is used.
    #' @return sagemaker.automl.AutoML: A ``AutoML`` instance with the attached automl job.
    attach = function(auto_ml_job_name,
                      sagemaker_session=NULL){

      sagemaker_session = sagemaker_session %||% Session$new()

      auto_ml_job_desc = sagemaker_session$describe_auto_ml_job(auto_ml_job_name)
      automl_job_tags = sagemaker_session$sagemaker$list_tags(
        ResourceArn=auto_ml_job_desc$AutoMLJobArn
      )$Tags

      aml_args = list(
        role=auto_ml_job_desc$RoleArn,
        target_attribute_name=auto_ml_job_desc$InputDataConfig[[1]]$TargetAttributeName,
        output_kms_key=auto_ml_job_desc$OutputDataConfig$KmsKeyId,
        output_path=auto_ml_job_desc$OutputDataConfig$S3OutputPath,
        base_job_name=auto_ml_job_name,
        compression_type=auto_ml_job_desc$InputDataConfig[[1]]$CompressionType,
        sagemaker_session=sagemaker_session,
        volume_kms_key=auto_ml_job_desc$AutoMLJobConfig$SecurityConfig$VolumeKmsKeyId,
        encrypt_inter_container_traffic= auto_ml_job_desc$AutoMLJobConfig$SecurityConfig$EnableInterContainerTrafficEncryption %||% FALSE,
        vpc_config=auto_ml_job_desc$AutoMLJobConfig$SecurityConfig$VpcConfig,
        problem_type=auto_ml_job_desc$ProblemType,
        max_candidates=auto_ml_job_desc$AutoMLJobConfig$CompletionCriteria$MaxCandidates,
        max_runtime_per_training_job_in_seconds=auto_ml_job_desc$AutoMLJobConfig$CompletionCriteria$MaxRuntimePerTrainingJobInSeconds,
        total_job_runtime_in_seconds=auto_ml_job_desc$AutoMLJobConfig$CompletionCriteria$MaxAutoMLJobRuntimeInSeconds,
        job_objective=auto_ml_job_desc$AutoMLJobObjective$MetricName,
        generate_candidate_definitions_only=auto_ml_job_desc$GenerateCandidateDefinitionsOnly %||% FALSE,
        tags=automl_job_tags
      )

      amlj = do.call(AutoML$new, aml_args)

      amlj$current_job_name = auto_ml_job_name
      amlj$latest_auto_ml_job = auto_ml_job_name
      amlj$.auto_ml_job_desc = auto_ml_job_desc
      return(amlj)
    },

    #' @description Returns the job description of an AutoML job for the given job name.
    #' @param job_name (str): The name of the AutoML job to describe.
    #'              If None, will use object's latest_auto_ml_job name.
    #' @return dict: A dictionary response with the AutoML Job description.
    describe_auto_ml_job = function(job_name = NULL){
      if (is.null(job_name))
        job_name = self$current_job_name
      self$.auto_ml_job_desc = self$sagemaker_session$describe_auto_ml_job(job_name)
      return(self$.auto_ml_job_desc)
    },

    #' @description Returns the best candidate of an AutoML job for a given name.
    #' @param job_name (str): The name of the AutoML job. If None, will use object's
    #'              .current_auto_ml_job_name.
    #' @return dict: A dictionary with information of the best candidate.
    best_candidate = function(job_name=NULL){
      if (!is.null(self$.best_candidate))
        return(self$.best_candidate)

      if (is.null(job_name))
        job_name = self$current_job_name
      if (is.null(self$.auto_ml_job_desc))
        self$.auto_ml_job_desc = self$sagemaker_session$describe_auto_ml_job(job_name)
      if (self$.auto_ml_job_desc$AutoMLJobName != job_name)
        self$.auto_ml_job_desc = self$sagemaker_session$describe_auto_ml_job(job_name)

      self$.best_candidate = self$.auto_ml_job_desc$BestCandidate
      return(self$.best_candidate)
    },

    #' @description Returns the list of candidates of an AutoML job for a given name.
    #' @param job_name (str): The name of the AutoML job. If None, will use object's
    #'              .current_job name.
    #' @param status_equals (str): Filter the result with candidate status, values could be
    #'              "Completed", "InProgress", "Failed", "Stopped", "Stopping"
    #' @param candidate_name (str): The name of a specified candidate to list.
    #'              Default to None.
    #' @param candidate_arn (str): The Arn of a specified candidate to list.
    #'              Default to None.
    #' @param sort_order (str): The order that the candidates will be listed in result.
    #'              Default to None.
    #' @param sort_by (str): The value that the candidates will be sorted by.
    #'              Default to None.
    #' @param max_results (int): The number of candidates will be listed in results,
    #'              between 1 to 100. Default to None. If None, will return all the candidates.
    #' @return list: A list of dictionaries with candidates information.
    list_candidates = function(job_name=NULL,
                               status_equals=NULL,
                               candidate_name=NULL,
                               candidate_arn=NULL,
                               sort_order=NULL,
                               sort_by=NULL,
                               max_results=NULL){
      if(is.null(job_name))
        job_name = self$current_job_name

      list_candidates_args = list("job_name"=job_name)
      list_candidates_args$status_equals = status_equals
      list_candidates_args$candidate_name = candidate_name
      list_candidates_args$candidate_arn = candidate_arn
      list_candidates_args$sort_order = sort_order
      list_candidates_args$sort_by = sort_by
      list_candidates_args$max_results = max_results

      return(do.call(self$sagemaker_session$list_candidates, list_candidates_args)$Candidates)
    },

    #' @description Creates a model from a given candidate or the best candidate from the job.
    #' @param name (str): The pipeline model name.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, the one originally associated with the ``AutoML`` instance is used.:
    #' @param candidate (CandidateEstimator or dict): a CandidateEstimator used for deploying
    #'              to a SageMaker Inference Pipeline. If None, the best candidate will
    #'              be used. If the candidate input is a dict, a CandidateEstimator will be
    #'              created from it.
    #' @param vpc_config (dict): Specifies a VPC that your training jobs and hosted models have
    #'              access to. Contents include "SecurityGroupIds" and "Subnets".
    #' @param enable_network_isolation (bool): Isolates the training container. No inbound or
    #'              outbound network calls can be made, except for calls between peers within a
    #'              training cluster for distributed training. Default: False
    #' @param model_kms_key (str): KMS key ARN used to encrypt the repacked
    #'              model archive file if the model is repacked
    #' @param predictor_cls (callable[string, sagemaker.session.Session]): A
    #'              function to call to create a predictor (default: None). If
    #'              specified, ``deploy()``  returns the result of invoking this
    #'              function on the created endpoint name.
    #' @param inference_response_keys (list): List of keys for response content. The order of the
    #'              keys will dictate the content order in the response.
    #' @return PipelineModel object.
    create_model = function(name,
                            sagemaker_session=NULL,
                            candidate=NULL,
                            vpc_config=NULL,
                            enable_network_isolation=FALSE,
                            model_kms_key=NULL,
                            predictor_cls=NULL,
                            inference_response_keys=NULL){
      sagemaker_session = sagemaker_session %||% self$sagemaker_session

      if (is.null(candidate)) {
        candidate_dict = self$best_candidate()
        candidate = CandidateEstimator$new(candidate_dict, sagemaker_session=sagemaker_session)
      } else if (inherits(candidate, "list")){
          candidate = CandidateEstimator$new(candidate, sagemaker_session=sagemaker_session)
      }

      inference_containers = candidate$containers

      inference_containers = self$validate_and_update_inference_response(inference_containers, inference_response_keys)

      # construct Model objects
      models = list()

      for (container in inference_containers){
        image_uri = container$Image
        model_data = container$ModelDataUrl
        env = container$Environment

        model = Model$new(
          image_uri=image_uri,
          model_data=model_data,
          role=self$role,
          env=env,
          vpc_config=vpc_config,
          sagemaker_session=sagemaker_session %||% self$sagemaker_session,
          enable_network_isolation=enable_network_isolation,
          model_kms_key=model_kms_key
        )
        models = c(models, model)
      }

      pipeline = PipelineModel$new(
        models=models,
        role=self$role,
        predictor_cls=predictor_cls,
        name=name,
        vpc_config=vpc_config,
        sagemaker_session=sagemaker_session %||% self$sagemaker_session
      )
      return(pipeline)
    },

    #' @description Deploy a candidate to a SageMaker Inference Pipeline.
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
    #' @param candidate (CandidateEstimator or dict): a CandidateEstimator used for deploying
    #'              to a SageMaker Inference Pipeline. If None, the best candidate will
    #'              be used. If the candidate input is a dict, a CandidateEstimator will be
    #'              created from it.
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, the one originally associated with the ``AutoML`` instance is used.
    #' @param name (str): The pipeline model name. If None, a default model name will
    #'              be selected on each ``deploy``.
    #' @param endpoint_name (str): The name of the endpoint to create (default:
    #'              None). If not specified, a unique endpoint name will be created.
    #' @param tags (List[dict[str, str]]): The list of tags to attach to this
    #'              specific endpoint.
    #' @param wait (bool): Whether the call should wait until the deployment of
    #'              model completes (default: True).
    #' @param vpc_config (dict): Specifies a VPC that your training jobs and hosted models have
    #'              access to. Contents include "SecurityGroupIds" and "Subnets".
    #' @param enable_network_isolation (bool): Isolates the training container. No inbound or
    #'              outbound network calls can be made, except for calls between peers within a
    #'              training cluster for distributed training. Default: False
    #' @param model_kms_key (str): KMS key ARN used to encrypt the repacked
    #'              model archive file if the model is repacked
    #' @param predictor_cls (callable[string, sagemaker.session.Session]): A
    #'              function to call to create a predictor (default: None). If
    #'              specified, ``deploy()``  returns the result of invoking this
    #'              function on the created endpoint name.
    #' @param inference_response_keys (list): List of keys for response content. The order of the
    #'              keys will dictate the content order in the response.
    #' @return callable[string, sagemaker.session.Session] or ``None``:
    #'              If ``predictor_cls`` is specified, the invocation of ``self.predictor_cls`` on
    #'              the created endpoint name. Otherwise, ``None``.
    deploy = function(initial_instance_count,
                      instance_type,
                      serializer=NULL,
                      deserializer=NULL,
                      candidate=NULL,
                      sagemaker_session=NULL,
                      name=NULL,
                      endpoint_name=NULL,
                      tags=NULL,
                      wait=TRUE,
                      vpc_config=NULL,
                      enable_network_isolation=FALSE,
                      model_kms_key=NULL,
                      predictor_cls=NULL,
                      inference_response_keys=NULL){
      sagemaker_session = sagemaker_session %||% self$sagemaker_session
      model = self$create_model(
        name=name,
        sagemaker_session=sagemaker_session,
        candidate=candidate,
        inference_response_keys=inference_response_keys,
        vpc_config=vpc_config,
        enable_network_isolation=enable_network_isolation,
        model_kms_key=model_kms_key,
        predictor_cls=predictor_cls
      )

      return(model$deploy(
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        serializer=serializer,
        deserializer=deserializer,
        endpoint_name=endpoint_name,
        tags=tags,
        wait=wait)
      )
    },

    #' @description Validates the requested inference keys and updates response content.
    #'              On validation, also updates the inference containers to emit appropriate response
    #'              content in the inference response.
    #' @param inference_containers (list): list of inference containers
    #' @param inference_response_keys (list): list of inference response keys
    validate_and_update_inference_response = function(inference_containers,
                                                      inference_response_keys){
      if (missing(inference_response_keys))
        return(invisible(NULL))

      private$.check_inference_keys(inference_response_keys, inference_containers)

      previous_container_output = NULL

      for (i in seq_along(inference_containers)) {
        supported_inference_keys_container = private$.get_supported_inference_keys(
          inference_containers[[i]], default=list())
        if (islistempty(supported_inference_keys_container)) {
          previous_container_output = NULL
          next}
        current_container_output = NULL
        for (key in inference_response_keys) {
          if (key %in% supported_inference_keys_container){
            current_container_output = if(!is.null(current_container_output)) paste0(current_container_output, ",",  key)  else key
          }
        }
        if (!islistempty(previous_container_output))
          inference_containers[[i]]$Environment$SAGEMAKER_INFERENCE_INPUT = previous_container_output
        if (!islistempty(current_container_output))
          inference_containers[[i]]$Environment$SAGEMAKER_INFERENCE_OUTPUT = current_container_output

        previous_container_output = current_container_output
      }
      return(inference_containers)
    },

    #' @description format class
    format = function(){
      return(format_class(self))
    }
  ),
  private = list(

    # Validate if problem_type and job_objective are both None or are both provided.
    # Args:
    #   problem_type (str): The type of problem of this AutoMLJob. Valid values are
    # "Regression", "BinaryClassification", "MultiClassClassification".
    # job_objective (dict): AutoMLJob objective, contains "AutoMLJobObjectiveType" (optional),
    # "MetricName" and "Value".
    # Raises (ValueError): raises ValueError if one of problem_type and job_objective is provided
    # while the other is None.
    .check_problem_type_and_job_objective = function(problem_type,
                                                     job_objective){
      if (!(islistempty(problem_type) && islistempty(job_objective)) &&
          (islistempty(problem_type) || islistempty(job_objective)))
        stop(
          "One of problem type and objective metric provided. ",
          "Either both of them should be provided or none of them should be provided.",
          call. = F
        )
    },

    # Set any values in the AutoMLJob that need to be set before creating request.
    # Args:
    #   job_name (str): The name of the AutoML job. If None, a job name will be
    # created from base_job_name or "sagemaker-auto-ml".
    .prepare_for_auto_ml_job = function(job_name=NULL){
      if (!is.null(job_name)){
        self$current_job_name = job_name
      } else {
        if (!is.null(self$base_job_name)){
          base_name = self$base_job_name
        } else {
          base_name = "automl"}
        # CreateAutoMLJob API validates that member length less than or equal to 32
        self$current_job_name = name_from_base(base_name, max_length=32)
        }
    if (is.null(self$output_path))
      self$output_path = sprintf("s3://%s/", self$sagemaker_session$default_bucket())
  },

  # Returns the inference keys supported by the container.
  # Args:
  #   container (dict): Dictionary representing container
  # default (object): The value to be returned if the container definition
  # has no marker environment variable
  # Returns:
  #   List of keys the container support or default
  # Raises:
  #   KeyError if the default is None and the container definition has
  # no marker environment variable SAGEMAKER_INFERENCE_SUPPORTED.
  .get_supported_inference_keys = function(container,
                                           default=NULL){
    tryCatch({
      return(trimws(split_str(container$Environment$SAGEMAKER_INFERENCE_SUPPORTED, ",")))},
      error= function(e){
        if (is.null(default))
          stop()
    })
    return(default)
  },

  # Checks if the pipeline supports the inference keys for the containers.
  # Given inference response keys and list of containers, determines whether
  # the keys are supported.
  # Args:
  #   inference_response_keys (list): List of keys for inference response content.
  # containers (list): list of inference container.
  # Raises:
  #   ValueError, if one or more keys in inference_response_keys are not supported
  # the inference pipeline.
  .check_inference_keys = function(inference_response_keys,
                                   containers){
    if (missing(inference_response_keys))
      return(invisible(NULL))
    tryCatch({
      supported_inference_keys = private$.get_supported_inference_keys(container=containers[[length(containers)]])},
      error = function(e) {
        stop(
          "The inference model does not support selection of inference content beyond ",
          "it's default content. Please retry without setting ",
          "inference_response_keys key word argument.", call. = F)
      }
    )

    bad_keys = list()
    for (key in inference_response_keys){
      if (!(key %in% supported_inference_keys))
        bad_keys = c(bad_keys, key)}

    if (!islistempty(bad_keys))
      stop(
        "Requested inference output keys [", paste(bad_keys, collapse = ", "), "] are unsupported. ",
        "The supported inference keys are [", paste(supported_inference_keys, collapse = ", "), "]",
        call. = F
      )
    }
  ),
  lock_objects = F
)

#' @title Accepts parameters that specify an S3 input for an auto ml job
#' @description Provides a method to turn those parameters into a dictionary.
#' @export
AutoMLInput = R6Class("AutoMLInput",
  public = list(

    #' @description Convert an S3 Uri or a list of S3 Uri to an AutoMLInput object.
    #' @param inputs (str, list[str]): a string or a list of string that points to (a)
    #'              S3 location(s) where input data is stored.
    #' @param  target_attribute_name (str): the target attribute name for regression
    #'              or classification.
    #' @param  compression (str): if training data is compressed, the compression type.
    #'              The default value is None.
    initialize = function(inputs,
                          target_attribute_name,
                          compression=NULL){
      self$inputs = inputs
      self$target_attribute_name = target_attribute_name
      self$compression = compression
    },

    #' @description Generates a request dictionary using the parameters provided to the class.
    to_request_list = function(){
      # Create the request dictionary.
      auto_ml_input = list()
      if (inherits(self$inputs, "character"))
        self$inputs = list(self$inputs)
      for(entry in self$inputs){
        input_entry = list(
          "DataSource"= list("S3DataSource" = list("S3DataType"= "S3Prefix", "S3Uri"= entry)),
          "TargetAttributeName" = self$target_attribute_name
        )
        if (!is.null(self$compression))
          input_entry$CompressionType = self$compression
        auto_ml_input = c(auto_ml_input, list(input_entry))}
      return(auto_ml_input)
    },

    #' @description format class
    format = function(){
      return(format_class(self))
    }
  ),
  lock_objects = F
)

#' @title AutoMLJob class
#' @description A class for interacting with CreateAutoMLJob API.
#' @export
AutoMLJob = R6Class("AutoMLJob",
  inherit = .Job,
  public = list(

    #' @description Initialize AutoMLJob class
    #' @param sagemaker_session (sagemaker.session.Session): A SageMaker Session
    #'              object, used for SageMaker interactions (default: None). If not
    #'              specified, the one originally associated with the ``AutoMLJob`` instance is used.
    #' @param job_name :
    #' @param inputs (str, list[str]): Parameters used when called
    #'              :meth:`~sagemaker.automl.AutoML.fit`.
    initialize = function(sagemaker_session,
                          job_name = NULL,
                          inputs= NULL){
      self$inputs = inputs
      self$job_name = job_name
      super$initialize(sagemaker_session=sagemaker_session, job_name=job_name)
    },

    #' @description Create a new Amazon SageMaker AutoML job from auto_ml.
    #' @param auto_ml (sagemaker.automl.AutoML): AutoML object
    #'              created by the user.
    #' @param inputs (str, list[str]): Parameters used when called
    #'              :meth:`~sagemaker.automl.AutoML.fit`.
    #' @return sagemaker.automl.AutoMLJob: Constructed object that captures
    #'              all information about the started AutoML job.
    start_new = function(auto_ml,
                         inputs){
      auto_ml_args = private$.load_config(inputs, auto_ml)
      auto_ml_args$job_name = auto_ml$current_job_name
      auto_ml_args$problem_type = auto_ml$problem_type
      auto_ml_args$job_objective = auto_ml$job_objective
      auto_ml_args$tags = auto_ml$tags

      do.call(auto_ml$sagemaker_session$auto_ml, auto_ml_args)

      cls = self$clone()
      cls$initialize(auto_ml$sagemaker_session, auto_ml$current_job_name, inputs)
      return(cls)
    },

    #' @description Prints out a response from the DescribeAutoMLJob API call.
    describe = function(){
      return(self$sagemaker_session$describe_auto_ml_job(self$job_name))
    },

    #' @description Wait for the AutoML job to finish.
    #' @param logs (bool): indicate whether to output logs.
    wait = function(logs=TRUE){
      if (logs)
        self$sagemaker_session$logs_for_auto_ml_job(self$job_name, wait=TRUE)
      else
        self$sagemaker_session$wait_for_auto_ml_job(self$job_name)
    },

    #' @description format class
    format = function(){
      return(format_class(self))
    }
  ),
  private = list(

    # Load job_config, input_config and output config from auto_ml and inputs.
    # Args:
    #   inputs (str): S3 Uri where the training data is stored, must start
    # with "s3://".
    # auto_ml (AutoML): an AutoML object that user initiated.
    # expand_role (str): The expanded role arn that allows for Sagemaker
    # executionts.
    # validate_uri (bool): indicate whether to validate the S3 uri.
    # Returns (dict): a config dictionary that contains input_config, output_config,
    # job_config and role information.
    .load_config = function(inputs,
                            auto_ml,
                            expand_role=TRUE,
                            validate_uri=TRUE){
      # JobConfig
      # InputDataConfig
      # OutputConfig

      if (inherits(inputs, "AutoMLInput"))
        input_config = inputs$to_request_list()
      else {
        input_config = private$.format_inputs_to_input_config(
          inputs, validate_uri, auto_ml$compression_type, auto_ml$target_attribute_name)
        }
      output_config = .Job$private_methods$.prepare_output_config(auto_ml$output_path, auto_ml$output_kms_key)

      role =if(expand_role) auto_ml$sagemaker_session$expand_role(auto_ml$role) else auto_ml$role

      stop_condition = private$.prepare_auto_ml_stop_condition(
        auto_ml$max_candidate,
        auto_ml$max_runtime_per_training_job_in_seconds,
        auto_ml$total_job_runtime_in_seconds
      )

      auto_ml_job_config = list(
        "CompletionCriteria" = stop_condition,
        "SecurityConfig" = list(
          "EnableInterContainerTrafficEncryption" = auto_ml$encrypt_inter_container_traffic
        )
      )

      auto_ml_job_config$SecurityConfig$VolumeKmsKeyId = auto_ml$volume_kms_key
      auto_ml_job_config$SecurityConfig$VpcConfig = auto_ml$vpc_config

      config = list(
        "input_config" = input_config,
        "output_config" = output_config,
        "auto_ml_job_config" = auto_ml_job_config,
        "role" = role,
        "generate_candidate_definitions_only" = auto_ml$generate_candidate_definitions_only
      )
      return(config)
    },

    # Convert inputs to AutoML InputDataConfig.
    # Args:
    #   inputs (str, list[str]): local path(s) or S3 uri(s) of input datasets.
    # validate_uri (bool): indicates whether it is needed to validate S3 uri.
    # compression (str): Compression type of the input data.
    # target_attribute_name (str): the target attribute name for classification
    # or regression.
    # Returns (dict): a dict of AutoML InputDataConfig
    .format_inputs_to_input_config = function(inputs,
                                              validate_uri=TRUE,
                                              compression=NULL,
                                              target_attribute_name=NULL){
      if (is.null(inputs))
        return(NULL)

      channels = list()
      if (inherits(inputs, "AutoMLInput")) {
        channels = c(channels, inputs$to_request_list())
      } else if (inherits(inputs, "character")) {
        channel = .Job$private_methods$.format_string_uri_input(
          inputs,
          validate_uri,
          compression=compression,
          target_attribute_name=target_attribute_name
          )$config
        channels = c(channels, list(channel))
      } else if (inherits(inputs, "list")) {
        for (input_entry in inputs) {
        channel = .Job$private_methods$.format_string_uri_input(
          input_entry,
          validate_uri,
          compression=compression,
          target_attribute_name=target_attribute_name
          )$config
        channels = c(channels, list(channel))
        }
      } else {
        msg = "Cannot format input %s. Expecting a string or a list of strings."
        stop(sprintf(msg,inputs), call. = F)
      }

      for (channel in channels){
        if (islistempty(channel$TargetAttributeName))
          stop("TargetAttributeName cannot be NULL", call. = F)
      }
      return(channels)
    },

    # Defines the CompletionCriteria of an AutoMLJob.
    # Args:
    #   max_candidates (int): the maximum number of candidates returned by an
    # AutoML job.
    # max_runtime_per_training_job_in_seconds (int): the maximum time of each
    # training job in seconds.
    # total_job_runtime_in_seconds (int): the total wait time of an AutoML job.
    # Returns (dict): an AutoML CompletionCriteria.
    .prepare_auto_ml_stop_condition = function(max_candidates,
                                               max_runtime_per_training_job_in_seconds = NULL,
                                              total_job_runtime_in_seconds = NULL){
      stopping_condition = list("MaxCandidates"= max_candidates)

      stopping_condition$MaxRuntimePerTrainingJobInSeconds = max_runtime_per_training_job_in_seconds
      stopping_condition$MaxAutoMLJobRuntimeInSeconds = total_job_runtime_in_seconds
      return(stopping_condition)
    }
  ),
  lock_objects = F
)
