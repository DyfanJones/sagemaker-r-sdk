# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/processing.py

#' @include utils.R
#' @include s3.R
#' @include session.R

#' @importFrom urltools url_parse
#' @import R6

#' @title Processor Class
#' @family Processor
#' @description Handles Amazon SageMaker Processing tasks.
Processor = R6Class("Processor",
  public = list(
    #' @field role
    #' An AWS IAM role name or ARN
    role = NULL,

    #' @field image_uri
    #' The URI of the Docker image to use
    image_uri = NULL,

    #' @field instance_count
    #' The number of instances to run
    instance_count = NULL,

    #' @field instance_type
    #' The type of EC2 instance to use
    instance_type = NULL,

    #' @field entrypoint
    #' The entrypoint for the processing job
    entrypoint = NULL,

    #' @field volume_size_in_gb
    #' Size in GB of the EBS volume
    volume_size_in_gb = NULL,

    #' @field volume_kms_key
    #' A KMS key for the processing
    volume_kms_key = NULL,

    #' @field output_kms_key
    #' The KMS key ID for processing job outputs
    output_kms_key = NULL,

    #' @field max_runtime_in_seconds
    #' Timeout in seconds
    max_runtime_in_seconds = NULL,

    #' @field base_job_name
    #' Prefix for processing job name
    base_job_name = NULL,

    #' @field sagemaker_session
    #' Session object which manages interactions with Amazon SageMaker
    sagemaker_session = NULL,

    #' @field env
    #' Environment variables
    env = NULL,

    #' @field tags
    #' List of tags to be passed
    tags = NULL,

    #' @field network_config
    #' A :class:`~sagemaker.network.NetworkConfig`
    network_config = NULL,

    #' @field jobs
    #' Jobs ran /running
    jobs = NULL,

    #' @field latest_job
    #' Previously ran jobs
    latest_job = NULL,

    #' @field .current_job_name
    #' Current job
    .current_job_name = NULL,

    #' @field arguments
    #' extra agruments
    arguments = NULL,
    #' @description Initializes a ``Processor`` instance. The ``Processor`` handles Amazon
    #'              SageMaker Processing tasks.
    #' @param role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
    #'              uses this role to access AWS resources, such as
    #'              data stored in Amazon S3.
    #' @param image_uri (str): The URI of the Docker image to use for the
    #'              processing jobs.
    #' @param instance_count (int): The number of instances to run
    #'              a processing job with.
    #' @param instance_type (str): The type of EC2 instance to use for
    #'              processing, for example, 'ml.c4.xlarge'.
    #' @param entrypoint (list[str]): The entrypoint for the processing job (default: NULL).
    #'              This is in the form of a list of strings that make a command.
    #' @param volume_size_in_gb (int): Size in GB of the EBS volume
    #'              to use for storing data during processing (default: 30).
    #' @param volume_kms_key (str): A KMS key for the processing
    #'              volume (default: NULL).
    #' @param output_kms_key (str): The KMS key ID for processing job outputs (default: NULL).
    #' @param max_runtime_in_seconds (int): Timeout in seconds (default: NULL).
    #'              After this amount of time, Amazon SageMaker terminates the job,
    #'              regardless of its current status. If `max_runtime_in_seconds` is not
    #'              specified, the default value is 24 hours.
    #' @param base_job_name (str): Prefix for processing job name. If not specified,
    #'              the processor generates a default job name, based on the
    #'              processing image name and current timestamp.
    #' @param sagemaker_session (:class:`~sagemaker.session.Session`):
    #'              Session object which manages interactions with Amazon SageMaker and
    #'              any other AWS services needed. If not specified, the processor creates
    #'              one using the default AWS configuration chain.
    #' @param env (dict[str, str]): Environment variables to be passed to
    #'              the processing jobs (default: NULL).
    #' @param tags (list[dict]): List of tags to be passed to the processing job
    #'              (default: NULL). For more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @param network_config (:class:`~sagemaker.network.NetworkConfig`):
    #'              A :class:`~sagemaker.network.NetworkConfig`
    #'              object that configures network isolation, encryption of
    #'              inter-container traffic, security group IDs, and subnets.
    initialize = function(role,
                          image_uri,
                          instance_count,
                          instance_type,
                          entrypoint=NULL,
                          volume_size_in_gb=30,
                          volume_kms_key=NULL,
                          output_kms_key=NULL,
                          max_runtime_in_seconds=NULL,
                          base_job_name=NULL,
                          sagemaker_session=NULL,
                          env=NULL,
                          tags=NULL,
                          network_config=NULL){
      self$role = role
      self$image_uri = image_uri
      self$instance_count = instance_count
      self$instance_type = instance_type
      self$entrypoint = entrypoint
      self$volume_size_in_gb = volume_size_in_gb
      self$volume_kms_key = volume_kms_key
      self$output_kms_key = output_kms_key
      self$max_runtime_in_seconds = max_runtime_in_seconds
      self$base_job_name = base_job_name
      self$sagemaker_session = sagemaker_session %||% Session()
      self$env = env
      self$tags = tags
      self$network_config = network_config

      self$jobs = list()
      self$latest_job = NULL
      self$.current_job_name = NULL
      self$arguments = NULL
    },

    #' @description Runs a processing job.
    #' @param inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
    #'              the processing job. These must be provided as
    #'              :class:`~sagemaker.processing.ProcessingInput` objects (default: NULL).
    #' @param outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
    #'              the processing job. These can be specified as either path strings or
    #'              :class:`~sagemaker.processing.ProcessingOutput` objects (default: NULL).
    #' @param arguments (list[str]): A list of string arguments to be passed to a
    #'              processing job (default: NULL).
    #' @param wait (bool): Whether the call should wait until the job completes (default: True).
    #' @param logs (bool): Whether to show the logs produced by the job.
    #'              Only meaningful when ``wait`` is True (default: True).
    #' @param job_name (str): Processing job name. If not specified, the processor generates
    #'              a default job name, based on the base job name and current timestamp.
    #' @param experiment_config (dict[str, str]): Experiment management configuration.
    #'              Dictionary contains three optional keys:
    #'              'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
    run = function(inputs=NULL,
                   outputs=NULL,
                   arguments=NULL,
                   wait=TRUE,
                   logs=TRUE,
                   job_name=NULL,
                   experiment_config=NULL){
      if (logs && !wait){
        stop("Logs can only be shown if wait is set to True. Please either set wait to True or set logs to False.",
             call. = FALSE)}

      self$.current_job_name = private$.generate_current_job_name(job_name=job_name)

      normalized_inputs = private$.normalize_inputs(inputs)
      normalized_outputs = private$.normalize_outputs(outputs)
      self.arguments = arguments

      self$latest_job = ProcessingJob$new()$start_new(
        processor=self,
        inputs=normalized_inputs,
        outputs=normalized_outputs,
        experiment_config=experiment_config)

      self$jobs = c(self$jobs, self$latest_job)
      if (wait) self$latest_job$wait(logs=logs)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      cat("<Processor>")
      invisible(self)
    }
  ),
  private = list(
    # Generates the job name before running a processing job.
    # Args:
    #   job_name (str): Name of the processing job to be created. If not
    # specified, one is generated, using the base name given to the
    # constructor if applicable.
    # Returns:
    #   str: The supplied or generated job name.
    .generate_current_job_name = function(job_name=NULL){
      if (!is.null(job_name))
        return(job_name)
      # Honor supplied base_job_name or generate it.
      if (self$base_job_name)
        base_name = self$base_job_name
      else
        base_name = base_name_from_image(self$image_uri)
      return(name_from_base(base_name))
    },

    # Ensures that all the ``ProcessingInput`` objects have names and S3 URIs.
    # Args:
    #   inputs (list[sagemaker.processing.ProcessingInput]): A list of ``ProcessingInput``
    # objects to be normalized (default: NULL). If not specified,
    # an empty list is returned.
    # Returns:
    #   list[sagemaker.processing.ProcessingInput]: The list of normalized
    # ``ProcessingInput`` objects.
    .normalize_inputs = function(inputs=NULL){
      # Initialize a list of normalized ProcessingInput objects.
      normalized_inputs = list()
      if (!is.null(inputs)){
        # Iterate through the provided list of inputs.
        for (count in 1:length(inputs)){
          if (!inherits(inputs[[count]], "ProcessingInput")){
            stop("Your inputs must be provided as ProcessingInput objects.", call. = F)}
          # Generate a name for the ProcessingInput if it doesn't have one.
          if (islistempty(inputs[[count]]$input_name)){
            inputs[[count]]$input_name = sprintf("input-%s",count)}
          # If the source is a local path, upload it to S3
          # and save the S3 uri in the ProcessingInput source.
          parse_result = parse_url(inputs[[count]]$source)
          if (parse_result$scheme != "s3"){
            desired_s3_uri = sprintf("s3://%s/%s/input/%s",
              self.sagemaker_session$default_bucket(),
              self$.current_job_name,
              inputs[[count]]$input_name)
            s3_uri = S3Uploader$new()$upload(
              local_path=inputs[[count]]$source,
              desired_s3_uri=desired_s3_uri,
              session=self$sagemaker_session)
            inputs[[count]]$source = s3_uri}
          normalized_inputs = c(normalized_inputs, inputs[[count]])}
      }
      return(normalized_inputs)
    },

    # Ensures that all the outputs are ``ProcessingOutput`` objects with
    # names and S3 URIs.
    # Args:
    #   outputs (list[sagemaker.processing.ProcessingOutput]): A list
    # of outputs to be normalized (default: NULL). Can be either strings or
    # ``ProcessingOutput`` objects. If not specified,
    # an empty list is returned.
    # Returns:
    #   list[sagemaker.processing.ProcessingOutput]: The list of normalized
    # ``ProcessingOutput`` objects.
    .normalize_outputs = function(output = NULL){
      # Initialize a list of normalized ProcessingOutput objects.
      normalized_outputs = list()
      if (!is.null(outputs)){
        # Iterate through the provided list of outputs.
        for(count in 1:length(outputs)){
          if (!inherits(outputs[[count]], "ProcessingOutput")){
            stop("Your outputs must be provided as ProcessingOutput objects.", call.=FALSE)}
          # Generate a name for the ProcessingOutput if it doesn't have one.
          if (islistempty(outputs[[count]]$output_name)){
            outputs[[count]]$output_name = sprintf("output-%s",count)}
            # If the output's destination is not an s3_uri, create one.
          parse_result = parse_url(outputs[[count]]$destination)
          if (parse_result$scheme != "s3"){
            s3_uri = sprintf("s3://%s/%s/output/%s",
              self$sagemaker_session$default_bucket(),
              self$.current_job_name,
              outputs[[count]]$output_name,
            )
            outputs[[count]]$destination = s3_uri}
          normalized_outputs = c(normalized_outputs, outputs[[count]])}
        }
      return(normalized_outputs)
    }
  )
)

#' @title Script Processor class
#' @family Processor
#' @description Handles Amazon SageMaker processing tasks for jobs using a machine learning framework.
#' @export
ScriptProcessor = R6Class("ScriptProcessor",
  inherit = Processor,
  public = list(

    #' @description Initializes a ``ScriptProcessor`` instance. The ``ScriptProcessor``
    #'              handles Amazon SageMaker Processing tasks for jobs using a machine learning framework,
    #'              which allows for providing a script to be run as part of the Processing Job.
    #' @param role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
    #'              uses this role to access AWS resources, such as
    #'              data stored in Amazon S3.
    #' @param image_uri (str): The URI of the Docker image to use for the
    #'              processing jobs.
    #' @param command ([str]): The command to run, along with any command-line flags.
    #'              Example: ["python3", "-v"].
    #' @param instance_count (int): The number of instances to run
    #'              a processing job with.
    #' @param instance_type (str): The type of EC2 instance to use for
    #'              processing, for example, 'ml.c4.xlarge'.
    #' @param volume_size_in_gb (int): Size in GB of the EBS volume
    #'              to use for storing data during processing (default: 30).
    #' @param volume_kms_key (str): A KMS key for the processing
    #'              volume (default: NULL).
    #' @param output_kms_key (str): The KMS key ID for processing job outputs (default: NULL).
    #' @param max_runtime_in_seconds (int): Timeout in seconds (default: NULL).
    #'              After this amount of time, Amazon SageMaker terminates the job,
    #'              regardless of its current status. If `max_runtime_in_seconds` is not
    #'              specified, the default value is 24 hours.
    #' @param base_job_name (str): Prefix for processing name. If not specified,
    #'              the processor generates a default job name, based on the
    #'              processing image name and current timestamp.
    #' @param sagemaker_session (:class:`~sagemaker.session.Session`):
    #'              Session object which manages interactions with Amazon SageMaker and
    #'              any other AWS services needed. If not specified, the processor creates
    #'              one using the default AWS configuration chain.
    #' @param env (dict[str, str]): Environment variables to be passed to
    #'              the processing jobs (default: NULL).
    #' @param tags (list[dict]): List of tags to be passed to the processing job
    #'              (default: NULL). For more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @param network_config (:class:`~sagemaker.network.NetworkConfig`):
    #'              A :class:`~sagemaker.network.NetworkConfig`
    #'              object that configures network isolation, encryption of
    #'              inter-container traffic, security group IDs, and subnets.
    initialize = function(role,
                          image_uri,
                          command,
                          instance_count,
                          instance_type,
                          volume_size_in_gb=30,
                          volume_kms_key=NULL,
                          output_kms_key=NULL,
                          max_runtime_in_seconds=NULL,
                          base_job_name=NULL,
                          sagemaker_session=NULL,
                          env=NULL,
                          tags=NULL,
                          network_config=NULL){
      self$.CODE_CONTAINER_BASE_PATH = "/opt/ml/processing/input/"
      self$.CODE_CONTAINER_INPUT_NAME = "code"
      self.command = command

      super$initialize(role=role,
                       image_uri=image_uri,
                       instance_count=instance_count,
                       instance_type=instance_type,
                       volume_size_in_gb=volume_size_in_gb,
                       volume_kms_key=volume_kms_key,
                       output_kms_key=output_kms_key,
                       max_runtime_in_seconds=max_runtime_in_seconds,
                       base_job_name=base_job_name,
                       sagemaker_session=sagemaker_session,
                       env=env,
                       tags=tags,
                       network_config=network_config)
    },

    #' @description Runs a processing job.
    #' @param code (str): This can be an S3 URI or a local path to
    #'              a file with the framework script to run.
    #' @param inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
    #'              the processing job. These must be provided as
    #'              :class:`~sagemaker.processing.ProcessingInput` objects (default: NULL).
    #' @param outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
    #'              the processing job. These can be specified as either path strings or
    #'              :class:`~sagemaker.processing.ProcessingOutput` objects (default: NULL).
    #' @param arguments (list[str]): A list of string arguments to be passed to a
    #'              processing job (default: NULL).
    #' @param wait (bool): Whether the call should wait until the job completes (default: True).
    #' @param logs (bool): Whether to show the logs produced by the job.
    #'              Only meaningful when wait is True (default: True).
    #' @param job_name (str): Processing job name. If not specified, the processor generates
    #'              a default job name, based on the base job name and current timestamp.
    #' @param experiment_config (dict[str, str]): Experiment management configuration.
    #'              Dictionary contains three optional keys:
    #'              'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
    run = function(code,
                   inputs=NULL,
                   outputs=NULL,
                   arguments=NULL,
                   wait=TRUE,
                   logs=TRUE,
                   job_name=NULL,
                   experiment_config=NULL){
      self$.current_job_name = self$.generate_current_job_name(job_name=job_name)

      user_code_s3_uri = private$.handle_user_code_url(code)
      user_script_name = private$.get_user_code_name(code)

      inputs_with_code = private$.convert_code_and_add_to_inputs(inputs, user_code_s3_uri)

      private$.set_entrypoint(self$command, user_script_name)

      normalized_inputs = private$.normalize_inputs(inputs_with_code)
      normalized_outputs = private$.normalize_outputs(outputs)
      self.arguments = arguments

      self.latest_job = ProcessingJob$new()$start_new(
        processor=self,
        inputs=normalized_inputs,
        outputs=normalized_outputs,
        experiment_config=experiment_config)

      self$jobs = c(self$jobs, self$latest_job)
      if (wait)
        self$latest_job$wait(logs=logs)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      cat("<ScriptProcessor>")
      invisible(self)
    }
  ),
  private = list(
    # Gets the basename of the user's code from the URL the customer provided.
    # Args:
    #     code (str): A URL to the user's code.
    # Returns:
    #   str: The basename of the user's code.
    .get_user_code_name = function(code){
      code_url = url_parse(code)
      return (file.path(code_url$path))
    },

    # Gets the S3 URL containing the user's code.
    #    Inspects the scheme the customer passed in ("s3://" for code in S3, "file://" or nothing
    #    for absolute or local file paths. Uploads the code to S3 if the code is a local file.
    # Args:
    #     code (str): A URL to the customer's code.
    # Returns:
    #   str: The S3 URL to the customer's code.
    .handle_user_code_url = function(code){
      code_url = url_parse(code)
      if (code_url$scheme == "s3") {
        user_code_s3_uri = code
      } else if(code_url$scheme == "" || code_url$scheme == "file") {
          # Validate that the file exists locally and is not a directory.
          if (!file.exists(code)){
            stop(sprintf("code {} wasn't found. Please make sure that the file exists.",
                         code), call. = F)}
        if (!file_test("-f", code)){
          stop(sprintf("code %s must be a file, not a directory. Please pass a path to a file.",
                       code), call. = F)}
        user_code_s3_uri = private$.upload_code(code)
        } else {
          stop(sprintf("code %s url scheme %s is not recognized. Please pass a file path or S3 url",
                       code, code_url$scheme),
               call. = F)}
        return(user_code_s3_uri)
    },

    # Uploads a code file or directory specified as a string
    # and returns the S3 URI.
    # Args:
    #   code (str): A file or directory to be uploaded to S3.
    # Returns:
    #   str: The S3 URI of the uploaded file or directory.
    .upload_code = function(code){
      desired_s3_uri = sprintf("s3://%s/%s/input/%s",
        self$sagemaker_session$default_bucket(),
        self$.current_job_name,
        self$.CODE_CONTAINER_INPUT_NAME)
      return(S3Uploader$new()$upload(
        local_path=code, desired_s3_uri=desired_s3_uri, session=self$sagemaker_session))
    },

    # Creates a ``ProcessingInput`` object from an S3 URI and adds it to the list of inputs.
    # Args:
    #   inputs (list[sagemaker.processing.ProcessingInput]):
    #   List of ``ProcessingInput`` objects.
    # s3_uri (str): S3 URI of the input to be added to inputs.
    # Returns:
    #   list[sagemaker.processing.ProcessingInput]: A new list of ``ProcessingInput`` objects,
    # with the ``ProcessingInput`` object created from ``s3_uri`` appended to the list.
    .convert_code_and_add_to_inputs = function(inputs, s3_uri){
      code_file_input = ProcessingInput$new(
        source=s3_uri,
        destination=sprintf("%s%s",
          self$.CODE_CONTAINER_BASE_PATH, self$.CODE_CONTAINER_INPUT_NAME),
        input_name=self$.CODE_CONTAINER_INPUT_NAME)

      output = list(inputs %||% list(), code_file_input)
      return(output)
    },

    # Sets the entrypoint based on the user's script and corresponding executable.
    # Args:
    #     user_script_name (str): A filename with an extension.
    .set_entrypoint = function(command, user_script_name){
      user_script_location = sprintf("%s%s/%s",
        self$.CODE_CONTAINER_BASE_PATH, self$.CODE_CONTAINER_INPUT_NAME, user_script_name)
      self$entrypoint = list(command, user_script_location)
    }
  ),
  lock_objects = F
)

#' @title ProccesingJob Class
#' @family Processor
#' @description Provides functionality to start, describe, and stop processing jobs.
#' @export
ProcessingJob = R6Class("ProcessingJob",
  inherit = .Job,
  public = list(
    #' @field inputs
    #' A list of :class:`~sagemaker.processing.ProcessingInput` objects.
    inputs = NULL,

    #' @field outputs
    #' A list of :class:`~sagemaker.processing.ProcessingOutput` objects.
    outputs = NULL,

    #' @field output_kms_key
    #' The output KMS key associated with the job
    output_kms_key = NULL,

    #' @description Initializes a Processing job.
    #' @param sagemaker_session (:class:`~sagemaker.session.Session`):
    #'              Session object which manages interactions with Amazon SageMaker and
    #'              any other AWS services needed. If not specified, the processor creates
    #'              one using the default AWS configuration chain.
    #' @param job_name (str): Name of the Processing job.
    #' @param inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): A list of
    #'              :class:`~sagemaker.processing.ProcessingInput` objects.
    #' @param outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): A list of
    #'              :class:`~sagemaker.processing.ProcessingOutput` objects.
    #' @param output_kms_key (str): The output KMS key associated with the job (default: None).
    initialize = function(sagemaker_session,
                          job_name,
                          inputs,
                          outputs,
                          output_kms_key=NULL){
      self$inputs = inputs
      self$outputs = outputs
      self$output_kms_key = output_kms_key

      super$initialize(sagemaker = sagemaker_session, job_name = job_name)
    },

    #' @description Starts a new processing job using the provided inputs and outputs.
    #' @param processor (:class:`~sagemaker.processing.Processor`): The ``Processor`` instance
    #'              that started the job.
    #' @param inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): A list of
    #'              :class:`~sagemaker.processing.ProcessingInput` objects.
    #' @param outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): A list of
    #'              :class:`~sagemaker.processing.ProcessingOutput` objects.
    #' @param experiment_config (dict[str, str]): Experiment management configuration.
    #'              Dictionary contains three optional keys:
    #'              'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
    #' @return :class:`~sagemaker.processing.ProcessingJob`: The instance of ``ProcessingJob`` created
    #'              using the ``Processor``.
    start_new = function(processor,
                         inputs,
                         outputs,
                         experiment_config){
      # Initialize an empty dictionary for arguments to be passed to sagemaker_session.process.
      process_request_args = list()

      # Add arguments to the dictionary.
      process_request_args$inputs = lapply(inputs, function(input) input$.to_request_list())

      process_request_args$output_config = list("Outputs"= lapply(outputs, function(output) output._to_request_dict()))
      if (!islistempty(processor$output_kms_key))
        process_request_args$output_config$KmsKeyId = processor$output_kms_key

      process_request_args$experiment_config = experiment_config
      process_request_args$job_name = processor$.current_job_name

      process_request_args$resources = list(
        "ClusterConfig" = list(
          "InstanceType"= processor$instance_type,
          "InstanceCount"= processor$instance_count,
          "VolumeSizeInGB"= processor$volume_size_in_gb,
        )
      )

      if(!islistempty(processor$volumene_kms_key)){
        process_request_args$resources$ClusterConfig$VolumeKmsKeyId = processor$volume_kms_key}

      if(!islistempty(processor$max_runtime_in_seconds)){
        process_request_args$stopping_condition = list(
          "MaxRuntimeInSeconds"= processor$max_runtime_in_seconds)}

      process_request_args$app_specification = list("ImageUri"= processor$image_uri)
      if (!islistempty(processor$arguments))
        process_request_args$app_specification$ContainerArguments = processor$arguments
      if (!islistempty(processor$entrypoint))
        process_request_args$app_specification$ContainerEntrypoint = processor$entrypoint

      process_request_args$environment = processor$env

      # TODO: create NetworkConfig Class
      if (!islistempty(processor$network_config))
        process_request_args$network_config = processor$network_config$to_request_list()

      process_request_args$role_arn = processor$sagemaker_session$expand_role(processor$role)

      process_request_args$tags = processor$tags

      # Print the job name and the user's inputs and outputs as lists of dictionaries.
      writeLines("")
      writeLines(sprintf("Job Name: %s", process_request_args$job_name))
      writeLines("Inputs: ")
      print(process_request_args$inputs)
      writeLines("Outputs: ")
      print(process_request_args$output_config$Outputs)

      # Call sagemaker_session.process using the arguments dictionary.
      do.call(processor$sagemaker_session$process, process_request_args)

      cls = self$clone()

      # update class super
      cls$sagemaker_session =  processor$sagemaker_session
      cls$job_name = processor$.current_job_name

      # update class
      cls$inputs = inputs
      cls$outputs = outputs
      cls$output_kms_key = processor$output_kms_key

      return(cls)
    },

    #' @description Initializes a ``ProcessingJob`` from a processing job name.
    #' @param sagemaker_session (:class:`~sagemaker.session.Session`):
    #'              Session object which manages interactions with Amazon SageMaker and
    #'              any other AWS services needed. If not specified, the processor creates
    #'              one using the default AWS configuration chain.
    #' @param processing_job_name (str): Name of the processing job.
    #' @return :class:`~sagemaker.processing.ProcessingJob`: The instance of ``ProcessingJob`` created
    #'              from the job name.
    from_processing_name = function(sagemaker_session,
                                    processing_job_name){
      job_desc = sagemaker_session$describe_processing_job(job_name=processing_job_name)

      inputs = NULL
      if (!islistempty(job_desc$ProcessingInputs)){
        inputs = lapply(job_desc$ProcessingInputs, function(processing_input) ProcessingInput$new(
                                    source=processing_input$S3Input$S3Uri,
                                    destination=processing_input$S3Input$LocalPath,
                                    input_name=processing_input$InputName,
                                    s3_data_type=processing_input$S3Input$S3DataType,
                                    s3_input_mode=processing_input$S3Input$S3InputMode,
                                    s3_data_distribution_type=processing_input$S3Input$S3DataDistributionType,
                                    s3_compression_type=processing_input$S3Input$S3CompressionType))
      }

      outputs = NULL
      if (!islistempty(job_desc$ProcessingOutputConfig) && !islistempty(job_desc$ProcessingOutputConfig$Outputs)){
        outputs = lapply(job_desc$ProcessingOutputConfig$Outputs, function(processing_output) ProcessingOutput$new(
                                    source=processing_output$S3Output$LocalPath,
                                    destination=processing_outputS3Output$S3Uri,
                                    output_name=processing_output$OutputName))
        }

      output_kms_key = NULL
      if (!islistempty(job_desc$ProcessingOutputConfig))
        output_kms_key = job_desc$ProcessingOutputConfig$KmsKeyId

      cls = self$clone()

      # update class super
      cls$sagemaker_session = sagemaker_session
      cls$job_name = processing_job_name

      # update class
      cls$inputs = inputs
      cls$outputs = outputs
      cls$output_kms_key = output_kms_key

      return(cls)
    },

    #' @description Initializes a ``ProcessingJob`` from a Processing ARN.
    #' @param sagemaker_session (:class:`~sagemaker.session.Session`):
    #'              Session object which manages interactions with Amazon SageMaker and
    #'              any other AWS services needed. If not specified, the processor creates
    #'              one using the default AWS configuration chain.
    #' @param processing_job_arn (str): ARN of the processing job.
    #' @return :class:`~sagemaker.processing.ProcessingJob`: The instance of ``ProcessingJob`` created
    #'              from the processing job's ARN.
    from_processing_arn = function(sagemaker_session,
                                   processing_job_arn){
      processing_job_name = split_str(processing_job_arn, ":")[6]
      processing_job_name = substring(processing_job_name, nchar("processing-job/") + 1, nchar(processing_job_name)) # This is necessary while the API only vends an arn.
      return(self$from_processing_name(sagemaker_session=sagemaker_session, processing_job_name=processing_job_name))
    },

    #' @description Waits for the processing job to complete.
    #' @param logs (bool): Whether to show the logs produced by the job (default: True).
    wait = function(logs = TRUE){
      if (logs)
        self$sagemaker_session$logs_for_processing_job(self$job_name, wait=TRUE)
      else
        self$sagemaker_session$wait_for_processing_job(self$job_name)
    },

    #' @description Prints out a response from the DescribeProcessingJob API call.
    describe = function(){
      return(self$sagemaker_session$describe_processing_job(self$job_name))
    },

    #' @description the processing job.
    stop = function(){
      return(self$sagemaker_session$stop_processing_job(self$name))
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      cat("<ProcessingJob>")
      invisible(self)
    }
  ),
  private = list(
    # Used for Local Mode. Not yet implemented.
    # Args:
    #   input_url (str): input URL
    .is_local_channel = function(input_url){
      stop("This method is not yet implemented.", call. = F)
    }
  )
)

#' @title ProcessingInput Class
#' @family Processor
#' @description Accepts parameters that specify an Amazon S3 input for a processing job and
#'              provides a method to turn those parameters into a dictionary.
#' @export
ProcessingInput = R6Class("ProcessingInput",
  public = list(

    #' @description Initializes a ``ProcessingInput`` instance. ``ProcessingInput`` accepts parameters
    #'              that specify an Amazon S3 input for a processing job and provides a method
    #'              to turn those parameters into a dictionary.
    #' @param source (str): The source for the input. If a local path is provided, it will
    #'              automatically be uploaded to S3 under:
    #'              "s3://<default-bucket-name>/<job-name>/input/<input-name>".
    #' @param destination (str): The destination of the input.
    #' @param input_name (str): The name for the input. If a name
    #'              is not provided, one will be generated (eg. "input-1").
    #' @param s3_data_type (str): Valid options are "ManifestFile" or "S3Prefix".
    #' @param s3_input_mode (str): Valid options are "Pipe" or "File".
    #' @param s3_data_distribution_type (str): Valid options are "FullyReplicated"
    #'              or "ShardedByS3Key".
    #' @param s3_compression_type (str): Valid options are "None" or "Gzip".
    initialize = function(source,
                          destination,
                          input_name=NULL,
                          s3_data_type=c("S3Prefix", "ManifestFile"),
                          s3_input_mode=c("File", "Pipe"),
                          s3_data_distribution_type=c("FullyReplicated", "ShardedByS3Key"),
                          s3_compression_type=c("None", "Gzip")){
      self$source = source
      self$destination = destination
      self$input_name = input_name
      self$s3_data_type = match.arg(s3_data_type)
      self$s3_input_mode = match.arg(s3_input_mode)
      self$s3_data_distribution_type = match.arg(s3_data_distribution_type)
      self$s3_compression_type = match.arg(s3_compression_type)
    },

    #' @description Generates a request dictionary using the parameters provided to the class.
    to_request_list = function(){
      # Create the request dictionary.
      s3_input_request = list(
        "InputName"= self$input_name,
        "S3Input"= list(
          "S3Uri"= self$source,
          "LocalPath"= self$destination,
          "S3DataType"= self$s3_data_type,
          "S3InputMode"= self$s3_input_mode,
          "S3DataDistributionType"= self$s3_data_distribution_type
        )
      )


      # Check the compression type, then add it to the dictionary.
      if (self$s3_compression_type == "Gzip" && self$s3_input_mode != "Pipe")
        stop("Data can only be gzipped when the input mode is Pipe.", call. = F)
      if (self$s3_compression_type != "None")
        s3_input_request$S3Input$S3CompressionType = self$s3_compression_type

        # Return the request dictionary.
        return (s3_input_request)
    },

    #' @description
    #' Printer.
    #' @param ... (ignored).
    print = function(...){
      cat("<ProcessingInput>")
      invisible(self)
    }
  ),
  lock_objects = F
)

#' @title ProcessingOutput Class
#' @family Processor
#' @description Accepts parameters that specify an Amazon S3 output for a processing job and provides
#'              a method to turn those parameters into a dictionary.
ProcessingOutput = R6Class("ProcessingOutput",
  public = list(

   #' @description Initializes a ``ProcessingOutput`` instance. ``ProcessingOutput`` accepts parameters that
   #'              specify an Amazon S3 output for a processing job and provides a method to turn
   #'              those parameters into a dictionary.
   #' @param source (str): The source for the output.
   #' @param destination (str): The destination of the output. If a destination
   #'              is not provided, one will be generated:
   #'              "s3://<default-bucket-name>/<job-name>/output/<output-name>".
   #' @param output_name (str): The name of the output. If a name
   #'              is not provided, one will be generated (eg. "output-1").
   #' @param s3_upload_mode (str): Valid options are "EndOfJob" or "Continuous".
   initialize = function(source,
                         destination=NULL,
                         output_name=NULL,
                         s3_upload_mode=c("EndOfJob", "Continuous")){
     self$source = source
     self$destination = destination
     self$output_name = output_name
     self$s3_upload_mode = match.arg(s3_upload_mode)
   },

   #' @description Generates a request dictionary using the parameters provided to the class.
   to_request_list = function(){
     # Create the request dictionary.
     s3_output_request = list(
       "OutputName"= self$output_name,
       "S3Output"= list(
         "S3Uri"= self$destination,
         "LocalPath"= self.source,
         "S3UploadMode"= self.s3_upload_mode)
     )

     # Return the request dictionary.
     return(s3_output_request)
   },

   #' @description
   #' Printer.
   #' @param ... (ignored).
   print = function(...){
     cat("<ProcessingOutput>")
     invisible(self)
   }
  ),
  lock_objects = F
)

