# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/03bc33794a509cd364a9e658d2596c10ca9efa07/src/sagemaker/model_monitor/model_monitoring.py

#' @include utils.R
#' @include
#'
#' @import httr


#' @title ModelMonitor Class
#' @description Sets up Amazon SageMaker Monitoring Schedules and baseline suggestions. Use this class when
#'              you want to provide your own container image containing the code you'd like to run, in order
#'              to produce your own statistics and constraint validation files.
#'              For a more guided experience, consider using the DefaultModelMonitor class instead.
#' @export
ModelMonitor = R6Class("ModelMonitor",
                       public = list(

                         #' @description Initializes a ``Monitor`` instance. The Monitor handles baselining datasets and
                         #'              creating Amazon SageMaker Monitoring Schedules to monitor SageMaker endpoints.
                         #' @param role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
                         #' @param image_uri (str): The uri of the image to use for the jobs started by
                         #'              the Monitor.
                         #' @param instance_count (int): The number of instances to run
                         #'              the jobs with.
                         #' @param instance_type (str): Type of EC2 instance to use for
                         #'              the job, for example, 'ml.m5.xlarge'.
                         #' @param entrypoint ([str]): The entrypoint for the job.
                         #' @param volume_size_in_gb (int): Size in GB of the EBS volume
                         #'              to use for storing data during processing (default: 30).
                         #' @param volume_kms_key (str): A KMS key for the job's volume.
                         #' @param output_kms_key (str): The KMS key id for the job's outputs.
                         #' @param max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                         #'              time, Amazon SageMaker terminates the job regardless of its current status.
                         #'              Default: 3600
                         #' @param base_job_name (str): Prefix for the job name. If not specified,
                         #'              a default name is generated based on the training image name and
                         #'              current timestamp.
                         #' @param sagemaker_session (sagemaker.session.Session): Session object which
                         #'              manages interactions with Amazon SageMaker APIs and any other
                         #'              AWS services needed. If not specified, one is created using
                         #'              the default AWS configuration chain.
                         #' @param env (dict): Environment variables to be passed to the job.
                         #' @param tags ([dict]): List of tags to be passed to the job.
                         #' @param network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                         #'              object that configures network isolation, encryption of
                         #'              inter-container traffic, security group IDs, and subnets.
                         initialize = function(role =NULL,
                                               image_uri = NULL,
                                               instance_count=1,
                                               instance_type="ml.m5.xlarge",
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

                           self$baselining_jobs = list()
                           self$latest_baselining_job = NULL
                           self$arguments = NULL
                           self$latest_baselining_job_name = NULL
                           self$monitoring_schedule_name = NULL
                         },

                         #' @description Run a processing job meant to baseline your dataset.
                         #' @param baseline_inputs ([sagemaker.processing.ProcessingInput]): Input files for the processing
                         #'              job. These must be provided as ProcessingInput objects.
                         #' @param output (sagemaker.processing.ProcessingOutput): Destination of the constraint_violations
                         #'              and statistics json files.
                         #' @param arguments ([str]): A list of string arguments to be passed to a processing job.
                         #' @param wait (bool): Whether the call should wait until the job completes (default: True).
                         #' @param logs (bool): Whether to show the logs produced by the job.
                         #'              Only meaningful when wait is True (default: True).
                         #' @param job_name (str): Processing job name. If not specified, the processor generates
                         #'              a default job name, based on the image name and current timestamp.
                         run_baseline = function(baseline_inputs,
                                                 output,
                                                 arguments=NULL,
                                                 wait=TRUE,
                                                 logs=TRUE,
                                                 job_name=NULL){
                           self$latest_baselining_job_name = self$.generate_baselining_job_name(job_name=job_name)
                           self$arguments = arguments
                           normalized_baseline_inputs = self$.normalize_baseline_inputs(
                                        baseline_inputs=baseline_inputs)
                           normalized_output = self$.normalize_processing_output(output=output)

                           baselining_processor = Processor$new(
                             role=self$role,
                             image_uri=self$image_uri,
                             instance_count=self$instance_count,
                             instance_type=self$instance_type,
                             entrypoint=self$entrypoint,
                             volume_size_in_gb=self$volume_size_in_gb,
                             volume_kms_key=self$volume_kms_key,
                             output_kms_key=self$output_kms_key,
                             max_runtime_in_seconds=self$max_runtime_in_seconds,
                             base_job_name=self$base_job_name,
                             sagemaker_session=self$sagemaker_session,
                             env=self$env,
                             tags=self$tags,
                             network_config=self$network_config)

                           baselining_processor$run(
                             inputs=normalized_baseline_inputs,
                             outputs=list(normalized_output),
                             arguments=self$arguments,
                             wait=wait,
                             logs=logs,
                             job_name=self$latest_baselining_job_name)

                           self$latest_baselining_job = BaseliningJob$new()$from_processing_job(
                             processing_job=baselining_processor$latest_job)

                           self$baselining_jobs = c(self$baselining_jobs, self$latest_baselining_job)
                         }
                       ),

                       # Generate the job name before running a suggestion processing job.
                       # Args:
                       #   job_name (str): Name of the suggestion processing job to be created. If not
                       # specified, one is generated using the base name given to the
                       # constructor, if applicable.
                       # Returns:
                       #   str: The supplied or generated job name.
                       private = list(
                         .generate_baselining_job_name = function(job_name = NULL){
                           if (!is.null(job_name))
                             return(job_name)

                           if (!is.null(self$base_job_name))
                             base_name = self$base_job_name
                           else
                             base_name = "baseline-suggestion-job"

                           return(name_from_base(base=base_name))
                         },

                         # Ensure that all the ProcessingInput objects have names and S3 uris.
                         # Args:
                         #   baseline_inputs ([sagemaker.processing.ProcessingInput]): A list of ProcessingInput
                         # objects to be normalized.
                         # Returns:
                         #   [sagemaker.processing.ProcessingInput]: The list of normalized
                         # ProcessingInput objects.
                         .normalize_baseline_inputs = function(baseline_inputs=NULL){
                           normalized_inputs = list()
                           if (!is.null(baseline_inputs)){
                             # Iterate through the provided list of inputs.
                             for (#c(count, file_input)
                                  file_input
                                  in enumerate(baseline_inputs, 1)){
                               # TODO: Create ProcessingInput Class
                               if (!inherits(file_input, "ProcessingInput")){
                                 stop("Your inputs must be provided as ProcessingInput objects.", call. = F)}
                                # Generate a name for the ProcessingInput if it doesn't have one.
                                if (is.null(file_input$input_name)){
                                   file_input$input_name = sprintf("input-%s",count)}
                                # If the source is a local path, upload it to S3
                                # and save the S3 uri in the ProcessingInput source.
                                parse_result = parse_url(file_input$source)
                                if (parse_result$scheme != "s3"){
                                  s3_uri = file.path(
                                    "s3://",
                                     self$sagemaker_session$default_bucket(),
                                     self$latest_baselining_job_name,
                                     file_input$input_name)

                                 S3Uploader$new()$upload(
                                   local_path=file_input.source,
                                   desired_s3_uri=s3_uri,
                                   session=self.sagemaker_session)

                                 file_input.source = s3_uri
                                 normalized_inputs.append(file_input)}
                                }
                             }

                             return(normalized_inputs)
                             }



                       ),
                       lock_objects = F
)
