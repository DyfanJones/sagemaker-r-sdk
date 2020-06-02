#' @import paws
#' @import jsonlite
#' @import R6
#' @import logger
#' @import utils

EstimatorBase = R6Class("EstimatorBase",
                        public = list(
                          initialize = function(role,
                                                train_instance_count,
                                                train_instance_type,
                                                train_volume_size = 30,
                                                train_volume_kms_key = NULL,
                                                train_max_run = 24 * 60 * 60,
                                                input_mode = "File",
                                                output_path = NULL,
                                                output_kms_key = NULL,
                                                base_job_name = NULL,
                                                sagemaker_session = NULL,
                                                tags = NULL,
                                                subnets = NULL,
                                                security_group_ids = NULL,
                                                model_uri = NULL,
                                                model_channel_name = "model",
                                                metric_definitions = NULL,
                                                encrypt_inter_container_traffic = FALSE,
                                                train_use_spot_instances =FALSE,
                                                train_max_wait = NULL,
                                                checkpoint_s3_uri = NULL,
                                                checkpoint_local_path = NULL,
                                                rules = NULL,
                                                debugger_hook_config = NULL,
                                                tensorboard_output_config = NULL,
                                                enable_sagemaker_metrics = NULL,
                                                enable_network_isolation = FALSE) {
                            self$role = role
                            self$train_instance_count = train_instance_count
                            self$train_instance_type = train_instance_type
                            self$train_volume_size = train_volume_size
                            self$train_volume_kms_key = train_volume_kms_key
                            self$train_max_run = train_max_run
                            self$input_mode = input_mode
                            self$tags = tags
                            self$metric_definitions = metric_definitions
                            self$model_uri = model_uri
                            self$model_channel_name = model_channel_name
                            self$code_uri = NULL
                            self$code_channel_name = "code"

                            if (self$train_instance_type %in% c("local", "local_gpu")) {
                              if (self$train_instance_type == "local_gpu" && self$train_instance_count > 1) stop("Distributed Training in Local GPU is not supported", call. = FALSE)
                              stop("Currently don't support local sagemaker", call. = F)
                              self$sagemaker_session = sagemaker_session #  LocalSession()
                              if (!inherist(self$sagemaker_session, "Session")) stop("instance_type local or local_gpu is only supported with an instance of LocalSession", call. = FALSE)
                            } else  {self$sagemaker_session = sagemaker_session %||% Session$new()}


                            self$base_job_name = base_job_name
                            self$.current_job_name = NULL

                            if (inherits(self$sagemaker_session, "Session") # need to change this part to local mode :S
                              && !is.null(output_path)
                              && startsWith(output_path, "file://")) {
                              stop("file:// output paths are only supported in Local Mode", call. = F)}

                            self$output_path = output_path
                            self$output_kms_key = output_kms_key
                            self$latest_training_job = None
                            self$jobs = list()
                            self$deploy_instance_type = NULL

                            self$.compiled_models = list()

                            # VPC configurations
                            self$subnets = subnets
                            self$security_group_ids = security_group_ids

                            self$encrypt_inter_container_traffic = encrypt_inter_container_traffic
                            self$train_use_spot_instances = train_use_spot_instances
                            self$train_max_wait = train_max_wait
                            self$checkpoint_s3_uri = checkpoint_s3_uri
                            self$checkpoint_local_path = checkpoint_local_path

                            self$rules = rules
                            self$debugger_hook_config = debugger_hook_config
                            self$tensorboard_output_config = tensorboard_output_config

                            self$debugger_rule_configs = None
                            self$collection_configs = None

                            self$enable_sagemaker_metrics = enable_sagemaker_metrics
                            self$.enable_network_isolation = enable_network_isolation
                          },

                          train_image = function() {stop("I'm an abstract interface method")},
                          hyperparameters = function() {stop("I'm an abstract interface method")},
                          enable_network_isolation = function() {return(self$.enable_network_isolation)},
                          prepare_workflow_for_training = function(job_name = NULL){
                            private$.prepare_for_training(job_name=job_name)
                          }

                          ),
                        private = list(
                          .prepare_for_training = function(job_name = NULL) {
                            if(!is.null(job_name)){
                              self$.current_job_name = job_name
                              } else {
                                # honor supplied base_job_name or generate it
                                if (!is.null(self$base_job_name)){
                                base_name = self$base_job_name
                                } else if (inherits(self$base_job_name, "AlgorithmEstimator")){ # need to work on this alittle more :)
                                  base_name = split_str(self$algorithm_arn, "/")[length(split_str(self$algorithm_arn, "/"))]
                                } else {
                                  base_name = base_name_from_image(self$train_image())}
                                self$.current_job_name = name_from_base(base_name)}

                            # if output_path was specified we use it otherwise initialize here.
                            # For Local Mode with local_code=True we don't need an explicit output_path

                            if(is.null(self$output_path)) {
                              local_code = get_config_value("local.local_code", self$sagemaker_session$config)
                              if (self$sagemaker_session$local_mode && !is.null(local_code)) {
                                self$output_path = ""
                              } else {self$output_path = sprintf("s3://%s",self$sagemaker_session$default_bucket())}
                            }

                            # Prepare rules and debugger configs for training.
                            if (!is.null(self$rules) && is.null(self$debugger_hook_config)) {
                              self$debugger_hook_config = DebuggerHookConfig$new(s3_output_path=self$output_path)}
                            # If an object was provided without an S3 URI is not provided, default it for the customer.
                            if (is.null(self$debugger_hook_config) && !is.null(self$debugger_hook_config$s3_output_path))
                              self$debugger_hook_config$s3_output_path = self$output_path
                            self._prepare_rules()
                            self._prepare_collection_configs()
                            }
                        # .prepare_rules <- function(){
                        #
                        #   """Set any necessary values in debugger rules, if they are provided."""
                        #   self.debugger_rule_configs = []
                        #   if self.rules is not None:
                        #     # Iterate through each of the provided rules.
                        #     for rule in self.rules:
                        #     # Set the image URI using the default rule evaluator image and the region.
                        #     if rule.image_uri == "DEFAULT_RULE_EVALUATOR_IMAGE":
                        #     rule.image_uri = get_rule_container_image_uri(
                        #       self$sagemaker_session$paws_region()
                        #     )
                        #   rule.instance_type = None
                        #   rule.volume_size_in_gb = None
                        #   # If source was provided as a rule parameter, upload to S3 and save the S3 uri.
                        #   if "source_s3_uri" in (rule.rule_parameters or {}):
                        #     parse_result = urlparse(rule.rule_parameters["source_s3_uri"])
                        #   if parse_result.scheme != "s3":
                        #     desired_s3_uri = os.path.join(
                        #       "s3://",
                        #       self.sagemaker_session.default_bucket(),
                        #       rule.name,
                        #       str(uuid.uuid4()),
                        #     )
                        #   s3_uri = S3Uploader.upload(
                        #     local_path=rule.rule_parameters["source_s3_uri"],
                        #     desired_s3_uri=desired_s3_uri,
                        #     session=self.sagemaker_session,
                        #   )
                        #   rule.rule_parameters["source_s3_uri"] = s3_uri
                        #   # Save the request dictionary for the rule.
                        #   self.debugger_rule_configs.append(rule.to_debugger_rule_config_dict())
                        #
                        # }
                        ),
                        lock_objects = F
                        )

