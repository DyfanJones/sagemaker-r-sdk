# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/clarify.py

#' @include processing.R
#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common
#' @import jsonlite

#' @title DataConfig Class
#' @description Config object related to configurations of the input and output dataset.
#' @export
DataConfig = R6Class("DataConfig",
  public = list(

    #' @field s3_data_input_path
    #' Dataset S3 prefix/object URI.
    s3_data_input_path = NULL,

    #' @field s3_output_path
    #' S3 prefix to store the output.
    s3_output_path = NULL,

    #' @field s3_data_distribution_type
    #' Valid options are "FullyReplicated" or "ShardedByS3Key".
    s3_data_distribution_type = NULL,

    #' @field s3_compression_type
    #' Valid options are "None" or "Gzip".
    s3_compression_type = NULL,

    #' @field label
    #' Target attribute of the model required by bias metrics
    label = NULL,

    #' @field features
    #' JSONPath for locating the feature columns
    features = NULL,

    #' @field analysis_config
    #' Analysis config dictionary
    analysis_config = NULL,

    #' @description Initializes a configuration of both input and output datasets.
    #' @param s3_data_input_path (str): Dataset S3 prefix/object URI.
    #' @param s3_output_path (str): S3 prefix to store the output.
    #' @param label (str): Target attribute of the model required by bias metrics (optional for SHAP)
    #'              Specified as column name or index for CSV dataset, or as JSONPath for JSONLines.
    #' @param headers (list[str]): A list of column names in the input dataset.
    #' @param features (str): JSONPath for locating the feature columns for bias metrics if the
    #'              dataset format is JSONLines.
    #' @param dataset_type (str): Format of the dataset. Valid values are "text/csv" for CSV
    #'              and "application/jsonlines" for JSONLines.
    #' @param s3_data_distribution_type (str): Valid options are "FullyReplicated" or
    #'              "ShardedByS3Key".
    #' @param s3_compression_type (str): Valid options are "None" or "Gzip".
    initialize = function(s3_data_input_path,
                          s3_output_path,
                          label=NULL,
                          headers=NULL,
                          features=NULL,
                          dataset_type="text/csv",
                          s3_data_distribution_type="FullyReplicated",
                          s3_compression_type=c("None", "Gzip")){
      self$s3_data_input_path = s3_data_input_path
      self$s3_output_path = s3_output_path
      self$s3_data_distribution_type = s3_data_distribution_type
      self$s3_compression_type = match.arg(s3_compression_type)
      self$label = label
      self$headers = headers
      self$features = features
      self$analysis_config = list(
        "dataset_type"= dataset_type)
      self$analysis_config[["features"]] = features
      self$analysis_config[["headers"]] = headers
      self$analysis_config[["label"]] = label
    },

    #' @description Returns part of an analysis config dictionary.
    get_config = function(){
      return(self$analysis_config)
    },

    #' @description format class
    format = function(){
      return(format_class(self))
    }
  )
)

#' @title BiasConfig Class
#' @description Config object related to bias configurations of the input dataset.
#' @export
BiasConfig = R6Class("BiasConfig",
  public = list(

    #' @field analysis_config
    #' Analysis config dictionary
    analysis_config = NULL,

    #' @description Initializes a configuration of the sensitive groups in the dataset.
    #' @param label_values_or_threshold (Any): List of label values or threshold to indicate positive
    #'              outcome used for bias metrics.
    #' @param facet_name (str): Sensitive attribute in the input data for which we like to compare
    #'              metrics.
    #' @param facet_values_or_threshold (list): Optional list of values to form a sensitive group or
    #'              threshold for a numeric facet column that defines the lower bound of a sensitive
    #'              group. Defaults to considering each possible value as sensitive group and
    #'              computing metrics vs all the other examples.
    #' @param group_name (str): Optional column name or index to indicate a group column to be used
    #'              for the bias metric 'Conditional Demographic Disparity in Labels - CDDL' or
    #'              'Conditional Demographic Disparity in Predicted Labels - CDDPL'.
    initialize = function(label_values_or_threshold,
                          facet_name,
                          facet_values_or_threshold=NULL,
                          group_name=NULL){
      facet = list("name_or_index"= facet_name)
      facet[["value_or_threshold"]] = facet_values_or_threshold
      self$analysis_config = list(
        "label_values_or_threshold"= label_values_or_threshold,
        "facet"= list(facet))
      self$analysis_config[["group_variable"]] = group_name
    },

    #' @description Returns part of an analysis config dictionary.
    get_config = function(){
      self$analysis_config
    },

    #' @description format class
    format = function(){
      return(print_class(self))
    }
  )
)

#' @title Model Config
#' @description Config object related to a model and its endpoint to be created.
#' @export
ModelConfig = R6Class("ModelConfig",
  public = list(

    #' @field predictor_config
    #' Predictor dictionary of the analysis config
    predictor_config = NULL,

    #' @description Initializes a configuration of a model and the endpoint to be created for it.
    #' @param model_name (str): Model name (as created by 'CreateModel').
    #' @param instance_count (int): The number of instances of a new endpoint for model inference.
    #' @param instance_type (str): The type of EC2 instance to use for model inference,
    #'              for example, 'ml.c5.xlarge'.
    #' @param accept_type (str): The model output format to be used for getting inferences with the
    #'              shadow endpoint. Valid values are "text/csv" for CSV and "application/jsonlines".
    #'              Default is the same as content_type.
    #' @param content_type (str): The model input format to be used for getting inferences with the
    #'              shadow endpoint. Valid values are "text/csv" for CSV and "application/jsonlines".
    #'              Default is the same as dataset format.
    #' @param content_template (str): A template string to be used to construct the model input from
    #'              dataset instances. It is only used when "model_content_type" is
    #'              "application/jsonlines". The template should have one and only one placeholder
    #'              $features which will be replaced by a features list for to form the model inference
    #'              input.
    initialize = function(model_name,
                          instance_count,
                          instance_type,
                          accept_type=NULL,
                          content_type=NULL,
                          content_template=NULL){
      self$predictor_config = list(
        "model_name"= model_name,
        "instance_type"= instance_type,
        "initial_instance_count"= instance_count)
      if (!is.null(accept_type)){
        if (!(accept_type %in% c("text/csv", "application/jsonlines"))){
          stop(sprintf("Invalid accept_type %s.", accept_type),
            " Please choose text/csv or application/jsonlines.", call. = F)
          }
        self$predictor_config[["accept_type"]] = accept_type
      }
      if (!is.null(content_type)){
        if (!(content_type %in% c("text/csv", "application/jsonlines"))){
          stop(sprintf("Invalid content_type %s.", content_type),
               " Please choose text/csv or application/jsonlines.", call. = F)
          }
        self$predictor_config[["content_type"]] = content_type
      }
      if (!is.null(content_template)){
        if (!("$features" %in% content_template)){
          stop(sprintf("Invalid content_template %s.", content_template),
               " Please include a placeholder $features.", call. = F)
        }
        self$predictor_config[["content_template"]] = content_template
      }
    },

    #' @description Returns part of the predictor dictionary of the analysis config.
    get_predictor_config = function(){
      return(self$predictor_config)
    },

    #' @description format class
    format = function(){
      return(format_class(self))
    }
  )
)

#' @title ModelPredictedLabelConfig Class
#' @description Config object to extract a predicted label from the model output.
#' @export
ModelPredictedLabelConfig = R6Class("ModelPredictedLabelConfig",
  public = list(

    #' @field label
    #' Predicted label of the same type as the label in the dataset
    label = NULL,

    #' @field probability
    #' Optional index or JSONPath location in the model
    probability = NULL,

    #' @field probability_threshold
    #' An optional value for binary prediction task
    probability_threshold = NULL,

    #' @field predictor_config
    #' Predictor dictionary of the analysis config.
    predictor_config = NULL,

    #' @description Initializes a model output config to extract the predicted label.
    #'              The following examples show different parameter configurations depending on the endpoint:
    #'              * Regression Task: The model returns the score, e.g. 1.2. we don't need to specify
    #'              anything. For json output, e.g. {'score': 1.2} we can set 'label='score''.
    #'              * Binary classification:
    #'              * The model returns a single probability and we would like to classify as 'yes'
    #'              those with a probability exceeding 0.2.
    #'              We can set 'probability_threshold=0.2, label_headers='yes''.
    #'              * The model returns {'probability': 0.3}, for which we would like to apply a
    #'              threshold of 0.5 to obtain a predicted label in {0, 1}. In this case we can set
    #'              'label='probability''.
    #'              * The model returns a tuple of the predicted label and the probability.
    #'              In this case we can set 'label=0'.
    #'              * Multiclass classification:
    #'              * The model returns
    #'              {'labels': ['cat', 'dog', 'fish'], 'probabilities': [0.35, 0.25, 0.4]}.
    #'              In this case we would set the 'probability='probabilities'' and
    #'              'label='labels'' and infer the predicted label to be 'fish.'
    #'              * The model returns {'predicted_label': 'fish', 'probabilities': [0.35, 0.25, 0.4]}.
    #'              In this case we would set the 'label='predicted_label''.
    #'              * The model returns [0.35, 0.25, 0.4]. In this case, we can set
    #'              'label_headers=['cat','dog','fish']' and infer the predicted label to be 'fish.'
    #' @md
    #' @param label (str or int or list[int]): Optional index or JSONPath location in the model
    #'              output for the prediction. In case, this is a predicted label of the same type as
    #'              the label in the dataset no further arguments need to be specified.
    #' @param probability (str or int or list[int]): Optional index or JSONPath location in the model
    #'              output for the predicted scores.
    #' @param probability_threshold (float): An optional value for binary prediction tasks in which
    #'              the model returns a probability, to indicate the threshold to convert the
    #'              prediction to a boolean value. Default is 0.5.
    #' @param label_headers (list): List of label values - one for each score of the ``probability``.
    initialize =function(label=NULL,
                         probability=NULL,
                         probability_threshold=NULL,
                         label_headers=NULL){
      self$label = label
      self$probability = probability
      self$probability_threshold = probability_threshold
      if (!is.null(probability_threshold)){
        tryCatch({
          as.numeric(probability_threshold)},
          error = function(e){
            stop(sprintf("Invalid probability_threshold %s. ", probability_threshold),
                 "Please choose one that can be cast to float.", call. = F)
          })
      }
      self$predictor_config = list()
      self$predictor_config[["label"]] =  label
      self$predictor_config[["probability"]] = probability
      self$predictor_config[["label_headers"]] = label_headers
    },

    #' @description Returns probability_threshold, predictor config.
    get_predictor_config = function(){
      return(list(self$probability_threshold, self$predictor_config))
    },

    #' @description format class
    format = function(){
      return(print_class(self))
    }
  )
)

#' @title ExplainabilityConfig Class
#' @description Abstract config class to configure an explainability method.
#' @export
ExplainabilityConfig = R6Class("ExplainabilityConfig",
  public = list(

    #' @description Returns config.
    get_explainability_config = function(){
      return(NULL)
    },

    #' @description format class
    format = function(){
      return(print_class(self))
    }
  )
)

#' @title SHAPConfig Class
#' @description Config class of SHAP.
#' @export
SHAPConfig = R6Class("SHAPConfig",
  inherit = ExplainabilityConfig,
  public = list(

    #' @field shap_config
    #' Shap Config
    shap_config = NULL,

    #' @description Initializes config for SHAP.
    #' @param baseline (str or list): A list of rows (at least one) or S3 object URI to be used as
    #'              the baseline dataset in the Kernel SHAP algorithm. The format should be the same
    #'              as the dataset format. Each row should contain only the feature columns/values
    #'              and omit the label column/values.
    #' @param num_samples (int): Number of samples to be used in the Kernel SHAP algorithm.
    #'              This number determines the size of the generated synthetic dataset to compute the
    #'              SHAP values.
    #' @param agg_method (str): Aggregation method for global SHAP values. Valid values are
    #'              "mean_abs" (mean of absolute SHAP values for all instances),
    #'              "median" (median of SHAP values for all instances) and
    #'              "mean_sq" (mean of squared SHAP values for all instances).
    #' @param use_logit (bool): Indicator of whether the logit function is to be applied to the model
    #'              predictions. Default is False. If "use_logit" is true then the SHAP values will
    #'              have log-odds units.
    #' @param save_local_shap_values (bool): Indicator of whether to save the local SHAP values
    #'              in the output location. Default is True.
    initialize = function(baseline,
                          num_samples,
                          agg_method = c("mean_abs", "median", "mean_sq"),
                          use_logit=FALSE,
                          save_local_shap_values=TRUE){
      agg_method = match.arg(agg_method)
      self$shap_config = list(
        "baseline"= baseline,
        "num_samples"= num_samples,
        "agg_method"= agg_method,
        "use_logit"= use_logit,
        "save_local_shap_values"= save_local_shap_values)
    },

    #' @description Returns config.
    get_explainability_config = function(){
      return(list("shap": self$shap_config))
    }
  )
)

#' @title SageMakerClarifyProcessor Class
#' @description Handles SageMaker Processing task to compute bias metrics and explain a model.
#' @export
SageMakerClarifyProcessor = R6Class("SageMakerClarifyProcessor",
  inherit = Processor,
  public = list(

    #' @description Initializes a ``Processor`` instance, computing bias metrics and model explanations.
    #' @param role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
    #'              uses this role to access AWS resources, such as
    #'              data stored in Amazon S3.
    #' @param instance_count (int): The number of instances to run
    #'              a processing job with.
    #' @param instance_type (str): The type of EC2 instance to use for
    #'              processing, for example, 'ml.c4.xlarge'.
    #' @param volume_size_in_gb (int): Size in GB of the EBS volume
    #'              to use for storing data during processing (default: 30).
    #' @param volume_kms_key (str): A KMS key for the processing
    #'              volume (default: None).
    #' @param output_kms_key (str): The KMS key ID for processing job outputs (default: None).
    #' @param max_runtime_in_seconds (int): Timeout in seconds (default: None).
    #'              After this amount of time, Amazon SageMaker terminates the job,
    #'              regardless of its current status. If `max_runtime_in_seconds` is not
    #'              specified, the default value is 24 hours.
    #' @param sagemaker_session (:class:`~sagemaker.session.Session`):
    #'              Session object which manages interactions with Amazon SageMaker and
    #'              any other AWS services needed. If not specified, the processor creates
    #'              one using the default AWS configuration chain.
    #' @param env (dict[str, str]): Environment variables to be passed to
    #'              the processing jobs (default: None).
    #' @param tags (list[dict]): List of tags to be passed to the processing job
    #'              (default: None). For more, see
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
    #' @param network_config (:class:`~sagemaker.network.NetworkConfig`):
    #'              A :class:`~sagemaker.network.NetworkConfig`
    #'              object that configures network isolation, encryption of
    #'              inter-container traffic, security group IDs, and subnets.
    initialize = function(role,
                          instance_count,
                          instance_type,
                          volume_size_in_gb=30,
                          volume_kms_key=NULL,
                          output_kms_key=NULL,
                          max_runtime_in_seconds=NULL,
                          sagemaker_session=NULL,
                          env=NULL,
                          tags=NULL,
                          network_config=NULL){
      container_uri = ImageUris$new()$retrieve("clarify", sagemaker_session$paws_region_name)
      super$initialize(
        role,
        container_uri,
        instance_count,
        instance_type,
        NULL,  # We manage the entrypoint.
        volume_size_in_gb,
        volume_kms_key,
        output_kms_key,
        max_runtime_in_seconds,
        NULL,  # We set method-specific job names below.
        sagemaker_session,
        env,
        tags,
        network_config)
    },

    #' @description Overriding the base class method but deferring to specific run_* methods.
    run = function(){
      stop(
        "Please choose a method of run_pre_training_bias, run_post_training_bias or ",
        "run_explainability.", call. = F)
    },

    #' @description Runs a ProcessingJob to compute the requested bias 'methods' of the input data.
    #'              Computes the requested methods that compare 'methods' (e.g. fraction of examples) for the
    #'              sensitive group vs the other examples.
    #' @param data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
    #' @param data_bias_config (:class:`~sagemaker.clarify.BiasConfig`): Config of sensitive groups.
    #' @param methods (str or list[str]): Selector of a subset of potential metrics:
    #'              ["CI", "DPL", "KL", "JS", "LP", "TVD", "KS", "CDDL"]. Defaults to computing all.
    # TODO: Provide a pointer to the official documentation of those.
    #' @param wait (bool): Whether the call should wait until the job completes (default: True).
    #' @param logs (bool): Whether to show the logs produced by the job.
    #'              Only meaningful when ``wait`` is True (default: True).
    #' @param job_name (str): Processing job name. If not specified, a name is composed of
    #'              "Clarify-Pretraining-Bias" and current timestamp.
    #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
    #'              user code file (default: None).
    run_pre_training_bias = function(data_config,
                                     data_bias_config,
                                     methods="all",
                                     wait=TRUE,
                                     logs=TRUE,
                                     job_name=NULL,
                                     kms_key=NULL){
      analysis_config = data_config$get_config()
      analysis_config.update(data_bias_config$get_config())
      analysis_config[["methods"]] = list("pre_training_bias"= list("methods"= methods))
      if (is.null(job_name))
        job_name = name_from_base("Clarify-Pretraining-Bias")
      private$.run(data_config, analysis_config, wait, logs, job_name, kms_key)
    },

    #' @description Runs a ProcessingJob to compute the requested bias 'methods' of the model predictions.
    #'              Spins up a model endpoint, runs inference over the input example in the
    #'              's3_data_input_path' to obtain predicted labels. Computes a the requested methods that
    #'              compare 'methods' (e.g. accuracy, precision, recall) for the sensitive group vs the other
    #'              examples.
    #' @param data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
    #' @param data_bias_config (:class:`~sagemaker.clarify.BiasConfig`): Config of sensitive groups.
    #' @param model_config (:class:`~sagemaker.clarify.ModelConfig`): Config of the model and its
    #'              endpoint to be created.
    #' @param model_predicted_label_config (:class:`~sagemaker.clarify.ModelPredictedLabelConfig`):
    #'              Config of how to extract the predicted label from the model output.
    #' @param methods (str or list[str]): Selector of a subset of potential metrics:
    # TODO: Provide a pointer to the official documentation of those.
    #'              ["DPPL", "DI", "DCA", "DCR", "RD", "DAR", "DRR", "AD", "CDDPL", "TE", "FT"].
    #'              Defaults to computing all.
    #' @param wait (bool): Whether the call should wait until the job completes (default: True).
    #' @param logs (bool): Whether to show the logs produced by the job.
    #'              Only meaningful when ``wait`` is True (default: True).
    #' @param job_name (str): Processing job name. If not specified, a name is composed of
    #'              "Clarify-Posttraining-Bias" and current timestamp.
    #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
    #'              user code file (default: None).
    run_post_training_bias = function(data_config,
                                      data_bias_config,
                                      model_config,
                                      model_predicted_label_config,
                                      methods="all",
                                      wait=TRUE,
                                      logs=TRUE,
                                      job_name=NULL,
                                      kms_key=NULL){
      analysis_config = data_config$get_config()
      analysis_config = c(analysis_config, data_bias_config$get_config())

      ll = model_predicted_label_config$get_predictor_config()
      names(ll) = c("probability_threshold", "predictor_config")

      ll$predictor_config = c(ll$predictor_config, model_config$get_predictor_config())
      analysis_config[["methods"]] = list("post_training_bias"= list("methods"= methods))
      analysis_config[["predictor"]] = ll$predictor_config
      ll$probability_threshold[["probability_threshold"]] = analysis_config
      if (is.null(job_name))
        job_name = name_from_base("Clarify-Posttraining-Bias")
      private$.run(data_config, analysis_config, wait, logs, job_name, kms_key)
    },

    #' @description Runs a ProcessingJob to compute the requested bias 'methods' of the model predictions.
    #'              Spins up a model endpoint, runs inference over the input example in the
    #'              's3_data_input_path' to obtain predicted labels. Computes a the requested methods that
    #'              compare 'methods' (e.g. accuracy, precision, recall) for the sensitive group vs the other
    #'              examples.
    #' @param data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
    #' @param bias_config (:class:`~sagemaker.clarify.BiasConfig`): Config of sensitive groups.
    #' @param model_config (:class:`~sagemaker.clarify.ModelConfig`): Config of the model and its
    #'              endpoint to be created.
    #' @param model_predicted_label_config (:class:`~sagemaker.clarify.ModelPredictedLabelConfig`):
    #'              Config of how to extract the predicted label from the model output.
    #' @param pre_training_methods (str or list[str]): Selector of a subset of potential metrics:
    # TODO: Provide a pointer to the official documentation of those.
    #'              ["DPPL", "DI", "DCA", "DCR", "RD", "DAR", "DRR", "AD", "CDDPL", "TE", "FT"].
    #'              Defaults to computing all.
    #' @param post_training_methods (str or list[str]): Selector of a subset of potential metrics:
    # TODO: Provide a pointer to the official documentation of those.
    #'              ["DPPL", "DI", "DCA", "DCR", "RD", "DAR", "DRR", "AD", "CDDPL", "TE", "FT"].
    #'              Defaults to computing all.
    #' @param wait (bool): Whether the call should wait until the job completes (default: True).
    #' @param logs (bool): Whether to show the logs produced by the job.
    #'              Only meaningful when ``wait`` is True (default: True).
    #' @param job_name (str): Processing job name. If not specified, a name is composed of
    #'              "Clarify-Bias" and current timestamp.
    #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
    #'              user code file (default: None).
    run_bias = function(data_config,
                        bias_config,
                        model_config,
                        model_predicted_label_config=NULL,
                        pre_training_methods="all",
                        post_training_methods="all",
                        wait=TRUE,
                        logs=TRUE,
                        job_name=NULL,
                        kms_key=NULL){
      analysis_config = data_config$get_config()
      analysis_config = c(analysis_config, bias_config$get_config())
      analysis_config[["predictor"]] = model_config$get_predictor_config()
      if (!is.null(model_predicted_label_config)){
        ll = model_predicted_label_config$get_predictor_config()
        names(ll) = c("probability_threshold", "predictor_config")
      if (!islistempty(ll$predictor_config))
        analysis_config[["predictor"]] = c(analysis_config[["predictor"]], ll$predictor_config)
      if (!islistempty(ll$probability_threshold))
        analysis_config[["probability_threshold"]] = ll$probability_threshold
      }
      analysis_config[["methods"]] = list(
        "pre_training_bias"= list("methods"= pre_training_methods),
        "post_training_bias"= list("methods"= post_training_methods))
      if (is.null(job_name))
        job_name = name_from_base("Clarify-Bias")
      private$.run(data_config, analysis_config, wait, logs, job_name, kms_key)
    },

    #' @description Runs a ProcessingJob computing for each example in the input the feature importance.
    #'              Currently, only SHAP is supported as explainability method.
    #'              Spins up a model endpoint.
    #'              For each input example in the 's3_data_input_path' the SHAP algorithm determines
    #'              feature importance, by creating 'num_samples' copies of the example with a subset
    #'              of features replaced with values from the 'baseline'.
    #'              Model inference is run to see how the prediction changes with the replaced features.
    #'              If the model output returns multiple scores importance is computed for each of them.
    #'              Across examples, feature importance is aggregated using 'agg_method'.
    #' @param data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
    #' @param model_config (:class:`~sagemaker.clarify.ModelConfig`): Config of the model and its
    #'              endpoint to be created.
    #' @param explainability_config (:class:`~sagemaker.clarify.ExplainabilityConfig`): Config of the
    #'              specific explainability method. Currently, only SHAP is supported.
    #' @param model_scores :  Index or JSONPath location in the model output for the predicted scores
    #'              to be explained. This is not required if the model output is a single score.
    #' @param wait (bool): Whether the call should wait until the job completes (default: True).
    #' @param logs (bool): Whether to show the logs produced by the job.
    #'              Only meaningful when ``wait`` is True (default: True).
    #' @param job_name (str): Processing job name. If not specified, a name is composed of
    #'              "Clarify-Explainability" and current timestamp.
    #' @param kms_key (str): The ARN of the KMS key that is used to encrypt the
    #'              user code file (default: None).
    run_explainability = function(data_config,
                                  model_config,
                                  explainability_config,
                                  model_scores=NULL,
                                  wait=TRUE,
                                  logs=TRUE,
                                  job_name=NULL,
                                  kms_key=NULL){
      analysis_config = data_config$get_config()
      predictor_config = model_config$get_predictor_config()
      predictor_config[["label"]] = model_scores
      analysis_config[["methods"]] = explainability_config$get_explainability_config()
      analysis_config[["predictor"]] = predictor_config
      if (is.null(job_name))
        job_name = name_from_base("Clarify-Explainability")
      private$.run(data_config, analysis_config, wait, logs, job_name, kms_key)
    }
  ),
  private = list(
    .CLARIFY_DATA_INPUT = "/opt/ml/processing/input/data",
    .CLARIFY_CONFIG_INPUT = "/opt/ml/processing/input/config",
    .CLARIFY_OUTPUT = "/opt/ml/processing/output",

    # Runs a ProcessingJob with the Sagemaker Clarify container and an analysis config.
    # Args:
    #   data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
    # analysis_config (dict): Config following the analysis_config.json format.
    # wait (bool): Whether the call should wait until the job completes (default: True).
    # logs (bool): Whether to show the logs produced by the job.
    # Only meaningful when ``wait`` is True (default: True).
    # job_name (str): Processing job name.
    # kms_key (str): The ARN of the KMS key that is used to encrypt the
    # user code file (default: None).
    .run = function(data_config,
                    analysis_config,
                    wait,
                    logs,
                    job_name,
                    kms_key){
      analysis_config[["methods"]][["report"]] = list("name"= "report", "title"= "Analysis Report")

      tmpdirname = tempdir()
      on.exit(unlink(tmpdirname, recursive = T))
      analysis_config_file = file.path(tmpdirname, "analysis_config.json")
      write_json(analysis_config, analysis_config_file, auto_unbox = T)

      config_input = ProcessingInput$new(
        input_name="analysis_config",
        source=analysis_config_file,
        destination=private$.CLARIFY_CONFIG_INPUT,
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_compression_type="None")

      data_input = ProcessingInput$new(
        input_name="dataset",
        source=data_config.s3_data_input_path,
        destination=self._CLARIFY_DATA_INPUT,
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type=data_config$s3_data_distribution_type,
        s3_compression_type=data_config$s3_compression_type)

      result_output = ProcessingOutput$new(
        source=private$.CLARIFY_OUTPUT,
        destination=data_config$s3_output_path,
        output_name="analysis_result",
        s3_upload_mode="EndOfJob")

      super$run(
        inputs=list(data_input, config_input),
        outputs=list(result_output),
        wait=wait,
        logs=logs,
        job_name=job_name,
        kms_key=kms_key)
    }
  )
)
