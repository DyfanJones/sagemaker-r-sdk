# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/af7f75ae336f0481e52bb968e4cc6df91b1bac2c/src/sagemaker/amazon/amazon_estimator.py


#' @include utils.R
#' @include xgboost_estimator.R
#' @include fw_registry.R
#' @include s3.R
#' @include amazon_hyperparameter.R
#' @include amazon_validation.R

#' @import R6
#' @importFrom urltools url_parse
#' @import paws
#' @import jsonlite

#' @title AmazonAlgorithmEstimatorBase Class
#' @description Base class for Amazon first-party Estimator implementations. This class
#'              isn't intended to be instantiated directly.
#' @export
AmazonAlgorithmEstimatorBase = R6Class("AmazonAlgorithmEstimatorBase",
  inherit = EstimatorBase,
  public = list(
    #' @field repo_name
    #' The repo name for the account
    repo_name = NULL,

    #' @field repo_version
    #' Version fo repo to call
    repo_version = NULL,

    #' @field .feature_dim
    #' descriptor class
    .feature_dim = NULL,

    #' @field .mini_batch_size
    #' descriptor class
    .mini_batch_size = NULL,

    #' @description Initialize an AmazonAlgorithmEstimatorBase.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if it needs to access an AWS resource.
    #' @param train_instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param train_instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.c4.xlarge'.
    #' @param data_location (str or None): The s3 prefix to upload RecordSet
    #'              objects to, expressed as an S3 url. For example
    #'              "s3://example-bucket/some-key-prefix/". Objects will be saved in
    #'              a unique sub-directory of the specified location. If None, a
    #'              default data location will be used.
    #' @param enable_network_isolation (bool): Specifies whether container will
    #'              run in network isolation mode. Network isolation mode restricts
    #'              the container access to outside networks (such as the internet).
    #'              Also known as internet-free mode (default: ``False``).
    #' @param ... : Additional parameters passed to
    #'             :class:`~sagemaker.estimator.EstimatorBase`.
    initialize = function(role,
                          train_instance_count,
                          train_instance_type,
                          data_location=NULL,
                          enable_network_isolation=FALSE,
                          ...){
      super$initialize(role,
                       train_instance_count,
                       train_instance_type,
                       enable_network_isolation=enable_network_isolation,
                       ...)

      data_location = data_location %||% sprintf("s3://%s/sagemaker-record-sets/", self$sagemaker_session$default_bucket())

      self$.data_location = data_location

      self$.feature_dim = Hyperparameter$new("feature_dim", Validation$new()$gt(0), data_type=as.integer, obj = self)
      self$.mini_batch_size = Hyperparameter$new("mini_batch_size", Validation$new()$gt(0), data_type=as.integer, obj = self)
    },

    #' @description Return algorithm image URI for the given AWS region, repository name, and
    #'              repository version
    train_image = function(){
      return(get_image_uri(
        self$sagemaker_session$paws_region_name, self$repo_name, self$repo_version)
      )
    },

    #' @description Return all non-None ``hyperparameter`` values on ``obj`` as a
    #'              ``dict[str,str].``
    hyperparameters = function(){
      return(Hyperparameter$public_methods$serialize_all(self))
    },

    #' @description Calls _prepare_for_training. Used when setting up a workflow.
    #' @param records (:class:`~RecordSet`): The records to train this ``Estimator`` on.
    #' @param mini_batch_size (int or None): The size of each mini-batch to use when
    #'              training. If ``None``, a default value will be used.
    #' @param job_name (str): Name of the training job to be created. If not
    #'              specified, one is generated, using the base name given to the
    #'              constructor if applicable.
    prepare_workflow_for_training = function(records=NULL,
                                             mini_batch_size=NULL,
                                             job_name=NULL){
      private$.prepare_for_training(
        records=records, mini_batch_size=mini_batch_size, job_name=job_name
      )
    },

    #' @description Fit this Estimator on serialized Record objects, stored in S3.
    #'              ``records`` should be an instance of :class:`~RecordSet`. This
    #'              defines a collection of S3 data files to train this ``Estimator`` on.
    #'              Training data is expected to be encoded as dense or sparse vectors in
    #'              the "values" feature on each Record. If the data is labeled, the label
    #'              is expected to be encoded as a list of scalas in the "values" feature of
    #'              the Record label.
    #'              More information on the Amazon Record format is available at:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html
    #'              See :meth:`~AmazonAlgorithmEstimatorBase.record_set` to construct a
    #'              ``RecordSet`` object from :class:`~numpy.ndarray` arrays.
    #' @param records (:class:`~RecordSet`): The records to train this ``Estimator`` on
    #' @param mini_batch_size (int or None): The size of each mini-batch to use
    #'              when training. If ``None``, a default value will be used.
    #' @param wait (bool): Whether the call should wait until the job completes
    #'              (default: True).
    #' @param logs (bool): Whether to show the logs produced by the job. Only
    #'              meaningful when wait is True (default: True).
    #' @param job_name (str): Training job name. If not specified, the estimator
    #'              generates a default job name, based on the training image name
    #'              and current timestamp.
    #' @param experiment_config (dict[str, str]): Experiment management configuration.
    #'              Dictionary contains three optional keys, 'ExperimentName',
    #'              'TrialName', and 'TrialComponentName'
    #'              (default: ``None``).
    fit = function(records,
                   mini_batch_size=NULL,
                   wait=TRUE,
                   logs=TRUE,
                   job_name=NULL,
                   experiment_config=NULL){
      private$.prepare_for_training(records, job_name=job_name, mini_batch_size=mini_batch_size)

      self$latest_training_job = gsub(".*/","", private$.start_new(records, experiment_config)$TrainingJobArn)
      if (wait)
        self$wait(logs=logs)
    },

    #' @description Build a :class:`~RecordSet` from a numpy :class:`~ndarray` matrix and
    #'              label vector.
    #'              For the 2D ``ndarray`` ``train``, each row is converted to a
    #'              :class:`~Record` object. The vector is stored in the "values" entry of
    #'              the ``features`` property of each Record. If ``labels`` is not None,
    #'              each corresponding label is assigned to the "values" entry of the
    #'              ``labels`` property of each Record.
    #'              The collection of ``Record`` objects are protobuf serialized and
    #'              uploaded to new S3 locations. A manifest file is generated containing
    #'              the list of objects created and also stored in S3.
    #'              The number of S3 objects created is controlled by the
    #'              ``train_instance_count`` property on this Estimator. One S3 object is
    #'              created per training instance.
    #' @param train (numpy.ndarray): A 2D numpy array of training data.
    #' @param labels (numpy.ndarray): A 1D numpy array of labels. Its length must
    #'              be equal to the number of rows in ``train``.
    #' @param channel (str): The SageMaker TrainingJob channel this RecordSet
    #'              should be assigned to.
    #' @param encrypt (bool): Specifies whether the objects uploaded to S3 are
    #'              encrypted on the server side using AES-256 (default: ``False``).
    #' @return RecordSet: A RecordSet referencing the encoded, uploading training
    #'              and label data.
    record_set = function(train,
                          labels=NULL,
                          channel="train",
                          encrypt=FALSE){
      s3 = paws::s3(config = self$sagemaker_session$paws_credentials$credentials)

      parsed_s3_url = url_parse(self$data_location)
      bucket = parsed_s3_url$domain
      key_prefix = parsed_s3_url$path
      key_prefix = paste0(key_prefix, sprintf("%s-%s/", class(self)[1], sagemaker_timestamp()))
      key_prefix = trimws(key_prefix, "left", "/")
      log_debug("Uploading to bucket %s and key_prefix %s", bucket, key_prefix)
      # TODO: upload_numpy_to_s3_shards function
      manifest_s3_file = upload_numpy_to_s3_shards(
        self$train_instance_count, s3, bucket, key_prefix, train, labels, encrypt
      )

      log_debug("Created manifest file %s", manifest_s3_file)

      return(RecordSet$new(
        manifest_s3_file,
        num_records=dim(train)[1],
        feature_dim=dim(train)[2],
        channel=channel)
      )
    },

    #' @description Wait for an Amazon SageMaker job to complete.
    #' @param logs ([str]): A list of strings specifying which logs to print. Acceptable
    #'              strings are "All", "NULL", "Training", or "Rules". To maintain backwards
    #'              compatibility, boolean values are also accepted and converted to strings.
    wait = function(logs = "All"){
      if(inherits(logs, "logical")) logs = ifelse(logs, "All", "NULL")

      if(logs != "NULL"){
        self$sagemaker_session$logs_for_job(job_name = self$latest_training_job, wait=TRUE, log_type=logs)
      } else {
        self$sagemaker_session$wait_for_job(job = self$latest_training_job)}
    }
  ),
  active = list(
    #' @field data_location
    #' The s3 prefix to upload RecordSet objects to, expressed as an S3 url
    data_location = function(data_location){
      if(missing(data_location))
        return(self$.data_location)


      if(!startsWith(data_location, "s3://"))
        stop(sprintf('Expecting an S3 URL beginning with "s3://". Got "%s"',data_location), call. = F)

      if (!grepl("/$", data_location))
        data_location = paste0(data_location, "/")

      self$.data_location = data_location
    },

    # --------- User Active binding to mimic Python's Descriptor Class

    #' @field feature_dim
    #' Hyperparameter class for feature_dim
    feature_dim = function(value){
      if(missing(value))
        return(self$.feature_dim$descriptor)
      self$.feature_dim$descriptor = value
    },

    #' @field mini_batch_size
    #' Hyperparameter class for mini_batch_size
    mini_batch_size = function(value){
      if(missing(value))
        return(private$.mini_batch_size$descriptor)
      private$.mini_batch_size$descriptor = value
    }
  ),
  private = list(
    # Convert the job description to init params that can be handled by the
    # class constructor
    # Args:
    #   job_details: the returned job details from a describe_training_job
    # API call.
    # model_channel_name (str): Name of the channel where pre-trained
    # model data will be downloaded.
    # Returns:
    #   dictionary: The transformed init_params
    .prepare_init_params_from_job_description = function(job_details,
                                                         model_channel_name=NULL){
      init_params = super$.prepare_init_params_from_job_description(
        job_details, model_channel_name)

      # The hyperparam names may not be the same as the class attribute that holds them,
      # for instance: local_lloyd_init_method is called local_init_method. We need to map these
      # and pass the correct name to the constructor.
      cls_list = as.list(self)
      for (i in seq_along(cls_list)){
        attribute = names(cls_list)[i]
        value = cls_list[[i]]
        if (inherits(value, "Hyperparameter")){
          if (names(value) %in% names(init_params$hyperparameters))
            init_params[[attribute]] = init_params$hyperparameters[[names(value)]]
        }
      }

      init_params["hyperparameters"] = NULL
      init_params["image"] = NULL
      return(init_params)
    },

    # Set hyperparameters needed for training.
    # Args:
    #     records (:class:`~RecordSet`): The records to train this ``Estimator`` on.
    #     mini_batch_size (int or None): The size of each mini-batch to use when
    #         training. If ``None``, a default value will be used.
    #     job_name (str): Name of the training job to be created. If not
    #         specified, one is generated, using the base name given to the
    #         constructor if applicable.
    .prepare_for_training = function(records,
                                     mini_batch_size=NULL,
                                     job_name=NULL){
      super$.prepare_for_training(job_name=job_name)

      feature_dim = NULL

      if (inherits(records, "list")){
        for (record in records){
          if (record$channel == "train"){
            feature_dim = record$feature_dim
            break}
        }
        if (islistempty(feature_dim))
          stop("Must provide train channel.", call. = F)
      } else {
        feature_dim = records$feature_dim
      }

      self$feature_dim = feature_dim
      self$mini_batch_size = mini_batch_size
    },

    # ------------------------ incorporate _TrainingJob.start_new calls -------------------

    # Create a new Amazon SageMaker training job from the estimator.
    # Args:
    #   estimator (sagemaker.estimator.EstimatorBase): Estimator object
    # created by the user.
    # inputs (str): Parameters used when called
    # :meth:`~sagemaker.estimator.EstimatorBase.fit`.
    # experiment_config (dict[str, str]): Experiment management configuration used when called
    # :meth:`~sagemaker.estimator.EstimatorBase.fit`.  Dictionary contains
    # three optional keys, 'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
    # Returns:
    #   sagemaker.estimator._TrainingJob: Constructed object that captures
    # all information about the started training job.
    .start_new = function(inputs,
                          experiment_config = NULL){
      local_mode = self$sagemaker_session$local_mode
      model_uri = self$model_uri

      # Allow file:// input only in local mode
      if (private$.is_local_channel(inputs) || private$.is_local_channel(model_uri)){
        if (!local_mode) stop("File URIs are supported in local mode only. Please use a S3 URI instead.", call. = F)
      }

      config = .Job$new()$load_config(inputs, self)

      if (!islistempty(self$hyperparameters())){
        hyperparameters = self$hyperparameters()}


      train_args = config
      train_args[["input_mode"]] = self$input_mode
      train_args[["job_name"]] = self$.current_job_name
      train_args[["hyperparameters"]] = hyperparameters
      train_args[["tags"]] = self$tags
      train_args[["metric_definitions"]] = self$metric_definitions
      train_args[["experiment_config"]] = experiment_config

      if (inherits(inputs, "s3_input")){
        if ("InputMode" %in% inputs$config){
          log_debug("Selecting s3_input's input_mode (%s) for TrainingInputMode.",
                    inputs$config$InputMode)
          train_args[["input_mode"]] = inputs$config$InputMod}
      }


      if (self$enable_network_isolation()){
        train_args[["enable_network_isolation"]] = TRUE}

      if (self$encrypt_inter_container_traffic){
        train_args[["encrypt_inter_container_traffic"]] = TRUE}

      if (inherits(self, "Algorithmself")){
        train_args[["algorithm_arn"]] = self$algorithm_arn
      } else {
        train_args[["image"]] = self$train_image()}


      if (!islistempty(self$debugger_rule_configs))
        train_args[["debugger_rule_configs"]] = self$debugger_rule_configs

      if (!islistempty(self$debugger_hook_config)){
        self$debugger_hook_config[["collection_configs"]] = self$collection_configs
        train_args[["debugger_hook_config"]] = self$debugger_hook_config$to_request_list()}

      if (!islistempty(self$tensorboard_output_config))
        train_args[["tensorboard_output_config"]] = self$tensorboard_output_config$to_request_list()

      train_args = c(train_args, private$.add_spot_checkpoint_args(local_mode, train_args))


      if (!islistempty(self$enable_sagemaker_metrics))
        train_args[["enable_sagemaker_metrics"]] = self$enable_sagemaker_metrics
      do.call(self$sagemaker_session$train, train_args)
    }

  ),
  lock_object = F
)

#' @title RecordSet Class
#' @export
RecordSet = R6Class("RecordSet",
  public = list(
    #' @description A collection of Amazon :class:~`Record` objects serialized and stored
    #'              in S3.
    #' @param s3_data (str): The S3 location of the training data
    #' @param num_records (int): The number of records in the set.
    #' @param feature_dim (int): The dimensionality of "values" arrays in the
    #'              Record features, and label (if each Record is labeled).
    #' @param s3_data_type (str): Valid values: 'S3Prefix', 'ManifestFile'. If
    #'              'S3Prefix', ``s3_data`` defines a prefix of s3 objects to train
    #'              on. All objects with s3 keys beginning with ``s3_data`` will be
    #'              used to train. If 'ManifestFile', then ``s3_data`` defines a
    #'              single s3 manifest file, listing each s3 object to train on.
    #' @param channel (str): The SageMaker Training Job channel this RecordSet
    #'              should be bound to
    initialize = function(){
      self$s3_data = s3_data
      self$feature_dim = feature_dim
      self$num_records = num_records
      self$s3_data_type = s3_data_type
      self$channel = channel
    },

    #' @description Return a dictionary to represent the training data in a channel for
    #'              use with ``fit()``
    data_channel = function(){
      output = list(self$records_s3_input())
      names(output) = self$channel
      return(output)
    },

    #' @description Return a s3_input to represent the training data
    recods_s3_input = function(){
      return(s3_input$new(self$s3_data, distribution="ShardedByS3Key", s3_data_type=self$s3_data_type))
    },

    #' @description Return an unambiguous representation of this RecordSet
    print = function(){
      class_list = private$.str_list(RecordSet)
      return(cat(paste("class <'RecordSet'>,", class_list), "\n"))
    }
  ),
  private = list(
    .str_list = function(cls_gen){
      output = c(names(cls_gen$public_fields),
                 names(cls_gen$public_methods),
                 names(cls_gen$private_fields),
                 names(cls_gen$private_methods))
      output = as.list(self)[setdiff(ls(self), output)]
      return(toJSON(output, auto_unbox = T))
    }
  ),
  lock_object = F
)

#' @title FileSystemRecordSet Class
#' @description Amazon SageMaker channel configuration for a file system data source
#'              for Amazon algorithms.
#' @export
FileSystemRecordSet = R6Class("FileSystemRecordSet",
  public = list(

    #' @field feature_dim
    #' The dimensionality of "values" arrays in the Record features
    feature_dim = NULL,

    #' @field num_records
    #' The number of records in the set
    num_records = NULL,

    #' @field channel
    #' The SageMaker Training Job channel this RecordSet should be bound to
    channel = NULL,

    #' @description Initialize a ``FileSystemRecordSet`` object.
    #' @param file_system_id (str): An Amazon file system ID starting with 'fs-'.
    #' @param file_system_type (str): The type of file system used for the input.
    #'              Valid values: 'EFS', 'FSxLustre'.
    #' @param directory_path (str): Absolute or normalized path to the root directory (mount point) in
    #'              the file system. Reference:
    #'              https://docs.aws.amazon.com/efs/latest/ug/mounting-fs.html and
    #'              https://docs.aws.amazon.com/efs/latest/ug/wt1-test.html
    #' @param num_records (int): The number of records in the set.
    #' @param feature_dim (int): The dimensionality of "values" arrays in the Record features,
    #'              and label (if each Record is labeled).
    #' @param file_system_access_mode (str): Permissions for read and write.
    #'              Valid values: 'ro' or 'rw'. Defaults to 'ro'.
    #' @param channel (str): The SageMaker Training Job channel this RecordSet should be bound to
    initialize = function(file_system_id,
                          file_system_type,
                          directory_path,
                          num_records,
                          feature_dim,
                          file_system_access_mode="ro",
                          channel="train"){
      self$file_system_input = FileSystemInput$new(
        file_system_id, file_system_type, directory_path, file_system_access_mode
      )
      self$feature_dim = feature_dim
      self$num_records = num_records
      self$channel = channel
    },

    #' @description Return an unambiguous representation of this RecordSet
    #' @description Return an unambiguous representation of this RecordSet
    print = function(){
      class_list = private$.str_list(RecordSet)
      return(cat(paste("class <'FileSystemRecordSet'>,", class_list), "\n"))
    },

    #' @description Return a dictionary to represent the training data in a channel for use with ``fit()``
    data_channel = function(){
      output = list(self$file_system_input)
      names(output) = self$channel
      return(output)
    }
  ),
  private = list(
    .str_list = function(cls_gen){
      output = c(names(cls_gen$public_fields),
                 names(cls_gen$public_methods),
                 names(cls_gen$private_fields),
                 names(cls_gen$private_methods))
      output = as.list(self)[setdiff(ls(self), output)]
      return(toJSON(output, auto_unbox = T))
      }
    )
)


# Return docker registry for the given AWS region
# Note: Not all the algorithms listed below have an Amazon Estimator
# implemented. For full list of pre-implemented Estimators, look at:
#   https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/amazon
# Args:
#   region_name (str): The region name for the account.
# algorithm (str): The algorithm for the account.
# Raises:
#   ValueError: If invalid algorithm passed in or if mapping does not exist for given algorithm
# and region.
registry <- function(region_name,
                     algorithm=NULL){
  region_to_accounts = list()
  if (is.null(algorithm)
      || algorithm %in% c("pca",
                       "kmeans",
                       "linear-learner",
                       "factorization-machines",
                       "ntm",
                       "randomcutforest",
                       "knn",
                       "object2vec",
                       "ipinsights")){
    region_to_accounts = list(
      "us-east-1"= "382416733822",
      "us-east-2"= "404615174143",
      "us-west-2"= "174872318107",
      "eu-west-1"= "438346466558",
      "eu-central-1"= "664544806723",
      "ap-northeast-1"= "351501993468",
      "ap-northeast-2"= "835164637446",
      "ap-southeast-2"= "712309505854",
      "us-gov-west-1"= "226302683700",
      "ap-southeast-1"= "475088953585",
      "ap-south-1"= "991648021394",
      "ca-central-1"= "469771592824",
      "eu-west-2"= "644912444149",
      "us-west-1"= "632365934929",
      "us-iso-east-1"= "490574956308",
      "ap-east-1"= "286214385809",
      "eu-north-1"= "669576153137",
      "eu-west-3"= "749696950732",
      "sa-east-1"= "855470959533",
      "me-south-1"= "249704162688",
      "cn-north-1"= "390948362332",
      "cn-northwest-1"= "387376663083")
  } else if (algorithm %in% c("lda")){
    region_to_accounts = list(
      "us-east-1"= "766337827248",
      "us-east-2"= "999911452149",
      "us-west-2"= "266724342769",
      "eu-west-1"= "999678624901",
      "eu-central-1"= "353608530281",
      "ap-northeast-1"= "258307448986",
      "ap-northeast-2"= "293181348795",
      "ap-southeast-2"= "297031611018",
      "us-gov-west-1"= "226302683700",
      "ap-southeast-1"= "475088953585",
      "ap-south-1"= "991648021394",
      "ca-central-1"= "469771592824",
      "eu-west-2"= "644912444149",
      "us-west-1"= "632365934929",
      "us-iso-east-1"= "490574956308")
  } else if (algorithm %in% c("forecasting-deepar")){
    region_to_accounts = list(
      "us-east-1"= "522234722520",
      "us-east-2"= "566113047672",
      "us-west-2"= "156387875391",
      "eu-west-1"= "224300973850",
      "eu-central-1"= "495149712605",
      "ap-northeast-1"= "633353088612",
      "ap-northeast-2"= "204372634319",
      "ap-southeast-2"= "514117268639",
      "us-gov-west-1"= "226302683700",
      "ap-southeast-1"= "475088953585",
      "ap-south-1"= "991648021394",
      "ca-central-1"= "469771592824",
      "eu-west-2"= "644912444149",
      "us-west-1"= "632365934929",
      "us-iso-east-1"= "490574956308",
      "ap-east-1"= "286214385809",
      "eu-north-1"= "669576153137",
      "eu-west-3"= "749696950732",
      "sa-east-1"= "855470959533",
      "me-south-1"= "249704162688",
      "cn-north-1"= "390948362332",
      "cn-northwest-1"= "387376663083")
  } else if (algorithm %in% c("xgboost",
                            "seq2seq",
                            "image-classification",
                            "blazingtext",
                            "object-detection",
                            "semantic-segmentation")){
    region_to_accounts = list(
      "us-east-1"= "811284229777",
      "us-east-2"= "825641698319",
      "us-west-2"= "433757028032",
      "eu-west-1"= "685385470294",
      "eu-central-1"= "813361260812",
      "ap-northeast-1"= "501404015308",
      "ap-northeast-2"= "306986355934",
      "ap-southeast-2"= "544295431143",
      "us-gov-west-1"= "226302683700",
      "ap-southeast-1"= "475088953585",
      "ap-south-1"= "991648021394",
      "ca-central-1"= "469771592824",
      "eu-west-2"= "644912444149",
      "us-west-1"= "632365934929",
      "us-iso-east-1"= "490574956308",
      "ap-east-1"= "286214385809",
      "eu-north-1"= "669576153137",
      "eu-west-3"= "749696950732",
      "sa-east-1"= "855470959533",
      "me-south-1"= "249704162688",
      "cn-north-1"= "390948362332",
      "cn-northwest-1"= "387376663083")
  } else if (algorithm %in% c("image-classification-neo", "xgboost-neo")){
    region_to_accounts = NEO_IMAGE_ACCOUNT
  } else {
    stop(sprintf("Algorithm class:%s does not have mapping to account_id with images",algorithm), call.=F)
  }

  if (region_name %in% names(region_to_accounts)){
    account_id = region_to_accounts[[region_name]]
    return (get_ecr_image_uri_prefix(account_id, region_name))
  }

  stop(sprintf("Algorithm (%s) is unsupported for region (%s).", algorithm, region_name), call. = F)
}


#' Return algorithm image URI for the given AWS region, repository name, and
#' repository version
#' @param region_name (str): The region name for the account.
#' @param repo_name (str): The repo name for the account
#' @param repo_version (str): Version fo repo to call
#' @export
get_image_uri <- function(region_name, repo_name, repo_version = "1.0-1"){
  stopifnot(is.character(region_name), is.character(repo_name), is.character(repo_version))
  log_warn(
    "'get_image_uri' method will be deprecated in favor of 'ImageURIProvider' class to align with SageMaker Python SDK v2."
  )

  if(repo_name == XGBOOST_NAME){
    if(repo_version %in% XGBOOST_1P_VERSIONS){
      .warn_newer_xgboost_image()
      return(sprintf("%s/%s:%s", registry(region_name, repo_name), repo_name, repo_version))
    }

    if(!grepl("-", repo_version)) {
      xgboost_version_matches = lapply(XGBOOST_SUPPORTED_VERSIONS, function(version) split_str(version, split = "-"))
      xgboost_version_matches = xgboost_version_matches[sapply(xgboost_version_matches, function(x) x[1] == repo_version)]

      if(length(xgboost_version_matches) > 0) {
        # Assumes that XGBOOST_SUPPORTED_VERSION is sorted from oldest version to latest.
        # When SageMaker version is not specified, use the oldest one that matches
        # XGBoost version for backward compatibility.
        repo_version = xgboost_version_matches[1]}
    }

    supported_framework_versions = sapply(XGBOOST_SUPPORTED_VERSIONS, .generate_version_equivalents)
    supported_framework_versions = supported_framework_versions[sapply(supported_framework_versions, function(x) repo_version %in% x)]

    if (length(supported_framework_versions) == 0){
      stop(sprintf("SageMaker XGBoost version %s is not supported. Supported versions: %s",
                   repo_version, paste(XGBOOST_SUPPORTED_VERSIONS, collapse = ", ")), call. = F)
    }

    if (!.is_latest_xgboost_version(repo_version)) .warn_newer_xgboost_image()

    return(get_xgboost_image_uri(region_name, unlist(supported_framework_versions)[length(unlist(supported_framework_versions))]))
  }

  repo = sprintf("%s:%s", repo_name, repo_version)
  return (sprintf("%s/%s",registry(region_name, repo_name), repo))
}

#' Return algorithm image URI for the given ecr repository
#' @description Decided to help R users integrate "bring your own R models" in sagemaker
#' @param region_name (str): The region name for the account.
#' @param repo_version (str): Version fo repo to call
#' @param sagemaker_session (sagemaker.session.Session): Session object which
#'              manages interactions with Amazon SageMaker APIs and any other
#'              AWS services needed. If not specified, the estimator creates one
#'              using the default AWS configuration chain.
#' @export
get_ecr_image_uri = function(repo_name, repo_version = NULL, sagemaker_session = NULL){
  stopifnot(is.character(repo_name),
            is.character(repo_version) || is.null(repo_version),
            is.null(sagemaker_session) || inherits(sagemaker_session, "session"))

  session = sagemaker_session %||% Session$new()

  ecr = paws::ecr(config = session$paws_credentials$credentials)

  nextToken = NULL
  repos = list()
  # get list of repositories in ecr
  while (!identical(nextToken, character(0))){
    repo_chunk = ecr$describe_repositories()
    repo_dt = rbindlist(repo_chunk$repositories)
    repos = list(repos, repo_dt)
    nextToken = repo_chunk$nextToken
  }
  repos = rbindlist(repos)

  # check if repo_name exists in registered ecr repositories
  if(nrow(repos[repositoryName == repo_name]) == 0)
    stop(sprintf("Custom repository %s doesn't exist in AWS ECR", repo_name))

  # after repo_name check only use repo_name
  repos = repos[repositoryName == repo_name]

  nextToken = NULL
  image_meta = list()
  # get all tags from repository
  while(!identical(nextToken, character(0))){
    image_chunk = ecr$describe_images(repos[, registryId], repos[, repositoryName])
    nextToken = image_chunk$nextToken
    image_chunk = lapply(image_chunk$imageDetails, function(x)
      data.table(imageTags = x$imageTags, imagePushedAt= x$imagePushedAt))
    image_meta = c(image_meta, image_chunk)
  }
  image_meta = rbindlist(image_meta)

  # check if repo_version matches existing tags
  if(!is.null(repo_version) && nrow(image_meta[imageTags == repo_version]) == 0)
    stop(sprintf("Repository version %s doesn't exist", repo_version))

  if(is.null(repo_version)) repo_version = image_meta[order(-imagePushedAt)][1,imageTags]

  paste(repos$repositoryUri, repo_version, sep = ":")
}

.is_latest_xgboost_version <- function(repo_version){
  # Compare xgboost image version with latest version
  if(repo_version %in% XGBOOST_1P_VERSIONS) return(FALSE)
  return(repo_version %in% unlist(.generate_version_equivalents(XGBOOST_LATEST_VERSION)))
}

.warn_newer_xgboost_image <- function(){
  log_warn(sprintf(paste0("There is a more up to date SageMaker XGBoost image. ",
                   "To use the newer image, please set 'repo_version'=",
                   "'%s'.\nFor example:",
                   "\tget_image_uri(region, '%s', '%s')."),XGBOOST_LATEST_VERSION,
                   XGBOOST_NAME, XGBOOST_LATEST_VERSION))
}

.generate_version_equivalents <- function(version){
  # Returns a list of version equivalents for XGBoost
  lapply(XGBOOST_VERSION_EQUIVALENTS, function(suffix) c(paste0(version, suffix), version))
}
