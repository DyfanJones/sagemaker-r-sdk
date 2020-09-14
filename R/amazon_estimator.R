# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/af7f75ae336f0481e52bb968e4cc6df91b1bac2c/src/sagemaker/amazon/amazon_estimator.py

#' @include utils.R
#' @include estimator.R
#' @include fw_registry.R
#' @include s3.R
#' @include amazon_hyperparameter.R
#' @include amazon_validation.R
#' @include amazon_common.R
#' @include image_uris.R

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

    #' @description Initialize an AmazonAlgorithmEstimatorBase.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if it needs to access an AWS resource.
    #' @param instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param instance_type (str): Type of EC2 instance to use for training,
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
                          instance_count,
                          instance_type,
                          data_location=NULL,
                          enable_network_isolation=FALSE,
                          ...){
      super$initialize(role = role,
                       instance_count = instance_count,
                       instance_type = instance_type,
                       enable_network_isolation=enable_network_isolation,
                       ...)

      data_location = data_location %||% sprintf("s3://%s/sagemaker-record-sets/", self$sagemaker_session$default_bucket())
      private$.feature_dim = Hyperparameter$new("feature_dim", Validation$new()$gt(0), data_type=DataTypes$new()$int, obj = self)
      private$.mini_batch_size = Hyperparameter$new("mini_batch_size", Validation$new()$gt(0), data_type=DataTypes$new()$int, obj = self)
      self$.data_location = data_location
    },

    #' @description Return algorithm image URI for the given AWS region, repository name, and
    #'              repository version
    training_image_uri = function(){
      image_uri = ImageUris$new(self$sagemaker_session)
      return(image_uri$retrieve(self$repo_name, version = self$repo_version))
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
    #'              ``instance_count`` property on this Estimator. One S3 object is
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
      if(is.vector(train)) train = as.array(train)
      parsed_s3_url = url_parse(self$data_location)
      bucket = parsed_s3_url$domain
      key_prefix = parsed_s3_url$path
      key_prefix = paste0(key_prefix, sprintf("%s-%s/", class(self)[1], sagemaker_timestamp()))
      key_prefix = trimws(key_prefix, "left", "/")
      log_debug("Uploading to bucket %s and key_prefix %s", bucket, key_prefix)
      manifest_s3_file = upload_matrix_to_s3_shards(
        self$instance_count, self$sagemaker_session$s3, bucket, key_prefix, train, labels, encrypt
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

    # --------- User Active binding to mimic Python's Descriptor Class ---------

    #' @field feature_dim
    #' Hyperparameter class for feature_dim
    feature_dim = function(value){
      if(missing(value))
        return(private$.feature_dim$descriptor)
      private$.feature_dim$descriptor = value
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
    # --------- initializing private objects of r python descriptor class ---------
    .feature_dim = NULL,
    .mini_batch_size = NULL,

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

      config = .Job$private_methods$.load_config(inputs, self)

      if (!islistempty(self$hyperparameters())){
        hyperparameters = self$hyperparameters()}


      train_args = config
      train_args[["input_mode"]] = self$input_mode
      train_args[["job_name"]] = self$.current_job_name
      train_args[["hyperparameters"]] = hyperparameters
      train_args[["tags"]] = self$tags
      train_args[["metric_definitions"]] = self$metric_definitions
      train_args[["experiment_config"]] = experiment_config

      if (inherits(inputs, "TrainingInput")){
        if ("InputMode" %in% inputs$config){
          log_debug("Selecting TrainingInput's input_mode (%s) for TrainingInputMode.",
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
        train_args[["image"]] = self$training_image_uri()}


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
    initialize = function(s3_data,
                          num_records,
                          feature_dim,
                          s3_data_type="ManifestFile",
                          channel="train"){
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

    #' @description Return a TrainingInput to represent the training data
    recods_s3_input = function(){
      return(TrainingInput$new(self$s3_data, distribution="ShardedByS3Key", s3_data_type=self$s3_data_type))
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

# needs to be compatible with vectors
.build_shards = function(num_shards,
                         array){
  if (num_shards < 1)
    stop("num_shards must be >= 1")

  if(is.vector(array))
    array = as.array(array)
  # ensure matrix gets split into same number of shards
  shard_size = ceiling(dim(array)[1] / num_shards)
  if (shard_size == 0)
    stop("Array length is less than num shards")

  max_row = dim(array)[1]
  split_vec <- seq(1, max_row, shard_size)

  if(length(dim(array)) == 1)
    lapply(split_vec, function(i) array[i:min(max_row,(i+shard_size-1))])
  else
    lapply(split_vec, function(i) array[i:min(max_row,(i+shard_size-1)),])
}

# Upload the training ``array`` and ``labels`` arrays to ``num_shards`` S3
# objects, stored in "s3:// ``bucket`` / ``key_prefix`` /". Optionally
# ``encrypt`` the S3 objects using AES-256.
upload_matrix_to_s3_shards = function(num_shards,
                                      s3,
                                      bucket,
                                      key_prefix,
                                      array,
                                      labels=NULL,
                                      encrypt=FALSE){
  # initialise protobuf
  initProtoBuf()

  shards = .build_shards(num_shards, array)

  if (!is.null(labels))
    label_shards = .build_shards(num_shards, labels)
  uploaded_files = list()
  if (!grepl("/$", key_prefix))
    key_prefix = paste0(key_prefix, "/")
  extra_put_kwargs = if(encrypt) list("ServerSideEncryption"= "AES256") else list()

  tryCatch({
    for(shard_index in seq_along(shards)){
      obj = raw(0)
      buf = rawConnection(obj, open = "wb")
      if (!is.null(labels))
        write_matrix_to_dense_tensor(buf, shards[[shard_index]], label_shards[[shard_index]])
      else
        write_matrix_to_dense_tensor(buf, shards[[shard_index]])

      obj = rawConnectionValue(buf)
      close(buf)

      shard_index_string = formatC(shard_index, width = nchar(nrow(shards[[shard_index]])), format = "d", flag = "0")
      file_name = sprintf("matrix_%s.pbr", shard_index_string)
      key = paste0(key_prefix, file_name)
      log_debug("Creating object %s in bucket %s", key, bucket)
      # Upload shard to s3
      s3$put_object(Bucket = bucket, Key = key, Body = obj, ServerSideEncryption = extra_put_kwargs$extra_put_kwargs)
      # update uploaded files
      uploaded_files = c(uploaded_files, file_name)
    }
    manifest_key = paste0(key_prefix,".amazon.manifest")
    manifest_str = toJSON(c(list(list("prefix" = sprintf("s3://%s/%s", bucket, key_prefix))), uploaded_files),
                          auto_unbox = T)
    s3$put_object(Bucket = bucket, Key = manifest_key,
                  Body = charToRaw(manifest_str),
                  ServerSideEncryption = extra_put_kwargs$extra_put_kwargs)
    return(sprintf("s3://%s/%s",bucket, manifest_key))
  },
  error = function(e){
    tryCatch({for(file in uploaded_files) s3$delete_object(bucket, paste0(key_prefix, file))},
             finally = function(f) stop(f, call. = F))
    }
  )
}
