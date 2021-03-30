# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/object2vec.py

#' @include image_uris.R
#' @include amazon_estimator.R
#' @include amazon_common.R
#' @include amazon_hyperparameter.R
#' @include amazon_validation.R
#' @include predictor.R
#' @include model.R
#' @include session.R
#' @include vpc_utils.R
#' @include utils.R

#' @import R6
#' @import lgr

#' @title A general-purpose neural embedding algorithm that is highly customizable.
#' @description It can learn low-dimensional dense embeddings of high-dimensional objects. The embeddings
#'              are learned in a way that preserves the semantics of the relationship between pairs of
#'              objects in the original space in the embedding space.
#' @export
Object2Vec = R6Class("Object2Vec",
  inherit = AmazonAlgorithmEstimatorBase,
  public = list(

    #' @field repo_name
    #' sagemaker repo name for framework
    repo_name = "object2vec",

    #' @field repo_version
    #' version of framework
    repo_version = 1,

    #' @field MINI_BATCH_SIZE
    #' The size of each mini-batch to use when training.
    MINI_BATCH_SIZE = 32,

    #' @description Object2Vec is :class:`Estimator` used for anomaly detection.
    #'              This Estimator may be fit via calls to
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
    #'              There is an utility
    #'              :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.record_set`
    #'              that can be used to upload data to S3 and creates
    #'              :class:`~sagemaker.amazon.amazon_estimator.RecordSet` to be passed to
    #'              the `fit` call.
    #'              After this Estimator is fit, model data is stored in S3. The model
    #'              may be deployed to an Amazon SageMaker Endpoint by invoking
    #'              :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
    #'              deploying an Endpoint, deploy returns a
    #'              :class:`~sagemaker.amazon.Predictor` object that can be used for
    #'              inference calls using the trained model hosted in the SageMaker
    #'              Endpoint.
    #'              Object2Vec Estimators can be configured by setting hyperparameters.
    #'              The available hyperparameters for Object2Vec are documented below.
    #'              For further information on the AWS Object2Vec algorithm, please
    #'              consult AWS technical documentation:
    #'              https://docs.aws.amazon.com/sagemaker/latest/dg/object2vec.html
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if accessing AWS resource.
    #' @param instance_count (int): Number of Amazon EC2 instances to use
    #'              for training.
    #' @param instance_type (str): Type of EC2 instance to use for training,
    #'              for example, 'ml.c4.xlarge'.
    #' @param epochs (int): Total number of epochs for SGD training
    #' @param enc0_max_seq_len (int): Maximum sequence length
    #' @param enc0_vocab_size (int): Vocabulary size of tokens
    #' @param enc_dim (int): Optional. Dimension of the output of the embedding
    #'              layer
    #' @param mini_batch_size (int): Optional. mini batch size for SGD training
    #' @param early_stopping_patience (int): Optional. The allowed number of
    #'              consecutive epochs without improvement before early stopping is
    #'              applied
    #' @param early_stopping_tolerance (float): Optional. The value used to
    #'              determine whether the algorithm has made improvement between two
    #'              consecutive epochs for early stopping
    #' @param dropout (float): Optional. Dropout probability on network layers
    #' @param weight_decay (float): Optional. Weight decay parameter during
    #'              optimization
    #' @param bucket_width (int): Optional. The allowed difference between data
    #'              sequence length when bucketing is enabled
    #' @param num_classes (int): Optional. Number of classes for classification
    #' @param training (ignored for regression problems)
    #' @param mlp_layers (int): Optional. Number of MLP layers in the network
    #' @param mlp_dim (int): Optional. Dimension of the output of MLP layer
    #' @param mlp_activation (str): Optional. Type of activation function for the
    #'              MLP layer
    #' @param output_layer (str): Optional. Type of output layer
    #' @param optimizer (str): Optional. Type of optimizer for training
    #' @param learning_rate (float): Optional. Learning rate for SGD training
    #' @param negative_sampling_rate (int): Optional. Negative sampling rate
    #' @param comparator_list (str): Optional. Customization of comparator
    #'              operator
    #' @param tied_token_embedding_weight (bool): Optional. Tying of token
    #'              embedding layer weight
    #' @param token_embedding_storage_type (str): Optional. Type of token
    #'              embedding storage
    #' @param enc0_network (str): Optional. Network model of encoder "enc0"
    #' @param enc1_network (str): Optional. Network model of encoder "enc1"
    #' @param enc0_cnn_filter_width (int): Optional. CNN filter width
    #' @param enc1_cnn_filter_width (int): Optional. CNN filter width
    #' @param enc1_max_seq_len (int): Optional. Maximum sequence length
    #' @param enc0_token_embedding_dim (int): Optional. Output dimension of token
    #'              embedding layer
    #' @param enc1_token_embedding_dim (int): Optional. Output dimension of token
    #'              embedding layer
    #' @param enc1_vocab_size (int): Optional. Vocabulary size of tokens
    #' @param enc0_layers (int): Optional. Number of layers in encoder
    #' @param enc1_layers (int): Optional. Number of layers in encoder
    #' @param enc0_freeze_pretrained_embedding (bool): Optional. Freeze pretrained
    #'              embedding weights
    #' @param enc1_freeze_pretrained_embedding (bool): Optional. Freeze pretrained
    #'              embedding weights
    #' @param ... : base class keyword argument values.
    initialize = function(role,
                          instance_count,
                          instance_type,
                          epochs,
                          enc0_max_seq_len,
                          enc0_vocab_size,
                          enc_dim=NULL,
                          mini_batch_size=NULL,
                          early_stopping_patience=NULL,
                          early_stopping_tolerance=NULL,
                          dropout=NULL,
                          weight_decay=NULL,
                          bucket_width=NULL,
                          num_classes=NULL,
                          mlp_layers=NULL,
                          mlp_dim=NULL,
                          mlp_activation=NULL,
                          output_layer=NULL,
                          optimizer=NULL,
                          learning_rate=NULL,
                          negative_sampling_rate=NULL,
                          comparator_list=NULL,
                          tied_token_embedding_weight=NULL,
                          token_embedding_storage_type=NULL,
                          enc0_network=NULL,
                          enc1_network=NULL,
                          enc0_cnn_filter_width=NULL,
                          enc1_cnn_filter_width=NULL,
                          enc1_max_seq_len=NULL,
                          enc0_token_embedding_dim=NULL,
                          enc1_token_embedding_dim=NULL,
                          enc1_vocab_size=NULL,
                          enc0_layers=NULL,
                          enc1_layers=NULL,
                          enc0_freeze_pretrained_embedding=NULL,
                          enc1_freeze_pretrained_embedding=NULL,
                          ...){
      private$.enc_dim = Hyperparameter$new("enc_dim", list(Validation$new()$ge(4), Validation$new()$le(10000)), "An integer in [4, 10000]", DataTypes$new()$int, obj = self)
      private$.mini_batch_size = Hyperparameter$new("mini_batch_size", list(Validation$new()$ge(1), Validation$new()$le(10000)), "An integer in [1, 10000]", DataTypes$new()$int, obj = self)
      private$.epochs = Hyperparameter$new("epochs", list(Validation$new()$ge(1), Validation$new()$le(100)), "An integer in [1, 100]", DataTypes$new()$int, obj = self)
      private$.early_stopping_patience = Hyperparameter$new(
        "early_stopping_patience", list(Validation$new()$ge(1), Validation$new()$le(5)), "An integer in [1, 5]", DataTypes$new()$int, obj = self
      )
      private$.early_stopping_tolerance = Hyperparameter$new(
        "early_stopping_tolerance", list(Validation$new()$ge(1e-06), Validation$new()$le(0.1)), "A float in [1e-06, 0.1]", DataTypes$new()$float, obj = self
      )
      private$.dropout = Hyperparameter$new("dropout", list(Validation$new()$ge(0.0), Validation$new()$le(1.0)), "A float in [0.0, 1.0]", DataTypes$new()$float, obj = self)
      private$.weight_decay = Hyperparameter$new("weight_decay", list(Validation$new()$ge(0.0), Validation$new()$le(10000.0)), "A float in [0.0, 10000.0]", DataTypes$new()$float, obj = self)
      private$.bucket_width = Hyperparameter$new("bucket_width", list(Validation$new()$ge(0), Validation$new()$le(100)), "An integer in [0, 100]", DataTypes$new()$int, obj = self)
      private$.num_classes = Hyperparameter$new("num_classes", list(Validation$new()$ge(2), Validation$new()$le(30)), "An integer in [2, 30]", DataTypes$new()$int, obj = self)
      private$.mlp_layers = Hyperparameter$new("mlp_layers", list(Validation$new()$ge(1), Validation$new()$le(10)), "An integer in [1, 10]", DataTypes$new()$int, obj = self)
      private$.mlp_dim = Hyperparameter$new("mlp_dim", list(Validation$new()$ge(2), Validation$new()$le(10000)), "An integer in [2, 10000]", DataTypes$new()$int, obj = self)
      private$.mlp_activation = Hyperparameter$new(
        "mlp_activation", Validation$new()$isin(c("tanh", "relu", "linear")), 'One of "tanh", "relu", "linear"', DataTypes$new()$str, obj = self
      )
      private$.output_layer = Hyperparameter$new(
        "output_layer",
        Validation$new()$isin(c("softmax", "mean_squared_error")),
        'One of "softmax", "mean_squared_error"',
        DataTypes$new()$str,
        obj = self
      )
      private$.optimizer = Hyperparameter$new(
        "optimizer",
        Validation$new()$isin(c("adagrad", "adam", "rmsprop", "sgd", "adadelta")),
        'One of "adagrad", "adam", "rmsprop", "sgd", "adadelta"',
        DataTypes$new()$str,
        obj = self
      )
      private$.learning_rate = Hyperparameter$new("learning_rate", list(Validation$new()$ge(1e-06), Validation$new()$le(1.0)), "A float in [1e-06, 1.0]", DataTypes$new()$float, obj = self)
      private$.negative_sampling_rate = Hyperparameter$new(
        "negative_sampling_rate", list(Validation$new()$ge(0), Validation$new()$le(100)), "An integer in [0, 100]", DataTypes$new()$int, obj = self
      )
      private$.comparator_list = Hyperparameter$new(
        "comparator_list",
        private$.list_check_subset(c("hadamard", "concat", "abs_diff")),
        'Comma-separated of hadamard, concat, abs_diff. E.g. "hadamard,abs_diff"',
        DataTypes$new()$str,
        obj = self
      )
      private$.tied_token_embedding_weight = Hyperparameter$new(
        "tied_token_embedding_weight", list(), "Either True or False", DataTypes$new()$bool, obj = self
      )
      private$.token_embedding_storage_type = Hyperparameter$new(
        "token_embedding_storage_type",
        Validation$new()$isin(c("dense", "row_sparse")),
        'One of "dense", "row_sparse"',
        DataTypes$new()$str,
        obj = self
      )
      private$.enc0_network = Hyperparameter$new(
        "enc0_network",
        Validation$new()$isin(c("hcnn", "bilstm", "pooled_embedding")),
        'One of "hcnn", "bilstm", "pooled_embedding"',
        DataTypes$new()$str,
        obj = self
      )
      private$.enc1_network = Hyperparameter$new(
        "enc1_network",
        Validation$new()$isin(c("hcnn", "bilstm", "pooled_embedding", "enc0")),
        'One of "hcnn", "bilstm", "pooled_embedding", "enc0"',
        DataTypes$new()$str,
        obj = self
      )
      private$.enc0_cnn_filter_width = Hyperparameter$new("enc0_cnn_filter_width", list(Validation$new()$ge(1), Validation$new()$le(9)), "An integer in [1, 9]", DataTypes$new()$int, obj = self)
      private$.enc1_cnn_filter_width = Hyperparameter$new("enc1_cnn_filter_width", list(Validation$new()$ge(1), Validation$new()$le(9)), "An integer in [1, 9]", DataTypes$new()$int, obj = self)
      private$.enc0_max_seq_len = Hyperparameter$new("enc0_max_seq_len", list(Validation$new()$ge(1), Validation$new()$le(5000)), "An integer in [1, 5000]", DataTypes$new()$int, obj = self)
      private$.enc1_max_seq_len = Hyperparameter$new("enc1_max_seq_len", list(Validation$new()$ge(1), Validation$new()$le(5000)), "An integer in [1, 5000]", DataTypes$new()$int, obj = self)
      private$.enc0_token_embedding_dim = Hyperparameter$new(
        "enc0_token_embedding_dim", list(Validation$new()$ge(2), Validation$new()$le(1000)), "An integer in [2, 1000]", DataTypes$new()$int, obj = self
      )
      private$.enc1_token_embedding_dim = Hyperparameter$new(
        "enc1_token_embedding_dim", list(Validation$new()$ge(2), Validation$new()$le(1000)), "An integer in [2, 1000]", DataTypes$new()$int, obj = self
      )
      private$.enc0_vocab_size = Hyperparameter$new("enc0_vocab_size", list(Validation$new()$ge(2), Validation$new()$le(3000000)), "An integer in [2, 3000000]", DataTypes$new()$int, obj = self)
      private$.enc1_vocab_size = Hyperparameter$new("enc1_vocab_size", list(Validation$new()$ge(2), Validation$new()$le(3000000)), "An integer in [2, 3000000]", DataTypes$new()$int, obj = self)
      private$.enc0_layers = Hyperparameter$new("enc0_layers", list(Validation$new()$ge(1), Validation$new()$le(4)), "An integer in [1, 4]", DataTypes$new()$int, obj = self)
      private$.enc1_layers = Hyperparameter$new("enc1_layers", list(Validation$new()$ge(1), Validation$new()$le(4)), "An integer in [1, 4]", DataTypes$new()$int, obj = self)
      private$.enc0_freeze_pretrained_embedding = Hyperparameter$new(
        "enc0_freeze_pretrained_embedding", list(), "Either True or False", DataTypes$new()$bool, obj = self
      )
      private$.enc1_freeze_pretrained_embedding = Hyperparameter$new(
        "enc1_freeze_pretrained_embedding", list(), "Either True or False", DataTypes$new()$bool, obj = self
      )

      super$initialize(role, instance_count, instance_type, ...)

      self$enc_dim = enc_dim
      self$mini_batch_size = mini_batch_size
      self$epochs = epochs
      self$early_stopping_patience = early_stopping_patience
      self$early_stopping_tolerance = early_stopping_tolerance
      self$dropout = dropout
      self$weight_decay = weight_decay
      self$bucket_width = bucket_width
      self$num_classes = num_classes
      self$mlp_layers = mlp_layers
      self$mlp_dim = mlp_dim
      self$mlp_activation = mlp_activation
      self$output_layer = output_layer
      self$optimizer = optimizer
      self$learning_rate = learning_rate

      self$negative_sampling_rate = negative_sampling_rate
      self$comparator_list = comparator_list
      self$tied_token_embedding_weight = tied_token_embedding_weight
      self$token_embedding_storage_type = token_embedding_storage_type

      self$enc0_network = enc0_network
      self$enc1_network = enc1_network
      self$enc0_cnn_filter_width = enc0_cnn_filter_width
      self$enc1_cnn_filter_width = enc1_cnn_filter_width
      self$enc0_max_seq_len = enc0_max_seq_len
      self$enc1_max_seq_len = enc1_max_seq_len
      self$enc0_token_embedding_dim = enc0_token_embedding_dim
      self$enc1_token_embedding_dim = enc1_token_embedding_dim
      self$enc0_vocab_size = enc0_vocab_size
      self$enc1_vocab_size = enc1_vocab_size
      self$enc0_layers = enc0_layers
      self$enc1_layers = enc1_layers
      self$enc0_freeze_pretrained_embedding = enc0_freeze_pretrained_embedding
      self$enc1_freeze_pretrained_embedding = enc1_freeze_pretrained_embedding
    },

    #' @description Return a :class:`~sagemaker.amazon.Object2VecModel` referencing the
    #'              latest s3 model data produced by this Estimator.
    #' @param vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
    #'              the model. Default: use subnets and security groups from this Estimator.
    #'              * 'Subnets' (list[str]): List of subnet ids.
    #'              * 'SecurityGroupIds' (list[str]): List of security group ids.
    #' @param ... : Additional kwargs passed to the Object2VecModel constructor.
    create_model = function(vpc_config_override="VPC_CONFIG_DEFAULT",
                            ...){
      return(Object2VecModel$new(
        self$model_data,
        self$role,
        sagemaker_session=self$sagemaker_session,
        vpc_config=self$get_vpc_config(vpc_config_override),
        ...
        )
      )
    },

    #' @description Set hyperparameters needed for training. This method will also
    #'              validate ``source_dir``.
    #' @param records (RecordSet) – The records to train this Estimator on.
    #' @param mini_batch_size (int or None) – The size of each mini-batch to use
    #'              when training. If None, a default value will be used.
    #' @param job_name (str): Name of the training job to be created. If not
    #'              specified, one is generated, using the base name given to the
    #'              constructor if applicable.
    .prepare_for_training = function(records,
                                     mini_batch_size=NULL,
                                     job_name=NULL){
      if (is.null(mini_batch_size))
        mini_batch_size = self$MINI_BATCH_SIZE

        super$.prepare_for_training(
          records, mini_batch_size=mini_batch_size, job_name=job_name
        )
    }
  ),
  private = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    .epochs=NULL,
    .enc_dim=NULL,
    .mini_batch_size=NULL,
    .early_stopping_patience=NULL,
    .early_stopping_tolerance=NULL,
    .dropout=NULL,
    .weight_decay=NULL,
    .bucket_width=NULL,
    .num_classes=NULL,
    .mlp_layers=NULL,
    .mlp_dim=NULL,
    .mlp_activation=NULL,
    .output_layer=NULL,
    .optimizer=NULL,
    .learning_rate=NULL,
    .negative_sampling_rate=NULL,
    .comparator_list=NULL,
    .tied_token_embedding_weight=NULL,
    .token_embedding_storage_type=NULL,
    .enc0_network=NULL,
    .enc1_network=NULL,
    .enc0_cnn_filter_width=NULL,
    .enc1_cnn_filter_width=NULL,
    .enc0_max_seq_len=NULL,
    .enc1_max_seq_len=NULL,
    .enc0_token_embedding_dim=NULL,
    .enc1_token_embedding_dim=NULL,
    .enc0_vocab_size=NULL,
    .enc1_vocab_size=NULL,
    .enc0_layers=NULL,
    .enc1_layers=NULL,
    .enc0_freeze_pretrained_embedding=NULL,
    .enc1_freeze_pretrained_embedding=NULL,
    # ---------

    .list_check_subset = function(valid_super_list){
      valid_superset = unique(valid_super_list)
      validate = function(value){
        if(!inherits(value, "character"))
          return(FALSE)

        val_list = lapply(split_str(value), trimws)
        return(all(val_list %in% valid_superset))
      }
      return(validate)
    }
  ),
  active = list(
    # --------- User Active binding to mimic Python's Descriptor Class ---------
    #' @field epochs
    #' Total number of epochs for SGD training
    epochs = function(value){
      if(missing(value))
        return(private$.epochs$descriptor)
      private$.epochs$descriptor = value
    },

    #' @field enc_dim
    #' Dimension of the output of the embedding layer
    enc_dim = function(value){
      if(missing(value))
        return(private$.enc_dim$descriptor)
      private$.enc_dim$descriptor = value
    },

    #' @field mini_batch_size
    #' mini batch size for SGD training
    mini_batch_size = function(value){
      if(missing(value))
        return(private$.mini_batch_size$descriptor)
      private$.mini_batch_size$descriptor = value
    },

    #' @field early_stopping_patience
    #' The allowed number of consecutive epochs without
    #'        improvement before early stopping is applied
    early_stopping_patience = function(value){
      if(missing(value))
        return(private$.early_stopping_patience$descriptor)
      private$.early_stopping_patience$descriptor = value
    },

    #' @field early_stopping_tolerance
    #' The value used to determine whether the algorithm
    #'        has made improvement between two consecutive epochs for early stopping
    early_stopping_tolerance = function(value){
      if(missing(value))
        return(private$.early_stopping_tolerance$descriptor)
      private$.early_stopping_tolerance$descriptor = value
    },

    #' @field dropout
    #' Dropout probability on network layers
    dropout = function(value){
      if(missing(value))
        return(private$.dropout$descriptor)
      private$.dropout$descriptor = value
    },

    #' @field weight_decay
    #' Weight decay parameter during optimization
    weight_decay = function(value){
      if(missing(value))
        return(private$.weight_decay$descriptor)
      private$.weight_decay$descriptor = value
    },

    #' @field bucket_width
    #' The allowed difference between data sequence length when bucketing is enabled
    bucket_width = function(value){
      if(missing(value))
        return(private$.bucket_width$descriptor)
      private$.bucket_width$descriptor = value
    },

    #' @field num_classes
    #' Number of classes for classification
    num_classes = function(value){
      if(missing(value))
        return(private$.num_classes$descriptor)
      private$.num_classes$descriptor = value
    },

    #' @field mlp_layers
    #' Number of MLP layers in the network
    mlp_layers = function(value){
      if(missing(value))
        return(private$.mlp_layers$descriptor)
      private$.mlp_layers$descriptor = value
    },

    #' @field mlp_dim
    #' Dimension of the output of MLP layer
    mlp_dim = function(value){
      if(missing(value))
        return(private$.mlp_dim$descriptor)
      private$.mlp_dim$descriptor = value
    },

    #' @field mlp_activation
    #' Type of activation function for the MLP layer
    mlp_activation = function(value){
      if(missing(value))
        return(private$.mlp_activation$descriptor)
      private$.mlp_activation$descriptor = value
    },

    #' @field output_layer
    #' Type of output layer
    output_layer = function(value){
      if(missing(value))
        return(private$.output_layer$descriptor)
      private$.output_layer$descriptor = value
    },

    #' @field optimizer
    #' Type of optimizer for training
    optimizer = function(value){
      if(missing(value))
        return(private$.optimizer$descriptor)
      private$.optimizer$descriptor = value
    },

    #' @field learning_rate
    #' Learning rate for SGD training
    learning_rate = function(value){
      if(missing(value))
        return(private$.learning_rate$descriptor)
      private$.learning_rate$descriptor = value
    },

    #' @field negative_sampling_rate
    #' Negative sampling rate
    negative_sampling_rate = function(value){
      if(missing(value))
        return(private$.negative_sampling_rate$descriptor)
      private$.negative_sampling_rate$descriptor = value
    },

    #' @field comparator_list
    #' Customization of comparator operator
    comparator_list = function(value){
      if(missing(value))
        return(private$.comparator_list$descriptor)
      private$.comparator_list$descriptor = value
    },

    #' @field tied_token_embedding_weight
    #' Tying of token embedding layer weight
    tied_token_embedding_weight = function(value){
      if(missing(value))
        return(private$.tied_token_embedding_weight$descriptor)
      private$.tied_token_embedding_weight$descriptor = value
    },

    #' @field token_embedding_storage_type
    #' Type of token embedding storage
    token_embedding_storage_type = function(value){
      if(missing(value))
        return(private$.token_embedding_storage_type$descriptor)
      private$.token_embedding_storage_type$descriptor = value
    },

    #' @field enc0_network
    #' Network model of encoder "enc0"
    enc0_network = function(value){
      if(missing(value))
        return(private$.enc0_network$descriptor)
      private$.enc0_network$descriptor = value
    },

    #' @field enc1_network
    #' Network model of encoder "enc1"
    enc1_network = function(value){
      if(missing(value))
        return(private$.enc1_network$descriptor)
      private$.enc1_network$descriptor = value
    },

    #' @field enc0_cnn_filter_width
    #' CNN filter width
    enc0_cnn_filter_width = function(value){
      if(missing(value))
        return(private$.enc0_cnn_filter_width$descriptor)
      private$.enc0_cnn_filter_width$descriptor = value
    },

    #' @field enc1_cnn_filter_width
    #' CNN filter width
    enc1_cnn_filter_width = function(value){
      if(missing(value))
        return(private$.enc1_cnn_filter_width$descriptor)
      private$.enc1_cnn_filter_width$descriptor = value
    },

    #' @field enc0_max_seq_len
    #' Maximum sequence length
    enc0_max_seq_len = function(value){
      if(missing(value))
        return(private$.enc0_max_seq_len$descriptor)
      private$.enc0_max_seq_len$descriptor = value
    },

    #' @field enc1_max_seq_len
    #' Maximum sequence length
    enc1_max_seq_len = function(value){
      if(missing(value))
        return(private$.enc1_max_seq_len$descriptor)
      private$.enc1_max_seq_len$descriptor = value
    },

    #' @field enc0_token_embedding_dim
    #' Output dimension of token embedding layer
    enc0_token_embedding_dim = function(value){
      if(missing(value))
        return(private$.enc0_token_embedding_dim$descriptor)
      private$.enc0_token_embedding_dim$descriptor = value
    },

    #' @field enc1_token_embedding_dim
    #' Output dimension of token embedding layer
    enc1_token_embedding_dim = function(value){
      if(missing(value))
        return(private$.enc1_token_embedding_dim$descriptor)
      private$.enc1_token_embedding_dim$descriptor = value
    },

    #' @field enc0_vocab_size
    #' Vocabulary size of tokens
    enc0_vocab_size = function(value){
      if(missing(value))
        return(private$.enc0_vocab_size$descriptor)
      private$.enc0_vocab_size$descriptor = value
    },

    #' @field enc1_vocab_size
    #' Vocabulary size of tokens
    enc1_vocab_size = function(value){
      if(missing(value))
        return(private$.enc1_vocab_size$descriptor)
      private$.enc1_vocab_size$descriptor = value
    },

    #' @field enc0_layers
    #' Number of layers in encoder
    enc0_layers = function(value){
      if(missing(value))
        return(private$.enc0_layers$descriptor)
      private$.enc0_layers$descriptor = value
    },

    #' @field enc1_layers
    #' Number of layers in encoder
    enc1_layers = function(value){
      if(missing(value))
        return(private$.enc1_layers$descriptor)
      private$.enc1_layers$descriptor = value
    },

    #' @field enc0_freeze_pretrained_embedding
    #' Freeze pretrained embedding weights
    enc0_freeze_pretrained_embedding = function(value){
      if(missing(value))
        return(private$.enc0_freeze_pretrained_embedding$descriptor)
      private$.enc0_freeze_pretrained_embedding$descriptor = value
    },

    #' @field enc1_freeze_pretrained_embedding
    #' Freeze pretrained embedding weights
    enc1_freeze_pretrained_embedding = function(value){
      if(missing(value))
        return(private$.enc1_freeze_pretrained_embedding$descriptor)
      private$.enc1_freeze_pretrained_embedding$descriptor = value
    }
  ),
  lock_objects = F
)

#' @title Reference Object2Vec s3 model data.
#' @description Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and returns a
#'              Predictor that calculates anomaly scores for datapoints.
#' @export
Object2VecModel = R6Class("Object2VecModel",
  inherit = Model,
  public = list(

    #' @description Initialize Object2VecModel class
    #' @param model_data (str): The S3 location of a SageMaker model data
    #'              ``.tar.gz`` file.
    #' @param role (str): An AWS IAM role (either name or full ARN). The Amazon
    #'              SageMaker training jobs and APIs that create Amazon SageMaker
    #'              endpoints use this role to access training data and model
    #'              artifacts. After the endpoint is created, the inference code
    #'              might use the IAM role, if it needs to access an AWS resource.
    #' @param sagemaker_session (sagemaker.session.Session): Session object which
    #'              manages interactions with Amazon SageMaker APIs and any other
    #'              AWS services needed. If not specified, the estimator creates one
    #'              using the default AWS configuration chain.
    #' @param ... : Keyword arguments passed to the ``FrameworkModel``
    #'              initializer.
    initialize = function(model_data,
                          role,
                          sagemaker_session=NULL,
                          ...){
      sagemaker_session = sagemaker_session %||% Session$new()
      image_uri = ImageUris$new()$retrieve(
        Object2Vec$public_fields$repo_name,
        sagemaker_session$paws_region_name,
        version=Object2Vec$public_fields$repo_version
      )
      super$initialize(
        image_uri,
        model_data,
        role,
        predictor_cls=Predictor,
        sagemaker_session=sagemaker_session,
        ...
      )
    }
  ),
  lock_objects = F
)
