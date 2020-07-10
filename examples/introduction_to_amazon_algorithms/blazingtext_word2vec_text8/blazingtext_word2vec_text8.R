#########################################################################
# This example is an adaptation from
# https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/blazingtext_word2vec_text8/blazingtext_word2vec_text8.ipynb
#
# Please look over the original example for more documentation
#########################################################################

#########################################################################
# packages required
#########################################################################
library(R6sagemaker)
library(jsonlite)
#########################################################################
# NOTE:
#
# R6sagemaker by default sets logging level to INFO.
# If you wish to change the level for the logging level please use the
# logger package, for example:
#
# >>> logger::log_threshold(logger::DEBUG)
#########################################################################

#########################################################################
# Setup
#########################################################################
sess = Session$new()
region = sess$paws_region_name
role = get_execution_role(sess)
bucket = sess$default_bucket()
prefix = 'sagemaker/DEMO-blazingtext-text8'
#########################################################################
# Data Ingestion
#########################################################################
download.file('http://mattmahoney.net/dc/text8.zip', "text8.zip")
unzip("text8.zip")

train_channel = file.path(prefix,'train')

sess$upload_data(path='text8', bucket=bucket, key_prefix=train_channel)

s3_train_data = sprintf('s3://%s/%s', bucket, train_channel)

s3_output_location = sprintf('s3://%s/%s/output', bucket, prefix)
#########################################################################
# Training Setup
#########################################################################
container = get_image_uri(region_name, "blazingtext", "latest")
cat(sprintf('Using SageMaker BlazingText container: %s (%s)', container, region_name))
#########################################################################
# Training the BlazingText model for generating word vectors
#########################################################################
bt_model = Estimator$new(container,
                         role,
                         train_instance_count=2,
                         train_instance_type='ml.c4.2xlarge',
                         train_volume_size = 5,
                         train_max_run = 360000,
                         input_mode= 'File',
                         output_path=s3_output_location,
                         sagemaker_session=sess)

bt_model$set_hyperparameters(mode="batch_skipgram",
                             epochs=5,
                             min_count=5,
                             sampling_threshold=0.0001,
                             learning_rate=0.05,
                             window_size=5,
                             vector_dim=100,
                             negative_samples=5,
                             batch_size=11, #  = (2*window_size + 1) (Preferred. Used only if mode is batch_skipgram)
                             evaluation=TRUE,# Perform similarity evaluation on WS-353 dataset at the end of training
                             subwords=FALSE) # Subword embedding learning is not supported by batch_skipgram

train_data = s3_input$new(s3_train_data, distribution='FullyReplicated',
                          content_type='text/plain', s3_data_type='S3Prefix')
data_channels = list('train'= train_data)

bt_model$fit(inputs=data_channels)
#########################################################################
# Hosting / Inference
#########################################################################
bt_endpoint = bt_model$deploy(initial_instance_count = 1,instance_type = 'ml.m4.xlarge')
#########################################################################
# Getting vector representations for words
# Use JSON format for inference
#########################################################################
words = c("awesome", "blazing")

payload = list("instances"= words)

# call predict directly from class
bt_endpoint$deserializer = json_deserializer
vecs = bt_endpoint$predict(toJSON(payload))

# Or call the predict s3 method
vecs = predict(bt_endpoint, toJSON(payload), NULL, json_deserializer)

#########################################################################
# Evaluation
#########################################################################
# get model_data
parsed_s3 = urltools::url_parse(bt_model$model_data)
sess$download_data("data/model", bucket = parsed_s3$domain, key_prefix = parsed_s3$path)
untar("data/model/model.tar.gz",exdir = "data/model")

fromJSON("data/model/eval.json")

# Going to use Rtsne to implement Van der Maaten’s Barnes-Hut implementation of t-Distributed Stochastic Neighbor: https://github.com/jkrijthe/Rtsne
# install.packages("Rtsne")
library(data.table) # read in data
library(Rtsne)

# data.table can handle unusual look text files (makes life easier)
dt = fread("data/model/vectors.txt")

set.seed(42) # Sets seed for reproducibility
word_vecs = normalize_input(as.matrix(dt[,-1])) # normalize data
tsne =  Rtsne(word_vecs[1:400,], perplexity = 40, dim = 2, pca = TRUE, max_iter = 10000) # fit tsne

library(ggplot2)
ggplot(data.table(tsne$Y),aes(V1, V2, label = dt[1:400][[1]], colour = unique(dt[1:400][[1]]))) +
  geom_point() + # plot scatter plot
  geom_text(hjust=0, vjust=0, size = 2.5, check_overlap = T) + # format label text
  theme_bw() + # use a theme you want (optional)
  theme(legend.position="none") + # remove colour label legend
  labs(x = "", y = "") # remove temporary x and y axis labels (optional)

