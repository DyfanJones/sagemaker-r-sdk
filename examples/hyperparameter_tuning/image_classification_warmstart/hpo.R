#########################################################################
# This example is an adaptation from
# https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/image_classification_warmstart/hpo_image_classification_warmstart.ipynb
#
#########################################################################

#########################################################################
# Packages required
#########################################################################
library(R6sagemaker)
library(data.table)
library(fastDummies)

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
prefix = 'sagemaker/DEMO-hpo-image-classification'

# To do!!!!
training_image = get_image_uri(sess.boto_region_name, 'image-classification', repo_version="1")
print (training_image)

#########################################################################
# Download data from AWS sample data 
#########################################################################
download.file('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec', 'train.rec')
download.file('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec', 'validation.rec')

# Upload into S3 
sess$upload_data(path = "train.rec", bucket = bucket, key_prefix = file.path(prefix, 'train'))
sess$upload_data(path = "validation.rec", bucket = bucket, key_prefix = file.path(prefix, 'validation'))

#########################################################################
# Set up Hyperparameter Tuning
#########################################################################

# set up initial estimator
imageclassification = Estimator$new(container,
                    role,
                    train_instance_count=1,
                    train_instance_type='ml.p3.2xlarge', # May need changing for our DSTK setup 
                    output_path=sprintf('s3://%s/%s/output',bucket, prefix),
                    sagemaker_session=sess)

# add static hyperparameters
imageclassification$set_hyperparameters(num_layers=18,
                                        image_shape='3,224,224',
                                        num_classes=257,
                                        num_training_samples=15420,
                                        mini_batch_size=128,
                                        epochs=10,
                                        optimizer='sgd',
                                        top_k='2',
                                        precision_dtype='float32',
                                        augmentation_type='crop')

# set up hyperparameter ranges
hyperparameter_ranges = list('learning_rate'= ContinuousParameter$new(0.0001, 0.05),
                             'momentum'= ContinuousParameter$new(0.0, 0.99),
                             'weight_decay'= ContinuousParameter$new(0.0, 0.99))

objective_metric_name = 'validation:accuracy'

tuner = HyperparameterTuner$new(imageclassification,
                                objective_metric_name,
                                hyperparameter_ranges,
                                objective_type='Maximize',
                                max_jobs=5,
                                max_parallel_jobs=2)


#########################################################################
# Launch Hyperparameter Tuning
#########################################################################
s3_input_train = s3_input$new(s3_data=sprintf('s3://%s/%s/train',bucket, prefix), content_type='csv')
s3_input_validation = s3_input$new(s3_data=sprintf('s3://%s/%s/validation/',bucket, prefix), content_type='csv')

tuner$fit(list('train'= s3_input_train, 'validation'= s3_input_validation), include_cls_metadata=FALSE)

# Optional: wait for hyperparameter job to finish
tuner$wait()

#########################################################################
# Analyse Hyperparametering Tuning
#########################################################################
tuner_analyse = HyperparameterTuningJobAnalytics$new(tuner$latest_tuning_job)



# Include the plotting section from 



#########################################################################
# Warm start 
#########################################################################



