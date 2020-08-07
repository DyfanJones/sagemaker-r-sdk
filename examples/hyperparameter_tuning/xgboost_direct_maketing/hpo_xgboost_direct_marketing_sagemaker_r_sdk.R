#########################################################################
# This example is an adaptation from
# https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/xgboost_direct_marketing/hpo_xgboost_direct_marketing_sagemaker_python_sdk.ipynb
#
#########################################################################

#########################################################################
# packages required
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
# Preparation
#########################################################################
sess = Session$new()
region = sess$paws_region_name
role = get_execution_role(sess)
bucket = sess$default_bucket()
prefix = 'sagemaker/DEMO-hpo-xgboost-dm'
#########################################################################
# Download data from AWS sample data
#########################################################################
download.file('https://sagemaker-sample-data-us-west-2.s3-us-west-2.amazonaws.com/autopilot/direct_marketing/bank-additional.zip', "bank-addition.zip")
unzip("bank-addition.zip")

# read in data
data = fread("bank-additional/bank-additional-full.csv")
#########################################################################
# Data Transformation
#########################################################################
# Format the data:
# Indicator variable to capture when pdays takes a value of 999
data[,no_previous_contact := fifelse(pdays == 999, 1,0)]

# Indicator for individuals not actively employed
data[,not_working := fifelse(job %in% c('student', 'retired', 'unemployed'), 1, 0)]

# Convert categorical variables to sets of indicators
model_data = dummy_cols(data, remove_selected_columns = T)

# remove unwanted columns
model_data = model_data[,-c('duration', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed')]

# Get train, validation and test data frames.
set.seed(42)
idx <- sample(1:3, size = nrow(model_data), replace = TRUE, prob = c(.7, .2, .1))
train_data <- model_data[idx == 1,]
validation_data <- model_data[idx == 2,]
test_data <- model_data[idx == 3,]
#########################################################################
# upload data
#########################################################################
fwrite(cbind(train_data[,'y_yes'], train_data[,-c('y_no', 'y_yes')]), "train.csv", col.names = FALSE)
fwrite(cbind(validation_data[,'y_yes'], validation_data[,-c('y_no', 'y_yes')]), "validation.csv", col.names = FALSE)

sess$upload_data(path = "train.csv", bucket = bucket, key_prefix = file.path(prefix, 'train'))
sess$upload_data(path = "validation.csv", bucket = bucket, key_prefix = file.path(prefix, 'validation'))
#########################################################################
# Set up Hyperparameter Tuning
#########################################################################
# Get xgboost container
container = get_image_uri(region, 'xgboost')

# set up initial estimator
xgb = Estimator$new(container,
                    role,
                    train_instance_count=1,
                    train_instance_type='ml.m4.xlarge',
                    output_path=sprintf('s3://%s/%s/output',bucket, prefix),
                    sagemaker_session=sess)

# add static hyperparameters
xgb$set_hyperparameters(eval_metric='auc',
                        objective='binary:logistic',
                        num_round=100,
                        rate_drop=0.3,
                        tweedie_variance_power=1.4)

# set up hyperparameter ranges
# xgboost parameters can be found: https://xgboost.readthedocs.io/en/latest/parameter.html
hyperparameter_ranges = list('eta'= ContinuousParameter$new(0, 1),
                             'min_child_weight'= ContinuousParameter$new(1, 10),
                             'alpha'= ContinuousParameter$new(0, 2),
                             'max_depth'= IntegerParameter$new(1, 10))

# Metrics Computed by the XGBoost Algorithm
# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html
objective_metric_name = 'validation:auc'

tuner = HyperparameterTuner$new(xgb,
                                objective_metric_name,
                                hyperparameter_ranges,
                                max_jobs=20,
                                max_parallel_jobs=3)

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


# return hyperparameter jobs metadata as a data frame
df = tuner_analyse$dataframe()

