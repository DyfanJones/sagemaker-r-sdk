#########################################################################
# This example is an adaptation from
# https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_applying_machine_learning/xgboost_direct_marketing/xgboost_direct_marketing_sagemaker.ipynb
#
#########################################################################

#########################################################################
# packages required
#########################################################################
library(R6sagemaker)
library(data.table)
library(fastDummies)
library(pROC)

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
# Set up session
#########################################################################
# Note: only require role when not working on sagemaker notebook
Cred = PawsCredentials$new()

# current method is temporary when issue: https://github.com/paws-r/paws/issues/253 is resolve this method will be streamlined
role = paws::sts(Cred$credentials)$assume_role(RoleArn = Sys.getenv("rathena_arn"),
                                               RoleSession = sprintf("sagemaker-%s", as.integer(Sys.time())))

Cred = PawsCredentials$new(aws_access_key_id = role$Credentials$AccessKeyId,
                           aws_secret_access_key = role$Credentials$SecretAccessKey,
                           aws_session_token = role$Credentials$SessionToken)

session = Session$new(paws_credentials = Cred)

# When working on sagemaker notebook
session = Session$new()

#########################################################################
# set AWS parameters
#########################################################################
my_region = session$paws_region_name
role = get_execution_role(session)
bucket = session$default_bucket()
prefix = "demo_sagemaker"
#########################################################################
# Download data from AWS sample data
#########################################################################
download.file('https://sagemaker-sample-data-us-west-2.s3-us-west-2.amazonaws.com/autopilot/direct_marketing/bank-additional.zip', "bank-addition.zip")
unzip("bank-addition.zip")

# read in data
data = fread("bank-additional/bank-additional-full.csv")
#########################################################################
# Data exploration
#########################################################################
skimr::skim(data)

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

session$upload_data(path = "train.csv", bucket = bucket, key_prefix = file.path(prefix, 'train'))
session$upload_data(path = "validation.csv", bucket = bucket, key_prefix = file.path(prefix, 'validation'))
#########################################################################
# prepare model
#########################################################################
container = get_image_uri(my_region, 'xgboost')

s3_input_train = s3_input$new(s3_data=sprintf('s3://%s/%s/train',bucket, prefix), content_type='csv')
s3_input_validation = s3_input$new(s3_data=sprintf('s3://%s/%s/validation/',bucket, prefix), content_type='csv')

xgb = Estimator$new(container,
                    role,
                    train_instance_count=1,
                    train_instance_type='ml.m4.xlarge',
                    output_path=sprintf('s3://%s/%s/output',bucket, prefix),
                    sagemaker_session=session)

xgb$set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        num_round=100)

xgb$fit(list('train'= s3_input_train, 'validation'= s3_input_validation))
#########################################################################
# Hosting
#########################################################################
xgb_predictor = xgb$deploy(initial_instance_count=1,
                           instance_type='ml.m4.xlarge')
#########################################################################
# Predict
#########################################################################
# s3 method
pred <- predict(xgb_predictor, test_data[,-c('y_no', 'y_yes')])

# Or call class directly
# Need to set serializer and deserializer
xgb_predictor$seralize = csv_serializer
xgb_predictor$deseralize = csv_deserializer

pred <- xgb_predictor$predict(test_data[,-c('y_no', 'y_yes')])

pred <- cbind(test_data[,'y_yes'], pred)

pred[, pred := fifelse(V1 >=.5 , 1, 0)]

# create a confusion matrix
table(pred$y_yes, pred$pred)

# create ROC curve plot
roc_obj <- roc(pred$y_yes, pred$pred)
plot(roc_obj)

# Get AUC of model
auc(roc_obj)
#########################################################################
# (Optional) Clean-up
#########################################################################
# When you are done with your model you can delete it so that it doesn't continue incurring cost
xgb$delete_endpoint()
