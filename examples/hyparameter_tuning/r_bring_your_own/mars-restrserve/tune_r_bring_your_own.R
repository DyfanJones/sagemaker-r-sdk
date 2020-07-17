#########################################################################
# This example is an adaptation from
# https://github.com/awslabs/amazon-sagemaker-examples/blob/master/r_examples/r_byo_r_algo_hpo/tune_r_bring_your_own.ipynb
#########################################################################

#########################################################################
# packages required
#########################################################################
library(R6sagemaker)
library(fastDummies)
library(data.table)

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
session = Session$new()
my_region = session$paws_region_name
role = get_execution_role(session)
bucket = session$default_bucket()
prefix = 'sagemaker/DEMO-hpo-r-byo'


#########################################################################
# Publish R Model
#########################################################################
# To get your custom model in to AWS. Please run the "build_and_publish.sh"

#########################################################################
# Data
#########################################################################
# create dummy columns for character/factor variables
iris_dt = data.table(iris)
iris_dt = dummy_cols(iris_dt, remove_selected_columns = T)

# Train/test split, 70%-30%
set.seed(42)
idx = iris_dt[,sample(.N *.7)]
train = iris_dt[idx]
test = iris_dt[-idx]

# write to S3
train_file = "iris_train.csv"
fwrite(train, train_file)
sess$upload_data(train_file, bucket = bucket, key_prefix = file.path(prefix, "train"))

#########################################################################
# Tune
#########################################################################
# R6sagemaker function to help integrate custom models into sagemaker
# This function scans AWS ECR and returns model's uri
img = get_ecr_image_uri("rmars")

# create Estimator with Static Parameter
estimator = Estimator$new(
  image_uri=img,
  role=role,
  train_instance_count=1,
  train_instance_type='ml.m4.xlarge',
  output_path=sprintf('s3://%s/%s/output', bucket, prefix),
  sagemaker_session=sess,
  hyperparameters=list('target' = 'Sepal.Length',
                       'degree'= 2))

# to set the degree as a varying HP to tune, use: 'degree': IntegerParameter$new(1, 3) and remove it from the Estimator
hyperparameter_ranges = list('thresh'= ContinuousParameter$new(0.001, 0.01),
                             'prune'= CategoricalParameter$new(c(TRUE, FALSE)))

# Will train model using metric: mse
# Regex will scan AWS CloudWatch logs for mse values from R model
objective_metric_name = 'mse'
metric_definitions = list(list('Name'= 'mse',
                               'Regex'= 'mse: ([0-9\\.]+)'))

tuner = HyperparameterTuner$new(estimator,
                                objective_metric_name,
                                hyperparameter_ranges,
                                metric_definitions,
                                objective_type='Minimize',
                                max_jobs=9,
                                max_parallel_jobs=3)

# Start hyperparameter tuning job
tuner$fit(list('train'= sprintf('s3://%s/%s/train',bucket, prefix)))

# Wait for tuning to complete
tuner$wait()

#########################################################################
# HPO Analysis
#########################################################################
# Set HyperparameterTuningJobAnalytics with tuning job
tuner_analyse = HyperparameterTuningJobAnalytics$new(tuner$latest_tuning_job)

# View Hyperparameter per tunning job
# As the object is returned as data.frame feel free to do any further analysis
tuner_analyse$dataframe()

#########################################################################
# Host
#########################################################################
mars_pred = tuner$deploy(initial_instance_count = 1, instance_type = 'ml.t2.medium')

#########################################################################
# Predict
#########################################################################
cols = names(test)[names(test)!= "Sepal.Length"]

# Create csv serializer that uses content_type "text/plain" when calling RestRserve endpoint
library(R6)
RestRserveCsv = R6Class("RestRserveCsv",
  inherit = CsvSerializer,
  public = list(
    initialize = function(){
      self$content_type = "text/plain"
    }
  )
)

# Get predictions from R model endpoint using standard S3 predict method
pred = predict(mars_pred, test[,..cols], RestRserveCsv$new())

# Alternatively you can call the predict method in mars_pred class
mars_pred$serializer = RestRserveCsv$new()
mars_pred$deserializer = csv_deserializer
pred = mars_pred$predict(test[,..cols])

# combined Actual and Predict
pred = cbind(test[,!..cols],pred)

# Calculate mse on test data
pred[, se := (Sepal.Length - prediction)**2]
pred[,.(mse = mean(se))]

# Plot Actual against Predicted
library(ggplot2)

ggplot(pred, aes(Sepal.Length, prediction)) +
  geom_point() +
  stat_smooth(method = "lm", alpha = 0, linetype = "dashed") +
  theme_classic() + # set theme to your liking (optional)
  labs(x = 'Sepal Length(Actual)',
       y = 'Sepal Length(Prediction)')


#########################################################################
# (Optional) Clean-up
#########################################################################
# When you are done with your model you can delete it so that it doesn't continue incurring cost
mars_pred$delete_endpoint()
