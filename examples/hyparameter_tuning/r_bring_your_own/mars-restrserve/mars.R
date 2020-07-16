# Adapted from: https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/r_bring_your_own/mars.R

# Use data.table to read in data fast
library(data.table)

# Bring in library that contains multivariate adaptive regression splines (MARS)
library(mda)

# Bring in library that allows parsing of JSON training parameters
library(jsonlite)

# Bring in library for prediction server
# Using RestRserve due high speed performance: https://restrserve.org/articles/benchmarks/Benchmarks.html
library(RestRserve)

# Setup parameters
# Container directories
prefix <- '/opt/ml'
input_path <- file.path(prefix, 'input/data')
output_path <- file.path(prefix, 'output')
model_path <- file.path(prefix, 'model')
param_path <- file.path(prefix, 'input/config/hyperparameters.json')

# Channel holding training data
channel_name <- 'train'
training_path <- file.path(input_path, channel_name)

train <- function(){

  # Read in hyperparameters
  training_params = read_json(param_path)

  # convert any TRUE or FALSE read in as string
  for(i in seq_along(training_params)){
    if(training_params[[i]] %in% c("TRUE", "FALSE"))
      training_params[[i]] = as.logical(training_params[[i]])
    }

  # stop if target variable not set correctly
  if(is.null(training_params$target))
    stop("Target variable not set")

  target = training_params$target
  training_params$target = NULL

  # Bring in data
  training_files = list.files(training_path, full.names=TRUE)
  training_data = rbindlist(lapply(training_files, fread))

  cols = names(training_data)[names(training_data)!= target]

  # Convert to model matrix
  training_X =model.matrix(~., data = training_data[, .SD, .SDcols = cols])

  character_var = names(training_data)[sapply(training_data, is.character)]

  # convert characters to factors
  for(x in character_var) set(training_data, j = x, value = as.factor(training_data[[x]]))

  # Save factor levels for scoring
  factor_levels <- lapply(training_data[, .SD, .SDcols = character_var],
                          function(x) {levels(x)})

  mod_param =  append(training_params, list(x = training_X, y=training_data[,.SD, .SDcols = target][[1]]))

  # This method give alot of flexibility for model build and is not constrained by specific hyperparameters
  model = do.call(mars, mod_param)

  save(model, factor_levels, file=file.path(model_path, 'model.RData'))

  write('success', file=file.path(output_path, 'success'))

  # Generate outputs
  print(summary(model))
  print(paste('gcv:', model$gcv))
  print(paste('mse:', mean(model$residuals**2)))
}

# Setup scoring function
serve <- function() {
  source(file.path(prefix, "server.R"))
  backend = BackendRserve$new()
  backend$start(app, 8080)}

# Run at start-up
args <- commandArgs()
if (any(grepl('train', args))) {
  train()}
if (any(grepl('serve', args))) {
  serve()}
