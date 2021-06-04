# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/model_monitor/cron_expression_generator.py

#' @include r_utils.R

#' @import R6
#' @import R6sagemaker.common

#' @title CronExpressionGenerator class
#' @description Generates cron expression strings for the SageMaker Model Monitoring Schedule API.
#' @export
CronExpressionGenerator = R6Class("CronExpressionGenerator",
  public = list(

    #' @description Generates hourly cron expression that denotes that a job runs at the top of every hour.
    #' @return str: The cron expression format accepted by the Amazon SageMaker Model Monitoring
    #'              Schedule API.
    hourly = function(){
      return("cron(0 * ? * * *)")
    },

    #' @description Generates daily cron expression that denotes that a job runs at the top of every hour.
    #' @param hour (int): The hour in HH24 format (UTC) to run the job at, on a daily schedule.
    #' @return str: The cron expression format accepted by the Amazon SageMaker Model Monitoring
    #'              Schedule API.
    daily = function(hour=0L){
      return(sprintf("cron(0 %i ? * * *)", hour))
    },

    #' @description Generates "daily every x hours" cron expression.
    #'              That denotes that a job runs every day at the specified hour, and then every x hours,
    #'              as specified in hour_interval.
    #' @param hour_interval (int): The hour interval to run the job at.
    #' @param starting_hour (int): The hour at which to begin in HH24 format (UTC).
    #' @return str: The cron expression format accepted by the Amazon SageMaker Model Monitoring
    #'              Schedule API.
    daily_every_x_hours = function(hour_interval,
                                   starting_hour=0L){
      return(sprintf("cron(0 %i/%i ? * * *)", starting_hour, hour_interval))
    },

    #' @description format class
    format = function(){
      format_class(self)
    }
  )
)
