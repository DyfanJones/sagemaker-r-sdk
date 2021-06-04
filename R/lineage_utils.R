# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/lineage/context.py

#' @include lineage_association.R

#' @import R6

LineageUtils = R6Class("LineageUtils",
  public = list(

    # Remove the association.
    # Remove incoming association when source_arn is provided, remove outgoing association when
    # destination_arn is provided.
    disassociate = function(source_arn=NULL,
                            destination_arn=NULL,
                            sagemaker_session=NULL){
      association_summaries = Association$new()$list(
        source_arn=source_arn, destination_arn=destination_arn, sagemaker_session=sagemaker_session
      )
      for (association_summary in association_summaries){
        curr_association = Association$new(
          sagemaker_session=sagemaker_session,
          source_arn=association_summary$source_arn,
          destination_arn=association_summary$destination_arn)
        curr_association$delete()
      }
    }
  )
)
