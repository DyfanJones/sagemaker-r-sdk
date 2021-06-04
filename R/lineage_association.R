# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/lineage/association.py

#' @include r_utils.R
#' @include apiutils_base_types.R
#' @include lineage_api_types.R

#' @import R6

# An Amazon SageMaker artifact, which is part of a SageMaker lineage.
# Examples:
#   .. code-block:: python
# from sagemaker.lineage import association
# my_association = association.Association.create(
#   source_arn=artifact_arn,
#   destination_arn=trial_component_arn,
#   association_type='ContributedTo')
# for assoctn in association.Association.list():
#   print(assoctn)
# my_association.delete()
Association = R6Class("Association",
  inherit = Record,
  public = list(
    # source_arn (str): The ARN of the source entity.
    source_arn = NULL,

    # source_type (str): The type of the source entity.
    source_type = NULL,

    # destination_arn (str): The ARN of the destination entity.
    destination_arn = NULL,

    # destination_type (str): The type of the destination entity.
    destination_type = NULL,

    # association_type (str): the type of the association.
    association_type = NULL,

    # Delete this Association from SageMaker.
    delete = function(){
      private$.invoke_api(private$.paws_delete_method, private$.paws_delete_members)
    },

    # Add a tag to the object.
    # Args:
    #   tag (obj): Key value pair to set tag.
    # Returns:
    #   list({str:str}): a list of key value pairs
    set_tag = function(tag = NULL){
      return(private$.set_tags(resource_arn=self$source_arn, tags=list(tag)))
    },

    # Add a tags to the object.
    # Args:
    #   tag (obj): Key value pair to set tag.
    # Returns:
    #   list({str:str}): a list of key value pairs
    set_tags = function(tags = NULL){
      return(private$.set_tags(resource_arn=self$source_arn, tags=tags))
    },

    # Add an association and return an ``Association`` object representing it.
    # Args:
    #   source_arn (str): The ARN of the source.
    # destination_arn (str): The ARN of the destination.
    # association_type (str): The type of the association. ContributedTo, AssociatedWith,
    # DerivedFrom, or Produced.
    # sagemaker_session (sagemaker.session.Session): Session object which
    # manages interactions with Amazon SageMaker APIs and any other
    # AWS services needed. If not specified, one is created using the
    # default AWS configuration chain.
    # Returns:
    #   association: A SageMaker ``Association`` object.
    create = function(source_arn,
                      destination_arn,
                      association_type=NULL,
                      sagemaker_session=NULL){
      return (super$.construct(
        private$.boto_create_method,
        source_arn=source_arn,
        destination_arn=destination_arn,
        association_type=association_type,
        sagemaker_session=sagemaker_session)
      )
    },

    # Return a list of context summaries.
    # Args:
    #   source_arn (str): The ARN of the source entity.
    # destination_arn (str): The ARN of the destination entity.
    # source_type (str): The type of the source entity.
    # destination_type (str): The type of the destination entity.
    # association_type (str): The type of the association.
    # created_after (datetime.datetime, optional): Return contexts created after this
    # instant.
    # created_before (datetime.datetime, optional): Return contexts created before this
    # instant.
    # sort_by (str, optional): Which property to sort results by.
    # One of 'SourceArn', 'CreatedBefore', 'CreatedAfter'
    # sort_order (str, optional): One of 'Ascending', or 'Descending'.
    # max_results (int, optional): maximum number of contexts to retrieve
    # next_token (str, optional): token for next page of results
    # sagemaker_session (sagemaker.session.Session): Session object which
    # manages interactions with Amazon SageMaker APIs and any other
    # AWS services needed. If not specified, one is created using the
    # default AWS configuration chain.
    # Returns:
    #   collections.Iterator[AssociationSummary]: An iterator
    # over ``AssociationSummary`` objects.
    list = function(source_arn=NULL,
                    destination_arn=NULL,
                    source_type=NULL,
                    destination_type=NULL,
                    association_type=NULL,
                    created_after=NULL,
                    created_before=NULL,
                    sort_by=NULL,
                    sort_order=NULL,
                    max_results=NULL,
                    next_token=NULL,
                    sagemaker_session=NULL){
      return(super$.list(
        "list_associations",
        AssociationSummary$new()$from_paws,
        "AssociationSummaries",
        source_arn=source_arn,
        destination_arn=destination_arn,
        source_type=source_type,
        destination_type=destination_type,
        association_type=association_type,
        created_after=created_after,
        created_before=created_before,
        sort_by=sort_by,
        sort_order=sort_order,
        max_results=max_results,
        next_token=next_token,
        sagemaker_session=sagemaker_session)
      )
    }
  ),

  private = list(
    .paws_create_method = "add_association",
    .paws_delete_method = "delete_association",

    .custom_paws_types = list(),

    .paws_delete_members = list("source_arn", "destination_arn")
  ),
  lock_objects = F
)
