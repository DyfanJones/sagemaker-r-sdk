# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/lineage/action.py

#' @include apiutils_base_types.R
#' @include lineage_api_types.R

#' @import R6

# An Amazon SageMaker action, which is part of a SageMaker lineage.
# Examples:
#   .. code-block:: python
# from sagemaker.lineage import action
# my_action = action.Action.create(
#   action_name='MyAction',
#   action_type='EndpointDeployment',
#   source_uri='s3://...')
# my_action.properties["added"] = "property"
# my_action.save()
# for actn in action.Action.list():
#   print(actn)
# my_action.delete()
Action = R6Class("Action",
  inherit = Record,
  public = list(
    # action_arn (str): The ARN of the action.
    action_arn = NULL,

    # action_name (str): The name of the action.
    action_name = NULL,

    # action_type (str): The type of the action.
    action_type = NULL,

    # description (str): A description of the action.
    description = NULL,

    # status (str): The status of the action.
    status = NULL,

    # source (obj): The source of the action with a URI and type.
    source = NULL,

    # properties (dict): Dictionary of properties.
    properties = NULL,

    # tags (List[dict[str, str]]): A list of tags to associate with the action.
    tags = NULL,

    # creation_time (datetime): When the action was created.
    creation_time = NULL,

    # created_by (obj): Contextual info on which account created the action.
    created_by = NULL,

    # last_modified_time (datetime): When the action was last modified.
    last_modified_time = NULL,

    # last_modified_by (obj): Contextual info on which account created the action.
    last_modified_by = NULL,

    # Save the state of this Action to SageMaker.
    # Returns:
    #   Action: A SageMaker ``Action``object.
    save = function(){
      private$.invoke_api(private$.paws_update_method, private$.paws_update_members)
    },

    # Delete the action.
    # Args:
    #   disassociate (bool): When set to true, disassociate incoming and outgoing association.
    delete = function(disassociate=FALSE){
      stopifnot(is.logical(disassociate))
      if (disassociate){
        LineageUtils$new()$disassociate(
          source_arn=self$action_arn, sagemaker_session=self$sagemaker_session)
        LineageUtils$new()$disassociate(
          destination_arn=self$action_arn,
          sagemaker_session=self$sagemaker_session)
      }

      private$.invoke_api(
        private$.paws_delete_method, private$.paws_delete_members)
    },

    # Load an existing action and return an ``Action`` object representing it.
    # Args:
    #   action_name (str): Name of the action
    # sagemaker_session (sagemaker.session.Session): Session object which
    # manages interactions with Amazon SageMaker APIs and any other
    # AWS services needed. If not specified, one is created using the
    # default AWS configuration chain.
    # Returns:
    #   Action: A SageMaker ``Action`` object
    load = function(action_name, sagemaker_session=NULL){
      stopifnot(is.character(action_name),
                is.null(sagemaker_session) ||
                inherits(sagemaker_session, "Session"))
      result = private$.construct(
        private$.paws_load_method,
        action_name=action_name,
        sagemaker_session=sagemaker_session)
      return(result)
    },

    # Add a tag to the object.
    # Args:
    # Returns:
    #   list({str:str}): a list of key value pairs
    set_tag = function(tag=NULL){
      return(private$.set_tags(resource_arn=self$action_arn, tags=list(tag)))
    },

    # Add tags to the object.
    # Args:
    #   tags ([{key:value}]): list of key value pairs.
    # Returns:
    #   list({str:str}): a list of key value pairs
    set_tags = function(tags=NULL){
      return(private$.set_tags(resource_arn=self$action_arn, tags=list(tag)))
    },

    # Create an action and return an ``Action`` object representing it.
    # Args:
    #   action_name (str): Name of the action
    # source_uri (str): Source URI of the action
    # source_type (str): Source type of the action
    # action_type (str): The type of the action
    # description (str): Description of the action
    # status (str): Status of the action.
    # properties (dict): key/value properties
    # tags (dict): AWS tags for the action
    # sagemaker_session (sagemaker.session.Session): Session object which
    # manages interactions with Amazon SageMaker APIs and any other
    # AWS services needed. If not specified, one is created using the
    # default AWS configuration chain.
    # Returns:
    #   Action: A SageMaker ``Action`` object.
    create = function(action_name=NULL,
                      source_uri=NULL,
                      source_type=NULL,
                      action_type=NULL,
                      description=NULL,
                      status=NULL,
                      properties=NULL,
                      tags=NULL,
                      sagemaker_session=NULL){
      return(super$.construct(
        private$.paws_create_method,
        action_name=action_name,
        source=ActionSource$new(source_uri=source_uri, source_type=source_type),
        action_type=action_type,
        description=description,
        status=status,
        properties=properties,
        tags=tags,
        sagemaker_session=sagemaker_session)
      )
    },

    # Return a list of action summaries.
    # Args:
    #   source_uri (str, optional): A source URI.
    # action_type (str, optional): An action type.
    # created_before (datetime.datetime, optional): Return actions created before this
    # instant.
    # created_after (datetime.datetime, optional): Return actions created after this instant.
    # sort_by (str, optional): Which property to sort results by.
    # One of 'SourceArn', 'CreatedBefore', 'CreatedAfter'
    # sort_order (str, optional): One of 'Ascending', or 'Descending'.
    # max_results (int, optional): maximum number of actions to retrieve
    # next_token (str, optional): token for next page of results
    # sagemaker_session (sagemaker.session.Session): Session object which
    # manages interactions with Amazon SageMaker APIs and any other
    # AWS services needed. If not specified, one is created using the
    # default AWS configuration chain.
    # Returns:
    #   collections.Iterator[ActionSummary]: An iterator
    # over ``ActionSummary`` objects.
    list = function(source_uri=NULL,
                    action_type=NULL,
                    created_after=NULL,
                    created_before=NULL,
                    sort_by=NULL,
                    sort_order=NULL,
                    sagemaker_session=NULL,
                    max_results=NULL,
                    next_token=NULL){
      return(super$.list(
        "list_actions",
        ActionSummary$new()$from_paws,
        "ActionSummaries",
        source_uri=source_uri,
        action_type=action_type,
        created_before=created_before,
        created_after=created_after,
        sort_by=sort_by,
        sort_order=sort_order,
        sagemaker_session=sagemaker_session,
        max_results=max_results,
        next_token=next_token)
      )
    }
  ),

  private = list(
    .paws_create_method = "create_action",
    .paws_load_method = "describe_action",
    .paws_update_method = "update_action",
    .paws_delete_method = "delete_action",

    .paws_update_members = list(
      "action_name",
      "description",
      "status",
      "properties",
      "properties_to_remove"),

    .paws_delete_members = "action_name",

    .custom_paws_types = list("source"= list(ActionSource, FALSE))
  ),
  lock_objects = F
)
