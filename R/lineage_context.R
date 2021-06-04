# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/lineage/context.py

#' @include r_utils.R
#' @include apiutils_base_types.R
#' @include lineage_api_types.R
#' @include lineage_utils.R
#' @include lineage_association.R

#' @import R6
#' @import R6sagemaker.common

# An Amazon SageMaker context, which is part of a SageMaker lineage.
Context = R6Class("Context",
  inherit = Record,
  public = list(
    # context_arn (str): The ARN of the context.
    context_arn = NULL,

    # context_name (str): The name of the context.
    context_name = NULL,

    # context_type (str): The type of the context.
    context_type = NULL,

    # description (str): A description of the context.
    description = NULL,

    # source (obj): The source of the context with a URI and type.
    source = NULL,

    # properties (dict): Dictionary of properties.
    properties = NULL,

    # tags (List[dict[str, str]]): A list of tags to associate with the context.
    tags = NULL,

    # creation_time (datetime): When the context was created.
    creation_time = NULL,

    # created_by (obj): Contextual info on which account created the context.
    created_by = NULL,

    # last_modified_time (datetime): When the context was last modified.
    last_modified_time = NULL,

    # last_modified_by (obj): Contextual info on which account created the context.
    last_modified_by = NULL,

    # Save the state of this Context to SageMaker.
    # Returns:
    #   obj: boto API response.
    save = function(){
      return(private$.invoke_api(
        private$.paws_update_method, private$.paws_update_members))
    },

    # Delete the context object.
    # Args:
    #   disassociate (bool): When set to true, disassociate incoming and outgoing association.
    # Returns:
    #   obj: boto API response.
    delete = function(disassociate = FALSE){
      if (disassociate){
        LineageUtils$new()$disassociate(
          source_arn=self$context_arn, sagemaker_session=self$sagemaker_session)
        LineageUtils$new()$disassociate(
          destination_arn=self$context_arn,
          sagemaker_session=self$sagemaker_session)
      }
      return(private$.invoke_api(
        private$.paws_delete_method, private$.paws_delete_members))
    },

    # Add tags to the object.
    # Args:
    #   tags ([{key:value}]): list of key value pairs.
    # Returns:
    #   list({str:str}): a list of key value pairs
    set_tag = function(tag = NULL){
      return(private$.set_tags(resource_arn=self$context_arn, tags=list(tag)))
    },

    # Add tags to the object.
    # Args:
    #   tags ([{key:value}]): list of key value pairs.
    # Returns:
    #   list({str:str}): a list of key value pairs
    set_tags = function(tags = NULL){
      return(private$.set_tags(resource_arn=self$context_arn, tags=tags))
    },

    # Load an existing context and return an ``Context`` object representing it.
    # Examples:
    #   .. code-block:: python
    # from sagemaker.lineage import context
    # my_context = context.Context.create(
    #   context_name='MyContext',
    #   context_type='Endpoint',
    #   source_uri='arn:aws:...')
    # my_context.properties["added"] = "property"
    # my_context.save()
    # for ctx in context.Context.list():
    #   print(ctx)
    # my_context.delete()
    # Args:
    #   context_name (str): Name of the context
    # sagemaker_session (sagemaker.session.Session): Session object which
    # manages interactions with Amazon SageMaker APIs and any other
    # AWS services needed. If not specified, one is created using the
    # default AWS configuration chain.
    # Returns:
    #   Context: A SageMaker ``Context`` object
    load = function(context_name,
                    sagemaker_session=NULL){
      context = private$.construct(
        private$.paws_load_method,
        context_name=context_name,
        sagemaker_session=sagemaker_session)
      return(context)
    },

    # Create a context and return a ``Context`` object representing it.
    # Args:
    #   context_name (str): The name of the context.
    # source_uri (str): The source URI of the context.
    # source_type (str): The type of the source.
    # context_type (str): The type of the context.
    # description (str): Description of the context.
    # properties (dict): Metadata associated with the context.
    # tags (dict): Tags to add to the context.
    # sagemaker_session (sagemaker.session.Session): Session object which
    # manages interactions with Amazon SageMaker APIs and any other
    # AWS services needed. If not specified, one is created using the
    # default AWS configuration chain.
    # Returns:
    #   Context: A SageMaker ``Context`` object.
    create = function(context_name=NULL,
                      source_uri=NULL,
                      source_type=NULL,
                      context_type=NULL,
                      description=NULL,
                      properties=NULL,
                      tags=NULL,
                      sagemaker_session=NULL){
      return(super$.construct(
        private$.paws_create_method,
        context_name=context_name,
        source=ContextSource$new(source_uri=source_uri, source_type=source_type),
        context_type=context_type,
        description=description,
        properties=properties,
        tags=tags,
        sagemaker_session=sagemaker_session)
      )
    },

    # Return a list of context summaries.
    # Args:
    #   source_uri (str, optional): A source URI.
    # context_type (str, optional): An context type.
    # created_before (datetime.datetime, optional): Return contexts created before this
    # instant.
    # created_after (datetime.datetime, optional): Return contexts created after this instant.
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
    #   collections.Iterator[ContextSummary]: An iterator
    # over ``ContextSummary`` objects.
    list = function(source_uri=NULL,
                    context_type=NULL,
                    created_after=NULL,
                    created_before=NULL,
                    sort_by=NULL,
                    sort_order=NULL,
                    max_results=NULL,
                    next_token=NULL,
                    sagemaker_session=NULL){
      return(super$.list(
        "list_contexts",
        ContextSummary$new()$from_paws,
        "ContextSummaries",
        source_uri=source_uri,
        context_type=context_type,
        created_before=created_before,
        created_after=created_after,
        sort_by=sort_by,
        sort_order=sort_order,
        max_results=max_results,
        next_token=next_token,
        sagemaker_session=sagemaker_session)
      )
    }
  ),
  private = list(
    .paws_load_method = "describe_context",
    .paws_create_method = "create_context",
    .paws_update_method = "update_context",
    .paws_delete_method = "delete_context",

    .custom_paws_types = list(
      "source"= list(ContextSource, FALSE)
      ),

    .paws_update_members = list(
      "context_name",
      "description",
      "properties",
      "properties_to_remove"),

    .paws_delete_members = "context_name"
  ),
  lock_objects = F
)

# An Amazon SageMaker endpoint context, which is part of a SageMaker lineage.
EndpointContext = R6Class("EndpointContext",
  inherit = Context,
  public = list(

    # Get all models deployed by all endpoint versions of the endpoint.
    # Returns:
    #   list of Associations: Associations that destination represents an endpoint's model.
    models = function(){
      endpoint_actions=Association$new()$list(
        sagemaker_session=self$sagemaker_session,
        source_arn=self$context_arn,
        destination_type="ModelDeployment")

      model_list = unlist(lapply(
        endpoint_actions,
        function(endpoint_action){
          Association$new()$list(
            source_arn=endpoint_action$destination_arn,
            destination_type="Model",
            sagemaker_session=self.sagemaker_session)}
        ), recursive = F)

      return(model_list)
    }
  ),
  lock_objects = F
)
