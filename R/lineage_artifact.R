# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/lineage/artifact.py

#' @include apiutils_base_types.R
#' @include apiutils_utils.R
#' @include lineage_api_types.R
#' @include lineage_utils.R
#' @include lineage_association.R

#' @import R6

# An Amazon SageMaker artifact, which is part of a SageMaker lineage.
# Examples:
#   .. code-block:: python
# from sagemaker.lineage import artifact
# my_artifact = artifact.Artifact.create(
#   artifact_name='MyArtifact',
#   artifact_type='S3File',
#   source_uri='s3://...')
# my_artifact.properties["added"] = "property"
# my_artifact.save()
# for artfct in artifact.Artifact.list():
#   print(artfct)
# my_artifact.delete()
Artifact = R6Class("Artifact",
  inherit = Record,
  public = list(
    # artifact_arn (str): The ARN of the artifact.
    artifact_arn=NULL,

    # artifact_name (str): The name of the artifact.
    artifact_name=NULL,

    # artifact_type (str): The type of the artifact.
    artifact_type=NULL,

    # source (obj): The source of the artifact with a URI and types.
    source=NULL,

    # properties (dict): Dictionary of properties.
    properties=NULL,

    # tags (List[dict[str, str]]): A list of tags to associate with the artifact.
    tags=NULL,

    # creation_time (datetime): When the artifact was created.
    creation_time=NULL,

    # created_by (obj): Contextual info on which account created the artifact.
    created_by=NULL,

    # last_modified_time (datetime): When the artifact was last modified.
    last_modified_time=NULL,

    # last_modified_by (obj): Contextual info on which account created the artifact.
    last_modified_by=NULL,

    # Save the state of this Artifact to SageMaker.
    # Note that this method must be run from a SageMaker context such as Studio or a training job
    # due to restrictions on the CreateArtifact API.
    # Returns:
    #   Artifact: A SageMaker `Artifact` object.
    save = function(){
      return(private$.invoke_api(private$.paws_update_method, private$.paws_update_members))
    },

    # Delete the artifact object.
    # Args:
    #   disassociate (bool): When set to true, disassociate incoming and outgoing association.
    delete = function(disassociate=FALSE){
      if (disassociate){
        LineageUtils$new()$disassociate(
          source_arn=self$artifact_arn,
          sagemaker_session=self$sagemaker_session)
        LineageUtils$new()$disassociate(
          destination_arn=self$artifact_arn,
          sagemaker_session=self$sagemaker_session)
      }
      return(private$.invoke_api(
        private$.paws_delete_method, private$.paws_delete_members))
    },

    # Load an existing artifact and return an ``Artifact`` object representing it.
    # Args:
    #   artifact_arn (str): ARN of the artifact
    # sagemaker_session (sagemaker.session.Session): Session object which
    # manages interactions with Amazon SageMaker APIs and any other
    # AWS services needed. If not specified, one is created using the
    # default AWS configuration chain.
    # Returns:
    #   Artifact: A SageMaker ``Artifact`` object
    load = function(artifact_arn, sagemaker_session=NULL){
      artifact = private$.construct(
        private$.paws_load_method,
        artifact_arn=artifact_arn,
        sagemaker_session=sagemaker_session)
      return(artifact)
    },

    # Retrieve all trial runs which that use this artifact.
    # Args:
    #   sagemaker_session (obj): Sagemaker Sesssion to use. If not provided a default session
    # will be created.
    # Returns:
    #   [Trial]: A list of SageMaker `Trial` objects.
    downstream_trials = function(sagemaker_session=NULL){
      # don't specify destination type because for Trial Components it could be one of
      # SageMaker[TrainingJob|ProcessingJob|TransformJob|ExperimentTrialComponent]
      outgoing_associations = Association$new()$list(
        source_arn=self$artifact_arn, sagemaker_session=sagemaker_session)
      trial_component_arns = list(Map(function(x) x$destination_arn, outgoing_associations))

      if (islistempty(trial_component_arns)){
        # no outgoing associations for this artifact
        return(list())
      }

      # TODO: smexperiments package: https://github.com/aws/sagemaker-experiments
      TrialComponent = pkg_method(fun="TrialComponent", pkg="smexperiments")
      search_expression = Map(
        pkg_method, c("Filter","Operator", "SearchExpression","BooleanOperator"),
        pkg = "smexperiments")

      max_search_by_arn = 60
      num_search_batches = ceiling(length(trial_component_arns) %% max_search_by_arn)
      trial_components = list()

      sagemaker_session = sagemaker_session %||% Session$new()
      sagemaker_client = sagemaker_session$sagemaker

      for (i in seq_len(num_search_batches)){
        start = as.integer(i * max_search_by_arn)
        end = start + max_search_by_arn
        arn_batch = as.list(trial_component_arns[start:end])
        se = private$.get_search_expression(arn_batch, search_expression)
        search_result = TrialComponent$new()$search(
          search_expression=se, sagemaker_paws_client=sagemaker_client)

        trial_components = c(trial_components, list(search_result))
      }

      for (tc in list(trial_components)){
        for (parent in tc$parents){
          trials = c(trials, parent[["TrialName"]])
        }
      }
      return(unquie(trials))
    },

    # Add a tag to the object.
    # Args:
    #   tag (obj): Key value pair to set tag.
    # Returns:
    #   list({str:str}): a list of key value pairs
    set_tag = function(tag=NULL){
      return(private$.set_tags(resource_arn=self$artifact_arn, tags=list(tag)))
    },

    # Add tags to the object.
    # Args:
    #   tags ([{key:value}]): list of key value pairs.
    # Returns:
    #   list({str:str}): a list of key value pairs
    set_tags = function(tags=NULL){
      return(private$.set_tags(resource_arn=self$artifact_arn, tags=tags))
    },

    # Create an artifact and return an ``Artifact`` object representing it.
    # Args:
    #   artifact_name (str, optional): Name of the artifact
    # source_uri (str, optional): Source URI of the artifact
    # source_types (list, optional): Source types
    # artifact_type (str, optional): Type of the artifact
    # properties (dict, optional): key/value properties
    # tags (dict, optional): AWS tags for the artifact
    # sagemaker_session (sagemaker.session.Session): Session object which
    # manages interactions with Amazon SageMaker APIs and any other
    # AWS services needed. If not specified, one is created using the
    # default AWS configuration chain.
    # Returns:
    #   Artifact: A SageMaker ``Artifact`` object.
    create = function(artifact_name=NULL,
                      source_uri=NULL,
                      source_types=NULL,
                      artifact_type=NULL,
                      properties=NULL,
                      tags=NULL,
                      sagemaker_session=NULL){
      return(super$.construct(
        private$.paws_create_method,
        artifact_name=artifact_name,
        source=ArtifactSource$new(source_uri=source_uri, source_types=source_types),
        artifact_type=artifact_type,
        properties=properties,
        tags=tags,
        sagemaker_session=sagemaker_session)
      )
    },

    # Return a list of artifact summaries.
    # Args:
    #   source_uri (str, optional): A source URI.
    # artifact_type (str, optional): An artifact type.
    # created_before (datetime.datetime, optional): Return artifacts created before this
    # instant.
    # created_after (datetime.datetime, optional): Return artifacts created after this
    # instant.
    # sort_by (str, optional): Which property to sort results by.
    # One of 'SourceArn', 'CreatedBefore','CreatedAfter'
    # sort_order (str, optional): One of 'Ascending', or 'Descending'.
    # max_results (int, optional): maximum number of artifacts to retrieve
    # next_token (str, optional): token for next page of results
    # sagemaker_session (sagemaker.session.Session): Session object which
    # manages interactions with Amazon SageMaker APIs and any other
    # AWS services needed. If not specified, one is created using the
    # default AWS configuration chain.
    # Returns:
    #   collections.Iterator[ArtifactSummary]: An iterator
    # over ``ArtifactSummary`` objects.
    list = function(source_uri=NULL,
                    artifact_type=NULL,
                    created_before=NULL,
                    created_after=NULL,
                    sort_by=NULL,
                    sort_order=NULL,
                    max_results=NULL,
                    next_token=NULL,
                    sagemaker_session=NULL){
      return(super$.list(
        "list_artifacts",
        ArtifactSummary$new()$from_paws,
        "ArtifactSummaries",
        source_uri=source_uri,
        artifact_type=artifact_type,
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
    .paws_create_method = "create_artifact",
    .paws_load_method = "describe_artifact",
    .paws_update_method = "update_artifact",
    .paws_delete_method = "delete_artifact",

    .paws_update_members = list(
      "artifact_arn",
      "artifact_name",
      "properties",
      "properties_to_remove"),
    .paws_delete_members = "artifact_arn",

    .custom_paws_types = list("source"= list(ArtifactSource, FALSE)),

    # Convert a set of arns to a search expression.
    # Args:
    #   arns (list): Trial Component arns to search for.
    # search_expression (obj): smexperiments.search_expression
    # Returns:
    #   search_expression (obj): Arns converted to a Trial Component search expression.
    .get_search_expression = function(arns, search_expression){
      max_arn_per_filter = 3L
      num_filters = ceiling(lengths(arns) / max_arn_per_filter)
      filters = list()

      for (i in seq_len(num_filters)){
        start = i * max_arn_per_filter
        end = i + max_arn_per_filter
        batch_arns: list = arns[start:end]
        search_filter = search_expression$Filter$new(
          name="TrialComponentArn",
          operator=search_expression$Operator$new()$EQUALS,
          value=paste(batch_arns, collapse = ","))

        filters = c(filters, search_filter)
      }

      search_expression = search_expression$SearchExpression$new(
        filters=filters,
        boolean_operator=search_expression$BooleanOperator$new()$OR)

      return(search_expression)
    }
  ),
  lock_objects = F
)

# A SageMaker lineage artifact representing a model.
# Common model specific lineage traversals to discover how the model is connected
# to otherentities.
ModelArtifact = R6Class("ModelArtifact",
  inherit = Artifact,
  public = list(

    # Given a model artifact, get all associated endpoint context.
    # Returns:
    #   [AssociationSummary]: A list of associations repesenting the endpoints using the model.
    endpoints = function(){
      endpoint_development_actions = Association$new()$list(
        source_arn=self$artifact_arn,
        destination_type="Action",
        sagemaker_session=self$sagemaker_session)

      endpoint_context_list = unlist(lapply(
        endpoint_development_actions,
        function(endpoint_development_action){
          Association$list$new(
            source_arn=endpoint_development_action$destination_arn,
            destination_type="Context",
            sagemaker_session=self$sagemaker_session)
          }), recursive = FALSE)

      return(endpoint_context_list)
    }
  ),
  lock_objects = F
)

# A SageMaker Lineage artifact representing a dataset.
# Encapsulates common dataset specific lineage traversals to discover how the dataset is
# connect to related entities.
DatasetArtifact = R6Class("DatasetArtifact",
  inherit = Artifcat,
  public = list(

    # Given a dataset artifact, get associated trained models.
    # Returns:
    #   list(Association): List of Contexts representing model artifacts.
    trained_models = function(){
      trial_components = Association$new()$list(
        source_arn=self$artifact_arn, sagemaker_session=self$sagemaker_session)

      result = list()
      for (trial_component in trial_components){
        if ("experiment-trial-component" %in% names(trial_component.destination_arn)){
          models = Association$new()$list(
            source_arn=trial_component$destination_arn,
            destination_type="Context",
            sagemaker_session=self$sagemaker_session)
          result = c(result, models)
        }
      }
      return(result)
    }
  ),
  lock_objects = F
)
