# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/lineage/_api_types.py

#' @include apiutils_base_types.R

#' @import R6

# ArtifactSource.
ArtifactSource = R6Class("ArtifactSource",
  inherit = ApiObject,
  public = list(

    # source_uri (str): The URI of the source.
    source_uri = NULL,

    # source_uri (str): The URI of the source.
    source_types = NULL,

    # Initialize ArtifactSource.
    # Args:
    #   source_uri (str): Source S3 URI of the artifact.
    # source_types (array): Array of artifact source types.
    # ... : Arbitrary keyword arguments.
    initialize = function(source_uri=NULL,
                          source_types=NULL,
                          ...){
      super$initialize(
        source_uri=source_uri, source_types=source_types, ...
      )
    }
  ),
  lock_objects = F
)

# ArtifactSourceType.
ArtifactSourceType = R6Class("ArtifactSourceType",
  inherit = ApiObject,
  public = list(

    # source_id_type (str): The source id type of artifact source.
    source_id_type = NULL,

    # value(str): The value of source
    value = NULL,

    # Initialize ArtifactSourceType.
    # Args:
    #   source_id_type (str): The type of the source id.
    # value (str): The source id.
    # ... : Arbitrary keyword arguments.
    initialize = function(source_id_type=NULL,
                          value=NULL,
                          ...){
      super$initialize(
        source_id_type=source_id_type, value=value, ...
      )
    }
  ),
  lock_objects = F
)

# ActionSource.
ActionSource = R6Class("ActionSource",
  inherit = ApiObject,
  public = list(
    # source_uri (str): The URI of the source.
    source_uri = NULL,

    # source_type (str):  The type of the source URI.
    source_type = NULL,

    # Initialize ActionSource.
    # Args:
    #   source_uri (str): The URI of the source.
    # source_type (str): The type of the source.
    # ... : Arbitrary keyword arguments.
    initialize = function(source_uri=NULL,
                          source_type=NULL,
                          ...){
      super$initialize(
        source_uri=source_uri, source_type=source_type, ...
      )
    }
  ),
  lock_objects = F
)

# ContextSource
ContextSource = R6Class("ContextSource",
  inherit = ApiObject,
  public = list(

    # source_uri (str): The URI of the source.
    source_uri = NULL,

    # source_type (str): The type of the source.
    source_type = NULL,

    # Initialize ContextSource.
    # Args:
    #   source_uri (str): The URI of the source.
    # source_type (str): The type of the source.
    # **kwargs: Arbitrary keyword arguments.
    initialize = function(source_uri=NULL,
                          source_type=NULL,
                          ...){
      super$initialize(
        source_uri=source_uri, source_type=source_type, ...
      )
    }
  ),
  lock_objects = F
)

# Summary model of an Artifact.
ArtifactSummary = R6Class("ArtifactSummary",
  inherit = ApiObject,
  public = list(
    # artifact_arn (str): ARN of artifact.
    artifact_arn = NULL,

    # artifact_name (str): Name of artifact.
    artifact_name = NULL,

    # source (obj): Source of artifact.
    source = NULL,

    # artifact_type (str): Type of artifact.
    artifact_type = NULL,

    # creation_time (datetime): Creation time.
    creation_time = NULL,

    # last_modified_time (datetime): Date last modified.
    last_modified_time = NULL
  ),
  private = list(
    .custom_paws_types = list("source"= c(ArtifactSource, FALSE))
  ),
  lock_objects = F
)

# Summary model of an action.
ActionSummary = R6Class("ActionSummary",
  inherit = ApiObject,
  public = list(
    # action_arn (str): ARN of action.
    action_arn = NULL,

    # action_name (str): Name of action.
    action_name = NULL,

    # source (obj): Source of action.
    source = NULL,

    # action_type (str): Type of action.
    action_type = NULL,

    # status (str): The status of the action.
    status = NULL,

    # creation_time (datetime): Creation time.
    creation_time = NULL,

    # last_modified_time (datetime): Date last modified.
    last_modified_time = NULL
  ),
  private = list(
    .custom_paws_types = list("source"= c(ActionSource, FALSE))
  ),
  lock_objects = F
)

# Summary model of an context.
ContextSummary = R6Class("ContextSummary",
  inherit = ApiObject,
  public = list(
    # context_arn (str): ARN of context.
    context_arn = NULL,

    # context_name (str): Name of context.
    context_name = NULL,

    # source (obj): Source of context.
    source = NULL,

    # context_type (str): Type of context.
    context_type = NULL,

    # creation_time (datetime): Creation time.
    creation_time = NULL,

    # last_modified_time (datetime): Date last modified.
    last_modified_time = NULL
  ),
  private = list(
    .custom_paws_types = list("source"= c(ContextSource, FALSE))
  ),
  lock_objects = F
)

# Summary model of an association.
AssociationSummary = R6Class("AssociationSummary",
  inherit = ApiObject,
  public = list(
    # source_arn (str): ARN of source entity.
    source_arn = NULL,

    # source_name (str): Name of the source entity.
    source_name = NULL,

    # destination_arn (str): ARN of the destination entity.
    destination_arn = NULL,

    # destination_name (str): Name of the destination entity.
    destination_name = NULL,

    # source_type (obj): Type of the source entity.
    source_type = NULL,

    # destination_type (str): Type of destination entity.
    destination_type = NULL,

    # association_type (str): The type of the association.
    association_type = NULL,

    # creation_time (datetime): Creation time.
    creation_time = NULL,

    # created_by (obj): Context on creator.
    created_by = NULL
  ),
  lock_objects = F
)
