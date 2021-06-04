# NOTE: This code has been modified from AWS Sagemaker Python:
# https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/lineage/visualizer.py

#' @include lineage_api_types.R
#' @include lineage_association.R

#' @import R6
#' @import lgr

# Creates a dataframe containing the lineage associations of a SageMaker object.
LineageTableVisualizer = R6Class("LineageTableVisualizer",
  public = list(

    # Init for LineageTableVisualizer.
    # Args:
    #   sagemaker_session (obj): The sagemaker session used for API requests.
    initialize = function(sagemaker_session){
      private$.session = sagemaker_session
    },

    # Generate a dataframe containing all incoming and outgoing lineage entities.
    # Examples:
    #   .. code-block:: python
    # viz = LineageTableVisualizer(sagemaker_session)
    # df = viz.show(training_job_name=training_job_name)
    # # in a notebook
    # display(df.to_html())
    # Args:
    #   trial_component_name (str, optional): Name of  a trial component. Defaults to None.
    # training_job_name (str, optional): Name of a training job. Defaults to None.
    # processing_job_name (str, optional): Name of a processing job. Defaults to None.
    # pipeline_execution_step (obj, optional): Pipeline execution step. Defaults to None.
    # model_package_arn (str, optional): Model package arn. Defaults to None.
    # endpoint_arn (str, optional): Endpoint arn. Defaults to None.
    # artifact_arn (str, optional): Artifact arn. Defaults to None.
    # context_arn (str, optional): Context arn. Defaults to None.
    # actions_arn (str, optional): Action arn. Defaults to None.
    # Returns:
    #   DataFrame: Pandas dataframe containing lineage associations.
    show = function(trial_component_name=NULL,
                    training_job_name=NULL,
                    processing_job_name=NULL,
                    pipeline_execution_step=NULL,
                    model_package_arn=NULL,
                    endpoint_arn=NULL,
                    artifact_arn=NULL,
                    context_arn=NULL,
                    actions_arn=NULL){
      start_arn=NULL

      if (!is.null(trial_component_name)){
        start_arn = private$.get_start_arn_from_trial_component_name(trial_component_name)
      } else if (!is.null(training_job_name)){
        trial_component_name = paste0(training_job_name, "-aws-training-job")
        start_arn = private$.get_start_arn_from_trial_component_name(trial_component_name)
      } else if (!is.null(processing_job_name)){
        trial_component_name = paste0(processing_job_name, "-aws-processing-job")
        start_arn = private$.get_start_arn_from_trial_component_name(trial_component_name)
      } else if (!is.null(pipeline_execution_step)){
        start_arn = private$.get_start_arn_from_pipeline_execution_step(pipeline_execution_step)
      } else if (!is.null(model_package_arn)){
        start_arn = private$.get_start_arn_from_model_package_arn(model_package_arn)
      } else if (!is.null(endpoint_arn)){
        start_arn = private$.get_start_arn_from_endpoint_arn(endpoint_arn)
      } else if (!is.null(artifact_arn)){
        start_arn = artifact_arn
      } else if (!is.null(context_arn)){
        start_arn = context_arn
      } else if (!is.null(actions_arn)){
        start_arn = actions_arn
      }

      return (private$.get_associations_dataframe(start_arn))
    }
  ),
  private = list(
    .session = NULL,

    # Given a pipeline exection step retrieve the arn of the lineage entity that represents it.
    # Args:
    #   pipeline_execution_step (obj): Pipeline execution step.
    # Returns:
    #   str: The arn of the lineage entity
    .get_start_arn_from_pipeline_execution_step = function(pipeline_execution_step){
      start_arn=NULL

      if (islistempty(pipeline_execution_step[["Metadata"]]))
        return(NULL)

      metadata = pipeline_execution_step[["Metadata"]]
      jobs = list("TrainingJob", "ProcessingJob", "TransformJob")

      for (job in jobs){
        if (!islistempty(metadata[[job]])){
          job_arn = metadata[[job]][["Arn"]]
          start_arn = private$.get_start_arn_from_job_arn(job_arn)
          break
        }
      }

      if ("RegisterModel" %in% names(metadata))
        start_arn = private$.get_start_arn_from_model_package_arn(
          metadata[["RegisterModel"]][["Arn"]])

      return(start_arn)
    },

    # Given a pipeline exection step retrieve the arn of the lineage entity that represents it.
    # Args:
    #   pipeline_execution_step (obj): Pipeline execution step.
    # Returns:
    #   str: The arn of the lineage entity
    .get_start_arn_from_job_arn = function(job_arn){
      start_arn=NULL
      response=private$.session$sagemaker$list_trial_components(SourceArn=job_arn)
      trial_components=response[["TrialComponentSummaries"]]
      if (!islistempty(trial_components)){
        start_arn = trial_components[[1]][["TrialComponentArn"]]
      } else {
        LOGGER$warn("No trial components found for %s", job_arn)
      }
      return(start_arn)
    },

    # Create a data frame containing lineage association information.
    # Args:
    #   arn (str): The arn of the lineage entity of interest.
    # Returns:
    #   DataFrame: A dataframe with association information.
    .get_associations_dataframe = function(arn){
      upstream_associations = private$.get_associations(dest_arn=arn)
      downstream_associations = private$.get_associations(src_arn=arn)
      inputs = as.list(Map(private$.convert_input_association_to_df_row, upstream_associations))
      outputs = as.list(
        Map(private$.convert_output_association_to_df_row, downstream_associations)
      )
      df = as.data.frame(
        c(inputs, outputs),
        colnames = c(
          "Name/Source",
          "Direction",
          "Type",
          "Association Type",
          "Lineage Type")
      )
      return(df)
    },

    # Given a trial component name retrieve a start arn.
    # Args:
    #   tc_name (str): Name of the trial compoonent.
    # Returns:
    #   str: The arn of the trial component.
    .get_start_arn_from_trail_component_name = function(tc_name){
      response = private$.session$sagemaker$describe_trial_component(
        TrialComponentName=tc_name)
      tc_arn = response[["TrialComponentArn"]]
      return(tc_arn)
    },

    # Given a model package arn retrieve the arn lineage entity.
    # Args:
    #   model_package_arn (str): The arn of a model package.
    # Returns:
    #   str: The arn of the lineage entity that represents the model package.
    .get_start_arn_from_model_package_arn = function(model_package_arn){
      response = private$.session$sagemaker$list_artifacts(SourceUri=model_package_arn)
      artifacts = response[["ArtifactSummaries"]]
      artifact_arn = NULL
      if (!islistempty(artifacts)){
        artifact_arn = artifacts[[1]][["ArtifactArn"]]
      } else{
        LOGGER$debug("No artifacts found for %s.", model_package_arn)
      }
      return(artifact_arn)
    },

    # Given an endpoint arn retrieve the arn of the lineage entity.
    # Args:
    #   endpoint_arn (str): The arn of an endpoint
    # Returns:
    #   str: The arn of the lineage entity that represents the model package.
    .get_start_arn_from_endpoint_arn = function(endpoint_arn){
      respons = private$.session$sagemaker$list_contexts(SourceUri=endpoint_arn)
      contexts = response[["ContextSummaries"]]
      context_arn = NULL
      if (!islistempty(contexts)){
        context_arn = contexts[[1]][["ContextArn"]]
      } else {
        LOGGER$debug("No contexts found for %s.", endpoint_arn)
      }
      return(context_arn)
    },

    # Given an arn retrieve all associated lineage entities.
    # The arn must be one of: experiment, trial, trial component, artifact, action, or context.
    # Args:
    #   src_arn (str, optional): The arn of the source. Defaults to None.
    # dest_arn (str, optional): The arn of the destination. Defaults to None.
    # Returns:
    #   array: An array of associations that are either incoming or outgoing from the lineage
    # entity of interest.
    .get_associations = function(src_arn=NULL,
                                 dest_arn=NULL){
      if (!is.null(src_arn)){
        associations = Association$new()$list(
          source_arn=src_arn, sagemaker_session=private$.session
        )
      } else {
        associations = Association$new()$list(
          destination_arn=dest_arn, sagemaker_session=private$.session
        )
      }
      return(associations)
    },

    # Convert an input association to a data frame row.
    # Args:
    #   association (obj): ``Association``
    # Returns:
    #   array: Array of column values for the association data frame.
    .convert_input_association_to_df_row = function(association){
      return(private$.convert_association_to_df_row(
        association$source_arn,
        association$source_name,
        "Input",
        association$source_type,
        association$association_type)
      )
    },

    # Convert an output association to a data frame row.
    # Args:
    #   association (obj): ``Association``
    # Returns:
    #   array: Array of column values for the association data frame.
    .convert_output_association_to_df_row = function(association){
      return(private$.convert_association_to_df_row(
        association$destination_arn,
        association$destination_name,
        "Output",
        association$destination_type,
        association$association_type)
      )
    },

    # Convert association data into a data frame row.
    # Args:
    #   arn (str): The arn of the associated entity.
    # name (str): The name of the associated entity.
    # direction (str): The direction the association is with the entity of interest. Values
    # are 'Input' or 'Output'.
    # src_dest_type (str): The type of the entity that is associated with the entity of
    # interest.
    # association_type ([type]): The type of the association.
    # Returns:
    #   [type]: [description]
    .convert_association_to_df_row = function(arn,
                                              name,
                                              direction,
                                              src_dest_type,
                                              association_type){
      arn_name = split_str(arn, split=":")[[6]]
      entity_type = split_str(arn_name, split="/")[[1]]
      name = private$.get_friendly_name(name, arn, entity_type)
      return(list(name, direction, src_dest_type, association_type, entity_type))
    },

    # Get a human readable name from the association.
    # Args:
    #   name (str): The name of the associated entity
    # arn (str): The arn of the associated entity
    # entity_type (str): The type of the associated entity (artifact, action, etc...)
    # Returns:
    #   str: The name for the association that will be displayed in the data frame.
    .get_friendly_name = function(name=NULL,
                                  arn=NULL,
                                  entity_type=NULL){
      if (!is.null(name))
        return(name)

      if (entity_type == "artifact"){
        artifact = private$.session$sagemaker$describe_artifact(ArtifactArn=arn)
        uri = artifact[["Source"]][["SourceUri"]]

        # shorten the uri if the length is more than 40,
        # e.g s3://flintstone-end-to-end-tests-gamma-us-west-2-069083975568/results/
        # canary-auto-1608761252626/preprocessed-data/tuning_data/train.txt
        # become s3://.../preprocessed-data/tuning_data/train.txt
        if (nchar(uri) > 48)
          name = paste0(substr(uri,1, 5), "...", substr(uri, (nchar(uri) - 39), nchar(uri)))

        # if not then use the full uri
        if (is.null(name))
          name = uri
      }

      # if still don't have name derive from arn
      if (is.null(name))
        name = split_str(split_str(arn, split=":")[[6]], split="/")[[2]]

      return(name)
    }
  ),
  lock_object = F
)
