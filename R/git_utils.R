# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/git_utils.py

#' @include utils.R

#' @title Clone Sagemaker repositories by calling git
#' @description Git clone repo containing the training code and serving code. This method
#'              also validate ``git_config``, and set ``entry_point``, ``source_dir`` and
#'              ``dependencies`` to the right file or directory in the repo cloned.
#' @param git_config (dict[str, str]): Git configurations used for cloning files,
#'              including ``repo``, ``branch``, ``commit``, ``2FA_enabled``,
#'              ``username``, ``password`` and ``token``. The ``repo`` field is
#'              required. All other fields are optional. ``repo`` specifies the Git
#'              repository where your training script is stored. If you don't
#'              provide ``branch``, the default value 'master' is used. If you don't
#'              provide ``commit``, the latest commit in the specified branch is
#'              used. ``2FA_enabled``, ``username``, ``password`` and ``token`` are
#'              for authentication purpose. If ``2FA_enabled`` is not provided, we
#'              consider 2FA as disabled.
#'              For GitHub and GitHub-like repos, when SSH URLs are provided, it
#'              doesn't matter whether 2FA is enabled or disabled; you should either
#'              have no passphrase for the SSH key pairs, or have the ssh-agent
#'              configured so that you will not be prompted for SSH passphrase when
#'              you do 'git clone' command with SSH URLs. When https URLs are
#'              provided: if 2FA is disabled, then either token or username+password
#'              will be used for authentication if provided (token prioritized); if
#'              2FA is enabled, only token will be used for authentication if
#'              provided. If required authentication info is not provided, python
#'              SDK will try to use local credentials storage to authenticate. If
#'              that fails either, an error message will be thrown.
#'              For CodeCommit repos, 2FA is not supported, so '2FA_enabled' should
#'              not be provided. There is no token in CodeCommit, so 'token' should
#'              not be provided too. When 'repo' is an SSH URL, the requirements are
#'              the same as GitHub-like repos. When 'repo' is an https URL,
#'              username+password will be used for authentication if they are
#'              provided; otherwise, python SDK will try to use either CodeCommit
#'              credential helper or local credential storage for authentication.
#' @param entry_point (str): A relative location to the Python source file which
#'              should be executed as the entry point to training or model hosting
#'              in the Git repo.
#' @param source_dir (str): A relative location to a directory with other training
#'              or model hosting source code dependencies aside from the entry point
#'              file in the Git repo (default: None). Structure within this
#'              directory are preserved when training on Amazon SageMaker.
#' @param dependencies (list[str]): A list of relative locations to directories
#'              with any additional libraries that will be exported to the container
#'              in the Git repo (default: []).
#' @return dict: A dict that contains the updated values of entry_point, source_dir
#'              and dependencies.
#' @export
git_clone_repo <- function(git_config,
                           entry_point,
                           source_dir=NULL,
                           dependencies=NULL){
  if(missing(entry_point)){
    stop("Please provide an entry point.", call. = F)}

  .valid_git_config(git_config)
  dest_dir = tempdir()
  .generate_and_run_clone_command(git_config, dest_dir)

  .checkout_branch_and_commit(git_config, dest_dir)

  updated_paths = list(
    "entry_point"= entry_point,
    "source_dir"= source_dir,
    "dependencies"= dependencies)


  # check if the cloned repo contains entry point, source directory and dependencies
  if (!is.null(source_dir)){
    if (dir.exists(file.path(dest_dir, source_dir)))
      stop("Source directory does not exist in the repo.", call. = F)
    if (!file_test("-f",(file.path(dest_dir, source_dir, entry_point))))
      stop("Entry point does not exist in the repo.", call. = F)
    updated_paths$source_dir = file.path(dest_dir, source_dir)
  } else {
    if (!file_test("-f", (file.path(dest_dir, entry_point)))){
      updated_paths$entry_point = file.path(dest_dir, entry_point)
    } else {stop("Entry point does not exist in the repo.", calll. = F)}
  }

  if (!islistempty(dependencies)) {
    updated_paths$dependencies = list()
  for (path in dependencies){
    if (file.exists(file.path(dest_dir, path)))
      updated_paths$dependencies = c(updated_paths$dependencies, file.path(dest_dir, path))
    else
      stop(sprintf("Dependency %s does not exist in the repo.",path), call. = F)
    }
  }
  return(updated_paths)
}


.valid_git_config <- function(git_config){
  if (!("repo" %in% names(git_config))){
    stop("Please provide a repo for git_config.", call. = F)}
  for (key in names(git_config)){
    if (key == "2FA_enabled"){
      if (!inherits(git_config[["2FA_enabled"]], "logical")){
        stop("Please enter a logical type for 2FA_enabled'.", call. = F)}
    } else if(!inherits(git_config[[key]], "character")){
      stop(sprintf("'%s' must be a string.",key), call. = F)}
  }
}


# check if a git_config param is valid, if it is, create the command to git
# clone the repo, and run it.
# Args:
#   git_config ((dict[str, str]): Git configurations used for cloning files,
#               including ``repo``, ``branch`` and ``commit``.
#               dest_dir (str): The local directory to clone the Git repo into.
.generate_and_run_clone_command <- function(git_config,
                                           dest_dir){
  if (startsWith(git_config$repo, "https://git-codecommit")
      || startsWith(git_config$repo, "ssh://git-codecommit")){
    .clone_command_for_codecommit(git_config, dest_dir)
  } else {
    .clone_command_for_github_like(git_config, dest_dir)}
}

# check if a git_config param is valid, if it is, create the command to git
# clone the repo, and run it.
# Args:
#   git_config ((dict[str, str]): Git configurations used for cloning files,
#               including ``repo``, ``branch`` and ``commit``.
#               dest_dir (str): The local directory to clone the Git repo into.
.generate_and_run_clone_command <- function(git_config,
                                           dest_dir){
  if (startsWith(git_config$repo,"https://git-codecommit")
      ||startsWith(git_config$repo,"ssh://git-codecommit"))
    .clone_command_for_codecommit(git_config, dest_dir)
  else
    .clone_command_for_github_like(git_config, dest_dir)
}

# check if a git_config param representing a GitHub (or like) repo is
# valid, if it is, create the command to git clone the repo, and run it.
# Args:
#   git_config ((dict[str, str]): Git configurations used for cloning files,
#               including ``repo``, ``branch`` and ``commit``.
#               dest_dir (str): The local directory to clone the Git repo into.
.clone_command_for_github_like <- function(git_config,
                                          dest_dir){
  is_https = startsWith(git_config$repo,"https://")
  is_ssh = startsWith(git_config$repo,"git@")
  if (!is_https && !is_ssh)
    stop("Invalid Git url provided.", call. = F)
  if (is_ssh)
    .clone_command_for_ssh(git_config, dest_dir)
  else if ("2FA_enabled" %in% names(git_config) && git_config[["2FA_enabled"]])
    .clone_command_for_github_like_https_2fa_enabled(git_config, dest_dir)
  else
    .clone_command_for_github_like_https_2fa_disabled(git_config, dest_dir)
}

.clone_command_for_ssh <- function(git_config, dest_dir){
  if ("username" %in% names(git_config) || "password" %in% names(git_config) || "token" %in% names(git_config))
    warning("SSH cloning, authentication information in git config will be ignored.")
  .run_clone_command(git_config$repo, dest_dir)
}

.clone_command_for_github_like_https_2fa_disabled <- function(git_config, dest_dir){
  updated_url = git_config$repo
  if ("token" %in% names(git_config)){
    if ("username" %in% names(git_config) || "password" %in% names(git_config))
      warning("Using token for authentication, other credentials will be ignored.")
    updated_url = .insert_token_to_repo_url(url=git_config$repo, token=git_config$token)
  } else if ("username" %in% names(git_config) && "password" %in% names(git_config)){
      updated_url = .insert_username_and_password_to_repo_url(
        url=git_config$repo, username=git_config$username, password=git_config$password)
  } else if ("username" %in% names(git_config) || "password" %in% names(git_config))
    warning("Credentials provided in git config will be ignored.")
  .run_clone_command(updated_url, dest_dir)
}

.clone_command_for_github_like_https_2fa_enabled <- function(git_config, dest_dir){
  updated_url = git_config$repo
  if ("token" %in% names(git_config)){
    if ("username" %in% names(git_config) || "password" %in% names(git_config))
      warning("Using token for authentication, other credentials will be ignored.")
    updated_url = .insert_token_to_repo_url(url=git_config$repo, token=git_config$token)}
  .run_clone_command(updated_url, dest_dir)
}

# check if a git_config param representing a CodeCommit repo is valid, if
# it is, create the command to git clone the repo, and run it.
# Args:
#   git_config ((dict[str, str]): Git configurations used for cloning files,
#               including ``repo``, ``branch`` and ``commit``.
#               dest_dir (str): The local directory to clone the Git repo into.
#               Raises:
#                 ValueError: If git_config['repo'] is in the wrong format.
#               CalledProcessError: If failed to clone git repo.
.clone_command_for_codecommit <- function(git_config, dest_dir){
  is_https = startsWith(git_config$repo,"https://git-codecommit")
  is_ssh = startsWith(git_config$repo,"ssh://git-codecommit")
  if (!is_https && !is_ssh)
    stop("Invalid Git url provided.", call. = F)
  if ("2FA_enabled" %in% names(git_config))
    warning("CodeCommit does not support 2FA, '2FA_enabled' will be ignored.")
  if ("token" %in% names(git_config))
    warning("There are no tokens in CodeCommit, the token provided will be ignored.")
  if (is_ssh)
    .clone_command_for_ssh(git_config, dest_dir)
  else
    .clone_command_for_codecommit_https(git_config, dest_dir)
}

.clone_command_for_codecommit_https <-
  function(git_config, dest_dir){
  updated_url = git_config$repo
  if ("username" %in% names(git_config) && "password" %in% names(git_config))
    updated_url = .insert_username_and_password_to_repo_url(
      url=git_config$repo, username=git_config$username, password=git_config$password)
  else if ("username" %in% names(git_config) || "password" %in% names(git_config))
    warning("Credentials provided in git config will be ignored.")
  .run_clone_command(updated_url, dest_dir)
}


# Run the 'git clone' command with the repo url and the directory to clone
# the repo into.
# Args:
#   repo_url (str): Git repo url to be cloned.
# dest_dir: (str): Local path where the repo should be cloned into.
.run_clone_command <-
  function(repo_url, dest_dir){
  output <- system2('git', args = shQuote(c("clone", repo_url, dest_dir)), stdout = TRUE)
  status <- attr(output, "status") %||% 0L
  if (status != 0L) {
    fmt <- "error in git command [status code %i]"
    stop(sprintf(fmt, status), call. = F)}
}

# Insert the token to the Git repo url, to make a component of the git
# clone command. This method can only be called when repo_url is an https url.
# Args:
#   url (str): Git repo url where the token should be inserted into.
# token (str): Token to be inserted.
# Returns:
#   str: the component needed fot the git clone command.
.insert_token_to_repo_url <- function(url, token){
  index = nchar("https://")
  if (gregexpr(pattern ='welcome',txt)[[1]][1] == index)
    return(url)
  return(gsub("https://", paste0("https://",token, "@"), url))
}

# Insert the username and the password to the Git repo url, to make a
# component of the git clone command. This method can only be called when
# repo_url is an https url.
# Args:
#   url (str): Git repo url where the token should be inserted into.
# username (str): Username to be inserted.
# password (str): Password to be inserted.
# Returns:
#   str: the component needed for the git clone command.
.insert_username_and_password_to_repo_url <- function(url, username, password){
  password = URLencode(password, reserved = T)
  index = nchar("https://")
  pt1 = substring(url, 1, index)
  pt4 = substring(url, index, char(url))
  return(paste0(pt1, username, ":", password, "@", pt4))
}

# Checkout the required branch and commit.
# Args:
#   git_config (dict[str, str]): Git configurations used for cloning files,
# including ``repo``, ``branch`` and ``commit``.
# dest_dir (str): the directory where the repo is cloned
.checkout_branch_and_commit <- function(git_config, dest_dir){
  # Current working directory
  cur <- getwd()
  # On exit, come back
  on.exit(setwd(cur))
  setwd(dest_dir)

  if ("branch" %in% names(git_config)){
    output <- system2('git', args = shQuote(c("checkout", git_config$branch)), stdout = TRUE)
    status <- attr(output, "status") %||% 0L
    if (status != 0L) {
      fmt <- "error in git command [status code %i]"
      stop(sprintf(fmt, status), call. = F)}
  }
  if ("commit" %in% names(git_config)){
    output <- system2('git', args = shQuote(c("checkout", git_config$commit)), stdout = TRUE)
    status <- attr(output, "status") %||% 0L
    if (status != 0L) {
      fmt <- "error in git command [status code %i]"
      stop(sprintf(fmt, status), call. = F)}
  }
}
