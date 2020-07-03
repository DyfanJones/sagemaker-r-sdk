# NOTE: This code has been modified from AWS Sagemaker Python: https://github.com/aws/sagemaker-python-sdk/blob/6a8bb6df1a0c34990a52fbfe4bbc6ec840d1bdcd/src/sagemaker/fw_registry.py

fw_registry <- function(region_name){
  tryCatch(account_id <- image_registry_map[[region_name]],
           error = function(e) stop("The specific image or region does not exist", call. = F))
  sprintf("%s.dkr.ecr.%s.amazonaws.com", account_id, region_name)
}

fw_default_framework_uri <- function(framework, region_name, image_tag){
  repository_name = sprintf("sagemaker-%s", framework)
  account_name = fw_registry(region_name)
  return(sprintf("%s/%s:%s",account_name, repository_name, image_tag))
}

image_registry_map = list("us-west-1" ="746614075791",
                          "us-west-2" = "246618743249",
                          "us-east-1" = "683313688378",
                          "us-east-2" = "257758044811",
                          "ap-northeast-1" = "354813040037",
                          "ap-northeast-2" = "366743142698",
                          "ap-southeast-1" = "121021644041",
                          "ap-southeast-2" = "783357654285",
                          "ap-south-1" = "720646828776",
                          "eu-west-1" = "141502667606",
                          "eu-west-2" = "764974769150",
                          "eu-central-1" = "492215442770",
                          "ca-central-1" = "341280168497",
                          "us-gov-west-1" = "414596584902",
                          "us-iso-east-1" = "833128469047",
                          "ap-east-1" = "651117190479",
                          "sa-east-1" = "737474898029",
                          "eu-north-1" = "662702820516",
                          "eu-west-3" = "659782779980",
                          "me-south-1" = "801668240914",
                          "cn-north-1" = "450853457545",
                          "cn-northwest-1" = "451049120500")
