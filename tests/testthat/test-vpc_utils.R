# Reference link https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_vpc_utils.py
# Actual file to compare https://github.com/DyfanJones/sagemaker-r-sdk/blob/master/R/vpc_utils.R

#########################################################################
# This example is an adaptation from
# https://github.com/aws/sagemaker-python-sdk/blob/master/tests/unit/test_vpc_utils.py
#
#########################################################################
context("vpc_utils helper function")


library(testthat)



#########################################################################
# Setup
#########################################################################
subnets <- c('subnet')
security_groups <- c('sg')

good_vpc_config <- list('SUBNETS_KEY' = subnets,
                        'SECURITY_GROUP_IDS_KEY' = security_groups)

foo_vpc_config <- list('SUBNETS_KEY' = subnets,
                       'SECURITY_GROUP_IDS_KEY' = security_groups,
                       'foo' = 1
                       )
#########################################################################

test_that("test legacy name_from_framework_image", {
  image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py3-gpu:2.5.6-gpu-py2"
  test_fw1 = framework_name_from_image(image_uri)
  exp_fw1 = list("mxnet", "py3", "2.5.6-gpu-py2", NULL)
  
  expect_equal(test_fw1, exp_fw1)
})




test_that(
  'First test', 
  
  {
  
expect_equal(NULL,NULL)
expect_equal(subnets,NULL)
expect_equal(NULL,security_groups)
expect_equal(subnets,security_groups) # Needs to be updated to reflect the assert to_dict

    
  })
 

test_that(
  
  
  expect_equal(
    vpc_from_list(good_vpc_config) ==list(subnets,security_groups)
    )
    
  expect_equal(
    vpc_from_list(foo_vpc_config) ==list(subnets,security_groups)
  )
  
  expect_equal(
    vpc_from_list(NULL) == list(NULL,NULL)
  )
  
  expect_equal(
    
    vpc_from_list(NULL, do_sanitize = TRUE) == list(NULL,NULL)
    
    )
  
  expect_error(
    
    vpc_from_list()
    
  )
  
  expect_error(
    
    vpc_from_list(SUBNETS_KEY,subnets)
    
  )

  expect_error(
    
    vpc_from_list(SECURITY_GROUP_IDS_KEY, security_groups)
    
  )
  
  
  expect_error(
    
    vpc_from_list(do_sanitize = TRUE)
    
  )
  
  )

  
  
# Sanitize section 
  test_that(
    
    expect_equal(
      
      vpc_sannitize(good_vpc_config) == good_vpc_config
      
    )
    
    expect_equal(
      
      vpc_sannitize(foo_vpc_config) == good_vpc_config
      
    )
    
    expect_equal(
      
      is.null(vpc_sannitize(NULL))
      
    )
    
    
    expect_error(
      
      vpc_sannitize()
      
    )
    
    
    expect_error(
      
      
      vpc_sannitize(list('SUBNETS_KEY' = 1))
      
    )
    
    expect_error(
      
      
      vpc_sannitize(list('SUBNETS_KEY' = NULL))
      
    )
    
    expect_error(
      
      
      vpc_sannitize(list('SECURITY_GROUP_IDS_KEY' = 1, 'SUBNETS_KEY' = subnets))
      
    )
    
    
    expect_error(
      
      
      vpc_sannitize(list('SECURITY_GROUP_IDS_KEY' = NULL, 'SUBNETS_KEY' = subnets))
      
    )
    
    
  )
  
  