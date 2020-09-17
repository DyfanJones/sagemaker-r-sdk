# Warning!!!
This repo is in constantly under development and is not currently stable. R6sagemaker is currently aligning it's api with sagemaker v2, apologises for any code breaking this causes.

# Brief overview

The idea is to rebuild [AWS Sagemaker Python SDK](https://github.com/aws/sagemaker-python-sdk) using [R6](https://github.com/r-lib/R6) classes and [paws](https://github.com/paws-r/paws) behind the scenes.

# API overview

This package aims to mimic python's AWS Sagemaker SDK api, but using `R6` and `paws`

# Installation:

Currently `R6sagemaker` can only be installed from Github. 
```r
remotes::install_github("DyfanJones/sagemaker-r-sdk")
```

# Learn from examples:

## Amazon Algorithms:

`R6sagemaker` is designed to minic python's sagemaker sdk. Therefore all examples for python's sagemaker should be able to accessible. 

### Examples:

* [Targeted Direct Marketing](https://github.com/DyfanJones/sagemaker-r-sdk/tree/master/examples/introduction_to_applying_machine_learning/xgboost_direct_marketing) predicts potential customers that are most likely to convert based on customer and aggregate level metrics, using Amazon SageMaker's implementation of XGBoost.
* [XGBoost Tuning](https://github.com/DyfanJones/sagemaker-r-sdk/blob/master/examples/hyparameter_tuning/xgboost_direct_maketing) shows how to use SageMaker hyperparameter tuning to improve your model fits for the [Targeted Direct Marketing](https://github.com/DyfanJones/sagemaker-r-sdk/tree/master/examples/introduction_to_applying_machine_learning/xgboost_direct_marketing) task.
* [BlazingText Word2Vec](https://github.com/DyfanJones/sagemaker-r-sdk/tree/master/examples/introduction_to_amazon_algorithms/blazingtext_word2vec_text8) generates Word2Vec embeddings from a cleaned text dump of Wikipedia articles using SageMaker's fast and scalable BlazingText implementation.

### R Model Examples:

* [R Multivariate Adaptive Regression Splines](https://github.com/DyfanJones/sagemaker-r-sdk/tree/master/examples/hyparameter_tuning/r_bring_your_own/mars-restrserve) example over iris data.frame


**Note: If a feature hasn't yet been implemented please feel free to raise a pull request or a ticket**

## For developers

To keep the package within the CRAN size limit of 5MB. R6sagemaker is currently using a separate repository ([sagemaker-r-test-data](https://github.com/DyfanJones/sagemaker-r-test-data)) to store R variants of test data stored in [sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk). sagemaker-r-test-data will only consist of data that can't be read into R natively i.e. python pickle files. For other test data R6sagemaker will read it directly from sagemaker-python-sdk.
