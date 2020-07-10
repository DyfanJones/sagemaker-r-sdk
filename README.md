# Warning!!!
This repo is in constantly under development and is not currently stable.

# Brief overview

The idea is to rebuild [AWS Sagemaker Python SDK](https://github.com/aws/sagemaker-python-sdk) using [R6](https://github.com/r-lib/R6) classes and [paws](https://github.com/paws-r/paws) behind the scenes.


# API overview

This package aims to mimic python's AWS Sagemaker SDK api, but using `R6` and `paws`


# Learn from examples:

## Amazon Algorithms:

`R6sagemaker` is designed to minic python's sagemaker sdk. Therefore all examples for python's sagemaker should be able to accessible. 

### Examples:

* [Targeted Direct Marketing](https://github.com/DyfanJones/sagemaker-r-sdk/tree/master/examples/introduction_to_applying_machine_learning/xgboost_direct_marketing) predicts potential customers that are most likely to convert based on customer and aggregate level metrics, using Amazon SageMaker's implementation of XGBoost.
* [XGBoost Tuning](https://github.com/DyfanJones/sagemaker-r-sdk/blob/master/examples/hyparameter_tuning/xgboost_direct_maketing) shows how to use SageMaker hyperparameter tuning to improve your model fits for the [Targeted Direct Marketing](https://github.com/DyfanJones/sagemaker-r-sdk/tree/master/examples/introduction_to_applying_machine_learning/xgboost_direct_marketing) task.
* [BlazingText Word2Vec](https://github.com/DyfanJones/sagemaker-r-sdk/tree/master/examples/introduction_to_amazon_algorithms/blazingtext_word2vec_text8) generates Word2Vec embeddings from a cleaned text dump of Wikipedia articles using SageMaker's fast and scalable BlazingText implementation.



**Note: If a feature hasn't yet been implemented please feel free to raise a pull request or a ticket**
