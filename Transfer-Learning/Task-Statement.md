*This task is taken originally from the Machine Learning Course (Spring 2020) lectured by Prof. Hung-yi Lee at National Taiwan University.*\
*For more information about the lecture on Transfer Learning, please watch it on [YouTube](https://www.youtube.com/watch?v=qD6iD4TFsdQ&ab_channel=Hung-yiLee)*.

# Introduction: Transfer Learning
Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. This area of research bears some relation to the long history of psychological literature on the transfer of learning, although formal ties between the two fields are limited. From the practical standpoint, reusing or transferring information from previously learned tasks for the learning of new tasks has the potential to significantly improve the sample efficiency of a reinforcement learning agent. ([from Wikipedia](https://en.wikipedia.org/wiki/Transfer_learning))\
The architecture of transfer learning is shown as below.
<p align="center"><img width="70%" src="https://i.imgur.com/02dhSS4.png" /></p>

There are several models that are regarded as transfer learning. Here, we'll focus on one specific type called *Domain-Adversarial Neural Networks (DaNN)*.
Generally speaking, the structure of a DaNN model composes three parts.

1. **The Feature Extractor**: for mapping the features from the source and target data to the same distribution.
2. **The Domain Classifier**: for classify the domain of the output from the feature extractor is from source or target data.
3. **The Label Predictor**: for predicting the label of the output from the feature extractor.

To be more specific, since the target and source data are normally two different domains even though they hold some similarities in between, the task of the feature extractor is to map the two domains to the same distribution. For the domain classifier, it is used to classify which domain of the feature extractor output coming from. However, we need to include a *gradient reversal layer* as the goal of the domain classifier is to classify the two distributions as apart as it can (The opposite direction of the feature extractor). Last but not least, the goal of the label predictor is to predict the label of the extracted feature.

# Task Description
In the tutorial, we will build the DaNN model to predict the class of a hand-written image.

# Dataset
The dataset consists of three folders.
1. Training (source data): 5,000 real images with labels. (32 x 32 RGB)
2. Testing (target data): 10,000 hand-written images **without** labels. (28 x 28 Gray scale)


The labels contain 10 classes, which are shown as below.
<p align="center"><img width="90%" src="https://i.imgur.com/jipBuvB.png" /></p>

# Reference
- [Machine Learning Course (Spring 2020) by Prof. Hung-yi Lee at National Taiwan University](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)
- [Domain-Adversarial Training of Neural Networks](https://arxiv.org/pdf/1505.07818.pdf)
- [Wikipedia: Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning)
