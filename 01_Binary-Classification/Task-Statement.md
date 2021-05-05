*This task is taken originally from the Machine Learning Course (Spring 2020) lectured by Prof. Hung-yi Lee at National Taiwan University.*\
*For more information about the lecture on Linear Regression, please watch it on YouTube [video 1](https://www.youtube.com/watch?v=fZAZUYEeIMg&ab_channel=Hung-yiLee) and [video 2](https://www.youtube.com/watch?v=hSXFuypLukA&ab_channel=Hung-yiLee)*

# Classification: Generative Model v.s. Discriminative Model
In statistical classification, two main approaches are called the generative approach and the discriminative approach. These compute classifiers by different approaches, differing in the degree of statistical modeling. Terminology is inconsistent, but three major types can be distinguished:
- Given an observable variable *X* and a target variable *Y*, a generative model is a statistical model of the joint probability distribution on *X × Y*, *P(X, Y)*;
- A discriminative model is a model of the conditional probability of the target *Y*, given an observation *x*, symbolically, P(Y|X=x); and
- Classifiers computed without using a probability model are also referred to loosely as "discriminative".\
([from Wikipedia](https://en.wikipedia.org/wiki/Generative_model))


# Task Description
In the tutorial, we will build the two classification models **from scratch** (Yes, we are going to code the model on our own, not from any open source library!) to predict whether the income of an individual exceeds $50,000 or not?\
The two classification models we'll build are the **Probabilistic Generative Model** and **Logistic Regression Model**.

# Dataset
The dataset is originally from [Census-Income (KDD) Dataset](https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)).
For convenience, we preprocessed the data to be `train.csv` and `test_no_label.csv`. These two files are text-based raw data. Besides, unnecessary attributes are removed and it is positive/negative ratio balanced.
There are several columns in the dataset, which can be categorized as `X_train`, `Y_train`, and `X_test`. Explanation below:
- Discrete features in `train.csv` -> one-hot encoding in `X_train` (education, martial state...).
- Continuous features in `train.csv` -> remain the same in `X_train` (age, capital losses...).
- `X_train` and `X_test` -> each row contains one 510-dim feature represents a sample.
- `Y_train` -> `label = 0` means  "<= 50K"  and `label = 1` means ">50K".

# Reference
- [Machine Learning Course (Spring 2020) by Prof. Hung-yi Lee at National Taiwan University](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)
