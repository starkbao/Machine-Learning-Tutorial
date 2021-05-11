*This task is taken originally from the Machine Learning Course (Spring 2020) lectured by Prof. Hung-yi Lee at National Taiwan University.*\
*For more information about the lecture on CNN, please watch it on [YouTube](https://www.youtube.com/watch?v=xCGidAeyS4M&feature=youtu.be&ab_channel=Hung-yiLee).*

# Recurrent Neural Network (RNN)
A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs. This makes them applicable to tasks such as unsegmented, connected handwriting recognition, or speech recognition. ([Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network))


# Task Description: Text Sentiment Classification
In the tutorial, we will build a RNN model with PyTorch to predict the classes of a sentence.
<p align="center"><img width="50%" src="https://i.imgur.com/2haQwFa.png"></p>

# Dataset
The dataset is crawled from tweets from Twitter. There are three main files of the dataset. Namely, the training data with labels (200,000 in total), the training data without labels (1,200,000 in total), and the testing data (200,000 in total; 100,000 for public and another half for private.).\
The format of the labeled training set is shown below. At the beginning of each sentence is its label. After the `+++$+++`, is the sentence.
<p align="center"><img width="70%" src="https://i.imgur.com/GKCv4Kc.png"></p>
The format of the testing set is also shown below. There is only a sentence at each line.
<p align="center"><img width="70%" src="https://i.imgur.com/RPh9T0z.png"></p>

# Reference
- [Machine Learning Course (Spring 2020) by Prof. Hung-yi Lee at National Taiwan University](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)
