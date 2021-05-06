*This task is taken originally from the Machine Learning Course (Spring 2020) lectured by Prof. Hung-yi Lee at National Taiwan University.*\
*For more information about the lecture on CNN, please watch it on [YouTube](https://www.youtube.com/watch?v=FrKWiRv254g&feature=youtu.be&ab_channel=Hung-yiLee)*

# Recurrent Neural Network (RNN)
A CNN is a neural network that has one or more convolutional layers and is used mainly for image processing, classification, segmentation, and also for other autocorrelated data. ([from towards data science](https://towardsdatascience.com/an-introduction-to-convolutional-neural-networks-eb0b60b58fd7))


# Task Description: Food Classification
In the tutorial, we will build a CNN model with PyTorch to predict the classes of a set of images.

# Dataset
The dataset file containing three folders, namely training, validation, and testing. The naming format of an image in the training and validation set is *class_number.jpg*. For example, *3_100.jpg* meaning that it's class 3. The naming format of a testing image is *number.jpg*.\
You can download the dataset in this [Google Drive link](https://drive.google.com/file/d/19CzXudqN58R3D-1G8KeFWk8UDQwlb8is/view).\
The predicted result should be store in a `.csv` file with `Id` starting with 0 in the 1st column, and `Category` in the 2nd column.

# Reference
- [Machine Learning Course (Spring 2020) by Prof. Hung-yi Lee at National Taiwan University](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)
- [An introduction to Convolutional Neural Networks by towards data science](https://towardsdatascience.com/an-introduction-to-convolutional-neural-networks-eb0b60b58fd7)
