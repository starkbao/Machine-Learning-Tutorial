*This task is taken originally from the Machine Learning Course (Spring 2020) lectured by Prof. Hung-yi Lee at National Taiwan University.*\
*For more information about the lecture on Clustering, please watch it on [YouTube](https://www.youtube.com/watch?v=iwh5o_M4BNU&ab_channel=Hung-yiLee).*

# Clustering
Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). It is the main task of exploratory data analysis, and a common technique for statistical data analysis, used in many fields, including pattern recognition, image analysis, information retrieval, bioinformatics, data compression, computer graphics, and machine learning. ([from WikiPedia](https://en.wikipedia.org/wiki/Cluster_analysis))

# Task Description
In this tutorial, we'll use first use the Dimension Reduction method to decrease the dimension of an image dataset. Then, do Clustering on the lower dimension dataset.\
The reason why Dimension Reduction is necessary is that the original dataset contains a large amount of useless information. If we do Clustering on the original dataset, the result will not be good.\
There are several methods for Dimension Reduction. For example, Auto-encoder, PCA, SVD, and t-SNE.\
There are three tasks to be done.
1. Plot the result of the validation set and their corresponding labels.
2. Take the Auto-encoder with the highest accuracy and plot the original image and the reconstructed image with index 1, 2, 3, 6, 7, 9.
3. Plot two figures (Reconstruction Error (MSE) and validation accuracy) with 10 checkpoints.

# Dataset
The dataset contains three files.
1. `trainX.npy`: 8,500 images with the size of 32 x 32 x 3. The shape is `(8500, 32, 32, 3)`.
2. `valX.npy`: 500 images with the size of 32 x 32 x 3. The shape is `(500, 32, 32, 3)`. (DO NOT use this for training!)
3. `valY.npy`: the labels corresponding to `valX.npy`. The shape is `(500,)`.  (DO NOT use this for training!)


# Reference
- [Machine Learning Course (Spring 2020) by Prof. Hung-yi Lee at National Taiwan University](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)
- [Clustering from Wikipedia](https://en.wikipedia.org/wiki/Cluster_analysis)
