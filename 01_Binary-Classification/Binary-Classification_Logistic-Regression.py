###############################
#### Binary Classification ####
###############################

"""
Task 1: Logistic Regression
"""
# Importing the libraries
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset
X_train_fpath = './data/X_train'
y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './output_{}.csv'

# parse the csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:]
                        for line in f], dtype=float)

with open(y_train_fpath) as f:
    next(f)
    y_train = np.array([line.strip('\n').split(',')[1]
                        for line in f], dtype=float)

with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:]
                       for line in f], dtype=float)

# check the size of the dataset
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

# Defining data preprocessing functions


def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    """
    This function normalizes specific columns of X.
    The mean and standard variance of training data will be reused when processing testing data.

    Arguments:
        X: data to be processed.
        train: 'True' when processing training data. 'False' when processing testing data.
        specified_column: indexes of the columns that will be normalized. If 'None', all columns will be normalized.
        X_mean: mean value of the training data, used when train='False'.
        X_std: standard deviation of the training data, used when train='False'.

    Outputs:
        X: normalized data.
        X_mean: computed mean value of the training data.
        X_std: computed standard deviation of the training data.
    """
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)
    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)

    return X, X_mean, X_std


def _train_dev_split(X, y, dev_ratio=0.25):
    """
    This function spilts data into training set and development set.
    """
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]


# Data preprocessing
# Normalizing the training and testing data
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(
    X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

# Spliting the data into training and development set
dev_ratio = 0.1
X_train, y_train, X_dev, y_dev = _train_dev_split(
    X_train, y_train, dev_ratio=dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]

# check the size of the dataset
print('Size of the training set: {}'.format(train_size))
print('Size of the development set: {}'.format(dev_size))
print('Size of the testing set: {}'.format(test_size))
print('Size of the data dimension: {}'.format(data_dim))

# Defining some useful functions
# Some functions that will be repeatedly used when iteratively updating the parameters.


def _shuffle(X, y):
    """
    This function shuffles two equal-length list/array, X and Y, together.
    """
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], y[randomize])


def _sigmoid(z):
    """
    Sigmoid function can be used to calculate probability.
    To avoid overflow, minimum/maximum output value is set.
    """
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1-(1e-8))


def _f(X, w, b):
    """
    This is the logistic regression function, parameterized by w and b.

    Arguments:
        X: input data, shape=[batch_size, data_dimension]
        w: weight vector, shape=[data_dimension]
        b: bias, scalar

    Output:
        predicted probability of each row of X being postively labeled, shape=[batch_size, ]
    """
    return _sigmoid(np.matmul(X, w) + b)


def _predict(X, w, b):
    """
    This function returns a truth value prediction for each row of X by rounding the result of logistic regression function.
    """
    return np.round(_f(X, w, b)).astype(np.int)


def _accuracy(y_pred, y_label):
    """
    This function calculates prediction accuracy
    """
    acc = 1 - np.mean(np.abs(y_pred - y_label))
    return acc

# Functions about gradient and loss


def _cross_entropy_loss(y_pred, y_label):
    """
    This function computes the cross entropy.

    Arguments:
        y_pred: probabilistic predictions, float vector.
        y_label: ground truth labels, bool vector.

    Outputs:
        cross entropy, scalar.
    """
    cross_entropy = -np.dot(y_label, np.log(y_pred)) - \
        np.dot((1 - y_label), np.log(1 - y_pred))
    return cross_entropy


def _gradient(X, y_label, w, b):
    """
    This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    """
    y_pred = _f(X, w, b)
    pred_error = y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


# Training the model
"""
We'll use the gradient descent method with small batches for training.
The training data is divided into many small batches.
For each small batch, we calculate the gradient and loss separately.
Then, update the model parameters according to the batch.
When a loop is completed, that is, after all the small batches of the entire training set have been used once,
we will break up all the training data and re-divide them into new small batches.
Then, proceed to the next loop until finishing all loops.
"""
# zero initialization for weights and bias
w = np.zeros((data_dim, ))
b = np.zeros((1, ))

# some parameters for training
max_iter = 10
batch_size = 8
learning_rate = 0.2

# keep the loss and accuracy for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# calculate the number of parameter updates
step = 1

# iterative training
for epoch in range(max_iter):
    # random shuffle at the beginning of each epoch
    X_train, y_train = _shuffle(X_train, y_train)

    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx * batch_size: (idx + 1) * batch_size]
        y = y_train[idx * batch_size: (idx + 1) * batch_size]

        # compute the gradient
        w_grad, b_grad = _gradient(X, y, w, b)

        # gradient descent updates
        # learning rate decay with time
        w = w - learning_rate / np.sqrt(step) * w_grad
        b = b - learning_rate / np.sqrt(step) * b_grad

        step += 1

    # compute the loss and accuracy of the training set and development set
    y_train_pred = _f(X_train, w, b)  # float
    Y_train_pred = np.round(y_train_pred)  # bool
    train_acc.append(_accuracy(Y_train_pred, y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, y_train) / train_size)

    y_dev_pred = _f(X_dev, w, b)  # float
    Y_dev_pred = np.round(y_dev_pred)  # bool
    dev_acc.append(_accuracy(Y_dev_pred, y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, y_dev) / dev_size)

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))

# Plotting loss and accuracy curve
# loss curve
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()

# Predicting the testing labels

predictions = _predict(X_test, w, b)
with open('output_logistic.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'label']
    print(header)
    csv_writer.writerow(header)
    for i in range(len(predictions)):
        row = [str(i+1), predictions[i]]
        csv_writer.writerow(row)
        print(row)
    print()

# Print out the most significant weights
# Arrange the array in an ascending order and take it from the end to the front
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0: 10]:
    print(features[i], w[i])
