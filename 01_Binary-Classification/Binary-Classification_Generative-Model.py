###############################
#### Binary Classification ####
###############################

"""
Task 2: Porbabilistic Generative Model
"""
# Importing the libraries
import csv
import numpy as np

# Loading the dataset
X_train_fpath = './data/X_train'
y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'

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

# Defining some useful functions


def _predict(X, w, b):
    """
    This function returns a truth value prediction for each row of X by rounding the result of logistic regression function.
    """
    return np.round(_f(X, w, b)).astype(np.int)


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


def _accuracy(y_pred, y_label):
    """
    This function calculates prediction accuracy
    """
    acc = 1 - np.mean(np.abs(y_pred - y_label))
    return acc


# Data preprocessing
# Normalizing the training and testing data
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(
    X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

# Calculating the Mean and Covariance
# In the generative model, we need to calculate the average and covariance of the data in the two categories separately.
# compute in-class mean
X_train_0 = np.array([x for x, y in zip(X_train, y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, y_train) if y == 1])

mean_0 = np.mean(X_train_0, axis=0)
mean_1 = np.mean(X_train_1, axis=0)

# compute the in-class covariance
data_dim = X_train.shape[1]
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))

for x in X_train_0:
    # np.transpose([x - mean_0]).shape -> (510, 1)
    # [x - mean_0].shape -> (1, 510)
    # np.dot(np.transpose([x - mean_0]), [x - mean_0]).shape -> (510, 510)
    cov_0 += np.dot(np.transpose([x - mean_0]),
                    [x - mean_0]) / X_train_0.shape[0]

for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_0]),
                    [x - mean_0]) / X_train_1.shape[0]

# Shared covariance is taken as a weighted average of individual in-class covariance.
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]
       ) / (X_train_0.shape[0] + X_train_1.shape[0])

# Computing weights and bias
# The weight matrix and deviation vector can be directly calculated.
# Compute the inverse of covariance matrix
# Since the covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error
# Via SVD decomposition, one can get matrix inverse efficiently and accurately
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

# Directly compute weights and bias
w = np.dot(inv, mean_0 - mean_1)
b = -0.5 * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1)) \
    + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

# Compute accuracy on training set
y_train_pred = 1 - _predict(X_train, w, b)
print('Training accuracy: {}'.format(_accuracy(y_train_pred, y_train)))

# Predicting testing labels

predictions = _predict(X_test, w, b)
with open('output_generative.csv', mode='w', newline='') as submit_file:
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
