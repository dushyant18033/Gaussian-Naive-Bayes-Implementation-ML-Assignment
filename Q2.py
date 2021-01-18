import pandas as pd
import numpy as np


def MSE_LOSS(X,y,theta):
    """
    Returns Average MSE loss obtained using X : input features
    y : output variable and theta : the model parameters.
    """
    X_theta_minus_y = np.dot(X,theta) - y
    sq_sum = np.dot(X_theta_minus_y.T, X_theta_minus_y)
    return sq_sum/X.shape[0]

# import and shuffle
data = pd.read_csv("Datasets/weight-height.csv", usecols=["Height","Weight"])
data = data.sample(frac=1, random_state=19)

# hyperparameters
num_test = data.shape[0]//5
sample_size = 200
num_samples = 10


# splitting test set before-hand
test = data[:num_test]
train = data[num_test:]


# separate X and y test set
Xtest = test[["Height"]].to_numpy()
ytest = test["Weight"].to_numpy()

# add columns of ones to Xtest
Xtest = np.insert(Xtest,0,np.ones(Xtest.shape[0]),axis=1)

# accumulator matrices for getting avg,sq of predictions
ypred_avg = np.zeros(Xtest.shape[0])
ypred_sq = np.zeros(Xtest.shape[0])

# accumulator for MSE
MSE_avg = 0

for i in range(num_samples):

    # get a bootstrap sample
    sample = train.sample(n=sample_size, replace=True)

    # separate X and y sampled train set
    Xtrain = sample[["Height"]].to_numpy()
    ytrain = sample["Weight"].to_numpy()

    # add columns of ones to Xtrain
    Xtrain = np.insert(Xtrain,0,np.ones(Xtrain.shape[0]),axis=1)

    # train a linear regression model
    xtx = np.dot(Xtrain.T,Xtrain)
    xty = np.dot(Xtrain.T,ytrain)
    xtx_inv = np.linalg.pinv(xtx)
    theta = np.dot(xtx_inv, xty)

    # make predictions on the test set
    ypred = np.dot(Xtest,theta)

    # for getting avg predictions
    ypred_avg += ypred
    ypred_sq += ypred**2

    # for getting avg Mean Squared Error
    MSE_avg += MSE_LOSS(Xtest,ytest,theta)


# calculate the average dividing by number of samples
ypred_avg /= num_samples
MSE_avg /= num_samples

B = num_samples
# variance for every test data sample wrt different hypothesis
variance = (ypred_sq - B*(ypred_avg**2))/(B-1)

# bias for every test data sample
bias = (ypred_avg-ytest)

# absolute mean bias
bs = np.mean(np.abs(bias))

# mean variance
var = np.mean(variance)

# unavoidable model uncertainty
uncertainty = MSE_avg - bs**2 - var

# printing the results
print("Bias:",bs)
print("MSE:",MSE_avg)
print("Variance:",var)
print("Uncertainty:",uncertainty)