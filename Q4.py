import pandas as pd
import numpy as np
import h5py
import decimal
from math import exp,pi,sqrt

from sklearn.naive_bayes import GaussianNB

class GaussianNaiveBayes:
    def __init__(self):
        pass

    def fit(self, Xtrain, ytrain):
        """
        Fits a Gaussian NB model on the given train data

        Parameters:
            Xtrain : input feature array with shape (n_samples, n_features)
            ytrain : binary profiled class label array with shape (n_samples, n_classes)
        
        Returns:
            None
        """
        # prior probability
        self.prior_prob = np.sum(ytrain,axis=0)/ytrain.shape[0]

        # number of classes
        self.C = ytrain.shape[1]

        # binary profiled 2d class labels --> 1d class labels array
        ytrain = np.argmax(ytrain,axis=1)

        # init 2d arrays to store means and variances
        self.mean = np.zeros((Xtrain.shape[1],self.C))
        self.var = np.zeros((Xtrain.shape[1],self.C))
        
        # calc mean, variance per feature, per class
        for c in range(self.C):
            temp = Xtrain[ytrain==c]
            self.mean[:,c] = np.mean(temp,axis=0)
            self.var[:,c] = np.var(temp,axis=0)*temp.shape[0]/(temp.shape[0]-1)

        # handling variance=0 case
        self.var += (self.var==0)*0.0000000001

    def predict(self,Xtest):
        """
        Uses the fitted Gaussian NB model to predict the class label for the given test data

        Parameters:
            Xtest : input feature array with shape (n_samples, n_features)
        
        Returns:
            ytest : 1d class label array with shape (n_samples,)
        """

        # initializing
        ypred = np.ones((Xtest.shape[0],self.C))

        # calculating class-wise probabilities
        for j in range(Xtest.shape[0]):
            for c in range(self.C):
                prob = decimal.Decimal(self.prior_prob[c])
                for i in range(Xtest.shape[1]):
                    temp = decimal.Decimal((Xtest[j][i] - self.mean[i][c])**2) / decimal.Decimal(2*self.var[i][c])
                    prob *= (-temp).exp() / decimal.Decimal(self.var[i][c]*pi*2).sqrt()
                
                ypred[j][c] = prob
        
        # return class giving max probability per sample
        return np.argmax(ypred,axis=1)
    
    def score(self,Xtest,ytest):
        """
        Uses the fitted Gaussian NB model to predict the class label for
        the given test data and calculates the accuracy by comparing it
        with the actual class label.

        Parameters:
            Xtest : input feature array with shape (n_samples, n_features)
            ytest : binary profiled class label array with shape (n_samples, n_classes)
        
        Returns:
            accuracy : fraction of samples correctly classified by the model.
        """
        # get the predictions
        ypred = self.predict(Xtest)

        # compare with the actual values and calc accuracy
        return np.sum(ypred==np.argmax(ytest,axis=1))/ytest.shape[0]


if __name__=="__main__":
    
    # importing the dataset (just change part_A_train.h5 to 
    # part_B_train.h5 in the line below to run for dataset-B.)

    dataA = h5py.File("Datasets/part_A_train.h5","r")

    X = np.array(dataA['X'])
    y = np.array(dataA['Y'])


    # shuffling the data consistently
    np.random.seed(42)
    shuff = np.random.permutation(X.shape[0])

    X=X[shuff]
    y=y[shuff]


    # making 80-20 train-test split
    split = X.shape[0]//5

    Xtrain = X[split:]
    ytrain = y[split:]

    Xtest = X[:split]
    ytest = y[:split]

    # Comparing my and sklearn implementations

    print("My Implementation running...")
    GNB = GaussianNaiveBayes()
    GNB.fit(Xtrain,ytrain)
    print(GNB.score(Xtest,ytest))
    
    print("SkLearn Implementation running...")
    ytrain = np.argmax(ytrain,axis=1)
    ytest = np.argmax(ytest,axis=1)
    model = GaussianNB()
    model.fit(Xtrain,ytrain)
    print(model.score(Xtest,ytest))