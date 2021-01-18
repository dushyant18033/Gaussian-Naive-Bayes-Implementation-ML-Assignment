import h5py
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


def class_distribution(y):
    """
    Input Parameter:
    
    y : class labels vector having shape as (n_samples,)

    Output: Prints the class distribution

    Returns: None
    """
    unique, counts = np.unique(y, return_counts=True)
    distri = dict(zip(unique, counts))
    print("Count\t\tFraction")
    total = sum(counts)
    for c in range(len(unique)):
        print(distri[c],"\t",distri[c]/total)
    print("Total:",total)


if __name__=="__main__":
    # importing the dataset
    data = h5py.File("Datasets/part_A_train.h5","r")

    # separating input features and class labels
    X = np.array(data['X'])
    Y = np.array(data['Y'])

    # number of classes
    C = Y.shape[1]

    # converting one-hot encoded y to 1d class-label vector
    y = np.argmax(Y,axis=1)


    ## 1(d)
    # stratified train test split
    Xtrain,Xtest,ytrain,ytest = train_test_split(X, y, test_size=0.2, stratify=y, random_state=19)

    # commenting on the distribution
    print("Train Data Class Distribution")
    class_distribution(ytrain)
    print("Test Data Class Distribution")
    class_distribution(ytest)



    ## 1(e)
    # pca decomposition
    pca = PCA(100)  #desired dimensionality of 100 features
    pca.fit(Xtrain)
    pca_Xtrain = pca.transform(Xtrain)
    pca_Xtest = pca.transform(Xtest)
    print("# features before:",Xtrain.shape[1],"and after PCA:",pca_Xtrain.shape[1])

    # t-SNE analysis
    pca_tsne = TSNE(n_components=2).fit_transform(pca_Xtrain)
    for c in range(C):
        select = pca_tsne[ytrain==c]
        plt.scatter(select[:,0],select[:,1],label="Class "+str(c))
    plt.suptitle("t-SNE Analysis after PCA")
    plt.legend()
    plt.xlabel("Axis-1")
    plt.ylabel("Axis-2")
    plt.show()

    # feature scaling
    pca_scaler = StandardScaler()
    pca_scaler.fit(pca_Xtrain)
    pca_Xtrain = pca_scaler.transform(pca_Xtrain)
    pca_Xtest = pca_scaler.transform(pca_Xtest)

    pca_log = LogisticRegression()
    pca_log.fit(pca_Xtrain,ytrain)
    print("Test Accuracy after PCA:",pca_log.score(pca_Xtest,ytest))



    ## 1(f)
    # svd decomposition
    svd = TruncatedSVD(100) #desired dimensionality of 100 features
    svd.fit(Xtrain)
    svd_Xtrain = svd.transform(Xtrain)
    svd_Xtest = svd.transform(Xtest)
    print("# features before:",Xtrain.shape[1],"and after SVD:",svd_Xtrain.shape[1])

    # t-SNE analysis
    svd_tsne = TSNE(n_components=2).fit_transform(svd_Xtrain)
    for c in range(C):
        select = svd_tsne[ytrain==c]
        plt.scatter(select[:,0],select[:,1],label="Class "+str(c))
    plt.suptitle("t-SNE Analysis after SVD")
    plt.legend()
    plt.xlabel("Axis-1")
    plt.ylabel("Axis-2")
    plt.show()

    # feature scaling
    svd_scaler = StandardScaler()
    svd_scaler.fit(svd_Xtrain)
    svd_Xtrain = svd_scaler.transform(svd_Xtrain)
    svd_Xtest = svd_scaler.transform(svd_Xtest)
    
    # training and testing model
    svd_log = LogisticRegression()
    svd_log.fit(svd_Xtrain,ytrain)
    print("Test Accuracy after SVD:",svd_log.score(svd_Xtest,ytest))



    ## JUST FOR COMPARISON (NO SVD/PCA)
    # feature scaling
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(Xtrain,ytrain)
    print("Accuracy without PCA/SVD:",logistic.score(Xtest,ytest))



