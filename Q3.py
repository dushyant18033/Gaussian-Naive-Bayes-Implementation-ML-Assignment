import numpy as np
import h5py
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix as cm
import copy
import pickle


"""Utility Functions Start Here"""

def kfoldCrossValidation(model, X,y, k):
    """
    Performing K-Fold Cross Validation

    Parameters
    ----------
    model : model to use
    
    k : k in k-fold
    
    X : input features

    y : output variable

    Returns
    -------
    2-tuple : (avg train accuracy, avg validation accuracy)
    """
    
    # average accuracy
    avg_train_acc = 0
    avg_val_acc = 0

    for j in range(k):

        # Creating the splits into training and valdation folds

        split = X.shape[0]//k

        Xval = X[ split*j : split*(j+1), : ]
        yval = y[ split*j : split*(j+1) ]

        Xtrain = None
        ytrain = None

        if j==0:        #corner case 1
            Xtrain = X[split:,:]
            ytrain = y[split:]
        elif j==k-1:    #corner case 2
            Xtrain = X[:split*(k-1),:]
            ytrain = y[:split*(k-1)]
        else:           #remaining cases
            Xtrain = X[ : split*j, :]
            np.append(Xtrain, X[ split*(j+1) :, : ], axis=0)
            ytrain = y[ : split*j]
            np.append(ytrain, y[ split*(j+1) : ], axis=0)
        
        # copy and init the model provided
        model1 = copy.deepcopy(model)
        model1.set_params(**model.get_params())

        # train the model
        model1.fit(Xtrain,ytrain)
        avg_train_acc += model1.score(Xtrain,ytrain)
        avg_val_acc += model1.score(Xval, yval)
    
    # take the avg
    avg_train_acc/=k
    avg_val_acc/=k

    return (avg_train_acc,avg_val_acc)


def gridSearch(X,y,model,params,plot=False):
    """
    Performs grid search for 1 parameter specified in params.
    Returns 2 lists containing train, validation accuracies 
    for each value of attribute given in params.
    
    Parameters:
        X : input features
        y : output class label
        model : machine learning model instance
        params : parameter to search on
        plot : (default=False), whether to plot or not
    
    Returns:
        train_acc,val_acc : 2-tuple of lists
    """
    train_acc=list()    # train accuracy list
    val_acc=list()      # validation accuracy list

    attr = list(params.keys())[0]   # attribute to work with
    for value in params[attr]:      # iterate for all specified values
        model1 = copy.deepcopy(model)   # copy and init the model
        param = dict()
        param[attr] = value
        model1.set_params(**param)      # apply parameter value
        train,val = kfoldCrossValidation(model1,X,y,4)  # run k-fold
        train_acc.append(train)         # save avg train accuracy
        val_acc.append(val)             # save avg validation accuracy
    
    # plot accuracy vs parameter value if asked (using plot=True)
    if(plot):
        plt.plot(params[attr], train_acc, label="Train Accuracy")
        plt.plot(params[attr], val_acc, label="Validation Accuracy")
        plt.xlabel("Parameter")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
    
    return train_acc,val_acc


def save_model(Xtrain,ytrain,model1,filename,param=None):
    """
    Train a fresh model using specified instance on the given data and param (if specified)
    Dump the model instance to a binary file using object serialization.

    Parameters:
        Xtrain : input features
        ytrain : output variable
        model1 : model instance to work with
        filename : path to the file where the model should be saved
        param : parameters to applyon model instance if specified
    
    Returns: 
        None
    """
    # init & copy model instance
    model = copy.deepcopy(model1)

    # if parameters provided, apply them
    if param is not None:
        model.set_params(**param)
    
    # train a fresh model with entire train data
    model.fit(Xtrain,ytrain)

    # save to the file specified
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    """
    Reads a serialized object from the path specified in filename.
    Returns the object as a model instance.

    Parameters:
        filename : path to the file where the model is saved
    
    Returns: 
        model instance
    """
    return pickle.load(open(filename,'rb'))


def classification_report(ytrue1, ypred1):
    """
    Prints macro/micro avg precision,recall,f1,accuracy scores
    Returns own, scikit-learn confusion matrix

    Parameters
        ytrue1 : true class labels (binary profiled)
        ypred1 : predicted class labels (binary profiled)
    
    Returns
        conf : confusion matrix calculated by my own implementation
        cm(ytrue,ypred) : confusion matrix calculated by sklearn library function
    """
    C = ytrue1.shape[1]     # counting classes

    # converting shape from (n_samples,C) to (n_samples,)
    ytrue = np.argmax(ytrue1,axis=1)
    ypred = np.argmax(ypred1,axis=1)

    # init np arrays with 0
    conf = np.zeros((C,C), dtype=int)
    precision = np.zeros(C)
    recall = np.zeros(C)

    # temp variables for calculation
    tp_total = np.sum(ytrue==ypred,axis=0)
    fp_total = 0
    fn_total = 0

    # for every class
    for c in range(C):
        true_c = (ytrue==c)     # idx of truly class c samples
        pred_as_c = (ypred==c)  # idx of predicted class c samples

        conf[c,:] = np.sum(ypred1[true_c],axis=0)   # c-th row takes truly class c samples and distributes among predicted class columns

        precision[c] = np.sum(ytrue[pred_as_c]==c, axis=0)/np.sum(pred_as_c,axis=0) # class-wise precision
        recall[c] = np.sum(ypred[true_c]==c, axis=0)/np.sum(true_c,axis=0)          # class-wise recall

        fp_total += np.sum(ytrue[pred_as_c]!=c, axis=0) # total fp for micro average calculations
        fn_total += np.sum(ypred[true_c]!=c, axis=0)    # total fn for micro average calculations
    
    accuracy = np.sum(ytrue==ypred)/ytrue.shape[0]  # accuracy calculation is same no matter what
    
    # macro avg calculations
    prec_macro = np.mean(precision)
    rec_macro = np.mean(recall)    
    f1_macro = (2*prec_macro*rec_macro) / (prec_macro + rec_macro)

    # micro avg calculations
    prec_micro = tp_total/(tp_total+fp_total)
    rec_micro = tp_total/(tp_total+fn_total)
    f1_micro = 2 * (prec_micro * rec_micro) / (prec_micro + rec_micro)

    # printing the results
    print("\n\nAccuracy Score:",accuracy)

    if(C>2):
        print("\nMacro Values")
        print("Precision:",prec_macro,"\tRecall:",rec_macro,"\tF1:",f1_macro)
        print("\nMicro Values")
        print("Precision:",prec_micro,"\tRecall:",rec_micro,"\tF1:",f1_micro)
    else:
        print()
        print("Precision:",prec_macro,"\tRecall:",rec_macro,"\tF1:",f1_macro)

    # return the confusion matrix
    return conf,cm(ytrue,ypred)


def plot_ROC(ytrue1, ypred_proba1, pos_class):
    """
    Plots the ROC curve using pos_class as the positive class
    Generates thresholds from unique values out of ypred_proba1

    Parameters:
        ytrue1 : true class labels
        ypred_proba1 : class-wise probability matrix
        pos_class : class label to consider as the positive class
    
    Returns:
        None
    """
    ytrue = ytrue1[:,pos_class]
    ypred_proba = ypred_proba1[pos_class,:,1]

    thresholds = set(np.unique(ypred_proba))
    thresholds.add(0)
    thresholds.add(0.5)
    thresholds.add(1)

    FPR = list()
    TPR = list()

    for t in thresholds:
        ypred_pos = ypred_proba>=t
        ypred_neg = ypred_proba<t
        TP = np.sum(ytrue[ypred_pos]==1)
        TN = np.sum(ytrue[ypred_neg]==0)
        FP = np.sum(ytrue[ypred_pos]==0)
        FN = np.sum(ytrue[ypred_neg]==1)

        TPR.append(TP/(TP+FN))
        FPR.append(FP/(TN+FP))

    idx = np.argsort(FPR)

    plt.plot([0]+np.array(FPR)[idx].tolist()+[1], [0]+np.array(TPR)[idx].tolist()+[1],color='blue')
    plt.scatter(np.array(FPR)[idx],np.array(TPR)[idx], color='red')
    plt.plot([0,1],[0,1], ':', color='orange')
    plt.suptitle("ROC Curve - TPR vs FPR - Single Class:"+str(pos_class))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()


def plot_ROC_multi(ytrue1, ypred_proba1, class_plots=True, leng=10):
    """
    Plots multi-class macro,micro avg ROC curve as well as ROC considering
    every class as the positive class at a time (if class_plots=True).
    Generates 'leng' threshold values at equal differences between 0 and 1.

    Parameters:
        ytrue1 : true class labels
        ypred_proba1 : class-wise probability matrix
        class_plots : whether to plot class-wise ROC curve (identical to single class plots)
        leng : number of threshold values to be generated
    
    Returns:
        None
    """
    C = ytrue1.shape[1]

    thresholds = [(i/leng) for i in range(leng)]    
    
    macro_FPR = np.zeros(leng)
    macro_TPR = np.zeros(leng)

    TP_t = np.zeros(leng)
    TN_t = np.zeros(leng)
    FP_t = np.zeros(leng)
    FN_t = np.zeros(leng)

    for pos_class in range(C):
        ytrue = ytrue1[:,pos_class]
        ypred_proba = ypred_proba1[pos_class,:,1]

        FPR = list()
        TPR = list()

        for t in thresholds:
            ypred_pos = ypred_proba>=t
            ypred_neg = ypred_proba<t
            TP = np.sum(ytrue[ypred_pos]==1)
            TN = np.sum(ytrue[ypred_neg]==0)
            FP = np.sum(ytrue[ypred_pos]==0)
            FN = np.sum(ytrue[ypred_neg]==1)

            TP_t[int(t*leng)] += TP
            TN_t[int(t*leng)] += TN
            FP_t[int(t*leng)] += FP
            FN_t[int(t*leng)] += FN

            TPR.append(TP/(TP+FN))
            FPR.append(FP/(TN+FP))
        
        macro_FPR += FPR
        macro_TPR += TPR

        if(class_plots):
            idx = np.argsort(FPR)
            plt.plot([0]+np.array(FPR)[idx].tolist()+[1], [0]+np.array(TPR)[idx].tolist()+[1], label='class '+str(pos_class))


    macro_FPR/=C
    macro_TPR/=C
    idx = np.argsort(macro_FPR)
    plt.plot([0]+(macro_FPR)[idx].tolist()+[1], [0]+(macro_TPR)[idx].tolist()+[1], '--', label='Macro Average')

    micro_TPR = TP_t/(TP_t+FN_t)
    micro_FPR = FP_t/(TN_t+FP_t)
    idx = np.argsort(micro_FPR)
    plt.plot([0]+(micro_FPR)[idx].tolist()+[1], [0]+(micro_TPR)[idx].tolist()+[1], '-.', label='Micro Average')


    plt.plot([0,1],[0,1], ':', label='Referenece Line')
    plt.suptitle("ROC Curve - TPR vs FPR - Multi Class")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()

"""Utility Functions End Here"""




"""Programming Question Tasks"""

def Q3_DT_A(just_load=False):
    # importing the dataset
    dataA = h5py.File("Datasets/part_A_train.h5","r")
    X = np.array(dataA['X'])
    y = np.array(dataA['Y'])

    # shuffling the data consistently
    np.random.seed(42)
    shuff = np.random.permutation(X.shape[0])
    X=X[shuff]
    y=y[shuff]

    # making train-test split
    split = X.shape[0]//5
    Xtrain = X[split:]
    ytrain = y[split:]
    Xtest = X[:split]
    ytest = y[:split]

    if not just_load:   # unless asked to train again, work with the saved model

        # run grid search on DT Depth
        train_acc,val_acc = gridSearch(Xtrain,ytrain,DecisionTreeClassifier(),{ 'max_depth':list(range(1,21))}, plot=True)
        
        # identify the depth value for maximum train/validation accuracy
        train_max, val_max = np.argmax(train_acc), np.argmax(val_acc)

        # print the results
        print("\tDepth\ttrain acc\tvalidate acc")
        print("Training",train_max,train_acc[train_max],val_acc[train_max])
        print("Validation",val_max,train_acc[val_max],val_acc[val_max])

        # store the model
        save_model(Xtrain,ytrain, DecisionTreeClassifier(), 'Weights/dataset_A_DT', param={'max_depth':17})

    
    # load the saved model
    model = load_model('Weights/dataset_A_DT')
    ypred = model.predict(Xtest)
    ypred_proba = np.array(model.predict_proba(Xtest))
    
    # print classification report, confusion matrix
    mycm, skcm = classification_report(ytest, ypred)
    print("\nConfusion Matrix\n")
    print("\nMy Implementation\n",mycm)
    print("\nSklearn Implementation\n",skcm)

    # plot the ROC curves without and with class-wise plots
    plot_ROC_multi(ytest,ypred_proba,class_plots=False)
    plot_ROC_multi(ytest,ypred_proba)


def Q3_DT_B(just_load=False):
    # importing the dataset
    dataA = h5py.File("Datasets/part_B_train.h5","r")
    X = np.array(dataA['X'])
    y = np.array(dataA['Y'])

    # shuffling the data consistently
    np.random.seed(42)
    shuff = np.random.permutation(X.shape[0])
    X=X[shuff]
    y=y[shuff]

    # making train-test split
    split = X.shape[0]//5
    Xtrain = X[split:]
    ytrain = y[split:]
    Xtest = X[:split]
    ytest = y[:split]

    if not just_load:   # unless asked to train again, work with the saved model
    
        # run grid search on DT Depth
        train_acc,val_acc = gridSearch(Xtrain,ytrain,DecisionTreeClassifier(),{ 'max_depth':list(range(1,21))}, plot=True)
        
        # identify the depth value for maximum train/validation accuracy
        train_max, val_max = np.argmax(train_acc), np.argmax(val_acc)

        # print the results
        print("\tDepth\ttrain acc\tvalidate acc")
        print("Training",train_max,train_acc[train_max],val_acc[train_max])
        print("Validation",val_max,train_acc[val_max],val_acc[val_max])
    
        # store the model
        save_model(Xtrain,ytrain, DecisionTreeClassifier(), 'Weights/dataset_B_DT', param={'max_depth':13})
    

    # load the saved model
    model = load_model('Weights/dataset_B_DT')
    ypred = model.predict(Xtest)
    ypred_proba = np.array(model.predict_proba(Xtest))
    
    # print classification report and confusion matrix
    mycm, skcm = classification_report(ytest, ypred)
    print("\nConfusion Matrix\n")
    print("\nMy Implementation\n",mycm)
    print("\nSklearn Implementation\n",skcm)

    # plot the ROC curve
    plot_ROC(ytest,ypred_proba,1)


def Q3_GNB_A():
    # import the dataset
    dataA = h5py.File("Datasets/part_A_train.h5","r")
    X = np.array(dataA['X'])
    y = np.array(dataA['Y'])

    # shuffling the data consistently
    np.random.seed(42)
    shuff = np.random.permutation(X.shape[0])
    X=X[shuff]
    y=np.argmax(y[shuff],axis=1)

    # run 5-fold cross validation
    train,val = kfoldCrossValidation(GaussianNB(),X,y,5)

    # print the results
    print("GNB Dataset A")
    print("Avg Training Accuracy:",train)
    print("Avg Validation Accuracy:",val)


def Q3_GNB_B():
    # import the dataset
    dataA = h5py.File("Datasets/part_B_train.h5","r")
    X = np.array(dataA['X'])
    y = np.array(dataA['Y'])

    # shuffling the data consistently
    np.random.seed(42)
    shuff = np.random.permutation(X.shape[0])
    X=X[shuff]
    y=np.argmax(y[shuff],axis=1)

    # run 5-fold cross validation
    train,val = kfoldCrossValidation(GaussianNB(),X,y,5)
    
    # print the results
    print("GNB Dataset B")
    print("Avg Training Accuracy:",train)
    print("Avg Validation Accuracy:",val)


if __name__=="__main__":
    Q3_DT_A(just_load=True)
    Q3_DT_B(just_load=True)
    Q3_GNB_A()
    Q3_GNB_B()