from audioop import cross
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection  import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_selection import SequentialFeatureSelector
import time

# dimension reduction using SequentialFeatureSelector
def greedyReduction(model_type, X, y):
    if model_type == 'KNN':
        model = KNeighborsClassifier(n_neighbors=1)
    elif model_type == 'decision tree':
        model = DecisionTreeClassifier(min_samples_split = 7)
    elif model_type == 'random forest':
        model = RandomForestClassifier(n_estimators = 95)
    elif model_type == 'SVM':
        model = SVC(kernel = 'rbf', C = 5.0)
    elif model_type == 'ANN':
        model = MLPClassifier(activation = 'tanh', max_iter = 1000)
    elif model_type == 'NB':
        model = GaussianNB(var_smoothing = 1e-9)
    else:
        return 
    
    sfs = SequentialFeatureSelector(model, direction='backward', n_features_to_select=4, cv = 5)
    reduced_X = sfs.fit_transform(X, y)
    
    return reduced_X


# knn classification
def knnClf():
    # knn classifier
    k_list = range(1, 30)
    k_error = []
    for k in k_list:
            knn = KNeighborsClassifier(n_neighbors = k)
            scores = cross_val_score(knn, X_train, y_train, scoring = 'accuracy', cv = 5)
            k_error.append(scores.mean())
    best_k = k_list[k_error.index(max(k_error))]
    print('best k is {}'.format(best_k))
    # plot error curves, x-axis is k, y-axis is error
    plt.figure()
    plt.plot(k_list, k_error)
    plt.xlabel("value of k in KNN")
    plt.ylabel("classification error")
    plt.show()

    # test, k=5 get the best classifier
    neigh = KNeighborsClassifier(n_neighbors = best_k)
    neigh.fit(X_train, y_train)
    start = time.time()
    # pred_y = neigh.predict(X_val)
    # score2 = cross_val_score(neigh, X_val, y_val, scoring = 'accuracy', cv=5)
    # print(score2.mean())
    score_test = neigh.score(X_val, y_val) # return mean accuracy
    end = time.time()
    print('feature reduction time is {}'.format(end - start))
    print(score_test)
    
        
# decision tree classification
def treeClf():
    tree_error = []
    depth = range(2, 30)
    split_list = range(2, 30)
    for samples in depth:
            decisionTree = DecisionTreeClassifier(max_depth = samples)
            scores = cross_val_score(decisionTree, X_train, y_train, scoring = 'accuracy', cv=5)
            tree_error.append(scores.mean())
    best_split = split_list[tree_error.index(max(tree_error))]
    # plot error curves
    plt.figure()
    plt.plot(split_list, tree_error)
    plt.xlabel("value of depth in decision tree")
    plt.ylabel("classification error")
    plt.show()
    # print(tree_error)
    # test, split=7 get the best classifier
    tree = DecisionTreeClassifier(min_samples_split = best_split)
    tree.fit(X_train, y_train)
    start = time.time()
    # pred_y = tree.predict(X_val)
    score_test = tree.score(X_val, y_val) # return mean accuracy
    end = time.time()
    print('feature reduction time is {}'.format(end - start))
    print(score_test)       

# random forest classification
def RFClf():
    rf_error = []
    est_list = range(80, 100)
    for n in est_list:
        rf = RandomForestClassifier(n_estimators = n)
        scores = cross_val_score(rf, X_train, y_train, scoring = 'accuracy', cv = 5)
        rf_error.append(scores.mean())
    best_est = est_list[rf_error.index(max(rf_error))]
    # plot error curves
    plt.figure()
    plt.plot(est_list, rf_error)
    plt.xlabel("num of trees in random forest")
    plt.ylabel("classification error")
    plt.show()
    # test, n_estimators = 95
    clf = RandomForestClassifier(n_estimators = best_est)
    clf.fit(X_train, y_train)
    # pred_y = clf.predict(X_val)
    start = time.time()
    score_test = clf.score(X_val, y_val) # return mean accuracy
    end = time.time()
    print('feature reduction time is {}'.format(end - start))
    print(score_test) 

    #  classification using RBF kernel 
def SVMClf():
    svm_error = []      
    C_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10]
    for n in C_list:
        clf = SVC(kernel = 'rbf', C = n)
        scores = cross_val_score(clf, X_train, y_train, scoring = 'accuracy', cv = 5)
        svm_error.append(scores.mean())
    best_C = C_list[svm_error.index(max(svm_error))]
    # plot error curves
    plt.figure()
    plt.plot(C_list, svm_error)
    plt.xlabel("num of trees in random forest")
    plt.ylabel("classification error")
    plt.show()
    # test, C = 5
    clf = SVC(kernel = 'rbf', C = best_C)
    clf.fit(X_train, y_train)
    # pred_y = clf.predict(X_val)
    start = time.time()
    score_test = clf.score(X_val, y_val) # return mean accuracy
    end = time.time()
    print('feature reduction time is {}'.format(end - start))
    print(score_test)
def ANNClf():
    ANN_error = []      
    acti = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    for n in acti:
        clf = MLPClassifier(activation = n, max_iter = 1000)
        scores = cross_val_score(clf, X_train, y_train, scoring = 'accuracy', cv = 5)
        ANN_error.append(scores.mean())
    best_acti = acti[ANN_error.index(max(ANN_error))]
    # plot error curves
    plt.figure()
    plt.plot(acti, ANN_error)
    plt.xlabel("activation function")
    plt.ylabel("classification error")
    plt.show()
    # test, activation = tanh
    clf = MLPClassifier(activation = best_acti, max_iter = 1000)
    clf.fit(X_train, y_train)
    # pred_y = clf.predict(X_val)
    start = time.time()
    score_test = clf.score(X_val, y_val) # return mean accuracy
    end = time.time()
    print('feature reduction time is {}'.format(end - start))
    print(score_test)
def NaiveBayesianClf():
    NB_error = []      
    var_smooth = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    for n in var_smooth:
        clf = GaussianNB(var_smoothing = n)
        scores = cross_val_score(clf, X_train, y_train, scoring = 'accuracy', cv = 5)
        NB_error.append(scores.mean())
    best_NB = var_smooth[NB_error.index(max(NB_error))]
    # plot error curves
    plt.figure()
    plt.plot(var_smooth, NB_error)
    plt.xlabel("var smoothing value")
    plt.ylabel("classification error")
    plt.show()
    # test, var_smoothing = default
    clf = GaussianNB(var_smoothing = best_NB)
    clf.fit(X_train, y_train)
    start = time.time()
    # pred_y = clf.predict(X_val)
    score_test = clf.score(X_val, y_val) # return mean accuracy
    end = time.time()
    print('feature reduction time is {}'.format(end - start))
    print(score_test)

if __name__ == '__main__':
    DATA_PATH = "./letter-recognition.data"
    DM_REDUCTION = False
    # 'HK' or 'MY' or 'WV'or 'multi'
    FLAG = 'multi'
    columnNames = ['letter', 'x-box', 'y-box', 
            'width', 'high', 'onpix', 
            'x-bar', 'y-bar', 'x2bar', 
            'y2bar', 'xybar', 'x2ybr', 
            'xy2br', 'x-ege', 'xegvy', 
            'y-ege', 'yegvx']
    # the first column is label, row number is not a column
    data = pd.read_csv(DATA_PATH, names = columnNames).values
    data_X = data[:,1:17]
    data_y = data[:,0]
    if DM_REDUCTION:
        data_X = greedyReduction('NB', data_X, data_y)
        data = np.concatenate((data_y[:,np.newaxis], data_X), axis = 1)
    binary_data = []
    
    clfdata = np.array([])
    # binary classification
    if FLAG != 'multi':
            arr1 = np.where(data_y == FLAG[0])
            arr2 = np.where(data_y == FLAG[1])
            binary_data.extend(list(arr1)[0])
            binary_data.extend(list(arr2)[0])
            binary_data.sort()
            clfdata = data[binary_data]
            clfdata[np.where(clfdata == FLAG[0])[0],0] = 0
            clfdata[np.where(clfdata == FLAG[1])[0],0] = 1
    # multi class classification
    elif FLAG == 'multi':
            arr1 = np.where(data_y == 'H')
            arr2 = np.where(data_y == 'K')
            arr3 = np.where(data_y == 'M')
            arr4 = np.where(data_y == 'Y')
            arr5 = np.where(data_y == 'W')
            arr6 = np.where(data_y == 'V')
            binary_data.extend(list(arr1)[0])
            binary_data.extend(list(arr2)[0])
            binary_data.extend(list(arr3)[0])
            binary_data.extend(list(arr4)[0])
            binary_data.extend(list(arr5)[0])
            binary_data.extend(list(arr6)[0])
            binary_data.sort()
            clfdata = data[binary_data]
            clfdata[np.where(clfdata == 'H')[0],0] = 0
            clfdata[np.where(clfdata == 'K')[0],0] = 1
            clfdata[np.where(clfdata == 'M')[0],0] = 2
            clfdata[np.where(clfdata == 'Y')[0],0] = 3
            clfdata[np.where(clfdata == 'W')[0],0] = 4
            clfdata[np.where(clfdata == 'V')[0],0] = 5       
    clfdata = clfdata.astype(int)
    X_train, X_val, y_train, y_val = train_test_split(clfdata[:,1:], 
                                            clfdata[:,0], 
                                            test_size = 0.1, 
                                            random_state=0)
    knnClf()
