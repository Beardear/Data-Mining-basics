import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# hyper parameters
HYPER_PARAMs = {
    # error type, 0 for MSE, 1 for MAE
    'Flag': 0,
    # normilize or not
    'Normalize': False,
    # maximum iteration
    'iters': 20000,
    # step size
    'alpha': 1e-6,
    # stop threshold
    'threshold': 1e-12,
    # training dataset size
    'training_size':900
}  

HYPER_PARAMs_NORMALIZE = {
    # step size
    'alpha': 1e-5,
    # stop threshold
    'threshold': 1e-20,
    'iters':30000
}    
    
    
def Generate_data(training_size, data, i=None):
    if i == None:
        X_train, X_test, y_train, y_test = data[:training_size, :8], data[training_size:, :8],\
                                            data[:training_size, 8], data[training_size:, 8]
        if HYPER_PARAMs['Normalize'] == True:
            mean_value = np.mean(X_train, axis = 0)
            std_value = np.std(X_train, axis = 0)
            X_train = (X_train - mean_value)/std_value
            X_test = (X_test - mean_value)/std_value

    else:
        print(i)
        X_train, X_test, y_train, y_test = data[:training_size, i:i + 1], data[training_size:, i:i + 1], \
                                            data[:training_size, 8], data[training_size:, 8]
        if HYPER_PARAMs['Normalize'] == True:
            mean_value = np.mean(X_train)
            std_value = np.std(X_train)
            X_train = (X_train - mean_value)/std_value
            X_test = (X_test - mean_value)/std_value

    X_train_new = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test_new = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
 
    y_train = y_train[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    return X_train_new, X_test_new, y_train, y_test

def cal_loss(weight, X, y, Flag):
    if Flag == 0: # mse loss
        y_pred = np.dot(X, weight)
        diff = y_pred - y
        loss = (1. / (2 * X.shape[0])) * np.dot(np.transpose(diff), diff)
        loss = loss[0][0]
    elif Flag == 1: # mae loss
        y_pred = np.dot(X, weight)
        loss = 1/X.shape[0] * np.sum(np.abs(y-y_pred)) 
    else:
        print('wrong FLAG value, check it again!')
        return       
    return loss

def R_squared(weight, X_test, y_test):
    
    y_pred = np.dot(X_test, weight)
    y_bar = np.mean(y_test)
    ss_tot = ((y_test-y_bar)**2).sum()
    ss_res = ((y_test-y_pred)**2).sum()
    return 1 - (ss_res/ss_tot)

def gradient_descent(X, y, weight, alpha, iters, Flag):
    def cal_gradient(X, y, weight, Flag):
        if Flag == 0:
            dif = np.dot(X, weight) - y
            gradient = (1. / X.shape[0]) * np.dot(np.transpose(X), dif)
        elif Flag == 1:
            y_pred = np.dot(X, weight)
            dif_mask = (y-y_pred).copy()
            dif_mask[y-y_pred > 0] = 1
            dif_mask[dif_mask <= 0] = -1
            gradient = -(1/X.shape[0]) * np.dot(np.transpose(X), dif_mask)
        else:
            print('wrong FLAG value, check it again!')
            return 
        return gradient
    gradient = cal_gradient(X, y, weight, Flag)

    error_results = []
    for i in range(iters):
        error_results.append(cal_loss(weight, X, y, HYPER_PARAMs['Flag']))
        if i % 1000 == 0:
            print(error_results[-1])
        if len(error_results) > 1 and abs(error_results[-2] - error_results[-1]) < HYPER_PARAMs['threshold']:
            break
        weight = weight - alpha * gradient
        gradient = cal_gradient(X, y, weight, Flag)
    return weight, error_results


def plot_results(index, X_train, y_train, trained_weight):
    labels = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']
    plt.figure()
    plt.xlabel(labels[index], fontsize=10)
    plt.ylabel('prediction', fontsize=10)
    plt.scatter(X_train[:, 1], y_train)
    plt.plot(X_train[:, 1], np.dot(X_train, trained_weight), color='r')
    plt.title('Univariate ' + labels[index], fontsize=10)
    plt.savefig('plotfigures' + str(index) + '.jpg', dpi=300)
    plt.show()


if __name__ == '__main__':
    path = './Concrete_Data.xls'
    data = pd.read_excel(path, header=0)
    data = np.array(data)
    # np.random.shuffle(data)
    
    training_size = HYPER_PARAMs['training_size']
    testing_size = np.shape(data)[0] - training_size

    R_squared_value = []
    # uni-variate GD
    uni_errors, uni_testing_errors = [], []
    labels = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']
    for i in range(np.shape(data)[1] - 1):
        X_train, X_test, y_train, y_test = Generate_data(training_size, data, i)
        
        weight = np.zeros((X_train.shape[1], 1))
        if (HYPER_PARAMs['Normalize']==False):
            trained_weight, error_results = gradient_descent(X_train, y_train, weight, HYPER_PARAMs['alpha'], \
                                            HYPER_PARAMs['iters'], HYPER_PARAMs['Flag'])
        elif(HYPER_PARAMs['Normalize']):
            trained_weight, error_results = gradient_descent(X_train, y_train, weight, HYPER_PARAMs_NORMALIZE['alpha'], \
                                            HYPER_PARAMs_NORMALIZE['iters'], HYPER_PARAMs['Flag'])
        else:
            print('normalizing value wrong!')
            break
        
        uni_errors.append(error_results)
        plot_results(i, X_train, y_train, trained_weight)
        uni_testing_errors.append(cal_loss(trained_weight, X_test, y_test, HYPER_PARAMs['Flag']))
        # draw R_square value
        R_squared_value.append(R_squared(trained_weight, X_test, y_test))
    
    # plt.figure()
    # plt.xlabel('uni variate training')
    # plt.ylabel('R-square value')
    # plt.plot(R_squared_value)
    # plt.show()
        
    
    # multi-variate GD
    X_train, X_test, y_train, y_test = Generate_data(training_size, data)
    
    
    weight = np.zeros((X_train.shape[1], 1))
    if (HYPER_PARAMs['Normalize']==False):
        trained_weight, multi_errors = gradient_descent(X_train, y_train, weight, HYPER_PARAMs['alpha'], \
                                        HYPER_PARAMs['iters'], HYPER_PARAMs['Flag'])
        iters = HYPER_PARAMs['iters']

    elif(HYPER_PARAMs['Normalize']):
        trained_weight, multi_errors = gradient_descent(X_train, y_train, weight, HYPER_PARAMs_NORMALIZE['alpha'], \
                                        HYPER_PARAMs_NORMALIZE['iters'], HYPER_PARAMs['Flag'])
        iters = HYPER_PARAMs_NORMALIZE['iters'] 

    else:
        print('normalizing value wrong!')
        raise SystemExit('It failed!')
        
    multi_testing_errors = [cal_loss(trained_weight, X_test, y_test, HYPER_PARAMs['Flag'])]
    R_squared_value.append(R_squared(trained_weight, X_test, y_test))
    print('R-square values are:' + str(R_squared_value))
    
    labels = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Muli_variables']

    for i, error in enumerate(uni_errors + [multi_errors]):
        plt.plot([i for i in range(int(iters))], error, label=labels[i])
    plt.legend()
    plt.xlabel('iteration number')
    plt.ylabel('value of losses')
    plt.title('training losses')
    plt.savefig('Loss.jpg', dpi=500)
    plt.show()

    print('testing errors are (uni for top 8, multi for the last)' + str(uni_testing_errors + multi_testing_errors))