import numpy as np

def load():
    x_train = np.load('cifar/data/npy/x_train.npy')
    t_train = np.load('cifar/data/npy/t_train.npy')
    x_test = np.load('cifar/data/npy/x_test.npy')
    t_test = np.load('cifar/data/npy/t_test.npy')
    return x_train, t_train, x_test, t_test
   
