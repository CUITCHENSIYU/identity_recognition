import numpy as np

def descale(data):
    return (data-8388608)/8388608*5000000/50

def normlize(data, mean, std):
    for i in range(np.array(data).shape[0]):
        data[i,:] = (data[i,:]-mean[i])/std[i]
    return data
