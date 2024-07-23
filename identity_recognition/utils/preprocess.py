import numpy as np

def descale(data):
    return (data-8388608)/8388608*5000000/50

def normlize(data, mean, std):
    data = np.clip(data, a_min=-50, a_max=50)
    data = (data + 50) / 100
    # for i in range(np.array(data).shape[0]):
    #     data[i,:] = (data[i,:]-mean[i])/std[i]
    return data
