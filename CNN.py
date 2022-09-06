import numpy as np
#import tensorflow as tf

ls = []
path = '/Users/yaoshunyu/Desktop/SMD_film_thinkness/ML_CNN/histfile.npz'
with np.load(path) as np_data:
    for i in range(3):
        run = 'arr_' + str(i)
        a = np_data[run]
        ls.append(a)

c = np.array(ls)
print(c)
print(c.shape)

