import h5py
import numpy as np

def savefile(path, x):
    f = h5py.File(path,'w')
    f['randx'] = x
    f.close()


for i in range(60000):
     path = '/data/tree/mnist/%d.h5' % i
     x = np.random.rand(10, 28, 28)
     x = x.astype(np.float32)
     print(i)
     savefile(path, x)
