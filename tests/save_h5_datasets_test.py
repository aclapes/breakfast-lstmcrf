import os
import h5py
import numpy as np
import time

# FEA_DIM = 224*224*3
#
# class Data(object):
#     def __init__(self):
#         self.vectors = []
#         self.labels = []
#         self.L = [70,58,120] #,667,968,444,5420,591,1860,3435,941,704,310,968,3141,3694,3874,2386,1860,979,1401,1105,461,1261,3313,3094,3017,5448,1850]
#         for i in range(len(self.L)):
#             length = self.L[i]
#             self.vectors.append(np.random.randint(0,10,(length,FEA_DIM)).astype(dtype=np.float32))
#             self.labels.append(np.random.randint(0,10,(length,)).astype(dtype=np.int64))
#
# data = Data()
#
# path = 'out.h5'
# os.remove(path)
#
# st_time = time.time()
# print 'started'
# with h5py.File(path, "a", libver='latest') as f:
#     # dt = h5py.special_dtype(vlen=np.dtype('uint8'))
#
#     st2_time = time.time()
#     dset = f.create_dataset('voltage284', (1000,10000,FEA_DIM), fillvalue=0, maxshape=(None,None,FEA_DIM),
#                             dtype=np.uint8, chunks=(32,100,FEA_DIM), compression='gzip', compression_opts=1)
#     print time.time() - st2_time
#     for i in range(len(data.L)):
#         # dset.resize((dset.shape[0] + 5 if i >= dset.shape[0] else dset.shape[0],
#         #              data.vectors[i].shape[0] if dset.shape[1] < data.vectors[i].shape[0] else dset.shape[1],
#         #              FEA_DIM))
#         dset[i, :data.vectors[i].shape[0], :] = data.vectors[i]
#         # dset[-1] = np.random.random(10**4)
# print time.time() - st_time

with h5py.File('test.hdf5', 'w') as f:
    image_dt = np.dtype('(240,320,3)<u1, <i4')
    dset = f.create_dataset('array', (100000,), dtype=image_dt, chunks=(1,), compression='gzip', compression_opts=9)

    a = dset[0]
    # dset[0] = np.zeros((224,224,3), dtype=np.uint8)

print  'done'