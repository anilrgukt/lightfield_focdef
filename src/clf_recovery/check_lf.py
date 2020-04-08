import h5py as h5py
import numpy as np 
import os
from scipy.misc import imsave


def lfsave(path,idx,key,img):
	if not os.path.isdir(path):
		os.mkdir(path)
			
	path = path+'/'+key+'_'
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			imsave(path+str(i)+str(j)+'.png',img[i,j])


data = None; 
print 'reading h5 file'
f = h5py.File("/media/data/susmitha/lf_data/TrainingData/patchLf/training_1.h5","r")
print 'done'

data = f['data']
#for value in f.values():
#	if(data is not None):
#		data = np.concatenate((data,value[...]),axis=0)
#	else:
#		data = value[...]
#f.close()

print data.shape

lf = data[678]
lf = np.array(lf)
lf = lf.reshape(64,3,60,60)
lf = lf[:,:,:,:,np.newaxis]
lf = lf.swapaxes(1,-1)
lf = lf.squeeze()

lf = lf.reshape(8,8,60,60,3)
print lf.shape
lf = lf.swapaxes(-2,-3)
lf = lf.swapaxes(0,1)
print lf.shape

lfsave('/home/susmitha/clf_recovery/lfcheck',0,'c',lf)


