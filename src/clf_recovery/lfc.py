import h5py as h5
import numpy as np 
from PIL import Image
import os
from scipy.misc import imsave, imread
import matplotlib.image as mplimg

def lfsave(path,idx,key,img):
	if not os.path.isdir(path):
		os.mkdir(path)
			
	path = path+'/'+key+'_'
	for i in range(img.shape[2]):
		for j in range(img.shape[3]):
			imsave(path+str(i)+str(j)+'.png',img[:,:,i,j,:])
'''
fname = '/media/data/susmitha/flowers_dataset/Flowers_8bit/IMG_8207_eslf.png'
h = 372
w = 540

I = Image.open(fname)
I = I.convert('RGB')
I = np.array(I)
print I.shape

I = I[:h*14,:w*14,:]
Ir = I.reshape(h,14,w,14,3)
print Ir.shape
Ir = Ir.transpose(0,2,1,3,4)
print Ir.shape	

lfsave('/home/susmitha/clf_recovery/lfcheck',0,'sdread',Ir[:,:,3:-3,3:-3,:])


def trainpath(n):
    trainPath = "/media/data/susmitha/lf_data/TrainingData/patchLf/training_"+str(n+1)+".h5"
    return trainPath

def getLf(data,lfsize):
    B,c,H,W = data.shape
    v = lfsize[3]
    u = lfsize[2]
    lf = data.reshape(B,-1,3,W,H)
    lf = lf.reshape(B,v,u,3,W,H)
    lf = lf.transpose(0,5,4,2,1,3) # B, H, W, v, u, 3

    return lf

print "Reading training file .... "
dataf = h5.File(trainpath(3),'r')
data = dataf['data']
dataCount = 0
print 'done', data.shape

idx = 123
lf = np.array(data[idx:idx+1])

lfsize = [372,540,8,8]

lf = getLf(lf,lfsize)
lfsave('/home/susmitha/clf_recovery/lfcheck',1,'h5read',lf.squeeze())

'''
print 'here'
fname = '/media/data/susmitha/flowers_dataset/Flowers_8bit/IMG_8207_eslf.png'
import sys
sys.path.insert(0, '/home/susmitha/clf_recovery/')
from readIlliumImages_v1 import illiumTools as tools
tools = tools()
Lf = tools.readIlliumImg(fname) # numViews,numViews,viewH,viewW,3
Lf = Lf[:,:,0:360,0:540,:]
lfsave('/home/susmitha/clf_recovery/lfcheck',2,'ilread',Lf.transpose(2,3,0,1,4))
print 'done'
