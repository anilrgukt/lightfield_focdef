import sys
import matplotlib.image as mplimg
sys.path.append('./')
import os
#os.environ['GLOG_minloglevel'] = '2' 
import caffe
from scipy.misc import imsave

import numpy as np
from pylab import *
import argparse

def swap_image(img):
	img = img[:,:,:,np.newaxis]
	img = img.swapaxes(0,3)
	return img.squeeze()

def lfsave(path,idx,key,img):
	if not os.path.isdir(path):
		os.mkdir(path)
			
	path = path+'/'+key+'_'
	for i in range(img.shape[2]):
		for j in range(img.shape[3]):
			imsave(path+str(i)+str(j)+'.png',img[:,:,i,j,:])

def save_img(data,name):
	count = 0
	for img in data:
		img = img.swapaxes(-1,-2)
		img = swap_image(img)
		imsave('tmp/'+str(count)+name, img)
		count+=1   

def getLf(data):
	B,c,H,W = data.shape
	v = int(np.sqrt(c/3))
	lf = data.reshape(B,-1,3,W,H)
	lf = lf.reshape(B,v,v,3,W,H)
	lf = lf.transpose(0,5,4,2,1,3) # B, H, W, v, u, 3	
	
	return lf

caffe.set_mode_gpu()
caffe.set_device(2)

solver_proto = 'def_model/check_solver.prototxt'
solver = caffe.get_solver(solver_proto)

print "training........"
for it in range(1):
	print 'iter', it	
	solver.step(1)  # SGD by Caffe

	data = solver.net.blobs['data'].data[...]
	print 'loaded data shape', data.shape
	lf = getLf(data)	
	lfsave('/home/susmitha/clf_recovery/lfcheck',1,'cfread',lf.squeeze())
	'''
	label = solver.net.blobs['label'].data[...]
	scene = solver.net.blobs['scene'].data[...]
	pos = solver.net.blobs['pos'].data[...]
	#pos = (pos - 1)*6 + 1

	save_img(data/49.0,'adata_.png')
	save_img(label,'clabel_.png')

	save_img(scene,'bscene.png')

	print pos.shape, data.max(), pos.max()
	'''
	#print pos.astype(int)
	#print np.histogram(pos.astype(int), bins=[1,2,3,4,5,6,7,8])

print 'done'
