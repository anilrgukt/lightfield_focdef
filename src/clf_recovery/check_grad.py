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

caffe.set_mode_gpu()
caffe.set_device(1)

solver_proto = 'def_model/check_solver.prototxt'
solver = caffe.get_solver(solver_proto)

print "training........"
for it in range(1):
	print 'iter', it	
	[n,c,h,w] = solver.net.params['grad_x'][0].data.shape
	print n,c,h,w
	solver.net.params['grad_x'][0].data[0,0] = np.array([1,-1])
	solver.net.params['grad_x'][0].data[0,1] = np.array([1,-1])
	solver.net.params['grad_x'][0].data[0,2] = np.array([1,-1])
	solver.step(1)  # SGD by Caffe

	#t = solver.net.params['conv1']
	#print type(t), t
	data = solver.net.blobs['label'].data[...]
	label = solver.net.blobs['grad_x'].data[...]

	count = 0
	for img in data:	
		#print img.shape
		img = img.swapaxes(-1,-2)
		img = swap_image(img)
		imsave('tmp/'+str(count)+'c_.png', img)
		count+=1

	count = 0
	for img in label:
		img = img.swapaxes(-1,-2)
		img = swap_image(img)
		imsave('tmp/'+str(count)+'l_.png', img)
		count+=1   

print 'done'
