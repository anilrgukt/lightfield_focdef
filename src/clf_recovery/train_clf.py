import sys
import matplotlib.image as mplimg
from scipy.misc import imsave, toimage
sys.path.append('./')
import os
#os.environ['GLOG_minloglevel'] = '2' 
import caffe
from tools import model

import numpy as np
from pylab import *
import argparse

def img_save(imgs,iter,start_id,type):
	if imgs.ndim<4:
		imgs = imgs[np.newaxis,:]
		
	for i, img in enumerate(imgs):
		#print 'saving the image', i
		img = img[:,:,:,np.newaxis]
		img = img.swapaxes(0,3)
		img = img.squeeze()
		if iter>=0:
			iter=str(iter)
		else:
			iter=''

		img = img.swapaxes(0,1)
		img_name = str(start_id+i).zfill(4)+'_'+type+iter+'.png'
		img = toimage(img)
		img.save(log_path+'val_qual_res/'+img_name)

parser = argparse.ArgumentParser()
parser.add_argument('--solver', type=str, required=True)
parser.add_argument('--maxiter', type=int, required=True)
parser.add_argument('--retrain', type=int, default=0)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--solver_state', type=str, required=False)
parser.add_argument('--vars', type=str, required=True, nargs='+')
parser.add_argument('--vars_it', type=str, required=True, nargs='+')
parser.add_argument('--gpu_id', type=int, required=True)
parser.add_argument('--test_interval', type=int, required=True)
parser.add_argument('--log_path', type=str, required=True)

args = parser.parse_args()

log_path = args.log_path

caffe.set_mode_gpu()
caffe.set_device(args.gpu_id)

test_interval = args.test_interval
solver_proto = args.solver
solver = caffe.get_solver(solver_proto)

if(args.retrain):
	#solver.net.copy_from(args.model)
	#solver.test_nets[0].copy_from(args.model)
	solver.restore(args.solver_state)
	solver.test_nets[0].share_with(solver.net)

show_results = True
show_interval = 800
print "training........"
idxs = ['a','b','c','d','e','f','g','h','i']

for it in range(args.maxiter):

	solver.step(1)  # SGD by Caffe

	#print 'done step'
	train_loss = solver.net.blobs['loss'].data
	fid = open(log_path+'train_log.txt','a')
	fid.write(str(train_loss)+'\n')
	fid.close()
	
	if it%100 == 0:
		print it, train_loss

	test_loss = 0

	if (it % test_interval == 0):
		#solver.test_nets[0].copy_from(solver.net)
		print 'Iteration', it, 'testing...'
		
		batch_size = solver.test_nets[0].blobs['data'].num
		test_iters = int(54720 / batch_size)
		for i in range(test_iters):
			solver.test_nets[0].forward()
			test_loss += solver.test_nets[0].blobs['loss'].data
							
			c = 0
			np.random.seed(345)
			ids = np.random.choice(batch_size,1)			
			if show_results and it%show_interval==0:
				#print 'saving the qualitative results'
				for x in args.vars:
					imgs = solver.test_nets[0].blobs[x].data[...]		
					img_save(imgs[ids],-1,i*1,idxs[c]+x+'_')
					c += 1
				for x in args.vars_it:
					imgs = solver.test_nets[0].blobs[x].data[...]		
					img_save(imgs[ids],it,i*1,idxs[c]+x+'_')
					c += 1

		test_loss /= test_iters
		fid = open(log_path+'val_log.txt','a')
		fid.write(str(test_loss)+'\n')
		fid.close()
		
print 'done'
