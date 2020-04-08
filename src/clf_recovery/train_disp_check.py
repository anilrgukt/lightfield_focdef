import sys
import matplotlib.image as mplimg
from scipy.misc import imsave, toimage
sys.path.append('./')
import os
#os.environ['GLOG_minloglevel'] = '2' 
import caffe

import numpy as np
from pylab import *
import argparse

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--solver', type=str, required=True)
parser.add_argument('--maxiter', type=int, required=True)
parser.add_argument('--retrain', type=int, default=0)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--solver_state', type=str, required=False)
parser.add_argument('--gpu_id', type=int, required=True)
parser.add_argument('--test_interval', type=int, required=True)
parser.add_argument('--log_path', type=str, required=True)
args = parser.parse_args()

log_path = args.log_path
print args 
#os.remove(log_path+'train_log.txt')
#os.remove(log_path+'val_log.txt')

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
		#imsave(log_path+'val_qual_res/'+img_name,toimage(img))

caffe.set_device(args.gpu_id)
caffe.set_mode_gpu()
test_interval = args.test_interval
solver_proto = args.solver
solver = caffe.get_solver(solver_proto)
if(args.retrain):
	#solver.net.copy_from('good_models/tr_bl2net_with_small_interpolation/clf_iter_250000.caffemodel')
	solver.restore('good_models/tr_bl2net_with_small_interpolation/clf_iter_250000.solverstate')

#fid = open(log_path+'val_log.txt','a')
#fid.write(str(0)+'\n')
#fid.close()
show_results = True
show_interval = 2000
print "training........"
for it in range(args.maxiter):
	#print it, 'before'
	
	solver.step(1)  # SGD by Caffe
	train_loss = solver.net.blobs['loss'].data
	#print 'check', solver.net.blobs['conv9'].num

	fid = open(log_path+'train_log.txt','a')
	fid.write(str(train_loss)+'\n')
	fid.close()

	print it, train_loss
	test_loss = 0
	if (it % test_interval == 0):
		print 'Iteration', it, 'testing...'
		batch_size = solver.test_nets[0].blobs['data'].num
		#print 'batchsize', batch_size
		test_iters = int(18000 / batch_size)
		for i in range(test_iters):
			solver.test_nets[0].forward()
			test_loss += solver.test_nets[0].blobs['loss'].data

			np.random.seed(345)
			idxs = np.random.choice(batch_size,1)
			#print i, idxs
			if show_results and it%show_interval==0:
				#print 'saving the qualitative results'				
				coded_imgs = solver.test_nets[0].blobs['data'].data[...] # n, c, h, w
				img_save(coded_imgs[idxs],-1,i*1,'acoded_')

				center_views = solver.test_nets[0].blobs['label'].data[...]
				img_save(center_views[idxs],-1,i*1,'ccenterview_')

				pred_disp = solver.test_nets[0].blobs['warped_cv'].data[...]
				img_save(pred_disp[idxs],it,i*1,'dwarpedviewcv_')

				pred_disp = solver.test_nets[0].blobs['disparity'].data[...]
				img_save(pred_disp[idxs],it,i*1,'bdisparity_')

				reconst_views = solver.test_nets[0].blobs['predicted_view'].data[...]
				img_save(reconst_views[idxs],it,i*1,'ereconst_')

				actual_views = solver.test_nets[0].blobs['crop_gt'].data[...]
				img_save(actual_views[idxs],-1,i*1,'ftopright_')

		test_loss /= test_iters
		fid = open(log_path+'val_log.txt','a')
		fid.write(str(test_loss)+'\n')
		fid.close()
	

print 'done'
