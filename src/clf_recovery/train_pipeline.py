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
parser.add_argument('--init', type=int, default=0)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--solver_state', type=str, required=False)
parser.add_argument('--gpu_id', type=int, required=True)
parser.add_argument('--test_interval', type=int, required=True)
parser.add_argument('--mode', type=int, required=True) # 1 for center view only, # 2 for disparity only, # 3 for whole pipeline
parser.add_argument('--log_path', type=str, required=True)

args = parser.parse_args()

log_path = args.log_path
#print args 
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
	#solver.net.copy_from(args.model)
	solver.restore(args.solver_state)
	solver.test_nets[0].share_with(solver.net)

if(args.init):
	solver.net.copy_from('pipeline/t3/cv2/cv_iter_65000.caffemodel')
	solver.net.copy_from('good_models/ssim_trblnet/full_c2/clf_iter_170000.caffemodel')
	solver.test_nets[0].share_with(solver.net)
	
#fid = open(log_path+'val_log.txt','a')
#fid.write(str(0)+'\n')
#fid.close()
show_results = True
show_interval = 800
print "training........"
for it in range(args.maxiter):
	#print it, 'before'
	
	solver.step(1)  # SGD by Caffe

	cv_loss = solver.net.blobs['loss_cv'].data[0]
	fid = open(log_path+'cv_train_log.txt','a')
	fid.write(str(cv_loss)+'\n')
	fid.close()
		
	train_loss = solver.net.blobs['loss'].data[0]
	fid = open(log_path+'train_log.txt','a')
	fid.write(str(train_loss)+'\n')
	fid.close()

	strain_loss = solver.net.blobs['loss_ssim'].data[0]
	fid = open(log_path+'ssim_train_log.txt','a')
	fid.write(str(strain_loss)+'\n')
	fid.close()

	if it % 100 == 0:
		print it, cv_loss, train_loss, strain_loss

	test_loss = 0
	cv_loss = 0
	sloss = 0
	if (it % test_interval == 0 and it!=0):
		print 'Iteration', it, 'testing...'
		batch_size = solver.test_nets[0].blobs['data'].num
		#print 'batchsize', batch_size
		test_iters = int(30000 / batch_size)
		for i in range(test_iters):
			solver.test_nets[0].forward()
			test_loss += solver.test_nets[0].blobs['loss'].data[0]
			
			cv_loss += solver.test_nets[0].blobs['loss_cv'].data[0]
			sloss += solver.test_nets[0].blobs['loss_ssim'].data[0]

			np.random.seed(345)
			idxs = np.random.choice(batch_size,1)
			if show_results and it%show_interval==0:
				#print 'saving the qualitative results'				
				coded_imgs = solver.test_nets[0].blobs['data'].data[...] # n, c, h, w
				img_save(coded_imgs[idxs],-1,i*1,'acoded_')

				reconst_views = solver.test_nets[0].blobs['cropped_cv'].data[...] # 50, 50
				img_save(reconst_views[idxs],-1,i*1,'breconst_')
				
				center_views = solver.test_nets[0].blobs['crop_label'].data[:,:,5:-5,5:-5] # 50,50
				img_save(center_views[idxs],-1,i*1,'ccenterview_')
				
				if args.mode==3:
					tr_views = solver.test_nets[0].blobs['predicted_view'].data[...] # 
					img_save(tr_views[idxs],-1,i*1,'etrview_')

					gt_views = solver.test_nets[0].blobs['cropped_gt'].data[...] # 
					img_save(gt_views[idxs],-1,i*1,'fact_trview_')					

					disparity = solver.test_nets[0].blobs['disparity'].data[...] #
					img_save(disparity[idxs],-1,i*1,'ddisp_')					


		test_loss /= test_iters
		fid = open(log_path+'val_log.txt','a')
		fid.write(str(test_loss)+'\n')
		fid.close()
		
		cv_loss /= test_iters		
		fid = open(log_path+'cv_val_log.txt','a')
		fid.write(str(cv_loss)+'\n')
		fid.close()

		sloss /= test_iters		
		fid = open(log_path+'ssim_val_log.txt','a')
		fid.write(str(sloss)+'\n')
		fid.close()

print 'done'