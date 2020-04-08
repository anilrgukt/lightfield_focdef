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
cv_solver = caffe.get_solver('/home/susmitha/clf_recovery/pipeline_v1/cv_solver.prototxt')
cv_solver.net.copy_from('/home/susmitha/clf_recovery/gautham/rednet/clf_iter_120000.caffemodel')
cv_solver.test_nets[0].share_with(cv_solver.net)

if(args.retrain):
	solver.net.copy_from(args.model)
	solver.restore(args.solver_state)

if(args.init):
	solver.net.copy_from('/home/susmitha/clf_recovery/good_models/tr_bl2net_with_small_interpolation/clf_iter_252000.caffemodel')
	solver.test_nets[0].share_with(solver.net)

'''
solver.net.params['pred_gxy'][0].data[0,0] = np.array([[1,0,1],[2,0,-2],[1,0,-1]])
solver.net.params['pred_gxy'][0].data[0,1] = np.array([[1,0,1],[2,0,-2],[1,0,-1]])
solver.net.params['pred_gxy'][0].data[0,2] = np.array([[1,0,1],[2,0,-2],[1,0,-1]])
solver.net.params['pred_gxy'][0].data[1,0] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
solver.net.params['pred_gxy'][0].data[1,1] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
solver.net.params['pred_gxy'][0].data[1,2] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

solver.net.params['label_gxy'][0].data[0,0] = np.array([[1,0,1],[2,0,-2],[1,0,-1]])
solver.net.params['label_gxy'][0].data[0,1] = np.array([[1,0,1],[2,0,-2],[1,0,-1]])
solver.net.params['label_gxy'][0].data[0,2] = np.array([[1,0,1],[2,0,-2],[1,0,-1]])
solver.net.params['label_gxy'][0].data[1,0] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
solver.net.params['label_gxy'][0].data[1,1] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
solver.net.params['label_gxy'][0].data[1,2] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

cv_solver.net.params['pred_gxy'][0].data[0,0] = np.array([[1,0,1],[2,0,-2],[1,0,-1]])
cv_solver.net.params['pred_gxy'][0].data[0,1] = np.array([[1,0,1],[2,0,-2],[1,0,-1]])
cv_solver.net.params['pred_gxy'][0].data[0,2] = np.array([[1,0,1],[2,0,-2],[1,0,-1]])
cv_solver.net.params['pred_gxy'][0].data[1,0] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
cv_solver.net.params['pred_gxy'][0].data[1,1] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
cv_solver.net.params['pred_gxy'][0].data[1,2] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

cv_solver.net.params['label_gxy'][0].data[0,0] = np.array([[1,0,1],[2,0,-2],[1,0,-1]])
cv_solver.net.params['label_gxy'][0].data[0,1] = np.array([[1,0,1],[2,0,-2],[1,0,-1]])
cv_solver.net.params['label_gxy'][0].data[0,2] = np.array([[1,0,1],[2,0,-2],[1,0,-1]])
cv_solver.net.params['label_gxy'][0].data[1,0] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
cv_solver.net.params['label_gxy'][0].data[1,1] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
cv_solver.net.params['label_gxy'][0].data[1,2] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
'''
#fid = open(log_path+'val_log.txt','a')
#fid.write(str(0)+'\n')
#fid.close()
show_results = True
show_interval = 800
c = 3
print "training........"
for it in range(args.maxiter):
	#print it, 'before'
	if it>15000:
		cv_solver.net.forward()
	else:
		cv_solver.step(1)
	#cv_solver.net.forward()
	#label = np.zeros([50,3,60,60])
	label = cv_solver.net.blobs['gen_cv'].data[...]
	solver.net.blobs['label'].data[...] = label[:,:,2:-2,2:-2]
	
	solver.step(1)  # SGD by Caffe
	
	if(args.mode==3):
		cv_loss = cv_solver.net.blobs['loss'].data
		fid = open(log_path+'cv_train_log.txt','a')
		fid.write(str(cv_loss)+'\n')
		fid.close()
		
	train_loss = solver.net.blobs['loss'].data
	fid = open(log_path+'train_log.txt','a')
	fid.write(str(train_loss)+'\n')
	fid.close()
	if (it%100==0):
		print it, train_loss, cv_loss
	
	test_loss = 0
	cv_loss = 0
	if (it % test_interval == 0):
		print 'Iteration', it, 'testing...'
		batch_size = solver.test_nets[0].blobs['data'].num
		#print 'batchsize', batch_size
		test_iters = int(43200 / batch_size)
		
		for i in range(test_iters):
			cv_solver.test_nets[0].forward()
			#label = np.zeros([batch_size,3,60,60])
			label = cv_solver.test_nets[0].blobs['gen_cv'].data[...]
			solver.test_nets[0].blobs['label'].data[...] = label[:,:,2:-2,2:-2]
			solver.test_nets[0].forward()
			
			test_loss += solver.test_nets[0].blobs['loss'].data
			if args.mode==3:
				cv_loss += cv_solver.test_nets[0].blobs['loss'].data

			np.random.seed(345)
			idxs = np.random.choice(batch_size,1)
			if show_results and it%show_interval==0:
				#print 'saving the qualitative results'				
				coded_imgs = solver.test_nets[0].blobs['data'].data[...] # n, c, h, w
				img_save(coded_imgs[idxs],-1,i*1,'acoded_')

				reconst_views = cv_solver.test_nets[0].blobs['gen_cv'].data[...]
				img_save(reconst_views[idxs],it,i*1,'breconst_')
				
				center_views = cv_solver.test_nets[0].blobs['label'].data[...]
				img_save(center_views[idxs],-1,i*1,'ccenterview_')
				
				if args.mode==3:
					tr_views = solver.test_nets[0].blobs['predicted_view'].data[...]
					img_save(tr_views[idxs],it,i*1,'dtrview_')

					disparity = solver.test_nets[0].blobs['disparity'].data[...]
					img_save(disparity[idxs],it,i*1,'edisp_')					


		test_loss /= test_iters
		fid = open(log_path+'val_log.txt','a')
		fid.write(str(test_loss)+'\n')
		fid.close()
		
		if args.mode==3:
			cv_loss /= test_iters		
			fid = open(log_path+'cv_val_log.txt','a')
			fid.write(str(cv_loss)+'\n')
			fid.close()

print 'done'
