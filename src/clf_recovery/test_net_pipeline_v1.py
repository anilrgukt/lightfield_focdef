import os
import sys
os.environ['GLOG_minloglevel'] = '2' 
import caffe
import math
import matplotlib.image as mplimg
from scipy.misc import imsave
from os import listdir
from scipy.io import savemat, loadmat
from tools.utils import swap, prep_batch
import numpy as np
from skimage.measure import structural_similarity as ssim
from skimage import color
import argparse

def overlap_reconst(args):
	from readIlliumImages_whole import illiumTools as tools

	folder_path = args.valData
	path = args.path
	fname = 'reconst_overlap/'

	if not os.path.isdir(path+fname):
		os.mkdir(path+fname)
	path += fname

	img_list = listdir(folder_path)
	n = 1 # number of views
	B = 5 # batch size
	v = 3 # ref view 3 center one currently not being used
	oP = 70 # output patch size from the network
	P = 90 # patch size
	S = oP + 1 # stride 
	code = loadmat(args.code)['fullCode']
	print 'code shape', code.shape

	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)

	proto = args.path+args.deploy
	model = args.path+'clf_iter_'+str(args.iter)+'.caffemodel'
	
	cv_model = '/home/susmitha/clf_recovery/pipeline_v1/cv_clf_iter_95000.caffemodel'
	cv_proto = '/home/susmitha/clf_recovery/gautham/rednet/deploy.prototxt'

	print 'loading disp net'	
	net = caffe.Net(proto,model,caffe.TEST) 
	print 'loading cv net'	
	cv_net = caffe.Net(cv_proto, cv_model, caffe.TEST)

	tools = tools()
	tools.verbosity = 0
	tools.op = oP
	tools.p = P
	tools.s = oP

	count = 0
	avg_loss = []
	c = 3	
	for img_name in img_list:
		print 'processing LF img', img_name
		if img_name[-4:] != '.png':
			print 'not an image'
			continue

		img_name = folder_path+img_name
		fullLF = tools.readIlliumImg(img_name) # numViews,numViews,viewH,viewW,3
		print 'read fullLF', fullLF.shape	

		selLF = tools.getRandViews(fullLF, n) # n,self.angRes,self.angRes,viewH,viewW,3
		print 'selected LFs size', selLF.shape	

		for viewLF in selLF:
			print 'count check', count
			viewLF = viewLF[:,:,0:360,0:540,:]
			tools.viewH = viewLF.shape[2]
			tools.viewW = viewLF.shape[3]
		
			codedImg = tools.getCodedImgWhole(viewLF, code) # H, W, 3		
			pCodedImg, rlist, clist = tools.extract_patches_strided(codedImg,60) # N, P, P, 3 		
			
			refView = np.squeeze(viewLF[2,2])
			trView = np.squeeze(viewLF[0,4])
			blView = np.squeeze(viewLF[4,0])
			prefView,rlist,clist = tools.extract_patches_strided(refView,60) # N,P,P,3
			ptr,rlist,clist = tools.extract_patches_strided(trView,60) # N,P,P,3
			pbl,rlist,clist = tools.extract_patches_strided(blView,60) # N,P,P,3
			
			numPatches = pCodedImg.shape[0]				

			outPatches = np.zeros((numPatches,oP,oP,3))
			dispPatches = np.zeros((numPatches,oP,oP,1))
			warpPatches = np.zeros((numPatches,oP,oP,3))

			batch_list = range(0,numPatches,B)
			cv_net.blobs['data'].reshape(B,3,P,P)
			cv_net.blobs['label'].reshape(B,3,P,P)			
			net.blobs['data'].reshape(B,3,P,P)
			net.blobs['label'].reshape(B,3,80,80)		
			net.blobs['label_tr'].reshape(B,3,P,P)		
			net.blobs['label_bl'].reshape(B,3,P,P)		
			net.reshape()

			#print 'forming batches',batch_list
			for batch_id in batch_list:
				#print 'bid', batch_id
				batch = prep_batch(pCodedImg[batch_id:batch_id+B]) # coded data
				batch_label = prep_batch(prefView[batch_id:batch_id+B]) # gt center view for loss
				batch_tr = prep_batch(ptr[batch_id:batch_id+B]) # gt center view for loss
				batch_bl = prep_batch(pbl[batch_id:batch_id+B]) # gt center view for loss
				if batch.shape[0] != B:
					cv_net.blobs['data'].reshape(batch.shape[0],3,P,P)
					cv_net.blobs['label'].reshape(batch.shape[0],3,P,P)
									
					net.blobs['data'].reshape(batch.shape[0],3,P,P)
					net.blobs['label'].reshape(batch.shape[0],3,80,80)
					net.blobs['label_tr'].reshape(batch.shape[0],3,P,P)
					net.blobs['label_bl'].reshape(batch.shape[0],3,P,P)
					net.reshape()

				#print 'forward pass..'		
				cv_net.blobs['data'].data[...] = batch
				cv_net.blobs['label'].data[...] = batch_label
				output = cv_net.forward()
				
				label = np.zeros([batch.shape[0],3,P,P])
				label = cv_net.blobs['gen_cv'].data[...]
					
				cr = 5
				net.blobs['data'].data[...] = batch
				net.blobs['label'].data[...] = label[:,:,2:-2,2:-2]
				net.blobs['label_tr'].data[...] = batch_tr
				net.blobs['label_bl'].data[...] = batch_bl
				output = net.forward()

				reconst = net.blobs['cropped_cv'].data[...] # B, 3, oP, oP # gen_cv2
				#reconst = reconst[:,:,cr:-cr,cr:-cr]
				reconst = swap(reconst)

				disparity = net.blobs['disparity'].data[...] # B, 3, oP, oP
				#disparity = disparity[:,:,cr:-cr,cr:-cr]
				disparity = swap(disparity)

				warped = net.blobs['predicted_view'].data[...] # B, 3, oP, oP
				#warped = warped[:,:,cr:-cr,cr:-cr]
				warped = swap(warped)

				outPatches[batch_id:batch_id+B,:,:,:] = reconst.swapaxes(-2,-3)
				dispPatches[batch_id:batch_id+B,:,:,:] = disparity.swapaxes(-2,-3)
				warpPatches[batch_id:batch_id+B,:,:,:] = warped.swapaxes(-2,-3)

				avg_loss.append(output['loss'])

			reconstView = tools.combine_patches(outPatches, rlist, clist)	
			imsave(path+str(count).zfill(3)+'_ccenter_'+str(args.iter/1000)+'k.png', reconstView)
			disparity = tools.combine_patches(dispPatches, rlist, clist)			
			imsave(path+str(count).zfill(3)+'_bdisp_'+str(args.iter/1000)+'k.png', disparity.squeeze())
			warpView = tools.combine_patches(warpPatches, rlist, clist)	
			imsave(path+str(count).zfill(3)+'_etr_'+str(args.iter/1000)+'k.png', warpView)
			imsave(path+str(count).zfill(3)+'_acoded.png', codedImg)
			imsave(path+str(count).zfill(3)+'_dref.png', refView)
			imsave(path+str(count).zfill(3)+'_fatr.png', trView)
			
			count+=1
	
	avg_loss = np.array(avg_loss)
	print 'avg loss', np.mean(avg_loss)
	
	return 0

def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--valData', type=str, default = '/media/data/susmitha/lf_data/TrainingData/Val/')
	parser.add_argument('--iter', type=int, required=True)
	parser.add_argument('--deploy', type=str, required=True)
	parser.add_argument('--code', type=str, required=True)
	parser.add_argument('--gpu_id', type=int, required=True)
	parser.add_argument('--patch', type=int, default=1)
	parser.add_argument('--path', type=str, required=True)
	args = parser.parse_args()		
	
	if not args.patch:
		return overlap_reconst(args)

	folder_path = args.valData
	path = args.path
	fname = 'reconst/'

	if not os.path.isdir(path+fname):
		os.mkdir(path+fname)
	path += fname

	img_list = listdir(folder_path)
	n = 1 # number of views
	B = 70 # batch size
	v = 3 # ref view 3 center one currently not being used
	oP = 30 # output patch size from the network
	P = 60 # patch size
	S = oP + 1 # stride 
	code = loadmat(args.code)['fullCode']
	print 'code shape', code.shape

	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)

	proto = args.path+args.deploy
	model = args.path+'clf_iter_'+str(args.iter)+'.caffemodel'
	
	cv_model = '/home/susmitha/clf_recovery/pipeline_v1/no_disp_cat/cv_clf_iter_15000.caffemodel'
	cv_proto = '/home/susmitha/clf_recovery/gautham/rednet/deploy.prototxt'

	print 'loading disp net'	
	net = caffe.Net(proto,model,caffe.TEST) 
	print 'loading cv net'	
	cv_net = caffe.Net(cv_proto, cv_model, caffe.TEST)
	
	from readIlliumImages import illiumTools as tools
	tools = tools()
	tools.verbosity = 0
	tools.op = oP
	tools.p = P
	tools.s = oP

	count = 0
	avg_loss = []
	c = 2

	for img_name in img_list:
		print 'processing LF img', img_name
		if img_name[-4:] != '.png':
			print 'not an image'
			continue

		img_name = folder_path+img_name
		fullLF = tools.readIlliumImg(img_name) # numViews,numViews,viewH,viewW,3
		print 'read fullLF', fullLF.shape	

		selLF = tools.getRandViews(fullLF, n) # n,self.angRes,self.angRes,viewH,viewW,3
		print 'selected LFs size', selLF.shape	
		

		for viewLF in selLF:
			# viewLF - v,v,H,W,3
			print 'count check', count
			pLFs, rlist, clist = tools.extract_patches(viewLF) # N,v,v P, P, 3		
			pCodedImg = tools.getCodedImg(pLFs, code) # N, P, P, 3
			numPatches = pCodedImg.shape[0]		
			refView = np.squeeze(viewLF[2,2])
			trView = np.squeeze(viewLF[0,4])			
			prefView = pLFs[:,2,2,:,:,:]
			ptr = pLFs[:,0,4,:,:,:]
			pbl = pLFs[:,4,0,:,:,:]
		
			codedImg = tools.combine_patches(pCodedImg, rlist, clist)

			outPatches = np.zeros((numPatches,oP,oP,3))
			dispPatches = np.zeros((numPatches,oP,oP,3))
			warpPatches = np.zeros((numPatches,oP,oP,3))
			
			batch_list = range(0,numPatches,B)

			cv_net.blobs['data'].reshape(B,3,P,P)
			cv_net.blobs['label'].reshape(B,3,P,P)
			net.blobs['data'].reshape(B,3,P,P)
			net.blobs['label'].reshape(B,3,50,50)		
			net.blobs['label_tr'].reshape(B,3,P,P)		
			net.blobs['label_bl'].reshape(B,3,P,P)		
			net.reshape()

			#print 'forming batches',batch_list
			for batch_id in batch_list:
				#print 'bid', batch_id
				batch = prep_batch(pCodedImg[batch_id:batch_id+B]) # coded data
				batch_label = prep_batch(prefView[batch_id:batch_id+B]) # gt center view for loss
				batch_tr = prep_batch(ptr[batch_id:batch_id+B]) # gt center view for loss
				batch_bl = prep_batch(pbl[batch_id:batch_id+B]) # gt center view for loss

				if batch.shape[0] != B:
					cv_net.blobs['data'].reshape(batch.shape[0],3,P,P)
					cv_net.blobs['label'].reshape(batch.shape[0],3,P,P)

					net.blobs['data'].reshape(batch.shape[0],3,P,P)					
					net.blobs['label'].reshape(batch.shape[0],3,50,50)
					net.blobs['label_tr'].reshape(batch.shape[0],3,P,P)
					net.blobs['label_bl'].reshape(batch.shape[0],3,P,P)
					net.reshape()

				#print 'forward pass..'			
				cv_net.blobs['data'].data[...] = batch
				cv_net.blobs['label'].data[...] = batch_label
				output = cv_net.forward()
				
				#label = np.zeros([batch.shape[0],3,P,P])
				label = cv_net.blobs['gen_cv'].data[...]
				label = label[:,:,c:-c,c:-c]
				net.blobs['data'].data[...] = batch
				net.blobs['label'].data[...] = label
				net.blobs['label_tr'].data[...] = batch_tr
				net.blobs['label_bl'].data[...] = batch_bl
				output = net.forward()

				cr = 5				
				reconst = net.blobs['cropped_cv'].data[...] # B, 3, oP, oP
				reconst = reconst[:,:,cr:-cr,cr:-cr]
				reconst = swap(reconst)

				disparity = net.blobs['disparity'].data[...] # B, 3, oP, oP
				disparity = disparity[:,:,cr:-cr,cr:-cr]
				disparity = swap(disparity)

				warped = net.blobs['predicted_view'].data[...] # B, 3, oP, oP
				warped = warped[:,:,cr:-cr,cr:-cr]
				warped = swap(warped)

				outPatches[batch_id:batch_id+B,:,:,:] = reconst.swapaxes(-2,-3)
				dispPatches[batch_id:batch_id+B,:,:,:] = disparity.swapaxes(-2,-3)
				warpPatches[batch_id:batch_id+B,:,:,:] = warped.swapaxes(-2,-3)

				avg_loss.append(output['loss'])

			reconstView = tools.combine_patches(outPatches, rlist, clist)	
			imsave(path+str(count).zfill(3)+'_ccenter_'+str(args.iter/1000)+'k.png', reconstView)
			disparity = tools.combine_patches(dispPatches, rlist, clist)	
			imsave(path+str(count).zfill(3)+'_bdisp_'+str(args.iter/1000)+'k.png', disparity)
			warpView = tools.combine_patches(warpPatches, rlist, clist)	
			imsave(path+str(count).zfill(3)+'_etr_'+str(args.iter/1000)+'k.png', warpView)
			imsave(path+str(count).zfill(3)+'_acoded.png', codedImg)
			imsave(path+str(count).zfill(3)+'_dref.png', refView)
			imsave(path+str(count).zfill(3)+'_fatr.png', trView)

			count+=1

	
	avg_loss = np.array(avg_loss)
	print 'avg loss', np.mean(avg_loss)


if __name__ == '__main__':
	sys.exit(main(sys.argv))
