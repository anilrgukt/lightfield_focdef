import os
os.environ['GLOG_minloglevel'] = '2' 
import caffe
import math
import matplotlib.image as mplimg
from scipy.misc import imsave
from os import listdir
from scipy.io import savemat, loadmat
from readIlliumImages import illiumTools as tools
import numpy as np
from skimage.measure import structural_similarity as ssim
from skimage import color

def swap(reconst):
	#reconst B, 3, oP, oP
	reconst = reconst.swapaxes(1,2) # B, oP, 3, oP
	reconst = reconst.swapaxes(2,3) # B, oP, oP, 3
	return reconst

def fwd_swap(a): # b,p,p,3 -> b,3,p,p
	a = np.expand_dims(a,axis=1) # b,1,p,p,3
	a = a.swapaxes(1,-1) # b,3,p,p,1
	return a.squeeze() # b,3,p,p

folder_path = '/media/data/susmitha/lf_data/TrainingData/Val/'
path = '/home/susmitha/clf_recovery/models/tr_bl_net2/v1/'

#if not os.path.isdir(path+'test_results'):
#	os.mkdir(path+'test_results')

#path += 'test_results/'

img_list = listdir(folder_path)
n = 1 # number of views
B = 300 # batch size
v = 3 # ref view 3 center one currently not being used
oP = 40 # output patch size from the network
P = 60 # patch size
odP = 50 # output disparity patches

code = loadmat('randn_p13_vcenter.mat')['fullCode']
print 'code shape', code.shape

caffe.set_mode_gpu()
caffe.set_device(0)

proto = 'models/tr_bl_net2/deploy.prototxt'
model = 'models/tr_bl_net2/v1/clf_iter_90000.caffemodel'
net = caffe.Net(proto,model,caffe.TEST) 

tools = tools()
tools.verbosity = 0
tools.p = P
tools.s = oP
tools.op = oP

count = 0
loss = 0
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
		print 'count', count
		pLFs, rlist, clist = tools.extract_patches(viewLF) # N,v,v P, P, 3		
		pCodedImg = tools.getCodedImg(pLFs, code) # N, P, P, 3
		numPatches = pCodedImg.shape[0]		
		label = pLFs[:,2,2,:,:,:] # N,P,P,3
		ptrView = pLFs[:,0,4,:,:,:] #N,P,P,3
		pblView = pLFs[:,4,0,:,:,:] #N,P,P,3
		refView = np.squeeze(viewLF[2,2])
		trView = np.squeeze(viewLF[0,4])		

		codedImg = tools.combine_patches(pCodedImg, rlist, clist)

		outPatches = np.zeros((numPatches,oP,oP,3)) 
		dispPatches = np.zeros((numPatches,odP,odP,1))
		warpPatches = np.zeros((numPatches,odP,odP,3))
		batch_list = range(0,numPatches,B)

		print 'forward pass of batches'
		for batch_id in batch_list:
			print 'bid', batch_id
			batch = pCodedImg[batch_id:batch_id+B] # B, P, P, 3
			batch_label = label[batch_id:batch_id+B] # B, P, P, 3
			batch_label_tr = ptrView[batch_id:batch_id+B] # B, P, P, 3
			batch_label_bl = pblView[batch_id:batch_id+B] # B, P, P, 3


			batch = fwd_swap(batch)
			batch_label = fwd_swap(batch_label)
			batch_label_tr = fwd_swap(batch_label_tr)
			batch_label_bl = fwd_swap(batch_label_bl)
		
			dB = net.blobs['data'].shape[0]
			if batch.shape[0] != dB:				
				net.reshape()
				net.blobs['data'].reshape(batch.shape[0],3,P,P)
				net.blobs['label'].reshape(batch.shape[0],3,P,P)
				net.blobs['label_tr'].reshape(batch.shape[0],3,P,P)
				net.blobs['label_bl'].reshape(batch.shape[0],3,P,P)

			net.blobs['data'].data[...] = batch.swapaxes(-1,-2)
			net.blobs['label'].data[...] = batch_label.swapaxes(-1,-2)
			net.blobs['label_tr'].data[...] = batch_label_tr.swapaxes(-1,-2)
			net.blobs['label_bl'].data[...] = batch_label_bl.swapaxes(-1,-2)
			output = net.forward()
		
			'''disp_reconst = net.blobs['disparity'].data.copy() # B, 3, odP, odP
			disp_reconst = swap(disp_reconst)
			dispPatches[batch_id:batch_id+B,:,:,:] = disp_reconst.swapaxes(-2,-3)

			warp_reconst = net.blobs['warped_centerview'].data.copy() # B, 3, odP, odP
			warp_reconst = swap(warp_reconst)
			warpPatches[batch_id:batch_id+B,:,:,:] = warp_reconst.swapaxes(-2,-3)'''
			#reconst = net.blobs['iconv5'].data.copy() # B, 3, oP, oP
			#reconst = swap(reconst)
			#outPatches[batch_id:batch_id+B,:,:,:] = reconst.swapaxes(-2,-3)

			loss += output['loss'] # scalar
			count+=1
			
		''''disp_reconst = tools.combine_patches(dispPatches, rlist, clist)
		disp_reconst = disp_reconst.squeeze()

		warp_reconst = tools.combine_patches(warpPatches, rlist, clist)
		#reconstView = tools.combine_patches(outPatches, rlist, clist)	

		#imsave(path+str(count).zfill(3)+'e_reconst.png', reconstView)
		imsave(path+str(count).zfill(3)+'a_coded.png', codedImg)
		imsave(path+str(count).zfill(3)+'c_centerv.png', refView)
		imsave(path+str(count).zfill(3)+'e_trv.png', trView)
		imsave(path+str(count).zfill(3)+'b_disparity.png', disp_reconst)
		imsave(path+str(count).zfill(3)+'d_warpedcv.png', warp_reconst)
		np.save(path+str(count).zfill(3)+'g_disparray', disp_reconst)'''


loss = loss/count
print "avg loss is", loss
	
