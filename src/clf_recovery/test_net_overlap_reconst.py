import os
os.environ['GLOG_minloglevel'] = '2' 
import caffe
import math
import matplotlib.image as mplimg
from scipy.misc import imsave
from os import listdir
from scipy.io import savemat, loadmat
from readIlliumImages_whole import illiumTools as tools
import numpy as np
from skimage.measure import structural_similarity as ssim
from skimage import color
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, required=True)

args = parser.parse_args()

iter = args.iter

def swap(reconst):
	#reconst B, 3, oP, oP
	reconst = reconst.swapaxes(1,2) # B, oP, 3, oP
	reconst = reconst.swapaxes(2,3) # B, oP, oP, 3
	return reconst

folder_path = '/media/data/susmitha/lf_data/TrainingData/Val/'
path = '/home/susmitha/clf_recovery/gautham/diff_models/densenet_2/'
fname = 'reconst/'

if not os.path.isdir(path+fname):
	os.mkdir(path+fname)

path += fname

img_list = listdir(folder_path)
n = 1 # number of views
B = 1 # batch size
v = 3 # ref view 3 center one currently not being used
oP = 54 # output patch size from the network
P = 60 # patch size
S = oP + 1 # stride 
code = loadmat('randn_clip01_p15_cs1.mat')['fullCode']
print 'code shape', code.shape

caffe.set_mode_gpu()
caffe.set_device(0)

proto = 'gautham/diff_models/densenet_2/deploy.prototxt'
model = 'gautham/diff_models/densenet_2/clf_iter_'+str(iter)+'.caffemodel'
net = caffe.Net(proto,model,caffe.TEST) 

tools = tools()
tools.verbosity = 0
tools.op = oP
tools.p = P
tools.s = oP

count = 0
avg_loss = []

for img_name in img_list:
	print 'processing LF img', img_name
	if img_name[-4:] != '.png':
		print 'not an image'
		continue

	img_name = folder_path+img_name
	fullLF = tools.readIlliumImg(img_name) # numViews,numViews,viewH,viewW,3
	print 'read fullLF', fullLF.shape	
	#fullLF = np.random.randn(8,8,375,541,3)
	
	selLF = tools.getRandViews(fullLF, n) # n,self.angRes,self.angRes,viewH,viewW,3
	print 'selected LFs size', selLF.shape	

	for viewLF in selLF:
		# viewLF - v,v,H,W,3
		print 'count', count
		viewLF = viewLF[:,:,0:360,0:540,:]
		tools.viewH = viewLF.shape[2]
		tools.viewW = viewLF.shape[3]
		
		codedImg = tools.getCodedImgWhole(viewLF, code) # H, W, 3		

		print 'coded lf image', codedImg.shape  
		pCodedImg, rlist, clist = tools.extract_patches_strided(codedImg,45) # N, P, P, 3 		
			
		refView = np.squeeze(viewLF[2,2])
		prefView,rlist,clist = tools.extract_patches_strided(refView,45) # N,P,P,3
		print 'check',prefView.shape, pCodedImg.shape 
		print net.blobs['gen_cv'].data[...].shape
		numPatches = pCodedImg.shape[0]
		outPatches = np.zeros((numPatches,oP,oP,3))
		inPatches = np.zeros((numPatches,oP,oP,3))
		batch_list = range(0,numPatches,B)
		net.blobs['data'].reshape(B,3,P,P)
		net.blobs['label'].reshape(B,3,P,P)		
		net.reshape()

		#print 'forming batches',batch_list
		for batch_id in batch_list:
			#print 'bid', batch_id
			batch = pCodedImg[batch_id:batch_id+B] # B, P, P, 3
			batch = np.expand_dims(batch,1) # B, 1, P, P, 3
			batch = batch.swapaxes(1,4)

			
			batch_label = prefView[batch_id:batch_id+B] # B, P, P, 3
			batch_label = np.expand_dims(batch_label,1) # B, 1, P, P, 3
			batch_label = batch_label.swapaxes(1,4)

			if batch.shape[0] != B:
				net.blobs['data'].reshape(batch.shape[0],3,P,P)
				net.blobs['label'].reshape(batch.shape[0],3,P,P)				
				net.reshape()

			#print 'forward pass..'			
			batch = batch.squeeze()
			batch_label = batch_label.squeeze()
				
			net.blobs['data'].data[...] = batch.swapaxes(-1,-2)
			net.blobs['label'].data[...] = batch_label.swapaxes(-1,-2)
			output = net.forward()

			reconst = net.blobs['gen_cv'].data[...] # B, 3, oP, oP
			reconst = swap(reconst)
			print "reconst", reconst.shape
			outPatches[batch_id:batch_id+B,:,:,:] = reconst.swapaxes(-2,-3)
			print "op", outPatches.shape
			
			batch_label = swap(batch_label[:,:,6:60,6:60])
			inPatches[batch_id:batch_id+B,:,:,:] = batch_label

			avg_loss.append(output['loss'])

		reconstView = tools.combine_patches(outPatches, rlist, clist)	
		actView = tools.combine_patches(inPatches, rlist, clist)	
		
		imsave(path+str(count).zfill(3)+'_reconst_'+str(iter)+'.png', reconstView)
		imsave(path+str(count).zfill(3)+'_coded.png', codedImg)
		imsave(path+str(count).zfill(3)+'_ref.png', refView)
		imsave(path+str(count).zfill(3)+'_rec_act.png', actView)

		count+=1

	
avg_loss = np.array(avg_loss)
print 'avg loss', np.mean(avg_loss)
