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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--valData', type=str, default = '/media/data/susmitha/lf_data/TrainingData/Val/')
parser.add_argument('--iter', type=int, required=True)
#parser.add_argument('--deploy', type=str, required=True)
parser.add_argument('--gpu_id', type=int, required=True)
parser.add_argument('--path', type=str, required=True)

args = parser.parse_args()

iter = args.iter

def swap(reconst):
	#reconst B, 3, oP, oP
	reconst = reconst.swapaxes(1,2) # B, oP, 3, oP
	reconst = reconst.swapaxes(2,3) # B, oP, oP, 3
	return reconst


folder_path = args.valData
path = args.path
fname = 'reconst/'

if not os.path.isdir(path+fname):
	os.mkdir(path+fname)

path += fname

img_list = listdir(folder_path)
n = 1 # number of views
B = 15 # batch size
v = 3 # ref view 3 center one currently not being used
oP = 50 # output patch size from the network
P = 60 # patch size
S = oP + 1 # stride 
code = loadmat('randn_clip01_p15_cs1.mat')['fullCode']
print 'code shape', code.shape

caffe.set_mode_gpu()
caffe.set_device(args.gpu_id)

proto = args.path+'deploy.prototxt'
model = args.path+'clf_iter_'+str(iter)+'.caffemodel'
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

	selLF = tools.getRandViews(fullLF, n) # n,self.angRes,self.angRes,viewH,viewW,3
	print 'selected LFs size', selLF.shape	

	for viewLF in selLF:
		# viewLF - v,v,H,W,3
		print 'count', count
		pLFs, rlist, clist = tools.extract_patches(viewLF) # N,v,v P, P, 3		
		pCodedImg = tools.getCodedImg(pLFs, code) # N, P, P, 3
		numPatches = pCodedImg.shape[0]		
		refView = np.squeeze(viewLF[2,2])
		prefView = pLFs[:,2,2,:,:,:]
		
		codedImg = tools.combine_patches(pCodedImg, rlist, clist)

		outPatches = np.zeros((numPatches,oP,oP,3))
		batch_list = range(0,numPatches,B)
		net.blobs['data'].reshape(B,3,P,P)
		net.blobs['label'].reshape(B,3,P,P)		
		
		net.reshape()

		#print 'forming batches',batch_list
		for batch_id in batch_list:
			print 'bid', batch_id
			print 'op', outPatches.shape
			batch = pCodedImg[batch_id:batch_id+B] # B, P, P, 3
			batch = np.expand_dims(batch,1) # B, 1, P, P, 3
			batch = batch.swapaxes(1,4)

			batch_label = prefView[batch_id:batch_id+B] # B, P, P, 3
			batch_label = np.expand_dims(batch_label,1) # B, 1, P, P, 3
			batch_label = batch_label.swapaxes(1,4)
			#print batch_id, batch.shape, batch_label.shape
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
			print 're', reconst.shape
			if oP != reconst.shape[-1]:
				cr = -(oP - reconst.shape[-1])/2
				#print cr
				reconst = reconst[:,:,cr:-cr,cr:-cr]	

			reconst = swap(reconst) # B, W, H, 3
			print 're', reconst.shape			
			#reconst = reconst[:,2:-2,2:-2,:]
			outPatches[batch_id:batch_id+B,:,:,:] = reconst.swapaxes(-2,-3) # B, H, W, 3
			
			
			avg_loss.append(output['loss'])
		
		reconstView = tools.combine_patches(outPatches, rlist, clist)
		#actView = tools.combine_patches(inPatches, rlist, clist)
			
		imsave(path+str(count).zfill(3)+'_reconst_'+str(iter)+'.png', reconstView)
		imsave(path+str(count).zfill(3)+'_coded.png', codedImg)
		imsave(path+str(count).zfill(3)+'_ref.png', refView)
		#imsave(path+str(count).zfill(3)+'_rec_act.png', actView)

		count+=1

	
avg_loss = np.array(avg_loss)
print 'avg loss', np.mean(avg_loss)
