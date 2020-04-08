import os
import sys
os.environ['GLOG_minloglevel'] = '2' 
import caffe
import math
import matplotlib.image as mplimg
from scipy.misc import imsave
from os import listdir
from scipy.io import savemat, loadmat
import numpy as np
from skimage.measure import structural_similarity as ssim
from skimage import color
import argparse
from tools import swap, prep_batch

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
	B = 70 # batch size
	v = 3 # ref view 3 center one currently not being used
	oP = 30 # output patch size from the network
	P = 70 # patch size
	stride = 20
	S = oP + 1 # stride 
	code = loadmat(args.code)['fullCode']
	print 'code shape', code.shape

	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)

	proto = args.path+args.deploy
	model = args.path+'clf_iter_'+str(args.iter)+'.caffemodel'
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
			print 'count', count
			viewLF = viewLF[:,:,25:25+350,25:25+490,:]
			tools.viewH = viewLF.shape[2]
			tools.viewW = viewLF.shape[3]
			#print 'cec', viewLF.shape
			codedImg = tools.getCodedImgWhole(viewLF, code) # H, W, 3		
			pCodedImg, rlist, clist = tools.extract_patches_strided(codedImg,stride) # N, P, P, 3 		
			
			refView = np.squeeze(viewLF[2,2])
			trView = np.squeeze(viewLF[0,4])
			blView = np.squeeze(viewLF[4,0])
			prefView,rlist,clist = tools.extract_patches_strided(refView,stride) # N,P,P,3
			ptr,rlist,clist = tools.extract_patches_strided(trView,stride) # N,P,P,3
			pbl,rlist,clist = tools.extract_patches_strided(blView,stride) # N,P,P,3
			
			numPatches = pCodedImg.shape[0]				

			outPatches = np.zeros((numPatches,oP,oP,3))
			dispPatches = np.zeros((numPatches,oP,oP,1))
			warpPatches = np.zeros((numPatches,oP,oP,3))

			batch_list = range(0,numPatches,B)
			net.blobs['data'].reshape(B,3,P,P)
			net.blobs['label'].reshape(B,3,P,P)		
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
					net.blobs['data'].reshape(batch.shape[0],3,P,P)
					net.blobs['label'].reshape(batch.shape[0],3,P,P)
					net.blobs['label_tr'].reshape(batch.shape[0],3,P,P)
					net.blobs['label_bl'].reshape(batch.shape[0],3,P,P)
					net.reshape()

				#print 'forward pass..'			
				net.blobs['data'].data[...] = batch
				net.blobs['label'].data[...] = batch_label
				net.blobs['label_tr'].data[...] = batch_tr
				net.blobs['label_bl'].data[...] = batch_bl
				output = net.forward()

				if args.op != net.blobs['cropped_cv'].data.shape[2]:
					c = net.blobs['cropped_cv'].data.shape[2] - args.op
					c = c/2
					reconst = net.blobs['cropped_cv'].data[:,:,c:-c,c:-c] # B, 3, oP, oP
					disparity = net.blobs['disparity'].data[:,:,c:-c,c:-c] # B, 3, oP, oP
					warped = net.blobs['predicted_view'].data[:,:,c:-c,c:-c] # B, 3, oP, oP
				else:				
					reconst = net.blobs['cropped_cv'].data[...] # B, 3, oP, oP
					disparity = net.blobs['disparity'].data[...] # B, 3, oP, oP
					warped = net.blobs['predicted_view'].data[...] # B, 3, oP, oP

				reconst = swap(reconst) # B, W, H, 3		
				disparity = swap(disparity)
				warped = swap(warped)

				outPatches[batch_id:batch_id+B,:,:,:] = reconst.swapaxes(-2,-3)
				dispPatches[batch_id:batch_id+B,:,:,:] = disparity.swapaxes(-2,-3)
				warpPatches[batch_id:batch_id+B,:,:,:] = warped.swapaxes(-2,-3)

				avg_loss.append(output['loss'])

			disparity = tools.combine_patches(dispPatches, rlist, clist)	
			imsave(path+str(count).zfill(3)+'_ddisp_'+str(args.iter/1000)+'k.png', disparity)
			savemat(path+str(count).zfill(3)+'_ddisp_'+str(args.iter/1000)+'k.mat',{'disp':disparity})

			reconst = tools.combine_patches(outPatches, rlist, clist)	
			imsave(path+str(count).zfill(3)+'_cgencv_'+str(args.iter/1000)+'k.png', reconst)

			warpView = tools.combine_patches(warpPatches, rlist, clist)			
			imsave(path+str(count).zfill(3)+'_epred_'+str(args.iter/1000)+'k.png', warpView)

			imsave(path+str(count).zfill(3)+'_fact_tr.png', trView)				
			imsave(path+str(count).zfill(3)+'_acoded.png', codedImg)
			imsave(path+str(count).zfill(3)+'_bref_cv.png', refView)


			count+=1
	
	avg_loss = np.array(avg_loss)
	print 'avg loss', np.mean(avg_loss)
	
	return 0

def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--valData', type=str, default = '/media/data/susmitha/lf_data/TrainingData/Val/')
	parser.add_argument('--iter', type=int, required=True)
	parser.add_argument('--deploy', type=str, required=True)
	parser.add_argument('--gpu_id', type=int, required=True)
	parser.add_argument('--patch', type=int, default=1)	
	parser.add_argument('--path', type=str, required=True)
	parser.add_argument('--code', type=str, required=True)
	parser.add_argument('--op', type=int, required=True)
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
	oP = args.op # output patch size from the network
	P = 70 # patch size
	S = oP + 1 # stride 

	code = loadmat(args.code)['fullCode']
	print 'code shape', code.shape

	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)

	proto = args.path+args.deploy
	model = args.path+'clf_iter_'+str(args.iter)+'.caffemodel'
	net = caffe.Net(proto,model,caffe.TEST) 

	from readIlliumImages import illiumTools as tools
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
			reftr = np.squeeze(viewLF[0,4])
			prefView = pLFs[:,2,2,:,:,:]
			ptr = pLFs[:,0,4,:,:,:]
			pbl = pLFs[:,4,0,:,:,:]

			codedImg = tools.combine_patches(pCodedImg, rlist, clist)

			outPatches = np.zeros((numPatches,oP,oP,3))
			dispPatches = np.zeros((numPatches,oP,oP,3))
			warpPatches = np.zeros((numPatches,oP,oP,3))

			batch_list = range(0,numPatches,B)
			net.blobs['data'].reshape(B,3,P,P)
			net.blobs['label'].reshape(B,3,P,P)		
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
					net.blobs['data'].reshape(batch.shape[0],3,P,P)
					net.blobs['label'].reshape(batch.shape[0],3,P,P)
					net.blobs['label_tr'].reshape(batch.shape[0],3,P,P)
					net.blobs['label_bl'].reshape(batch.shape[0],3,P,P)
					net.reshape()

				#print 'forward pass..'			
				net.blobs['data'].data[...] = batch
				net.blobs['label'].data[...] = batch_label
				net.blobs['label_tr'].data[...] = batch_tr
				net.blobs['label_bl'].data[...] = batch_bl
				output = net.forward()

				if args.op != net.blobs['cropped_cv'].data.shape[2]:
					c = net.blobs['cropped_cv'].data.shape[2] - args.op
					c = c/2
					reconst = net.blobs['cropped_cv'].data[:,:,c:-c,c:-c] # B, 3, oP, oP
					disparity = net.blobs['disparity'].data[:,:,c:-c,c:-c] # B, 3, oP, oP
					warped = net.blobs['predicted_view'].data[:,:,c:-c,c:-c] # B, 3, oP, oP
				else:				
					reconst = net.blobs['cropped_cv'].data[...] # B, 3, oP, oP
					disparity = net.blobs['disparity'].data[...] # B, 3, oP, oP
					warped = net.blobs['predicted_view'].data[...] # B, 3, oP, oP

				reconst = swap(reconst) # B, W, H, 3		
				disparity = swap(disparity)
				warped = swap(warped)

				outPatches[batch_id:batch_id+B,:,:,:] = reconst.swapaxes(-2,-3)
				dispPatches[batch_id:batch_id+B,:,:,:] = disparity.swapaxes(-2,-3)
				warpPatches[batch_id:batch_id+B,:,:,:] = warped.swapaxes(-2,-3)
			
			
				avg_loss.append(output['loss'])		
			
			disparity = tools.combine_patches(dispPatches, rlist, clist)	
			imsave(path+str(count).zfill(3)+'_ddisp_'+str(args.iter/1000)+'k.png', disparity)
			savemat(path+str(count).zfill(3)+'_ddisp_'+str(args.iter/1000)+'k.mat',{'disp':disparity})
			
			reconst = tools.combine_patches(outPatches, rlist, clist)	
			imsave(path+str(count).zfill(3)+'_cgencv_'+str(args.iter/1000)+'k.png', reconst)

			warpView = tools.combine_patches(warpPatches, rlist, clist)			
			imsave(path+str(count).zfill(3)+'_epred_'+str(args.iter/1000)+'k.png', warpView)

			imsave(path+str(count).zfill(3)+'_fact_tr.png', reftr)				
			imsave(path+str(count).zfill(3)+'_acoded.png', codedImg)
			imsave(path+str(count).zfill(3)+'_bref_cv.png', refView)

			count+=1

	
	avg_loss = np.array(avg_loss)
	print 'avg loss', np.mean(avg_loss)

if __name__ == '__main__':
	sys.exit(main(sys.argv))
