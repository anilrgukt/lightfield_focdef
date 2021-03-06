import matplotlib.image as mplimg
import numpy as np
import Image
import cv2
from scipy.misc import imsave

class illiumTools():
	def __init__(self):
		self.numXs = 14
		self.numYs = 14
		self.minView = 4
		self.maxView = 11
		self.angRes = 5
		self.p = 60
		self.op = 48
		self.s = self.op
		self.viewH = None
		self.viewW = None
		self.verbosity = 1

	def comp_list(self, r):
		p = self.p
		s = self.s
		l = range(0,r-p+1,s)
		#print r,p,s,l
		
		#if(r - (l[-1]+p) > 0):
		#	l.append(r-p)		
		return l

 	def comp_list_whole(self, r, stride):
		p = self.p
		s = stride
		l = range(0,r-p+1,s)
		#print r,p,s,l
		
		#if(r - (l[-1]+p) > 0):
		#	l.append(r-p)		
		return l

	def extract_patches_strided(self, lf, stride):
		if self.verbosity:
			print 'extracting patches'
		P = self.p
		S = stride

		patches = []
		#print lf.shape
		r_list = self.comp_list_whole(lf.shape[0], S) 
		c_list = self.comp_list_whole(lf.shape[1], S)

		for r in r_list:
			for c in c_list:
				patch = lf[r:r+P,c:c+P,:]
				patches.append(patch)

		return np.array(patches), r_list, c_list

	def extract_patches(self, lf):
		if self.verbosity:
			print 'extracting patches'
		P = self.p
		S = self.s

		patches = []
		r_list = self.comp_list(lf.shape[2])
		c_list = self.comp_list(lf.shape[2])

		for r in r_list:
			for c in c_list:
				patch = lf[:,:,r:r+P,c:c+P,:]
				patches.append(patch)

		return np.array(patches), r_list, c_list

  	def combine_patches(self, patches,rlist,clist):
		if self.verbosity:
			print 'combining the patches'
		H = self.viewH
		W = self.viewW
		oP = self.op # output patchsize
		offset = (self.p-oP)/2 # offset for 

		# p - 60, op- 40, patch - 50
		pH = patches.shape[1]
		pW = patches.shape[2]
		pC = patches.shape[3]

		#print 'patches shape', patches.shape	
		img = np.zeros((H,W,pC))
		count = np.zeros((H,W,pC))

		n = 0
		for r in rlist:
			for c in clist:
				#print r, c
				rstart = r+offset; rend = rstart+oP
				cstart = c+offset; cend = cstart+oP
				img[rstart:rend, cstart:cend] += patches[n]
				count[rstart:rend, cstart:cend] += 1
				n+=1
		#print 'combined patches', n
		count[count==0]=1
		return img/count.astype(float)

		
	def readIlliumImg(self,imagepath):
		'''
		fullLF = Image.open(imagepath)
		fullLF = fullLF.convert('RGB')
		fullLF = np.array(fullLF)
		'''

		fullLF = cv2.imread(imagepath,-1)
		fullLF = cv2.cvtColor(fullLF,cv2.COLOR_BGRA2RGB)
		
		#print 'check', fullLF.shape, fullLF.max(), fullLF.min(), fullLF.dtype
		#fullLF = fullLF/255.0
		fullLF = fullLF/65535.0

		viewH = fullLF.shape[0]/self.numXs
		viewW = fullLF.shape[1]/self.numYs
		self.viewH = viewH
		self.viewW = viewW
	
		numViews = self.maxView - self.minView + 1
		offset = self.minView - 1

		curLF = np.zeros((numViews,numViews,viewH,viewW,3))
	
		for x in range(numViews):
			for y in range(numViews):
				curLF[x,y] = fullLF[x+offset::self.numXs,y+offset::self.numYs,:]

		return curLF
		
	def getRandViews(self, fullLF, n):
		# inputLF - v,v,H,W,3
		# print 'getting rand views'
		numViews = fullLF.shape[0]
		viewH = fullLF.shape[2]
		viewW = fullLF.shape[3]
		# print 'H,W,', viewH, viewW
		indRange = numViews - (self.angRes/2)*2
		selInd = range(indRange**2)		
		import random
		random.seed(123)
		random.shuffle(selInd)
		viewInds = np.array(selInd[:n])

		selLF = np.zeros((n,self.angRes,self.angRes,viewH,viewW,3))
		for i, ind in enumerate(viewInds):
			r = 3#ind / (numViews - (self.angRes/2)*2)  
			c = 2#ind % (numViews - (self.angRes/2)*2) 
			selLF[i] = fullLF[r:r+self.angRes,c:c+self.angRes,:,:,:]				
			
		return selLF 
	
	def reshape_code(self, code):
		code_shape = code.shape # P, P, 3, v, v
		#print 'check', code.shape
		v = code_shape[-1]
		p = code_shape[0]
		_code = np.zeros((v,v,p,p,3))
		for i in range(v):
			for j in range(v):
				#print i, j 
				_code[i,j] = code[:,:,:,i,j]
		'''
		for i in range(self.angRes):
			for j in range(self.angRes):
				_img = np.squeeze(_code[i,j])
				print _img.shape
				imsave('code_'+str(i)+str(j)+'.png', _img)
		'''
		return _code

	def getCodedImgWhole(self, inputLF, code):
		in_shape = inputLF.shape # v1,v2,h,w,3

		code = code[np.newaxis,:] # p,p,3,v1,v2
		code = code.swapaxes(0,-1).squeeze() # v2,p,p,3,v1
		code = code[np.newaxis,:] # 1,v2,p,p,3,v1
		code = code.swapaxes(0,-1).squeeze()
		print 'code shape', code.shape # v1,v2,p,p,3
		
		x = in_shape[2]/code.shape[2]
		y = in_shape[3]/code.shape[3]
		# repeat the code for image size h,w 
		#code = code.repeat(in_shape[2]/code.shape[2],2)
		#code = code.repeat(in_shape[3]/code.shape[3],3) # now the code shape should be v,v,h,w,3
		code = np.tile(code,(1,1,x,y,1))
		
		print 'code shapef', code.shape 
		codedLF = inputLF*code

		codedLF = np.sum(codedLF,axis=0) # v2,H,W,3
		codedLF = np.sum(codedLF,axis=0) # H,W,3
		#print codedLF.shape

		return codedLF/float(self.angRes**2)

	
	def getmse(self,img,imgref):
		#imgrecon, imgref
		errr=np.sum((imgref.astype("float") - img.astype("float")) ** 2)
		errr /= float(imgref.shape[0] * imgref.shape[1])
		return errr		

	'''
	def writeh5(self,codedImg,centerImg,trView,outputDir,writeOrder,startInd,createFlag,arraySize)
		chunkSize = 1000)

		fileName = "%straining" %(outputDir)
		numTotalPatches = self.numPatches
		numH5 = self.numH5
		sizeH5 = floor(numTotalPatches/numH5)
		bins = 1:sizeH5:numTotalPatches;
		bins = [bins, numTotalPatches+1];
		numElements = self.numElements		

		for k = range(numElements)
    		j = k + startInd - 1;
			curInImgs = inImgs[:, :, :, k];
    		curRef = ref[:, :, :, k];
			w = ceil(writeOrder[j]/sizeH5);
    		fName = "%s_%s.h5" %(fileName,num2str[w]);

    		arraySize = bins(w+1)-bins(w);
    		startLoc = mod(writeOrder(j),sizeH5) + 1;
    		SaveHDF(fName, '/data_tr', single(curInImgs), PadWithOne(size(curInImgs), 4), [1, 1, 1, startLoc], chunkSize, createFlag(w), arraySize);
    		SaveHDF(fName, '/label_tr', single(curRef), PadWithOne(size(curRef), 4), [1, 1, 1, startLoc], chunkSize, createFlag(w), arraySize);

    		createFlag[w] = false;
		# SaveHDF(fileName, datasetName, input, inDims, startLoc, chunkSize, createFlag, arraySize)
		# h5create(fileName, datasetName, [inDims(1:end-1), arraySize], 'Datatype', 'single', 'ChunkSize', [inDims(1:end-1), chunkSize]);
		# h5write(fileName, datasetName, single(input), startLoc, inDims);
	'''
