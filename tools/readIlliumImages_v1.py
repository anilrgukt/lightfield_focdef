import matplotlib.image as mplimg
import numpy as np
import cv2
import time

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

	def extract_patches(self, lf):
		if self.verbosity:
			print('Extracting patches')
		P = self.p
		S = self.s

		patches = []
		r_list = self.comp_list(lf.shape[2])
		c_list = self.comp_list(lf.shape[3])

		for r in r_list:
			for c in c_list:
				patch = lf[:,:,r:r+P,c:c+P,:]
				patches.append(patch)
		patches = np.array(patches)
		#print patches.shape

		return patches, r_list, c_list

	def combine_patches(self, patches,rlist,clist,lf=0):
		if self.verbosity:
			print ('Combining patches')
		H = self.viewH
		W = self.viewW
		oP = self.coP# 90 #self.op # output patchsize
		offset = (self.p-oP)//2 # offset for 
		
		
		# p - 60, op- 40, patch - 50
		pH = patches.shape[-3]
		pW = patches.shape[-2]
		pC = patches.shape[-1]
		#print 'patch shape', patches.shape
		#print 'lf', lf, pH, oP
		#if pH != self.s:
		#	off = (pH-self.s)/2
		#	patches = patches[:,off:-off,off:-off,:]			
		#### This offset is to further crop the prediction and stride should be planned accordingly
		
		if pH != oP:
			offset2 = (pH-oP)//2
			if lf:
				v = patches.shape[1]
				patches = patches.reshape(patches.shape[0],v*v,pH,pW,pC)
				patches = patches[:,:,offset2:-offset2,offset2:-offset2,:]
			else:
				#print 'not lf', patches.shape, offset2
				patches = patches[:,offset2:-offset2,offset2:-offset2,:]
				#print 'not lf', patches.shape
		#patches = patches[:,offset:-offset,offset:-offset,:]
		#print 'patches shape', patches.shape

		if not lf:
			img = np.zeros((H,W,pC))
			count = np.zeros((H,W,pC))
		else:
			img = np.zeros((v*v,H,W,pC))
			count = np.zeros((v*v,H,W,pC))
		#print 'init done'
		#print 'img shape', img.shape,'offset', offset
		st_time = time.time()
		n = 0
		for r in rlist:
			for c in clist:
				#print r, c
				rstart = r+offset; rend = rstart+oP
				cstart = c+offset; cend = cstart+oP
				if not lf:
					img[rstart:rend, cstart:cend] += patches[n]
					count[rstart:rend, cstart:cend] += 1
				else:
					img[:,rstart:rend, cstart:cend,:] = img[:,rstart:rend, cstart:cend,:] + patches[n]
					count[:,rstart:rend, cstart:cend,:] = count[:,rstart:rend, cstart:cend,:] + 1

				n+=1
		#print 'combined patches', n, 'time', time.time()-st_time, 'secs'
		count[count==0]=1
		img = img/count.astype(float)
		if lf:
			img = img.reshape(v,v,img.shape[-3],img.shape[-2],img.shape[-1])
		return img			
		
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

		viewH = fullLF.shape[0]//self.numXs
		viewW = fullLF.shape[1]//self.numYs
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
		indRange = numViews - (self.angRes//2)*2
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
		 
	def getViews(self, fullLF, n):
		# inputLF - v,v,H,W,3
		# print 'getting rand views'
		numViews = fullLF.shape[0]
		viewH = fullLF.shape[2]
		viewW = fullLF.shape[3]
		
		# print 'H,W,', viewH, viewW

		selLF = np.zeros((n,self.angRes,self.angRes,viewH,viewW,3))
		
		selLF[0] = fullLF[1:,1:,:,:,:]
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

	def getCodedImg(self, inputLF, code):
		in_shape = inputLF.shape # v1,v2,h,w,3

		code = code[np.newaxis,:] # 1,p,p,3,v1,v2
		code = code.swapaxes(0,-1).squeeze() # v2,p,p,3,v1
		code = code[np.newaxis,:] # 1,v2,p,p,3,v1
		code = code.swapaxes(0,-1).squeeze()
		print ('code shape', code.shape) # v1,v2,p,p,3

		#code = np.expand_dims(code,0) # 1,v,v,P,P,3
		#code = code.repeat(in_shape[0], 0) # N, v, v, P, P, 3
		#code = code.repeat(6,2)
		#code = code.repeat(9,3)
		#print 'code check', code.shape		
		codedLF,r,c = self.extract_patches(inputLF*code)
		#print 'code LF check', codedLF.shape, inputLF.shape	
		#codedLF = inputLF*code
		codedLF = np.sum(codedLF,axis=1) # N,v2,P,P,3
		codedLF = np.sum(codedLF,axis=1) # N,P,P,3
		
		return codedLF#/float(self.angRes**2)

	
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
	
	
		
