from os import listdir
#from readIlliumImages import illiumTools as tools
import matplotlib.image as mplimg
import numpy as np
from PIL import Image

path='/home/susmitha/clf_recovery/models/fc/val_results/';
count=0
listed=listdir(path);
#print 'listed', listed

def getmse(img,imgref):
	errr=np.sum((imgref.astype("float") - img.astype("float")) ** 2)
	errr /= float(imgref.shape[0] * imgref.shape[1])
	return errr

def read_img(path):
	#print path
	img = Image.open(path)
	img = img.convert('RGB')
	
	return np.array(img)/255.0

coded_err = []
reconst_err = []

for imgname in listed:
	#print 'for loop'
	#print 'img name', imgname
	if(imgname[-9:-4]=='coded'):
		#print ' in if condition'
		coded = read_img(path+imgname)
		reconst = read_img(path+imgname[:4]+'reconst.png')
		ref = read_img(path+imgname[:4]+'ref.png')

		print 'sum', np.sum(coded[:,-1,1]), np.sum(coded[:,-2,1])
		tmp = np.sum(ref,2);
		N = 3*np.sum(tmp!=3)
		
		ref[tmp==3,:]=3
			
		err = np.sum((coded-ref)**2)/float(N)
		coded_err.append(err)
		
		err2 = np.sum((reconst-ref)**2)/float(N)
		reconst_err.append(err2)

		print ref.max(), ref.min()
		print imgname,N/3, err, err2
		count += 1 


coded_err = np.mean(np.array(coded_err))
reconst_err = np.mean(np.array(reconst_err))

print 'avg error', coded_err, reconst_err
