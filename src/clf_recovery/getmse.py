import os
os.environ['GLOG_minloglevel'] = '2'
from os import listdir
import caffe
import math
from readIlliumImages import illiumTools as tools
import matplotlib.image as mplimg
import numpy as np
import cv2
path='/home/susmitha/clf_recovery/models/fc/val_results/';
listed=listdir(path);
print listed;
for imgname in listed:
	if(imgname[-9:-4]=='coded'):
		img1=cv2.imread(path+imgname,1);
		img2=cv2.imread(path+imgname[:4]+'ref.png',1);
		mse=tools.get_mse(img1,img2);
		print mse


