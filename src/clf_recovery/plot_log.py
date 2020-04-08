import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default=None)
parser.add_argument('--i', type=int, default=None)
parser.add_argument('--vstep', type=int, default=None)
parser.add_argument('--ylim', type=float, default=None)
parser.add_argument('--i2', type=int, default=None)
parser.add_argument('--h', type=int, default=None)

args = parser.parse_args()

path = args.p
train_log = path+'train_log.txt'
val_log = path+'val_log.txt'

log_train = np.loadtxt(train_log)
log_val = np.loadtxt(val_log)

if args.h > 0:
	log_train = log_train / (args.h**2)
	log_val = log_val / (args.h**2)
avg_log_train = [np.mean(log_train[max(0,i-100):i])  for i, j in enumerate(log_train)]


#print log_train.shape, log_val.shape
l = log_val.size
l1 = log_train.size
s = args.vstep
#print l
#print len(range(0,700*l,700))
print 'plotting'
plt.plot(log_train,'-c',label='train',linewidth=0.1)
plt.plot(avg_log_train,'-b',label='avg_loss',linewidth=1.2)
plt.plot(range(0,s*l,s),log_val,'-r',label='val',linewidth=1.5)
plt.legend(loc='best')
if args.i2 > 0:
	plt.xlim(args.i,args.i2)
else:
	plt.xlim(args.i,l1)
plt.xlabel('iterations')
plt.ylabel('MSE')
plt.ylim(0.0,args.ylim)
plt.grid(b=True, which='major', color='g', linestyle='-')
plt.grid(b=True, which='minor', color='g', linestyle='--')
plt.savefig(path+'log')
plt.close()

