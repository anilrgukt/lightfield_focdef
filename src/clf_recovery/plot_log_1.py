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

args = parser.parse_args()

path = args.p
train_log = path+'train_log.txt'
val_log = path+'val_log.txt'

log_train = np.loadtxt(train_log)
log_val = np.loadtxt(val_log)

#print log_train.shape, log_val.shape
l = log_val.size
l1 = log_train.size
print l,l1
s = args.vstep
#print l
#print len(range(0,700*l,700))
print 'plotting'
fig = matplotlib.pyplot.figure(figsize=(38.0, 5.0)) # in inches!
plt.plot(log_train,label='train')
plt.plot(range(0,s*l,s),log_val,'-r',label='val',linewidth=2.0)
plt.legend(loc='best')
if args.i2 > 0:
	plt.xlim(args.i,args.i2)
else:
	plt.xlim(args.i,l1)
x=range(0,20000,1000)
print s,l
plt.xticks(np.arange(min(x), max(x)+1, 1000.0))
plt.xlabel('iterations')
plt.ylabel('MSE')
plt.ylim(0.0,args.ylim)
#plt.show(fig)
plt.savefig(path+'log_expand')
#plt.close()

