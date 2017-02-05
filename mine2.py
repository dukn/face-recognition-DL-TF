#
# run in python 2.7
#
# output in file "face.pkl.gz" content 2 dump X:180*80*80 and Y:180 
#
import cPickle,numpy as np,Image,gzip,pylab
from scipy import signal
from scipy import misc
#sub import 
import matplotlib.pyplot as plt
from PIL import Image


prepath = "facedatasets/"

def stand(arr):
	maxx = arr.shape[0]
	maxy = arr.shape[1]
	maxxy = min(maxx,maxy)
	tlx = 1.0*maxxy/80
	tly = 1.0*maxxy/80
	res = []
	for i in range(80):
		subres = []
		for j in range(80):
			tmp = arr[int(i*tlx)][int(j*tly)]
			subres.append(tmp)
		res.append(subres)
	rst = np.array(res)
	#print rst.shape
	return rst
Y = []
for i in range(6):
	for j in range(30):
		Y.append(i)

#print Y
X = []

for i in range(1,181):#181
	path = prepath + "%d.PNG" %(i)
	#img = Image.open(open(path))
	#arr = numpy.asarray(img,dtype = 'float32')
	img = misc.imread(path).astype(np.float)
	grayim = np.dot(img[...,:3],[0.299, 0.587, 0.114])
	grayim = stand(grayim)	
	#IMG = np.expand_dims(grayim,-1)
	X.append(grayim)

	if i%10 == 0:
		print i,

	#debug
	""" 
	plt.subplot(1, 2, 1)
	plt.imshow(256-img[...,:3])
	plt.xlabel(" Float Image ")
	print img.shape
	print img[0][0]
	plt.subplot(1, 2, 2)
	plt.imshow(grayim, cmap=plt.get_cmap("gray"))
	plt.xlabel(" Gray Scale Image ")
	plt.show()
	print grayim.shape
	"""
	#end debug.

from random import randint

tmpX = X[0]
tmpY = Y[0]

for i in range(180):
	n = randint(0,179)
	tmpX = X[i]; X[i] = X[n]; X[n] = tmpX
	tmpY = Y[i]; Y[i] = Y[n]; Y[n] = tmpY

with gzip.open("face.pkl.gz", 'wb') as f:
	cPickle.dump(X,f)
	cPickle.dump(Y,f)

print "110 label is : " , Y[110]
plt.imshow(X[110], cmap = plt.get_cmap("gray"))
plt.show()

