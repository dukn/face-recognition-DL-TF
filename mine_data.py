import cPickle,numpy,Image,gzip,pylab

prepath = "facedatasets/"

result = []

def for2three(arr):
	x = arr.shape[0]
	y = arr.shape[1]
	res = []
	for i in range(x):
		subres = []
		for j in range(y):
			tmp = arr[i][j][:3]
			subres.append(tmp)
		res.append(subres)
	rst =  numpy.array(res)
	print rst.shape
	return rst

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
	rst = numpy.array(res)
	print rst.shape
	return rst
	
def togray(arr):
	#
#run in here...
for i in range(1,2): #181
	path = prepath + "%d.PNG" %(i)
	img = Image.open(open(path))
	arr = numpy.asarray(img,dtype = 'float32')
	#print arr.shape
	newarr = for2three(arr)
	#print arr[1][1]
	#print newarr[1][1]
	newarr2 =stand(newarr)

	#pylab.imshow(255-newarr2)
	#/pylab.show()
	newarr3 = 1 - newarr2/255
	result.append(newarr3)
#result 
label = []
for nm in range(6):
	for i in range(30):
		label.append(nm)

	
with gzip.open("face_recognition.pkl.gz", 'wb') as f:
	cPickle.dump(result,f)
	cPickle.dump(label,f)


print "completed!"
