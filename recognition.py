import tensorflow as tf
import numpy as np
import pickle,numpy as np,gzip
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from subprocess import call

img = "me.jpg"

call(["python2", "testfacedetection.py",img])

f = open("runlog")
data =  f.read().split('\n')
num = int (data[0])
res = []
for i in range(1,num+1):
	curdata = data[i].split(" ")
	cur = [int(i) for i in curdata]
	res.append(cur)
#print (res)

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
#

im = misc.imread(img).astype(np.float)
grayim = np.dot(im[...,:3],[0.299, 0.587, 0.114])


theface = []

for i in res:
	inc = int(0.08*(i[2]-i[0]))
	left = i[0]-inc
	top = i[1]-inc
	right = i[2]+inc
	botton = i[3]+inc
	
	tmp1 = []
	for k1 in range(top,botton):	
		tmp2 = []
		for k2 in range(left,right):
			tmp2.append(grayim[k1][k2])
		tmp1.append(tmp2)
		tmp3 = np.array(tmp1)
		tmp3 = stand(tmp3)

	theface.append(tmp3)
for i in range(30):
	theface.append(theface[0])


theface = np.array(theface).reshape(-1, 80, 80, 1)/256.0

#plt.imshow(theface[1], cmap=plt.get_cmap("gray"))
#plt.show()

#
X = tf.placeholder("float", [None, 80, 80, 1])
Y = tf.placeholder("float", [None, 6])

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.2)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.5, shape=shape)
	return tf.Variable(initial)

# probability placeholder for dropout layer
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

sess = tf.Session()

def make_model(x):
	# First we have 7x7 Convolution with 10 filter 
	conv1_biases = tf.Variable(tf.zeros([10]))
	conv1_weights = tf.Variable(tf.truncated_normal([7,7,1,10], stddev=0.1))

	stage1 = tf.nn.conv2d(x,conv1_weights, strides= [1,1,1,1], padding = 'SAME')
	stage1 = tf.tanh(tf.nn.bias_add(stage1,conv1_biases))

	# Then 4x4 downscaling with maxpool layer
	pool1 = tf.nn.max_pool(stage1,ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME') # 40x40
	# Next is 5x5 convolution 
	conv2_biases = tf.Variable(tf.zeros([20]))
	conv2_weights = tf.Variable(tf.truncated_normal([5,5,10,20], stddev = 0.1))

	stage2 = tf.nn.conv2d(pool1,conv2_weights, strides = [1,1,1,1], padding = 'SAME')
	stage2 = tf.tanh(tf.nn.bias_add(stage2,conv2_biases))

	# Again maxpool
	pool2 = tf.nn.max_pool(stage2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME') # 20 x 20

	# Next is 5x5 convolution 
	conv3_biases = tf.Variable(tf.zeros([40]))
	conv3_weights = tf.Variable(tf.truncated_normal([5,5,20,40], stddev = 0.1))

	stage3 = tf.nn.conv2d(pool2,conv3_weights, strides = [1,1,1,1], padding = 'SAME')
	stage3 = tf.tanh(tf.nn.bias_add(stage3,conv3_biases))

	#Again maxpool
	pool3 = tf.nn.max_pool(stage3,ksize=[1,4,4,1],strides = [1,4,4,1],padding = 'SAME') # 5x5


	# Reshape and do fully-connected hidden layer using matrix multiplication
	pool_shape = pool3.get_shape().as_list()
	reshape = tf.reshape(pool3,[-1,pool_shape[1]* pool_shape[2]*pool_shape[3]])

	fc1_biases = tf.Variable(tf.constant(0.1, shape=[80]))
	fc1_weights = tf.Variable(tf.truncated_normal([5*5*40,80],stddev = 0.1)) #

	hidden = tf.matmul(tf.nn.dropout(reshape,keep_prob), fc1_weights) + fc1_biases
	hidden = tf.tanh(hidden)

	# Fully-connected softmax output layer
	fc2_biases = tf.Variable(tf.constant(0.1,shape=[6]))
	fc2_weights = tf.Variable(tf.truncated_normal([80,6], stddev = 0.1))

	y = tf.nn.softmax(tf.matmul(hidden,fc2_weights) + fc2_biases)
	return y

Y_pred = make_model(X)
# Cross entropy is a standard ML penalty funtion

cross_entropy = -tf.reduce_mean(Y*tf.log(tf.clip_by_value(Y_pred,1e-10,1.0)))

correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(Y_pred,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

step_size = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(step_size).minimize(cross_entropy)

sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

#saver.restore(sess, "model.data")
new_saver = tf.train.import_meta_graph('model.data.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.trainable_variables()

'''for v in all_vars:
    print(v.name)'''

result = sess.run(Y_pred,feed_dict = {
			X:theface[0:30],
			keep_prob:1.0
			}
		)
print result

named = ["Taylor Switf","Adam Levine","Miley Cyrus","Donald Trump","Hilary Clinton","Barack Obama"]
resultshow = []
for i in range(num):
	A = [tmp for tmp in result[i]]
	B = [tmp for tmp in named]
	for k in range(5):
		for l in range(k+1,6):
			if A[k]<A[l]:
				tA = A[k]
				A[k] = A[l]
				A[l] = tA 
				tB = B[k]
				B[k] = B[l]
				B[l] = tB

	#for j in range(len(A)):
	#	A[i] = 100.0*A[i]
	resultshow.append([A,B])

print resultshow

#import patches
import matplotlib.patches as patches
fig,ax = plt.subplots(1)

ax.imshow(256.0-im)
for dem in range(len(res)):
	i = res[dem]
	rect = patches.Rectangle((i[0],i[1]),i[2]-i[0],i[3]-i[1],linewidth=1,edgecolor='b',facecolor='none')
	ax.add_patch(rect)
	ax.text(i[0],i[3]+40, "%s:%0.2f"%(resultshow[dem][1][0],resultshow[dem][0][0]),color='r',fontsize = 13)
	ax.text(i[0],i[3]+60, "%s:%0.2f"%(resultshow[dem][1][1],resultshow[dem][0][1]),color='r',fontsize = 13)
	
plt.show()


