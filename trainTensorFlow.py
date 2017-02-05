import tensorflow as tf
import numpy as np
import pickle,numpy as np,gzip


#load data
f = gzip.open('face.pkl.gz','rb')

dataX = pickle.load(f)
dataY = pickle.load(f)
f.close()

length = 180
tmp = [[0 for i in range(6)] for i in range(length)]
for i in range(length):
	tmp[i][dataY[i]] = 1
dataY = tmp

#(dataY)
trX = np.array(dataX[:150]).reshape(-1, 80, 80, 1)/256.0
trY = np.array(dataY[:150])
teX = np.array(dataX[150:]).reshape(-1, 80, 80, 1)/256.0
teY = np.array(dataY[150:])
#print("shape Y : ",trY.shape)

#
X = tf.placeholder(tf.float32, [None, 80, 80, 1])
Y = tf.placeholder(tf.float32, [None, 6])

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
print"Training..."
for i in range(400):
	for start,end  in zip(range(0,len(trX),30), range(30,len(trX),30)):
		sess.run(train_step, 
			feed_dict = {
				X:trX[start:end], 
				Y:trY[start:end], 
				keep_prob:0.5,
				step_size:5e-5}
				)	
	ce = sess.run(cross_entropy,
		feed_dict = {
			X:teX, 
			Y:teY, 
			keep_prob:1.0
			}
			)
	ac = sess.run(accuracy,
		feed_dict = {
			X:teX, 
			Y:teY, 
			keep_prob:1.0
			}
			)
	if i%10 == 0: 
		print i , "(cross entropy, accuracy) :", ce, ac

saver.save(sess, "model2.data")
