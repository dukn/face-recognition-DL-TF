import numpy
import theano
import theano.tensor as T
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from model import LogReg, HiddenLayer, LeNetConvPoolLayer

def train_conv():
	learning_rate = 00.1 
	n_epochs = 1000
	nkerns = [20, 50] 
	batch_size = 500

	rng = numpy.random.RandomState(23455)
	datasets = load_data("")
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x,  test_set_y  = datasets[2]

	n_train_batches = train_set_x.get_value(borrow = True).shape[0] // batch_size
	n_valid_batches = valid_set_x.get_value(borrow = True).shape[0] // batch_size
	n_test_batches  = test_set_x.get_value( borrow = True).shape[0] // batch_size
	
	#realtime
	filepath = 'demo_data/' + nfile +'.png'
	test_real = load_demo_data(filepath)

	index = T.lscalar()

	x = T.matrix('x')
	y = T.ivector('y')

	print ('... rebuilding the model')

	#layer0_input = x.reshape((batch_size, 1,28,28))
	
	layer0 = LeNetConvPoolLayer(
		rng,
		input = x.reshape((batch_size, 1,28,28)), #layer0_input,
		image_shape = (batch_size, 1,28,28),
		filter_shape = (nkerns[0],1,5,5),
		poolsize = (2,2)
	)
	layer1 = LeNetConvPoolLayer(
		rng,
		input = layer0.output,
		image_shape = (batch_size, nkerns[0], 12, 12),
		filter_shape = (nkerns [1], nkerns[0], 5, 5),
		poolsize = (2,2)
	)
	
	#layer2_input = layer1.output.flatten(2)
	
	layer2 = HiddenLayer(
		rng,
		input =  layer1.output.flatten(2),#layer2_input,
		n_in = nkerns[1] * 4 * 4,
		n_out = 500,
		activation = T.tanh
	)
	layer3 = LogReg(input = layer2.output, n_in = 500, n_out = 10)
	f = open('best_model_conv.pkl', 'rb')
	layer0.W.set_value(numpy.asarray(cPickle.load(f), dtype=theano.config.floatX))
	layer0.b.set_value(numpy.asarray(cPickle.load(f), dtype=theano.config.floatX))
	layer1.W.set_value(numpy.asarray(cPickle.load(f), dtype=theano.config.floatX))
	layer1.b.set_value(numpy.asarray(cPickle.load(f), dtype=theano.config.floatX))
	layer2.W.set_value(numpy.asarray(cPickle.load(f), dtype=theano.config.floatX))
	layer2.b.set_value(numpy.asarray(cPickle.load(f), dtype=theano.config.floatX))
	layer3.W.set_value(numpy.asarray(cPickle.load(f), dtype=theano.config.floatX))
	layer3.b.set_value(numpy.asarray(cPickle.load(f), dtype=theano.config.floatX))
	f.close()

	pred_model = theano.function(
		inputs = [index],
		outputs = layer3.y_pred,
		givens={
			x: test_set_x[index * batch_size: (index + 1) * batch_size]
		}
	)

	#predict_value = pred_model (0)
	#print (predict_value)
	real_model = theano.function(
		inputs = [index],
		outputs = layer3.y_pred,
		givens = { 
			x: test_real[index * batch_size: (index + 1) * batch_size]
		}
	)
	# read file and solve
	#filepath = ['demo_data/%i.png' %i for i in range(1,19)]
	#print filepath
	real_value = real_model(0)
	print ('>> ',real_value[:5])

reuse_conv()