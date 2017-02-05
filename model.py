from __future__ import print_function
__docformat__ = 'restructedtext en'

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
#from loaddata import load_data

class LogReg(object):
	def __init__(self, input, n_in, n_out):
		self.W = theano.shared(
			value = numpy.zeros(
				(n_in, n_out),
				dtype = theano.config.floatX),
			name = 'W',
			borrow = True
		)
		self.b = theano.shared(
			value = numpy.zeros(
				(n_out,),
				dtype = theano.config.floatX),
			name = 'b',
			borrow = True
		)
		self.y_softmax = T.nnet.softmax(T.dot(input,self.W) + self.b)
		self.y_pred = T.argmax(self.y_softmax, axis = 1)
		self.params = [self.W, self.b]
		self.input = input
	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.y_softmax)[T.arange(y.shape[0]), y])
	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)

		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()


class HiddenLayer(object):
	"""docstring for HiddenLayer"""
	def __init__(self, rng, input, n_in, n_out, W = None, b = None, 
		activation = T.tanh):

		self.input = input
		if W is None:
			W_values = numpy.asarray(
				rng.uniform(
					low = -numpy.sqrt(6. / (n_in + n_out)),
					high = numpy.sqrt(6. / (n_in + n_out)),
					size = (n_in,n_out)
					),
				dtype = theano.config.floatX
			)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *=4

			W = theano.shared (value = W_values, name = 'w', borrow = True)
		if b is None:
			b_values = numpy.zeros((n_out,), dtype = theano.config.floatX)
			b = theano.shared(value = b_values, name = 'b', borrow = True)

		self.W = W
		self.b = b 
		lin_output = T.dot(input,self.W) + self.b 
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)
		self.params = [self.W, self.b]
class MLP(object):
	def __init__(self, rng, input, n_in, n_hidden, n_out):
		self.hiddenLayer = HiddenLayer(
			rng = rng,
			input = input,
			n_in = n_in,
			n_out = n_hidden,
			activation = T.tanh
		)

		self.logRegressionLayer = LogReg(
			input = self.hiddenLayer.output,
			n_in = n_hidden,
			n_out = n_out
		)
		
		self.L1 = (
			abs(self.hiddenLayer.W).sum() 
			+ abs (self.logRegressionLayer.W ).sum()
		)
		self.L2 = (
			abs(self.hiddenLayer.W**2).sum() 
			+ abs (self.logRegressionLayer.W**2).sum()
		)

		self.negative_log_likelihood = (
			self.logRegressionLayer.negative_log_likelihood
		)
		self.errors = self.logRegressionLayer.errors
		self.params = self.hiddenLayer.params + self.logRegressionLayer.params
		self.input = input


class LeNetConvPoolLayer(object):
	def __init__ (
		self, rng, input, filter_shape, image_shape, poolsize = (2,2)
				):

		assert image_shape[1] == filter_shape[1]
		self.input = input
		
		fan_in = numpy.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
					numpy.prod(poolsize))
		W_bound = numpy.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(
			numpy.asarray(
				rng.uniform(low = -W_bound, high = W_bound, size = filter_shape),
				dtype = theano.config.floatX
			),
			borrow = True
		)

		b_values = numpy.zeros((filter_shape[0],), dtype = theano.config.floatX)
		self.b = theano.shared(value = b_values, borrow = True)

		conv_out = conv2d(
			input = input,
			filters = self.W,
			filter_shape = filter_shape,
			input_shape = image_shape
		)
		pooled_out = downsample.max_pool_2d(
			input = conv_out,
			ds = poolsize,
			ignore_border = True
		)
		self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
		self.params = [self.W, self.b]
		self.input = input

