# Implemented from https://enlight.nyc/projects/neural-network/

import numpy as np

class NeuralNetwork:
    def __init__(self, setup=[[68,25,"sigmoid",0],[25,1,"sigmoid",0]],lr=.05,seed=1,error_rate=0,bias=1,iter=500,lamba=.00001,simple=0):
    	# create 8x3x8 encoder
    	self.inputSize = 8
    	self.outputSize = 8
    	self.hiddenSize = 3
		# two sets of weights required 
    	# 1) to go from input layer to hiden layer
    	# 2) to go from the hidden layer to the output layer
    	self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
    	self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def make_weights(self):
    	self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
    	self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 

    def feedforward(self, X):
    	#forward propagation through our network
    	self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 8x3 weights
    	self.z2 = self.sigmoid(self.z) # activation function applied to hidden layer to apply nonlinearity and is mapped 0-1
    	self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x8 weights
    	o = self.sigmoid(self.z3) # final activation function applied to output layer to apply nonlinearity and is mapped 0-1
    	return o

    def sigmoid(self, s):
    	# activation function
    	return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
    	# derivative of sigmoid
    	return s * (1-s)

    def backprop(self, X, y, o):
    	# Backpropagation works by using a loss(mean squared error) function to calculate how far the network was from the target output.
    	# since we started with random wieghts have to alter them so the output gets more accurate
    	self.o_error = y - o # y is the actual output, o is the predicted output (for autoencoder these should be the same)
    	self.o_delta = self.o_error*self.sigmoidPrime(o) # get delta output sum by applying derivative of sigmoid to error 

    	# dot product of delta output sum and the second set of weights (from hidden layer to output)
    	# not sure why this is transofrmed? probably ahs to do with dot product rules
    	self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much hidden layer weights contributed to output error

    	# calculate delta output sum for the hidden layer by applying the derivative of sigmoid activation layer
    	self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error


    	# adjust weights
    	# dot product of the input layer with the hidden delta output sum
    	self.W1 += X.T.dot(self.z2_delta) 

    	# perform a dot product of the hidden layer and the output of the delta output sum
    	self.W2 += self.z2.T.dot(self.o_delta) 

    def fit(self, X, y):
    	o = self.feedforward(X)
    	self.backprop(X, y, o)


    def predict(self):
    	return

NN = NeuralNetwork()

X = np.asarray([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]])
y = X

for i in range(5000): # trains the NN 5,000 times
  NN.feedforward(X)
  NN.fit(X, y)

print(NN.feedforward(X))

# def activation(x):
	# return

