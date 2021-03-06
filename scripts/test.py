# Implemented from https://enlight.nyc/projects/neural-network/

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
import random
from sklearn import metrics

class NeuralNetwork:
    def __init__(self, setup=[[68,25,"sigmoid",0],[25,1,"sigmoid",0]],lr=.05,seed=1,error_rate=0,bias=1,iter=500,lamba=.00001,simple=0):
    	# create 8x3x8 encoder
    	self.inputSize = 68
    	self.outputSize = 1
    	self.hiddenSize = 10
        # two sets of weights required 
    	# 1) to go from input layer to hiden layer
    	# 2) to go from the hidden layer to the output layer
    	self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
    	self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
    	# self.bias = np.zeros((self.outputSize, self.hiddenSize))
    	# self.bias2 = np.zeros((self.outputSize, self.outputSize))


    def make_weights(self):
    	self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
    	self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 

    # def softmax(self, s):
    # 	exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    # 	return exps/np.sum(exps, axis=1, keepdims=True)

    # was returning nan due to floating point limitation
    # def softmax(self, s):
    #     return np.exp(s)/np.sum(np.exp(s), axis=0)
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def feedforward(self, X):
    	#forward propagation through our network
    	self.z = np.dot(X, self.W1)#  + self.bias # dot product of X (input) and first set of 8x3 weights
    	self.z2 = self.sigmoid(self.z) # activation function applied to hidden layer to apply nonlinearity and is mapped 0-1
    	self.z3 = np.dot(self.z2, self.W2)#  + self.bias2 # dot product of hidden layer (z2) and second set of 3x8 weights
    	o = self.softmax(self.z3) # final activation function applied to output layer, use softmax to convert logit to probability
    	return o

    def sigmoid(self, s):
    	# activation function
    	return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
    	# derivative of sigmoid was trickier than I thought
    	# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
    	return self.sigmoid(s) * (1-self.sigmoid(s))

    def backprop(self, X, y, o):
    	# Backpropagation works by using a loss(mean squared error) function to calculate how far the network was from the target output.
    	# since we started with random wieghts have to alter them so the output gets more accurate

    	self.o_error = y - o # y is the actual output, o is the predicted output (for autoencoder these should be the same)
        print(self.o_error)
    	self.o_delta = self.o_error*self.sigmoidPrime(o) # get delta output sum by applying derivative of sigmoid to error 

    	# dot product of delta output sum and the second set of weights (from hidden layer to output)
    	# not sure why this is transofrmed? probably ahs to do with dot product rules
    	self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much hidden layer weights contributed to output error

    	# calculate delta output sum for the hidden layer by applying the derivative of sigmoid activation layer
    	self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error


    	# adjust weights
    	# dot product of the input layer with the hidden delta output sum
    	self.W1 += (X.T.dot(self.z2_delta)) * 0.001

    	# perform a dot product of the hidden layer and the output of the delta output sum
    	self.W2 += (self.z2.T.dot(self.o_delta)) * 0.001

    def fit(self, X, y):
    	o = self.feedforward(X) 
    	self.backprop(X, y, o)


    def predict(self, X):
    	return self.feedforward(X)


# NN = NeuralNetwork()

# X = np.identity(8)
# y = X

# for i in range(10000): # trains the NN 5,000 times
#   NN.feedforward(X)
#   NN.fit(X, y)

# answer = NN.feedforward(X)

# print(np.asarray(answer))

# autoencoder seems to be working
# for i in answer:
# 	print(sum(i))




# Describe and implement a data preprocessing approach

# read in test sequences positive, negative, and test sequences
# take reverse complement of positive sequences to create more positive data
# down sample negatives to match # of positives, and make 17bp long
# ensure that none of negatives are in the positives
# convert to binary
# split into testing and training

with open('../data/rap1-lieb-test.txt') as txt:
	all_seqs = txt.readlines()
	# strip /n
	test_sequences_list = [sequences.rstrip() for sequences in all_seqs]


# read in positive sequences
with open('../data/rap1-lieb-positives.txt') as txt:
	all_seqs = txt.readlines()
	# strip /n
	pos_seq_list = [sequences.rstrip() for sequences in all_seqs]

# generate more positive data by taking reverse complement using biopython https://biopython.org/wiki/Seq
seq_obj_list = [Seq(x) for x in pos_seq_list]
rev_comp_pos_seq_list = [str(x.reverse_complement()) for x in seq_obj_list]

positives = pos_seq_list + rev_comp_pos_seq_list
positives = positives + positives + positives + positives + positives + positives + positives

# there are 274 positive examples each with a length of 17

# read in negative sequences
# https://biopython.org/wiki/SeqIO
with open('../data/yeast-upstream-1k-negative.fa') as txt:
	all_negatives = [ ]
	for record in SeqIO.parse("../data/yeast-upstream-1k-negative.fa", "fasta"):
		all_negatives.append(str(record.seq))


# match the number of positives and negatives
long_negatives = random.sample(all_negatives, 274)

# randomly select 17mer from entire negative sequence
negatives = []
for seq in long_negatives:
	sequence_length = len(seq)
	start = random.randrange(0,sequence_length-17)
	end = start + 17
	negatives.append(seq[start:end])

# make sure that negatives and positives don't overlap

# l3 = [x for x in l1 if x not in l2]
cleaned_negatives = [x for x in negatives if x not in positives]

# ensure that none of positives and negatives overlap
# looks like zero overlap
# print(len(set(cleaned_negatives).intersection(set(positives))))

convert = {'A': [1,0,0,0], 'T': [0,1,0,0], 'G': [0,0,1,0], 'C': [0,0,0,1]}


def binary_conversion(sequence_list):
	test_seq = [ ]
	for sequen in sequence_list:
		test = [ ]
		for n in sequen:
			test.append(convert[n])
		flattened_list = [y for x in test for y in x]
		test_seq.append(flattened_list)
		# test_seq.append(np.asarray(test).T) # thought this had to be a 4x17 array?
	return test_seq

positive_binary = binary_conversion(positives)
negative_binary = binary_conversion(cleaned_negatives)
test_binary = binary_conversion(test_sequences_list)


x = positive_binary + negative_binary

x = np.asarray(x)

negative_binary = np.asarray(negative_binary)

pos_y = [1] * len(positive_binary)
neg_y = [0] * len(negative_binary)
combined_y = pos_y + neg_y
neg_y = [0] * len(negative_binary)
neg_y = np.asarray(neg_y)
neg_y = neg_y.reshape(neg_y.shape[0],1)
y = np.asarray(combined_y)
y = y.reshape(y.shape[0],1)


# print(len(positives))
# print(len(cleaned_negatives))
# print(len(neg_y))


# remove 
def split_train_test(x_data, y_data, test_ratio):
	shuffled_indicies = np.random.permutation(len(x_data))
	test_set_size = int(len(x_data) * test_ratio)
	test_indicies = shuffled_indicies[:test_set_size]
	train_indicies = shuffled_indicies[test_set_size:]
	return x_data[train_indicies], x_data[test_indicies], y_data[train_indicies], y_data[test_indicies]

# def cross_fold(x_data, y_data, folds):
# shuffled_indicies = np.random.permutation(len(x))
# test_set_size = int(len(x) * 0.2)
# test_indicies = shuffled_indicies[:test_set_size]
# train_indicies = shuffled_indicies[test_set_size:]

train_x, test_x, train_y, test_y = split_train_test(x,y,0.2)


NN = NeuralNetwork()


for i in range(10): # trains the NN 5,000 times
  NN.fit(train_x,train_y)

print('Prediction')
an = NN.predict(test_x)

for i in an:
    print(i)

print('hey')

for ii in test_y:
    print(ii)


fpr, tpr, thresholds = metrics.roc_curve(test_y, NN.predict(test_x))
auc = metrics.auc(fpr, tpr)
print('AUC is %s' %auc)


# print('Actual answer')
# print(test_y)

# answer = NN.feedforward(X)

# print(answer)

# for i in range(100): # trains the NN 5,000 times
#   NN.fit(x,y)


