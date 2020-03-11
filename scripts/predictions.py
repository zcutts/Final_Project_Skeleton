# Implemented from https://enlight.nyc/projects/neural-network/

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
import random
from sklearn import metrics
import matplotlib.pyplot as plt

class NeuralNetwork:
	def __init__(self, learning = 0.001, hidden_size = 10, setup=[[68,25,"sigmoid",0],[25,1,"sigmoid",0]],lr=.05,seed=10,error_rate=0,bias=1,iter=500,lamba=.00001,simple=0):
		# create 8x3x8 encoder
		self.inputSize = 68
		self.outputSize = 1
		self.hiddenSize = hidden_size
		random.seed(1)
		self.lr = learning
		# two sets of weights required 
		# 1) to go from input layer to hiden layer
		# 2) to go from the hidden layer to the output layer
		self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
		self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
		self.bias = np.zeros((self.outputSize, self.hiddenSize))
		self.bias2 = np.zeros((self.outputSize, self.outputSize))
		self.mse = 1


	def make_weights(self):
		self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
		self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 

	# def softmax(self, s):
	#   exps = np.exp(s - np.max(s, axis=1, keepdims=True))
	#   return exps/np.sum(exps, axis=1, keepdims=True)

	# was returning nan due to floating point limitation
	# def softmax(self, s):
	#     return np.exp(s)/np.sum(np.exp(s), axis=0)

	def softmax(self, Z):
		expZ = np.exp(Z - np.max(Z))
		return expZ / expZ.sum(axis=0, keepdims=True)

	def feedforward(self, X):
		#forward propagation through our network
		self.z = np.dot(X, self.W1) + self.bias # dot product of X (input) and first set of 8x3 weights
		self.z2 = self.sigmoid(self.z) # activation function applied to hidden layer to apply nonlinearity and is mapped 0-1
		self.z3 = np.dot(self.z2, self.W2) + self.bias2 # dot product of hidden layer (z2) and second set of weights
		o = self.softmax(self.z3) # final activation function applied to output layer, use softmax to convert logit to probability
		return self.sigmoid(self.z3)

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
		self.mse = ((y-o)**2).mean()
		# print(self.o_error)
		self.o_delta = self.o_error*self.sigmoidPrime(o) # get delta output sum by applying derivative of sigmoid to error 

		# dot product of delta output sum and the second set of weights (from hidden layer to output)
		# not sure why this is transofrmed? probably ahs to do with dot product rules
		self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much hidden layer weights contributed to output error

		# calculate delta output sum for the hidden layer by applying the derivative of sigmoid activation layer
		self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error


		# adjust weights
		# dot product of the input layer with the hidden delta output sum
		self.W1 += (X.T.dot(self.z2_delta))  * self.lr

		# perform a dot product of the hidden layer and the output of the delta output sum
		self.W2 += (self.z2.T.dot(self.o_delta))  * self.lr

	def fit(self, X, y):
		o = self.feedforward(X) 
		self.backprop(X, y, o)


	def predict(self, X):
		return self.feedforward(X)







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
positives = positives + positives

# there are 274 positive examples each with a length of 17

# read in negative sequences
# https://biopython.org/wiki/SeqIO
with open('../data/yeast-upstream-1k-negative.fa') as txt:
	all_negatives = [ ]
	for record in SeqIO.parse("../data/yeast-upstream-1k-negative.fa", "fasta"):
		all_negatives.append(str(record.seq))


# match the number of positives and negatives
long_negatives = random.sample(all_negatives, len(positives))

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


def split_train_test(x_data, y_data, test_ratio, k=5):
	shuffled_indicies = np.random.permutation(len(x_data))
	test_set_size = int(len(x_data) * test_ratio)
	test_indicies = shuffled_indicies[:test_set_size]
	train_indicies = shuffled_indicies[test_set_size:]
	return x_data[train_indicies], x_data[test_indicies], y_data[train_indicies], y_data[test_indicies]


def k_fold(x_data, y_data, test_ratio, k=5):
	shuffled_indicies = np.random.permutation(len(x_data))
	test_set_size = int(len(x_data) * test_ratio)
	test_indicies = shuffled_indicies[:test_set_size]
	train_indicies = shuffled_indicies[test_set_size:]
	test_indicies_list = []
	train_indicies_list = []
	for i in range(k):
		shuf = shuffled_indicies
		# print('shuffled indicies')
		# print(shuf)
		test_indicies = shuf[i*test_set_size:test_set_size * (i+1)]
		# print('test indicies')
		# print(len(test_indicies))
		# print(test_indicies)
		train_indicies = np.delete(shuf, list(range(i*test_set_size,test_set_size * (i+1))))
		# print('train indicies')
		# print(len(train_indicies))
		# print(train_indicies)
		test_indicies_list.append(test_indicies)
		train_indicies_list.append(train_indicies)
	return test_indicies_list, train_indicies_list



testing, training = k_fold(x,y,0.2)




k_fold_auc = []
model_error_list = [ ]
aur_list = [ ]
iterations = [1, 10, 100, 1000, 5000, 10000, 50000, 75000, 100000]

def train_model(x, y, ii, num_train):
	error = [ ]
	for i in range(num_train):
		NN.fit(x[training[ii]],y[training[ii]])
		error.append(NN.mse)
	return error


# Testing stop criterion for convergence in your learned parameters
# ideally I would test when the test set error starts to increase, 
# but I decided to cut off the training when the error started to stay the same

5 folds for K-folds validation
for i in range(0,5):
	print(i)
	model_error = [ ]
	auc_ = [ ]
	NN = NeuralNetwork()
	error = train_model(x,y,i, 10000)
	an = NN.predict(x[testing[i]])
	fpr, tpr, thresholds = metrics.roc_curve(y[testing[i]], an)
	auc = metrics.auc(fpr, tpr)
	model_error.append(error)
	auc_.append(auc)

plt.plot(error)
plt.show
print(error)



def train_model(x, y, ii, num_train):
	for i in range(num_train):
		NN.fit(x[training[ii]],y[training[ii]])


# Testing hyperparameters

learning_rate_list = [.0001, .00025, .0005, .001, .005, .01, .05, .1, .2, 0.3, 0.4, 0.5]
hidden_nodes = [2, 5, 6, 8, 10, 12, 18, 25, 30, 40, 50, 60, 70]


for learn in learning_rate_list:
	for node in hidden_nodes:
		for i in range(0,5):
				print(i)
				model_error = [ ]
				auc_ = [ ]
				NN = NeuralNetwork(learning = learn, hidden_size = node)
				train_model(x,y,i, 5000)
				an = NN.predict(x[testing[i]])
				fpr, tpr, thresholds = metrics.roc_curve(y[testing[i]], an)
				auc = metrics.auc(fpr, tpr)
				model_error.append(NN.mse)
				auc_.append(auc)
		k_fold_auc.append(np.mean(auc_))
		model_error_list.append(np.mean(model_error))


# print('Auc is')
# print(k_fold_auc)
# print('error is')
# print(model_error_list)


# auc_hyp = [0.9019130726631984, 0.9540191307266319, 0.9833864742406444, 0.9902668232924987, 0.9789394193656653, 0.9916932371203222, 0.9937069978184259, 0.9935391844269172, 0.9862393018962914, 0.9903507299882531, 0.9925323040778654, 0.9963920120825641, 0.9963081053868098, 0.9798623930189629, 0.9932874643396543, 0.9851485148514851, 0.9965598254740728, 0.9895955697264641, 0.9843933545896962, 0.9949655982547407, 0.9960563852995469, 0.9962241986910556, 0.9939587179056889, 0.9978184259103876, 0.9953012250377581, 0.9959724786037926, 0.6779661016949152, 0.9912737036415507, 0.9891760362476926, 0.9894277563349555, 0.9969793589528444, 0.9934552777311629, 0.9938748112099345, 0.9935391844269172, 0.9943782513844605, 0.9895955697264642, 0.9937909045141803, 0.9935391844269174, 0.9878335291156235, 0.9787716059741567, 0.9951334116462494, 0.9906024500755161, 0.9963081053868098, 0.9931196509481457, 0.990854170162779, 0.9938748112099346, 0.9948816915589864, 0.991357610337305, 0.9940426246014432, 0.9917771438160765, 0.9960563852995469, 0.9990770263467025, 0.9895955697264641, 0.9849807014599765, 0.9812888068467863, 0.9856519550260111, 0.9798623930189629, 0.9844772612854507, 0.9709682832690049, 0.9847289813727136, 0.9880852492028864, 0.9915254237288136, 0.9877496224198692, 0.990350729988253, 0.9869944621580802, 0.9692062426581641, 0.9765900318845444, 0.9645074676959221, 0.9892599429434469, 0.9750797113609666, 0.9860714885047828, 0.9778486323208593, 0.9787716059741568, 0.9819600604128209, 0.9797784863232086, 0.9799462997147173, 0.9850646081557308, 0.9881691558986407, 0.9719332102701795, 0.9870783688538345, 0.5635593220338984, 0.9690384292666555, 0.9757509649270012, 0.9894277563349555, 0.9745343178385635, 0.978310119147508, 0.9752475247524752, 0.9787716059741568, 0.9807853666722605, 0.9804497398892432, 0.9875818090283605, 0.9619483134754153, 0.9753314314482296, 0.9579627454270851, 0.9890082228561838, 0.9714717234435308, 0.9736532975331432, 0.9857358617217654, 0.979107232757174, 0.9794428595401913, 0.9775130055378419, 0.9619063601275383, 0.9730659506628629, 0.9687867091793926, 0.5635593220338984, 0.988504782681658, 0.974911897969458, 0.9750797113609666, 0.9758768249706327, 0.9719751636180567, 0.9805336465849974, 0.9807853666722605, 0.9834284275885216, 0.9751636180567209, 0.9808692733680149, 0.9910219835542876, 0.9738211109246518, 0.9710521899647592, 0.9766319852324216, 0.9640040275213962, 0.982757174022487, 0.9883789226380264, 0.9913995636851821, 0.9849807014599765, 0.9835123342842759, 0.9763383117972814, 0.9785618392347709, 0.9818342003691894, 0.9730239973149858, 0.9847289813727136, 0.9808692733680148, 0.9859875818090283, 0.9899311965094815, 0.9825474072831013, 0.9871622755495888, 0.9877496224198691, 0.9873300889410975, 0.978058399060245, 0.9810790401074007, 0.9864490686356772, 0.9833025675448901, 0.9837221010236616, 0.9834284275885216, 0.9833864742406444, 0.9829669407618729, 0.9894697096828327, 0.9911478435979191, 0.9844353079375734, 0.9890082228561838, 0.987959389159255, 0.9829249874139957, 0.9773871454942104, 0.9765480785366671, 0.9836381943279074, 0.9869105554623259, 0.9850646081557308]
# err_hyp = [0.20465249477068964, 0.07217837158882598, 0.05003107961785816, 0.03941041085408836, 0.04166847409642125, 0.027669118204887468, 0.028974918503862787, 0.028756677915645807, 0.025081802769862268, 0.01901541562356726, 0.025735417071499938, 0.01648319921068552, 0.017015485650599776, 0.03552131854123011, 0.034993205862476215, 0.020785792627905683, 0.020344303198651655, 0.02216874882931812, 0.020732417670248107, 0.015080604105313008, 0.014545513360464184, 0.013962086498009542, 0.010860894841618816, 0.008957028921197829, 0.010966327350386978, 0.008484887385359338, 0.22880453446603308, 0.03408013055960948, 0.016903239209271, 0.012447164916200034, 0.014010009574651393, 0.009073062644123922, 0.008369134374919965, 0.010893408980970375, 0.009448106372597945, 0.008926877998420573, 0.009287882998712528, 0.007795161964797222, 0.0062244026407449835, 0.012783919317924004, 0.01311461037337695, 0.00909830623667101, 0.009369578909386521, 0.013072857801906904, 0.02553239946496954, 0.008400302314841365, 0.006793289452690167, 0.004943125154141766, 0.00760427542759632, 0.004248900420765671, 0.0034752710122354373, 0.003342707768678919, 0.005692709564986115, 0.003916500847700629, 0.0038665709721929582, 0.003654948102838394, 0.003252515287689319, 0.0027916007299068216, 0.0025232787980316667, 0.0033192176320288765, 0.0014259200204495756, 0.0015786970648344661, 0.0005389329861994008, 0.0006044720780077827, 0.0014125838884440983, 0.0024706298976489706, 0.001076700864827448, 0.0017349430445184166, 0.0010302857138198545, 0.0015235707287563425, 0.0010353336317408075, 0.0006589147503595381, 0.0015095978479692338, 0.00034975880748033486, 0.0004666067213715877, 0.00039472152829354314, 0.00023263454745543684, 0.00011850458862377168, 0.0035266560582332103, 0.0013105456855381878, 0.2294757463082719, 0.0023967201346930876, 3.473975530519123e-05, 0.00011146380998798837, 0.00044917340089394286, 0.0008918110936429259, 9.163302392293738e-05, 8.854019595243968e-06, 5.014935131532646e-05, 6.651802344604063e-05, 5.024824421459919e-05, 0.010966038119522558, 3.495977946363401e-05, 0.00013452720371510507, 0.00015189126222974546, 0.008881161675859943, 0.0028205321696577875, 1.0840685612027556e-05, 9.750519158627858e-05, 0.00017676651084639093, 2.3197700485437896e-05, 8.31380440406274e-05, 1.3997069893204856e-05, 1.3172381640492768e-05, 0.22980685755601307, 0.022990437785485447, 0.009539519140022658, 6.036833752141562e-05, 0.013171245343078288, 0.003995800552675564, 0.00020888339233845855, 0.00508968060051983, 0.009474726298509734, 0.009186531757041537, 0.008011166048387964, 2.6698455641274247e-05, 0.005802199317742409, 0.018164745038407794, 0.008218527177296575, 0.008589919846927528, 0.008379484032451697, 0.0343455427744973, 0.008381310595960605, 0.00773397347913532, 0.009958634090988013, 0.012310091236451288, 0.011870140709261399, 0.017067491963626145, 0.010825450792327922, 0.010982375974515323, 0.015933337278560136, 0.012250323929169075, 0.020174572647194675, 0.013667933518002688, 0.007962304569555882, 0.01146074424549657, 0.006637277909845872, 0.00824497376218822, 0.012767606526616362, 0.008903093172166652, 0.010731427259606033, 0.010525198164627075, 0.005247356187652042, 0.016211087085967375, 0.020546989154768592, 0.02616974497656919, 0.010997297897363863, 0.011931684839274273, 0.009786794501710648, 0.012606201933134357, 0.008560982068345071, 0.00568515720630276, 0.011556369121662602, 0.011143641453247657, 0.009154511167261426, 0.014054202721822525]


# print('.0001')
# print(auc_hyp[0:13])
# print(err_hyp[0:13])
# print('.00025')
# print(auc_hyp[13:26])
# print(err_hyp[13:26])
# print('.0005')
# print(auc_hyp[26:39])
# print(err_hyp[26:39])
# print('.001')
# print(auc_hyp[39:52])
# print(err_hyp[39:52])
# print('.005')
# print(auc_hyp[52:65])
# print(err_hyp[52:65])

NN = NeuralNetwork(learning = 0.00025, hidden_size = 18)
for i in range(5000):
	NN.fit(x,y)

predictions_answer = list(NN.predict(test_binary).flatten())

df = pd.DataFrame({'sequences': test_sequences_list, 'predictions': predictions_answer})
print(df)

df.to_csv('zach_cutts_predictions3.tsv', sep='\t')





