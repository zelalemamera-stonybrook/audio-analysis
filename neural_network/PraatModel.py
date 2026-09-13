'''
The following code specifies a neural network that takes as its input a word (treated as a sequence of syllables) and outputs a sequence of probability distributions (one for each syllable).
Each syllable is treated as a vector which is obtained by summarizing a sequence of features generated from the raw audio. This summary is  obtained by an attention mechanism based on the position of the syllable within the
word.
'''

import torch
import torchaudio
import torch.nn as nn
from torch import Tensor
DEBUG = False

class Network(nn.Module):
	'''
	neural network implementation for the above
	'''
	def __init__(self):
		super().__init__()
		print('initializing parameters')
		self.name = 'PraatModel'

		g_cpu = torch.Generator()
		seed = 3529145006359120161
		self.conv1 = nn.Conv1d(1, 3, (5,), stride = 5)
		self.conv1.weight.data = nn.init.uniform_(self.conv1.weight.data, -0.5 * 9, 0.5 * 9, g_cpu.manual_seed(seed))
		self.conv1.bias.data = nn.init.uniform_(self.conv1.bias.data, -0.5 * 9, 0.5 * 9, g_cpu.manual_seed(seed))


		self.conv2 = nn.Conv1d(3,2, (5,), stride=2)
		self.conv2.weight.data = nn.init.uniform_(self.conv2.weight.data, -0.5 * 5, 0.5 * 5, g_cpu.manual_seed(seed))
		self.conv2.bias.data = nn.init.uniform_(self.conv2.bias.data, -0.5 * 5, 0.5 * 5, g_cpu.manual_seed(seed))

		self.conv3 = nn.Conv1d(2, 1, (6,), stride=2)
		self.conv3.weight.data = nn.init.uniform_(self.conv3.weight.data, -0.5 * 6, 0.5 * 6, g_cpu.manual_seed(seed))
		self.conv3.bias.data = nn.init.uniform_(self.conv3.bias.data, -0.5 * 6, 0.5 * 6, g_cpu.manual_seed(seed))


		self.conv4 = nn.Conv1d(1,1,(4,), stride = 2)
		self.conv4.weight.data = nn.init.uniform_(self.conv4.weight.data, -0.5 * 4, 0.5 * 4, g_cpu.manual_seed(seed))
		self.conv4.bias.data = nn.init.uniform_(self.conv4.bias.data, -0.5 * 4, 0.5 * 4, g_cpu.manual_seed(seed))


		self.conv5 = nn.Conv1d(1,1, (4,), stride = 2)
		self.conv5.weight.data = nn.init.uniform_(self.conv5.weight.data, -0.5 * 4, 0.5 * 4, g_cpu.manual_seed(seed))
		self.conv5.bias.data = nn.init.uniform_(self.conv5.bias.data, -0.5 * 4, 0.5 * 4, g_cpu.manual_seed(seed))

		self.conv6 = nn.Conv1d(1,1, (4,), stride = 2)
		self.conv6.weight.data = nn.init.uniform_(self.conv6.weight.data, -0.5 * 4, 0.5 * 4, g_cpu.manual_seed(seed))
		self.conv6.bias.data = nn.init.uniform_(self.conv6.bias.data, -0.5 * 4, 0.5 * 4, g_cpu.manual_seed(seed))


		self.attnlayer1 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((500,600 + 225)),  - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.attnlayer1_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((500)), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.attnlayer2 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((100, 500) ), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.attnlayer2_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((100) ), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.attnlayer3 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((1,100)), - 0.5 * 1.5, 0.5 * 1.5, g_cpu.manual_seed(seed)))
		self.attnlayer3_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((1,)), - 0.5, 0.5, g_cpu.manual_seed(seed)))

		self.attention_weights = []

		self.recurrent_left_in = nn.parameter.Parameter(nn.init.uniform_(torch.empty((300, 58) ),  - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_left_in_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((300) ), 0, 1, g_cpu.manual_seed(seed)))
		self.recurrent_left_hidden = nn.parameter.Parameter(nn.init.uniform_(torch.empty((300,300)), -0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_left_hidden_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((300)), 0, 1, g_cpu.manual_seed(seed)))

		self.recurrent_right_in = nn.parameter.Parameter(nn.init.uniform_(torch.empty((300, 58)), - 0.5, 0.5,  g_cpu.manual_seed(seed)))
		self.recurrent_right_in_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((300) ), 0, 1, g_cpu.manual_seed(seed)))
		self.recurrent_right_hidden = nn.parameter.Parameter(nn.init.uniform_(torch.empty((300,300)), -0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_right_hidden_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((300)), 0, 1, g_cpu.manual_seed(seed)))

		self.recurrent_out1 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((1000, 600 + 1800)), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_out1_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((1000)), 0, 1, g_cpu.manual_seed(seed)))
		self.recurrent_out2 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((500, 1000) ), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_out2_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((500,)), 0, 1, g_cpu.manual_seed(seed)))
		self.recurrent_out3 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((250, 500) ), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_out3_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((250,)), 0, 1, g_cpu.manual_seed(seed)))
		self.recurrent_out4 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((2, 250) ), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_out4_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((2,)), 0, 1, g_cpu.manual_seed(seed)))

		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, word: list):
		'''
		Passes the word once through the network. The architecture of the network is as follows: the input is assumed to be a sequence of syllables where each syllable is
		a list of feature embeddings. Feature embeddings can come from a variety of sources but they are all generated over the raw audio signal associated with the syllable.
		First all of these features must be summarized into one vector. For this a simple single headed attention mechanism is used with queries coming from a vector with two pieces of information:
		1. the position of the current syllable to be summarized, and 2. the total number of syllables that are present in this word. The values are the feature vectors of this syllable, and the keys will be a matrix
		learned by the network. The output of this step applied to each syllable is a sequence where each syllable is a single vector representing a feature summary.
		The weights used to generate this summary are used for interpreting feature importance.

		The next step is to apply a bidirectional RNN to the sequence which generates a binary distribution for each syllable interpreted as its likelihood of being stressed. This takes into account the full
		context surrounding the syllable. The output probabilities are used to make prediction about stress position in the word.
		'''
		sound_vec_embedding = []
		if DEBUG:
			print('embedding syllables')
		for syll in word:
			if DEBUG:
				print('syll received', syll.shape)
			sound_vec_embedding.append(self.convolution_forward(syll))
		output = self.rnn_forward(sound_vec_embedding, features)
		print('finished forward pass')
		return output


	def filter(self, features: Tensor, i: int):
		'''
		returns the ith element of each tensor in features
		'''
		filtered_list = []
		for word in features:
			filtered_list.append(word[i])
		return torch.stack(tuple(filtered_list))

	def convolution_forward(self, vec: Tensor):
		'''
		embeds the input tensor into a compact representation, we first add a channel since the input shape is channelless
		input shape: (30,000)
		output shape: (990)
		'''
		conv_first = self.tanh(self.conv1(vec.reshape((1,vec.shape[0]))))
		if DEBUG:
			print(conv_first.shape, torch.min(conv_first).item(), torch.max(conv_first).item(), torch.mean(conv_first).item())
		conv_second = self.tanh(self.conv2(conv_first))
		if DEBUG:
			print(conv_second.shape, torch.min(conv_second).item(), torch.max(conv_second).item(), torch.mean(conv_second).item())
		conv_third = self.tanh(self.conv3(conv_second))
		if DEBUG:
			print(conv_third.shape, torch.min(conv_third).item(), torch.max(conv_third).item(), torch.mean(conv_third).item())
		'''
		conv_fourth = self.tanh(self.conv4(conv_third))
		if DEBUG:
			print(conv_fourth.shape, torch.min(conv_fourth).item(), torch.max(conv_fourth).item(), torch.mean(conv_fourth).item())

		conv_fifth = self.tanh(self.conv5(conv_fourth))
		if DEBUG:
			print(conv_fifth.shape, torch.min(conv_fifth).item(), torch.max(conv_fifth).item(), torch.mean(conv_fifth).item())

		conv_sixth = self.sigmoid(self.conv6(conv_fifth))
		if DEBUG:
			print(conv_sixth.shape, torch.min(conv_sixth).item(), torch.max(conv_sixth).item(), torch.mean(conv_sixth).item())
		'''
		return conv_third.reshape(-1)
##########################################################################################################################################################################
# This function is not used in the implementation.
	def convolve(self, conv: Tensor, input: Tensor, bias: Tensor,  stride=1):
		'''
		slides conv once over the input signal with stride = n and returns the result
		'''
		print('convolution begins')
		width = len(conv)
		print('input received', len(input))
		print('width of filter', width)
		print('stride', stride)
		print('output dimension should be', ((len(input) - width) / stride) + 1)
		output = [torch.linalg.vecdot(conv, input[0 + stride * i : width + stride * i] ) + bias for i in range( int((len(input) - width) / stride + 1))]
		output = torch.stack(output).reshape(-1)
		print(output.shape, output, 'min', torch.min(output).item(), 'max', torch.max(output).item())
		return output
#
#############################################################################################################################################################################



	def attend(self, hidden: Tensor, feature_vecs: Tensor):
		'''
		computes the attention score of each element in the list with respect to the other elements, then returns the weighted sum of the whole
		'''
		if DEBUG:
			print('starting attention network')
		weight_list = []
		if DEBUG:
			print('compatibility is computed over', feature_vecs.shape)
		feature_vecs = [self.sigmoid(f) for f in feature_vecs]
		for attention_target in feature_vecs:
			weight_list.append(self.attention_forward(hidden, attention_target))
		weight_tensor = torch.stack(weight_list).reshape(-1)
		attention_vector = self.softmax(weight_tensor)
		print(attention_vector)
		i = 0
		weighted_list = []
		for feature in feature_vecs:
			weighted_list.append(attention_vector[i] * feature)
			i +=1
			if DEBUG:
				print(f'attention vector {i - 1}', attention_vector[i - 1])
				print('weighted feature', weighted_list[-1])
		weighted = torch.cat(weighted_list)
		if DEBUG:
			print(weighted.shape, torch.min(weighted).item(), torch.max(weighted).item(), torch.mean(weighted).item())
			print(weighted)
		output = torch.cat((hidden, weighted))
		return attention_vector, output



	def attention_forward(self, attention_source: Tensor, attention_target: Tensor):
		'''
		computes the compatibility score of the source to the target
		input shape: (400 + 30)
		output shape: (1)
		'''
		if DEBUG:
			print('attention forward begins')
		input = torch.cat((attention_source, attention_target))
		if DEBUG:
			print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
		first_layer = self.sigmoid(torch.matmul( self.attnlayer1, input ) + self.attnlayer1_bias)
		if DEBUG:
			print(first_layer.shape, torch.min(first_layer).item(), torch.max(first_layer).item(), torch.mean(first_layer).item())
		second_layer = self.sigmoid(torch.matmul(self.attnlayer2, first_layer ) + self.attnlayer2_bias)
		if DEBUG:
			print(second_layer.shape, torch.min(second_layer).item(), torch.max(second_layer).item(), torch.mean(second_layer).item())
		third_layer = torch.matmul(self.attnlayer3, second_layer ) + self.attnlayer3_bias
		if DEBUG:
			print(third_layer)
		return third_layer

	def rnn_forward(self, injected_list: Tensor, features = None):
		'''
		passes the list of feature injected and attention weighted tensors through one pass of a bi-directional reccurrent network, and returns the output sequence as a list of probability distributions over the two classes. 
		input shape: (n, 30)
		output shape: (n, 2)
		'''
		if DEBUG:
			print('final layer begins')
		prev = torch.ones((self.recurrent_left_hidden.shape[0]),)
		hidden_list = []
		if DEBUG:
			print('left rnn')
		for input in injected_list:
			if DEBUG:
				print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
			hidden = self.sigmoid(torch.matmul(self.recurrent_left_in, input) + self.recurrent_left_in_bias + torch.matmul(self.recurrent_left_hidden, prev) + self.recurrent_left_hidden_bias)
			if DEBUG:
				print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			hidden_list.append(hidden)
			prev = hidden
		reverse_hidden_list = []
		prev = torch.ones((self.recurrent_left_hidden.shape[0]),)
		n = len(injected_list) - 1
		if DEBUG:
			print('right rnn')
		while n >= 0:
			input = injected_list[n]
			if DEBUG:
				print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
			hidden = self.sigmoid(torch.matmul(self.recurrent_right_in, input) + self.recurrent_right_in_bias + torch.matmul(self.recurrent_right_hidden, prev) + self.recurrent_right_hidden_bias)
			if DEBUG:
				print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			reverse_hidden_list.append(hidden)
			prev = hidden
			n -= 1
		hidden_list2 = []
		n = len(reverse_hidden_list) - 1
		while n >= 0:
			hidden_list2.append(reverse_hidden_list[n])
			n-=1
		full_context = []
		for hidden1, hidden2 in zip(hidden_list, hidden_list2):
			full = torch.cat((hidden1, hidden2))
			if DEBUG:
				print('combined left and right RNN')
				print(full.shape, torch.min(full).item(), torch.max(full).item(), torch.mean(full).item())
			full_context.append(full)
		output_list = []
		if DEBUG:
			print('output layer')
		attention_list = []
		for i, hidden in enumerate(full_context):
			if DEBUG:
				print('received hidden vector')
				print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			if features:
				feature_vecs = self.filter(features, i)
				attention_vector, hidden = self.attend(hidden, feature_vecs)
				attention_list.append(attention_vector.tolist())
			if False:
				print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
				print('prediction layer')
			first_layer = self.sigmoid(torch.matmul(self.recurrent_out1, hidden) + self.recurrent_out1_bias)
			if DEBUG:
				print(first_layer.shape, torch.min(first_layer).item(), torch.max(first_layer).item(), torch.mean(first_layer).item())
			second_layer = self.sigmoid(torch.matmul(self.recurrent_out2, first_layer) + self.recurrent_out2_bias)
			if DEBUG:
				print(second_layer.shape, torch.min(second_layer).item(), torch.max(second_layer).item(), torch.mean(second_layer).item())
			third_layer = self.sigmoid(torch.matmul(self.recurrent_out3, second_layer) + self.recurrent_out3_bias)
			if DEBUG:
				print(third_layer.shape, torch.min(third_layer).item(), torch.max(third_layer).item(), torch.mean(third_layer).item())
			fourth_layer = self.softmax(torch.matmul(self.recurrent_out4, third_layer) + self.recurrent_out4_bias)
			output_list.append(fourth_layer)
		if features:
			self.attention_weights.append(attention_list)
		return torch.stack(output_list)
