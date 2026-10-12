'''
The following code specifies a neural network that takes as its input a word (treated as a sequence of syllables) and outputs a sequence of probability distributions (one for each syllable).
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
		self.name = 'RawModel'

		self.input_dim = 349
		self.hidden_dim = 512 // 2
		self.output_dim = 512
		self.nlayers = 5

		self.stride = 2
		self.filter_size = 5
		self.nchannels = 10
		self.nconvolutions = 3

		g_cpu = torch.Generator()

		seed = 3529145006359120161
		#seed = 9249168034906996919
		#seed = 13979071961427503144
		#seed = 13472831205784841605
		#seed = 8820161422455662510


		self.convin = nn.Conv1d(1, self.nchannels,(self.filter_size,), stride = self.stride)
		self.convin.weight.data = nn.init.uniform_(self.convin.weight.data, -self.filter_size * 0.5, self.filter_size * 0.5, g_cpu.manual_seed(seed))

		self.conv = nn.ParameterList([nn.Conv1d(self.nchannels, self.nchannels, (self.filter_size,), stride = self.stride) for i in range(self.nconvolutions)])
		for c in self.conv:
			c.weight.data = nn.init.uniform_(c.weight.data, -self.filter_size * 0.5, self.filter_size * 0.5, g_cpu.manual_seed(seed))

		self.convout = nn.Conv1d(self.nchannels, 1, (self.filter_size,), stride = self.stride)
		self.convout.weight.data = nn.init.uniform_(self.convout.weight.data, -self.filter_size * 0.5, self.filter_size * 0.5, g_cpu.manual_seed(seed))

		self.convout_reshape = nn.Parameter(nn.init.uniform_(torch.empty((self.output_dim, self.input_dim)), -0.5, 0.5, g_cpu.manual_seed(seed)))

		self.recurrent_left = nn.ParameterList(
						[nn.ParameterList(
						[nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim, self.output_dim) ),  - 0.5, 0.5, g_cpu.manual_seed(seed))),
						nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim) ), 0, 1, g_cpu.manual_seed(seed))),
						nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim,self.hidden_dim)), -0.5, 0.5, g_cpu.manual_seed(seed))),
						nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim)), 0, 1, g_cpu.manual_seed(seed)))]
						) for i in range(self.nlayers)]
						)

		self.recurrent_right = nn.ParameterList(
						[nn.ParameterList(
						[nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim, self.output_dim) ),  - 0.5, 0.5, g_cpu.manual_seed(seed))),
						nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim) ), 0, 1, g_cpu.manual_seed(seed))),
						nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim,self.hidden_dim)), -0.5, 0.5, g_cpu.manual_seed(seed))),
						nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim)), 0, 1, g_cpu.manual_seed(seed)))]
						) for i in range(self.nlayers)]
						)

		self.recurrent_out1 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.output_dim, self.hidden_dim * 2)), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_out1_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.output_dim)), 0, 1, g_cpu.manual_seed(seed)))
		self.recurrent_out2 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.output_dim // 2, self.output_dim) ), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_out2_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.output_dim // 2,)), 0, 1, g_cpu.manual_seed(seed)))
		self.recurrent_out3 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.output_dim // 4, self.output_dim // 2) ), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_out3_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.output_dim // 4,)), 0, 1, g_cpu.manual_seed(seed)))
		self.recurrent_out4 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((2, self.output_dim // 4) ), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_out4_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((2,)), 0, 1, g_cpu.manual_seed(seed)))

		self.attention_weights = []

		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, word: list):
		'''
		passes the word once through the network, and returns the output
		word shape: (n, 30000)
		features shape: (f, n, 250)
		output shape: (n, 2)
		where n is the number of syllables >= 2
		'''
		sound_vec_embedding = []
		if DEBUG:
			print('embedding syllables')
		for syll in word:
			if DEBUG:
				print('syll received', syll.shape)
			sound_vec_embedding.append(self.convolution_forward(syll))
			if DEBUG:
				print('syllable embedded')
		if DEBUG:
			print('starting bi-directional recurrent network')
		output = self.rnn_forward(sound_vec_embedding)
		print('finished forward pass')
		return output

	def convolution_forward(self, vec: Tensor):
		'''
		embeds the input tensor into a compact representation, we first add a channel since the input shape is channelless
		input shape: (30,000)
		output shape: (990)
		'''
		input = self.tanh(self.convin(vec.reshape((1, vec.shape[0]))))
		if DEBUG:
			print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
		for layer in self.conv:
			input = self.tanh(layer(input))
			if DEBUG:
				print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
		output = self.sigmoid(self.convout(input).reshape(-1))
		if DEBUG:
			print(output.shape, torch.min(output).item(), torch.max(output).item(), torch.mean(output).item())
		return torch.matmul(self.convout_reshape, output)


############################################################################################################################
# Not used in the implementation as Python code computation is slower than C
#
#
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
######################################################################################################################################

	def rnn_forward(self, sequence: list):
		'''
		passes two layers of a bidirectional RNN over sequence. The outputs are a sequence of probability distributions for each vector in sequence.
		'''
		if DEBUG:
			print('final layer begins')
		joined_states = sequence
		left_hidden_states = None
		right_hidden_states = None
		for i in range(self.nlayers):
			left_hidden_states = self.left_rnn_forward(joined_states, i)
			right_hidden_states = self.right_rnn_forward(joined_states, i)
			joined_states = self.combine_hidden_states(left_hidden_states, right_hidden_states)

		output_list = []
		if DEBUG:
			print('output layer')
		for i, hidden in enumerate(joined_states):
			if DEBUG:
				print('received hidden vector')
				print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
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
		return torch.stack(output_list)

	def left_rnn_forward(self, sequence: list, layer: int):
		'''
		implements one pass of the left rnn at the specified layer. Returns the hidden states of the rnn.
		'''

		recurrent_left_in, recurrent_left_in_bias, recurrent_left_hidden, recurrent_left_hidden_bias = self.recurrent_left[layer]
		prev = torch.ones((recurrent_left_hidden.shape[0]),)
		hidden_list = []
		if DEBUG:
			print('left rnn')
		for input in sequence:
			if DEBUG:
				print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
			hidden = self.sigmoid(torch.matmul(recurrent_left_in, input) + recurrent_left_in_bias + torch.matmul(recurrent_left_hidden, prev) + recurrent_left_hidden_bias)
			if DEBUG:
				print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			hidden_list.append(hidden)
			prev = hidden
		return hidden_list

	def right_rnn_forward(self, sequence: list, layer: int):
		'''
		Implements one pass of the right rnn a the specified layer. Returns the hidden states of the rnn.
		'''
		reverse_hidden_list = []
		recurrent_right_in, recurrent_right_in_bias, recurrent_right_hidden, recurrent_right_hidden_bias = self.recurrent_right[layer]
		prev = torch.ones((recurrent_right_hidden.shape[0]),)
		n = len(sequence) - 1
		if DEBUG:
			print('right rnn')
		while n >= 0:
			input = sequence[n]
			if DEBUG:
				print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
			hidden = self.sigmoid(torch.matmul(recurrent_right_in, input) + recurrent_right_in_bias + torch.matmul(recurrent_right_hidden, prev) + recurrent_right_hidden_bias)
			if DEBUG:
				print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			reverse_hidden_list.append(hidden)
			prev = hidden
			n -= 1
		return reverse_hidden_list


	def combine_hidden_states(self, left_hidden_states: list, right_hidden_states: list):
		'''
		combines the hidden states from the left rnn and right rnn. the right rnn states are in the wrong order (with the summary for the first entry at the end) so must be reversed first before stacking
		the states.
		'''
		full_context = []
		correct_order = []
		n = len(right_hidden_states) - 1
		while n >= 0:
			correct_order.append(right_hidden_states[n])
			n-=1
		full_context = []
		for hidden1, hidden2 in zip(left_hidden_states, correct_order):
			full = torch.cat((hidden1, hidden2))
			if DEBUG:
				print('combined left and right RNN')
				print(full.shape, torch.min(full).item(), torch.max(full).item(), torch.mean(full).item())
			full_context.append(full)
		return full_context
