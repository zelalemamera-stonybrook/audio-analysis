'''
This code implements the same neural network architecture of Model.py, however, this model is optimized for gpu computations. The algorithm attempts to batch all computations.
'''
import torch
import torchaudio
import torch.nn as nn
from torch import Tensor

###############################################
## important definitions

DEBUG = False
device = torch.device('cpu')
torch.set_default_device(device)

g_cpu = torch.Generator(device=device)

seeds = [3529145006359120161, 9249168034906996919, 13979071961427503144, 13472831205784841605, 8820161422455662510]
seed = seeds[0]
##
#################################################


class Network(nn.Module):

	def __init__(self):
		super().__init__()

		self.name = 'Model'

		self.input_dim = 354
		self.seq_len = 8
		self.nsyll = 4 # the number of maximum syllables in the word that has the most number of syllables in the dataset. All model computations are fixed to this syllable length and masking is used to deal with the variation.
		self.hidden_dim = 512 // 2
		self.output_dim = 512
		self.nheads = 5
		self.nlayers = 5

		self.keys = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.seq_len, self.input_dim)), -0.05, 0.05, g_cpu.manual_seed(seed)))
		self.heads = nn.ParameterList(
				[nn.ParameterList(
					[nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.input_dim // self.nheads, self.input_dim)), -0.05, 0.05, g_cpu.manual_seed(seed))),
					nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.input_dim // self.nheads, self.input_dim)), -0.05, 0.05, g_cpu.manual_seed(seed))),
					nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.input_dim // self.nheads, self.input_dim)), -0.05, 0.05, g_cpu.manual_seed(seed)))]
					) for i in range(self.nheads)
				]
				)
		self.multihead_out = nn.parameter.Parameter(nn.init.uniform_(torch.empty((((self.input_dim // self.nheads) * self.nheads), self.output_dim)), -0.05, 0.05, g_cpu.manual_seed(seed)))

		self.attention_weights = []

		self.recurrent_left = nn.ParameterList(
						[nn.ParameterList(
						[nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim, self.output_dim)),  - 0.5, 0.5, g_cpu.manual_seed(seed))),
						nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim)), 0, 1, g_cpu.manual_seed(seed))),
						nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim,self.hidden_dim)), -0.5, 0.5, g_cpu.manual_seed(seed))),
						nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim)), 0, 1, g_cpu.manual_seed(seed)))]
						) for i in range(self.nlayers)]
					)


		self.recurrent_right = nn.ParameterList(
						[nn.ParameterList(
						[nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim, self.output_dim)),  - 0.5, 0.5, g_cpu.manual_seed(seed))),
						nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim)), 0, 1, g_cpu.manual_seed(seed))),
						nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim,self.hidden_dim)), -0.5, 0.5, g_cpu.manual_seed(seed))),
						nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.hidden_dim)), 0, 1, g_cpu.manual_seed(seed)))]
						) for i in range(self.nlayers)]
					)

		self.recurrent_out1 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.output_dim, self.hidden_dim * 2)), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_out1_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.output_dim)), 0, 1, g_cpu.manual_seed(seed)))
		self.recurrent_out2 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.output_dim // 2, self.output_dim)), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_out2_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.output_dim // 2,)), 0, 1, g_cpu.manual_seed(seed)))
		self.recurrent_out3 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.output_dim // 4, self.output_dim // 2)), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_out3_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.output_dim // 4,)), 0, 1, g_cpu.manual_seed(seed)))
		self.recurrent_out4 = nn.parameter.Parameter(nn.init.uniform_(torch.empty((2, self.output_dim // 4)), - 0.5, 0.5, g_cpu.manual_seed(seed)))
		self.recurrent_out4_bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty((2,)), 0, 1, g_cpu.manual_seed(seed)))

		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=1)

	def __call__(self, x, mask):
		return self.forward(x, mask)

	def forward(self, batch, mask):

		batch_size, syll_size = batch.shape[0], batch.shape[1]
		queries = self.obtain_queries(batch, batch_size)
		syllable_batches = []

		for i in range(syll_size):
			syllable_batch = torch.stack([batch[k][i] for k in range(batch_size)])
			query_batch = torch.stack([torch.cat((self.position_encoder(i + 1), self.position_encoder(mask[k]))) + queries[k] for k in range(batch_size)])
			distributions, summaries = self.multi_attention_forward(syllable_batch, query_batch, batch_size)
			syllable_batches.append(summaries)

		output = self.rnn_forward(torch.stack(syllable_batches), batch_size)
		return output

	def obtain_queries(self, batch, batch_size):

		query_list = []
		for n in range(batch_size):
			word = batch[n]
			query_vector = torch.zeros((self.input_dim,))
			for syll in word:
				for vec in syll:
					query_vector += vec
			query_list.append(query_vector / word.shape[0] * word.shape[1])
		return torch.stack(query_list)

	def position_encoder(self, i: int):
		'''
		implements the same position encoding function from vaswani et. al
		'''
		position_vector = torch.empty((self.input_dim // 2,))
		for dim in range(len(position_vector)):
			if dim % 2 == 0:
				position_vector[dim] = torch.sin( torch.tensor(i / (10000 ** (dim / self.output_dim))))
			else:
				position_vector[dim] = torch.cos(torch.tensor(i / (10000 ** (dim / self.output_dim))))
		return position_vector

	def multi_attention_forward(self, syllable_batch, query_batch, batch_size):
		'''
		Uses all of the heads in the model to implement multiheaded attention from Vaswani et al.
		'''
		attention_vector = None
		head_outputs = []
		i = 0
		for Q, K, V in self.heads:
			projected_queries = torch.matmul(query_batch, Q.T)
			projected_key = torch.matmul(self.keys, K.T)
			batch_attention_weights = self.softmax(torch.matmul(projected_queries, projected_key.T))

			values_batch = syllable_batch.view((batch_size * self.seq_len, self.input_dim))
			values = torch.matmul(values_batch, V.T)
			values = values.view((batch_size, self.seq_len, self.input_dim // self.nheads))

			output = torch.stack([torch.matmul(batch_attention_weights[j], values[j]) for j in range(batch_size)])
			head_outputs.append(output)
			if i == 0:
				attention_vector = batch_attention_weights
			i+=1
		final = torch.matmul(torch.cat(head_outputs, 1), self.multihead_out)
		return attention_vector, final

	def attention_forward(self, query: Tensor, key: Tensor, values: Tensor):
		'''
		Generates a probability distribution over the values based on query.
		'''
		if DEBUG:
			print('attention forward begins')
			print('keys are mapped:', torch.matmul(key, query))
			print(self.softmax(torch.matmul(key, query)))
		attention_vector = self.softmax(torch.matmul(key, query))
		summary = torch.matmul(attention_vector, values)
		return attention_vector, summary

	def rnn_forward(self, sequence, batch_size):
		'''
		passes two layers of a bidirectional RNN over sequence. The outputs are a sequence of probability distributions for each vector in sequence.
		'''
		if DEBUG:
			print('final layer begins')
		joined_states = sequence
		left_hidden_states = None
		right_hidden_states = None
		for i in range(self.nlayers):
			left_hidden_states = self.left_rnn_forward(joined_states, i, batch_size)
			right_hidden_states = self.right_rnn_forward(joined_states, i, batch_size)
			joined_states = self.combine_hidden_states(left_hidden_states, right_hidden_states)

		output_list = []
		if DEBUG:
			print('output layer')
		for i, hidden in enumerate(joined_states):
			if DEBUG:
				print('received hidden vector')
				print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			first_layer = self.sigmoid(torch.matmul(hidden, self.recurrent_out1.T) + torch.stack([self.recurrent_out1_bias for i in range(batch_size)]))
			if DEBUG:
				print(first_layer.shape, torch.min(first_layer).item(), torch.max(first_layer).item(), torch.mean(first_layer).item())
			second_layer = self.sigmoid(torch.matmul(first_layer, self.recurrent_out2.T) + torch.stack([self.recurrent_out2_bias for i in range(batch_size)]))
			if DEBUG:
				print(second_layer.shape, torch.min(second_layer).item(), torch.max(second_layer).item(), torch.mean(second_layer).item())
			third_layer = self.sigmoid(torch.matmul(second_layer, self.recurrent_out3.T) + torch.stack([self.recurrent_out3_bias for i in range(batch_size)]))
			if DEBUG:
				print(third_layer.shape, torch.min(third_layer).item(), torch.max(third_layer).item(), torch.mean(third_layer).item())
			fourth_layer = self.softmax(torch.matmul(third_layer, self.recurrent_out4.T) + torch.stack([self.recurrent_out4_bias for i in range(batch_size)]))
			output_list.append(fourth_layer)
		return torch.stack(output_list)

	def left_rnn_forward(self, sequence, layer, batch_size):
		'''
		implements one pass of the left rnn at the specified layer. Returns the hidden states of the rnn.
		'''
		recurrent_left_in, recurrent_left_in_bias, recurrent_left_hidden, recurrent_left_hidden_bias = self.recurrent_left[layer]
		prev = torch.ones((batch_size, recurrent_left_hidden.shape[0]))
		hidden_list = []
		if DEBUG:
			print('left rnn')
		for input in sequence:
			if DEBUG:
				print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
			hidden = self.sigmoid(torch.matmul(input, recurrent_left_in.T) + torch.stack([recurrent_left_in_bias for i in range(batch_size)]) + torch.matmul(prev, recurrent_left_hidden) + torch.stack([recurrent_left_hidden_bias for i in range(batch_size)]))
			if DEBUG:
				print(hidden.shape, torch.min(hidden).item(), torch.max(hidden).item(), torch.mean(hidden).item())
			hidden_list.append(hidden)
			prev = hidden
		return hidden_list

	def right_rnn_forward(self, sequence: list, layer: int, batch_size):
		'''
		Implements one pass of the right rnn a the specified layer. Returns the hidden states of the rnn.
		'''
		reverse_hidden_list = []
		recurrent_right_in, recurrent_right_in_bias, recurrent_right_hidden, recurrent_right_hidden_bias = self.recurrent_right[layer]
		prev = torch.ones((batch_size, recurrent_right_hidden.shape[0]))
		n = len(sequence) - 1
		if DEBUG:
			print('right rnn')
		while n >= 0:
			input = sequence[n]
			if DEBUG:
				print(input.shape, torch.min(input).item(), torch.max(input).item(), torch.mean(input).item())
			hidden = self.sigmoid(torch.matmul(input, recurrent_right_in.T) + torch.stack([recurrent_right_in_bias for i in range(batch_size)]) + torch.matmul(prev, recurrent_right_hidden) + torch.stack([recurrent_right_hidden_bias for i in range(batch_size)]))
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
		correct_order = []
		n = self.nsyll - 1
		while n >= 0:
			correct_order.append(right_hidden_states[n])
			n-=1
		full_context = []
		for hidden1, hidden2 in zip(left_hidden_states, correct_order):
			full = torch.cat((hidden1, hidden2), 1)
			if DEBUG:
				print('combined left and right RNN')
				print(full.shape, torch.min(full).item(), torch.max(full).item(), torch.mean(full).item())
			full_context.append(full)
		return full_context

