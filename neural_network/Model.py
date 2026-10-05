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
		self.name = 'Model'

		self.input_dim = 354
		self.seq_len = 8
		self.hidden_dim = 512 // 2
		self.output_dim = 512
		self.nheads = 3
		self.nlayers = 3

		g_cpu = torch.Generator()

		seed = 3529145006359120161
		#seed = 9249168034906996919
		#seed = 13979071961427503144
		#seed = 13472831205784841605
		#seed = 8820161422455662510

		self.keys = nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.seq_len, self.input_dim)), -0.05, 0.05, g_cpu.manual_seed(seed)))
		self.heads = nn.ParameterList(
				[nn.ParameterList(
					[nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.input_dim // self.nheads, self.input_dim)), -0.05, 0.05, g_cpu.manual_seed(seed))),
					nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.input_dim // self.nheads, self.input_dim)), -0.05, 0.05, g_cpu.manual_seed(seed))),
					nn.parameter.Parameter(nn.init.uniform_(torch.empty((self.input_dim // self.nheads, self.input_dim)), -0.05, 0.05, g_cpu.manual_seed(seed)))]
					) for i in range(self.nheads)
				]
				)
		self.multihead_out = nn.parameter.Parameter(nn.init.uniform_(torch.empty(((self.input_dim // self.nheads) * self.nheads), self.output_dim), -0.05, 0.05, g_cpu.manual_seed(seed)))

		self.attention_weights = []

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

		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, word: list):
		'''
		Passes the word once through the network. The architecture of the network is as follows: the input is assumed to be a sequence of syllables where each syllable is
		a list of feature embeddings. Feature embeddings can come from a variety of sources but they are all generated over the raw audio signal associated with the syllable.
		First all of these features must be summarized into one vector. For this a simple multi-headed attention mechanism is used with queries coming from a vector with two pieces of information:
		1. the position of the current syllable to be summarized, and 2. the total number of syllables that are present in this word. The values to be summarized are the feature vectors in the list, and the keys will be a matrix
		learned by the network. This step is applied to each list to produce a sequence of syllable vectors.

		The next step is to apply bidirectional RNN layers to the syllable sequence. Followed by an output feedforward network which takes each hidden state and generates a binary probability
		distribution, to be interpreted as the syllable's likelihood of being stressed. This takes into account the full context surrounding the syllable as the left and right rnn hidden states are combined
		The output probabilities are used to make a prediction about stress position in the word.

		input -> attention -> RNN -> Feedforward -> output
		'''
		sound_vec_embedding = []
		if DEBUG:
			print('embedding syllables')
		attention_list = []
		query_vector = self.obtain_query(word)
		for i, syll in enumerate(word):
			if DEBUG:
				print('syll received', syll.shape)
			pos = i + 1
			total = len(word)
			feature_encoding = torch.cat([self.position_encoder(pos), self.position_encoder(total)])
			attention_vector, summary = self.multi_attention_forward(feature_encoding + query_vector, syll)
			sound_vec_embedding.append(summary)
			print(attention_vector)
			attention_list.append(attention_vector.tolist())
		self.attention_weights.append(attention_list)
		output = self.rnn_forward(sound_vec_embedding)
		print('finished forward pass')
		return output

	def obtain_query(self, word: list):
		'''
		The architecture of this model creates an awkward situation where we need to build an attention distribution over the sequence but do not
		have a query vector coming from previous decoder layers, nor do we want to use self attention because the output sequence cannot be the same length
		as the input. For this, the proposed solution is to use a special query vector which is built out of position and total syllable information. In addition to this,
		an average of all of the feature vectors is provided. The reasoning for this is that since the multihead attention is shared across the syllables, the query vector should include
		information from each syllable.
		'''
		query_vector = torch.zeros((self.input_dim,))
		n = 0
		for syll in word:
			for vec in syll:
				query_vector += vec
				n+=1
		return query_vector / n

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

	def multi_attention_forward(self, query: Tensor, values: Tensor):
		'''
		Uses all of the heads in the model to implement multiheaded attention from Vaswani et al.
		'''
		attention_vector = None
		head_outputs = []
		i = 0
		for Q, K, V in self.heads:
			vec, output = self.attention_forward(torch.matmul(Q ,query), torch.matmul(self.keys, K.T), torch.matmul(values, V.T))
			head_outputs.append(output)
			if i == 0:
				attention_vector = vec
			i+=1
		final = torch.matmul(torch.cat(head_outputs), self.multihead_out)
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
