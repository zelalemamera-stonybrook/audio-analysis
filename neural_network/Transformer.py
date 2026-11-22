'''
Implements the transformer architecture from Vaswani et al.

A transformer neural network is made up of multiple layers with self attention heads in each layer, and connections within  the layers.
Positional encodings are used for the input sequence, and the output sequence is decoded from the output of the encoder. Since our problem does not require a decoder,
a decoder is not implemented here, and instead the output of the encoder layer is passed directly to a feedforward layer.
'''
import torch
import torch.nn as nn
from definitions import N, N_LINEAR, N_HEADS, OUTPUT_DIM, SEQ_LEN, F_LEN, BATCH_SIZE, MIN, MAX
PDROP = 0.1

device = torch.device('cpu')
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)
g_cpu = torch.Generator(device=device)
seeds = [3529145006359120161, 9249168034906996919, 13979071961427503144, 13472831205784841605, 8820161422455662510]
seed = seeds[0]
debug = False

class Network(nn.Module):

	def __init__(self):
		super().__init__()
		self.inputdim = 391
		self.embedder = Embedder(2, self.inputdim)
		self.encoder = Encoder(N)
		self.output = Output()
		self.name = "Transformer"
		self.attention_weights = []

	def __call__(self, input, mask):
		return self.forward(input, mask)

	def forward(self, input, mask):
		'''
		First input is embedded using the embedder network, then it is encoded, then the output is decoded.
		'''

		embedded = []
		if debug:
			print("tranformer network begins")
			print("batch size", BATCH_SIZE)
		for i in range(SEQ_LEN):
			slice = input[:,i]
			embedded.append(self.embedder(slice.clone(), i, mask))

		embedded = torch.cat(embedded, 1)
		# the input is now assumed to be a fixed length sequence. The sequence length that we assume as our input here is 6. Data points that are fewer than this are masked so that the
		# computation is simplified.

		encoded = self.encoder(embedded)

		result = self.output(encoded)
		if debug:
			dprint(result, "transformer network complete")
		return result

class Embedder(nn.Module):
	'''
	this class implements the network that allows us to learn a single vector representation of a syllable from its multiple feature representations. Several self attention heads are used for
	this and the final output is obtained by using a special query key that is obtained from the properties of the vector we want to produce.
	This last step is a multiattention head whose learned distribution is used to interpret possible relevant feature representations.
	'''
	def __init__(self, N, inputdim):
		super().__init__()
		self.input_layer = Linear(OUTPUT_DIM,inputdim, 0, 1)
		self.layers = nn.ModuleList([Layer(N_HEADS, F_LEN, N_LINEAR, 0 , 1) for i in range(2)])
		self.output_head = Multihead(N_HEADS, F_LEN, 0, 1)
		self.norm = nn.LeakyReLU()
		self.dropout = Dropout(PDROP)

	def __call__(self, input, i, mask):

		return self.forward(input, i, mask)

	def forward(self, input, i, mask):

		if debug:
			dprint(input, "embedding layer begins")
		input = self.input_layer(input)

		for i, layer in enumerate(self.layers):
			input = layer(input, dropout=False)

		query = self.get_query(i, mask)
		output = self.norm(self.output_head(input, query))
		if debug:
			dprint(output, "embedding layer complete")
		return output

	def get_query(self, i, mask):
		query = torch.stack([torch.cat((self.position_encoder(i + 1), self.position_encoder(mask[k]))) for k in range(BATCH_SIZE)])
		return query.view(BATCH_SIZE, 1, query.shape[-1])

	def position_encoder(self, i: int):
		'''
		implements the same position encoding function from vaswani et. al
		'''
		position_vector = torch.empty((OUTPUT_DIM // 2,))
		for dim in range(len(position_vector)):
			if dim % 2 == 0:
				position_vector[dim] = torch.sin( torch.tensor(i / (10000 ** (dim / OUTPUT_DIM))))
			else:
				position_vector[dim] = torch.cos(torch.tensor(i / (10000 ** (dim / OUTPUT_DIM))))
		return position_vector


class Encoder(nn.Module):

	def __init__(self, N):
		super().__init__()
		self.layers = nn.ModuleList([Layer(N_HEADS, SEQ_LEN, N_LINEAR, 0, 1) for i in range(N)])
		self.dropout = Dropout(PDROP)

	def __call__(self, input):
		return self.forward(self.dropout(input, self.training))

	def forward(self, input):

		if debug:
			dprint(input, "encoder network begins")
		current = input
		for i, layer in enumerate(self.layers):
			current = layer(current)
		if debug:
			dprint(current, "encoder network complete")
		return current


class Output(nn.Module):

	def __init__(self):
		super().__init__()
		self.layers= nn.ModuleList([Linear(OUTPUT_DIM // 2, OUTPUT_DIM, 0, 1),
						Linear(OUTPUT_DIM // (2 ** 2), OUTPUT_DIM // 2, 0, 1)])
		self.output_layer = Linear(OUTPUT_DIM // (2 ** 8), OUTPUT_DIM // (2 ** 2), 0, 1)
		self.softmax = nn.Softmax(dim = -1)
		self.norm = nn.LeakyReLU()

	def __call__(self, input):
		return self.forward(input)

	def forward(self, input):
		if debug:
			dprint(input, "output network begins")
		for layer in self.layers:
			input = self.norm(layer(input))

		result = self.output_layer(input)
		result = self.softmax(result)
		if debug:
			dprint(result, "output network complete")
		return result


class Layer(nn.Module):
	'''
	implements a single transformer layer from vaswani et al.
	'''

	def __init__(self, nheads, seq_len, nlinear, min=MIN, max=MAX):
		super().__init__()
		self.multihead = Multihead(nheads, seq_len, min, max)
		self.linear = SquareLinear(min , max)
		self.norm = nn.LeakyReLU()
		self.dropout = Dropout(PDROP)


	def __call__(self, input, w=1, dropout=True):
		return self.forward(input, w, dropout)

	def layernorm(self, input):
		mean = torch.mean(input, 1)
		std = torch.std(input, 1)
		normal = []
		for i, j, k in zip(input, mean, std):
			normal.append( (i - j) / (k + (10 ** - 6)))

		return torch.stack(normal)

	def forward(self, input, w, dropout):
		if debug:
			dprint(input, "transformer layer begins")

		multiheadout = self.layernorm(self.dropout(self.multihead(input, input), self.training and dropout) + input * w)

		linearout = self.layernorm(self.dropout(self.linear(multiheadout), self.training and dropout) +  multiheadout * w)

		if debug:
			dprint(linearout, "transformer layer complete")

		return linearout


class  Multihead(nn.Module):

	def __init__(self, nheads, seq_len, min = MIN, max = MAX):
		super().__init__()
		self.query_transformations = parameter_list(nheads, OUTPUT_DIM // nheads, OUTPUT_DIM, min, max)
		self.key = parameter(seq_len, OUTPUT_DIM, min, max)
		self.key_transformations = parameter_list(nheads, OUTPUT_DIM, OUTPUT_DIM // nheads, min, max)
		self.value_transformations = parameter_list(nheads, OUTPUT_DIM // nheads, OUTPUT_DIM, min, max)
		self.output = parameter(OUTPUT_DIM, OUTPUT_DIM, min, max)
		self.softmax = nn.Softmax(dim = 1)

	def __call__(self, input, query):
		return self.forward(input, query)


	def forward(self, input, query):


		if debug:
			dprint(input, "multihead sublayer begins")
		out = []
		seq_len = input.shape[1] # for now, all input is assumed to be batched, so the next dimension after batch size is the sequence dimension
		if len(query.shape) < 3:
			raise ValueError("query must be at least three dimensional, if the query is a single vector, then reshape it to add the relevant dimensions.")
		query_len = query.shape[1]

		query = query.view(BATCH_SIZE * query_len, query.shape[-1])
		input = input.view(BATCH_SIZE * seq_len, input.shape[-1])

		for Q, K, V in zip(self.query_transformations, self.key_transformations, self.value_transformations):

			queries = torch.matmul(query, Q.T)

			if debug:
				dprint(queries, "query projected")
			key = torch.matmul(self.key, K)
			if debug:
				dprint(key, "key projected")
			values = torch.matmul(input, V.T)
			if debug:
				dprint(values, "values projected")

			distributions = self.softmax((OUTPUT_DIM ** -0.5) * torch.matmul(queries, key.T))

			if debug:
				dprint(distributions, "weight distribution generated")

			distributions = distributions.view(BATCH_SIZE, query_len, seq_len)
			values = values.view(BATCH_SIZE, seq_len, values.shape[-1])

			output = torch.stack([torch.matmul(distributions[i], values[i]) for i in range(BATCH_SIZE)])

			if debug:
				dprint(output, "output of head")
			out.append(output)

		result = torch.cat(out, -1)
		result = result.view(BATCH_SIZE * query_len, result.shape[-1])

		result = torch.matmul(result, self.output)
		if debug:
			dprint(result, "multi head sublayer complete")

		return result.view(BATCH_SIZE, query_len, result.shape[-1])



class SquareLinear(nn.Module):


	def __init__(self, min=MIN, max=MAX):
		super().__init__()
		self.layer1 = Linear(OUTPUT_DIM * 4 , OUTPUT_DIM, min, max)
		self.layer2 = Linear(OUTPUT_DIM, OUTPUT_DIM * 4, min, max)
		self.norm = nn.LeakyReLU()

	def __call__(self, input):
		return self.forward(input)

	def forward(self, input):

		input = self.norm(self.layer1(input))
		input = self.layer2(input)

		return input

class Linear(nn.Module):

	def __init__(self, dout, din, min=MIN, max=MAX):
		super().__init__()
		self.layer = parameter(dout, din, min, max)
		self.bias = parameter(1, dout, min, max)

	def __call__(self, input):
		return self.forward(input)

	def forward(self, input):
		if debug:
			dprint(input, "linear layer begins")
		seq_len = input.shape[1]
		input = input.view(BATCH_SIZE * seq_len, input.shape[-1])
		bias = torch.cat([self.bias for i in range(BATCH_SIZE * seq_len)], 0)
		input = torch.matmul(input, self.layer.T) + bias

		result = input.view(BATCH_SIZE, seq_len, input.shape[-1])

		if debug:
			dprint(result, "linear layer complete")
		return result

class Dropout():
	'''
	Implements a drop out regularizer. A probability is specified which is used to drop each unit output with that probability
	'''
	def __init__(self, pdrop):
		self.pdrop = pdrop

	def __call__(self, input, train=True):
		return self.forward(input, train)

	def forward(self, input, train):
		seq_len = input.shape[1]
		dim = input.shape[-1]
		if train:
			distribution = Dropout.get_distribution(self.pdrop, dim)
			input = input.view(BATCH_SIZE * seq_len, dim)

			result = input * distribution
			return result.view(BATCH_SIZE, seq_len, dim)
		else:
			scale = Dropout.get_scale(self.pdrop, dim)
			input = input.view(BATCH_SIZE * seq_len, dim)

			result = input * scale
			return result.view(BATCH_SIZE, seq_len, dim)

	def get_distribution(pdrop, dim):
		'''
		For now, the drop out is assumed to be the same probability for every unit.
		'''
		bernoulli = torch.full((dim,), 1 - pdrop)
		return torch.bernoulli(bernoulli)

	def get_scale(pdrop, dim):
		'''
		Under the assumption that each unit has an equal probability of being dropped, then each output is also averaged with the same weight
		'''
		return torch.full((dim,), 1 - pdrop)

def dprint(t, message=''):
	print(message)
	print(t.shape, torch.min(t).item(), torch.max(t).item(), torch.mean(t).item())
	print(t[:3])
	print("\n")

def parameter_list(n, output_shape, input_shape, min, max):
	'''
	produces and returns a parameter list from the uniform distribution drawn between min and max, the length of the list is specified by n. Each parameter in the list has an input and output
	dimension specified by input_shape and output_shape respectively.
	'''
	return nn.ParameterList([parameter(output_shape, input_shape, min, max) for i in range(n)])

def parameter(output_shape, input_shape, min, max):
	'''
	returns a parameter drawn from the uniform distribution between min and max of the specified shape.
	'''
	return nn.parameter.Parameter(nn.init.kaiming_uniform_(torch.empty((output_shape, input_shape)), a=0.01, generator = g_cpu.manual_seed(seed)))

