import torch as t
import neural_network as nn

class AND:
	def __init__(self):
		self.model = nn.NeuralNetwork([2,1])

	def __call__(self,x, y):
		retval = self.forward(int(x), int(y))
		if retval >= 0.5:
			return True
		else:
			return False

	def forward(self,a,b):
		inp = t.tensor([[a],[b]])
		a = self.model.forward(inp)
		return a

	def train(self):
		for i in range(50000):
			para1 = bool(t.rand(1) >= 0.5)
			para2 = bool(t.rand(1) >= 0.5)
			out = para1 and para2
			self.forward(int(para1), int(para2))
			self.model.backward(t.tensor([out]))
			self.model.updateParams(0.001)

class OR:
	def __init__(self):
		self.model = nn.NeuralNetwork([2,1])

	def __call__(self,x, y):
		retval = self.forward(int(x), int(y))
		if retval >= 0.5:
			return True
		else:
			return False

	def forward(self,a,b):
		inp = t.tensor([[a],[b]])
		a = self.model.forward(inp)
		return a

	def train(self):
		for i in range(50000):
			para1 = bool(t.rand(1) >= 0.5)
			para2 = bool(t.rand(1) >= 0.5)
			out = para1 or para2
			self.forward(int(para1), int(para2))
			self.model.backward(t.tensor([out]))
			self.model.updateParams(0.001)

class NOT:
	def __init__(self):
		self.model = nn.NeuralNetwork([1,1])

	def __call__(self,x):
		retval = self.forward(int(x))
		if retval > 0.5:
			return True
		else:
			return False

	def forward(self,a):
		inp = t.tensor([[a]])
		a = self.model.forward(inp)
		return a

	def train(self):
		for i in range(50000):
			para1 = bool(t.rand(1) >= 0.5)
			out = not para1
			self.forward(int(para1))
			self.model.backward(t.tensor([out]))
			self.model.updateParams(0.001)


class XOR:
	def __init__(self):
		self.model = nn.NeuralNetwork([2,2,1])
		layer1 = self.model.getLayer(0)

	def __call__(self,x, y):
		retval = self.forward(int(x), int(y))
		if retval > 0.5:
			return True
		else:
			return False

	def forward(self,a,b):
		inp = t.tensor([[a],[b]])
		a = self.model.forward(inp)
		return a

	def train(self):
		for i in range(50000):
			para1 = bool(t.rand(1) >= 0.5)
			para2 = bool(t.rand(1) >= 0.5)
			out = ((para1 and (not para2)) or (para2 and (not para1)))
			self.forward(int(para1), int(para2))
			self.model.backward(t.tensor([out]))
			self.model.updateParams(0.1)

