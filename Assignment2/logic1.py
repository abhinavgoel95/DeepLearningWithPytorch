import torch as t
import neural_network as nn

class AND:
	def __init__(self):
		self.model = nn.NeuralNetwork([2,1])
		layer = self.model.getLayer(0)
		layer[0] = -1.0
		layer[1] = 1.0
		layer[2] = 1.0

	def __call__(self,x, y):
		retval = self.forward(int(x), int(y))
		if retval > 0:
			return True
		else:
			return False

	def forward(self,a,b):
		inp = t.tensor([[a],[b]])
		a = self.model.forward(inp)
		return a

class OR:
	def __init__(self):
		self.model = nn.NeuralNetwork([2,1])
		layer = self.model.getLayer(0)
		layer[0] = 0.0
		layer[1] = 1.0
		layer[2] = 1.0

	def __call__(self,x, y):
		if type(x) == bool:
		x = [x]
		y = [y]
		for i in range(len(x)):
			x[i] = int(x[i])
		for i in range(len(y)):
			y[i] = int(y[i])			
		retval = self.forward(x, y)
		if retval > 0:
			return True
		else:
			return False

	def forward(self,a,b):
		inp = t.tensor([[a],[b]])
		a = self.model.forward(inp)
		return a

class NOT:
	def __init__(self):
		self.model = nn.NeuralNetwork([1,1])
		layer = self.model.getLayer(0)
		layer[0] = 1.0
		layer[1] = -1.0

	def __call__(self,x):
		retval = self.forward(int(x))
		if retval > 0:
			return True
		else:
			return False

	def forward(self,a):
		inp = t.tensor([[a]])
		a = self.model.forward(inp)
		return a


class XOR:
	def __init__(self):
		self.model = nn.NeuralNetwork([2,2,1])
		layer1 = self.model.getLayer(0)
		layer1[0] = t.tensor([-10.0, 30.0])
		layer1[1] = t.tensor([20.0, -20.0])
		layer1[2] = t.tensor([20.0, -20.0])
		
		layer2 = self.model.getLayer(1)
		layer2[0] = -30.0
		layer2[1] = 20.0
		layer2[2] = 20.0

	def __call__(self,x, y):
		retval = self.forward(int(x), int(y))
		if retval > 0:
			return True
		else:
			return False

	def forward(self,a,b):
		inp = t.tensor([[a],[b]])
		a = self.model.forward(inp)
		return a