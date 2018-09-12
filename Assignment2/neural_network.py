import torch as t
class NeuralNetwork:
	def __init__(self, inps):
		self.in1 = inps[0]
		self.out = inps[-1]
		self.theta = dict()
		self.inps = inps

		for i in range(0, len(inps)-1):
			self.theta[i] = t.tensor(t.normal(mean = t.zeros(inps[i]+1, inps[i+1]), 
											std = t.ones(inps[i]+1, inps[i+1])*1/inps[i+1]))

	def getLayer(self, layer):
		return self.theta[layer]

	def forward(self,inp):
		inp = inp.type(t.FloatTensor)
		inp = t.reshape(inp, [inp.size()[0], -1])
		y = t.tensor(t.ones(1, inp.size()[-1]))
		inp = t.cat((y,inp),0)
		sig = t.nn.Sigmoid()
		
		w = self.getLayer(0)
		a = inp
		i = 0
		while i in range(0,len(self.inps)-2):
			w = self.getLayer(i)
			a = t.mm(t.t(w), a)
			y = t.tensor(t.ones(1, a.size()[-1]))
			a = t.cat((y,a),0)
			a = sig(a)
			i+=1
		w = self.getLayer(i)
		a = t.mm(t.t(w), a)
		return a