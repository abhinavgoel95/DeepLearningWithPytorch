import torch as t
class NeuralNetwork:
	def __init__(self, inps):
		self.l = []
		self.in1 = inps[0]
		self.out = inps[-1]
		self.theta = dict()
		self.d_theta = dict()
		self.a = dict()
		self.z = dict()
		self.inps = inps

		for i in range(0, len(inps)-1):
			self.theta[i] = t.tensor(t.normal(mean = t.zeros(inps[i]+1, inps[i+1]), 
											std = t.ones(inps[i]+1, inps[i+1])*1/inps[i+1]))
			self.d_theta[i] = t.tensor(t.zeros(inps[i]+1, inps[i+1]))
			self.a[i+1] = t.tensor(0)
			self.z[i+1] = t.tensor(0)

	def getLayer(self, layer):
		return self.theta[layer]

	def forward(self,inp):
		inp = inp.type(t.FloatTensor)
		inp = t.reshape(inp, [inp.size()[0], -1])
		y = t.tensor(t.ones(1, inp.size()[-1]))
		inp = t.cat((y,inp),0)
		sig = t.nn.Sigmoid()	
		w = self.getLayer(0)
		self.a[0] = inp
		i = 0
		while i in range(0,len(self.inps)-2):
			w = self.getLayer(i)
			self.z[i+1] = t.mm(t.t(w), self.a[i])
			y = t.tensor(t.ones(1, self.z[i+1].size()[-1]))
			self.z[i+1] = t.cat((y,self.z[i+1]),0)
			self.a[i+1] = sig(self.z[i+1])
			i+=1
		w = self.getLayer(i)
		self.z[i+1] = t.mm(t.t(w), self.a[i])
		return self.z[i+1]

	def backward(self, target):
		target = target.type(t.FloatTensor)
		target = t.reshape(target, [-1, target.size()[0]])
		sig = t.nn.Sigmoid()

		i = len(self.inps)-2
		dz = self.z[i+1] - target
		self.l.append((self.z[i+1] - target)*(self.z[i+1] - target))
		self.d_theta[i] = t.mm((self.a[i]),t.t(dz))/target.shape[-1]
		i-=1	
		while i >= 0:
			dz = t.mm(self.getLayer(i+1),dz)[1:] * (sig(self.z[i+1][1:])*(1-sig(self.z[i+1][1:])))
			self.d_theta[i] = t.mm((self.a[i]),t.t(dz))/(target.shape[-1])
			i-=1

	def updateParams(self, eta):
		for i in range(len(self.theta)):
			self.theta[i] = self.theta[i] - eta*self.d_theta[i]