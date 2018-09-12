import neural_network as nn
import torch as t

a = nn.NeuralNetwork((2,3,3,1))
a.forward(t.tensor([[1,1,1,1,1],[0,0,1,1,1]]))
a.backward(t.tensor([0,1,1,1,1]))
a.updateParams(0.5)