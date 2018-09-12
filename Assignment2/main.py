import torch as t
import neural_network as nn

x = nn.NeuralNetwork([2,4,3,2,1])
inp = t.tensor([[1],[8]])
a = x.forward(inp)