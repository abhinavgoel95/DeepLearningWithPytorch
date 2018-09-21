import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

class img2num(nn.Module):
	def __init__(self):
		super(Img2Num, self).__init__()
		self.conv1 = t.nn.Conv2d(1,10, kernel_size = 5)
		self.conv2 = t.nn.Conv2d(10,20, kernel_size = 5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = t.nn.Linear(320, 50)
		self.fc2 = t.nn.Linear(50, 10)
		self.sig = t.nn.Sigmoid()
		self.acc = []

	def train(self):
		optimizer = optim.SGD(self.parameters(), lr=0.0001, momentum=0.9)
		loss_func = nn.NLLLoss()

		loss_log = []

		train_loader = t.utils.data.DataLoader(
						datasets.MNIST(
								'data', train = True, 
								download = True, 
								transform=transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.1307,), (0.3081,))])
							), 
						batch_size=1
						)

		for i in range(10):
			count = 0
			for data, target in train_loader:
				data = data.type(t.FloatTensor)
				
				optimizer.zero_grad()
				net_out = self(data)

				values, indices = t.max(net_out.data, 1)
				count += (indices == target.data).sum()

				loss = loss_func(net_out, target)
				loss.backward()
				optimizer.step()
			self.acc.append(count/600)
			print(count/600)

	def forward(self, img):
		img = img.type(t.FloatTensor)
		x = F.relu(F.max_pool2d(self.conv1(img),2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		#x = self.sig(x)
		x = F.log_softmax(x, dim = 1)
		return x