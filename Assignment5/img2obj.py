import cv2
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

class img2obj(nn.Module):
	def __init__(self):
		super(img2obj, self).__init__()
		self.conv1 = t.nn.Conv2d(3,10, kernel_size = 5)
		self.conv2 = t.nn.Conv2d(10,20, kernel_size = 5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = t.nn.Linear(500, 200)
		self.fc2 = t.nn.Linear(200, 100)
		self.acc = []

	def train(self):
		optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
		loss_func = nn.NLLLoss()
		loss_log = []
		train_loader = t.utils.data.DataLoader(
						datasets.CIFAR100(
								'dataCIFAR', train = True, 
								download = True, 
								transform=transforms.Compose([transforms.ToTensor()])
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
			self.acc.append(count/500)
			print(count/500)


	def forward(self, img):
		img = img.type(t.FloatTensor)
		x = F.relu(F.max_pool2d(self.conv1(img),2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
		x = x.view(-1, 500)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		x = F.log_softmax(x, dim = 1)
		return x

	def view(self, img):
		img = img.type(t.FloatTensor)
		net_out =  self(img)
		values, indices = t.max(net_out.data, 1)
		print("class", indices)
		cv2.imshow('image', img)
		if cv2.waitKey() == 27:
			cv2.destroyAllWindows()

	def cam(self, id = 0):
		cam1 = cv2.VideoCapture(0)
		while True:
			ret, img = cam1.read()
			cv2.imshow('camera', img)
			if cv2.waitKey() == 27:
				break
		cv2.destroyAllWindows()