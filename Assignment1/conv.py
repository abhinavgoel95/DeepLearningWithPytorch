import numpy as np
from scipy.signal import convolve2d

class Conv2D:
	def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
		self.in_channel = in_channel
		self.o_channel = o_channel
		self.kernel_size = kernel_size
		self.stride = stride
		self.mode = mode
		x = [[1,1, 1],[0,0,0],[-1,-1,-1]] 
		self.kernel = np.asarray(x).reshape([in_channel,kernel_size, kernel_size, o_channel])
"""
	def forward(self, input_image):
		c, h, w = input_image.shape
		out_conv = []
		conv = np.zeros(shape = [self.o_channel, int(h/self.stride), int(w/self.stride)])
		img = np.zeros((c, h + 2*int(self.kernel_size/2), w + 2*int(self.kernel_size/2)))
		img[:, int(self.kernel_size/2): -int(self.kernel_size/2), int(self.kernel_size/2): -int(self.kernel_size/2)] = input_image
		for y in range(self.o_channel):
			ims = []
			i = 0
			while i in range(h):
				j = 0
				while j in range(w):
					conv[y][i][j] = (self.kernel[:,:,:,y] * img[:, i : i + self.kernel_size, j : j + self.kernel_size]).sum()
					j += self.stride
					print(j, w)
				i += self.stride
			ims.append(conv)
		out_conv = np.stack(ims, axis=2).astype("float32").reshape(self.o_channel,h,w)

		return out_conv
"""


"""
		for y in range(self.o_channel):
			ims = []
			for x in range(self.kernel_size):
				im_conv_d = convolve2d(input_image[x,:,:], self.kernel[x,:,:,y], mode="same", boundary="symm")
				ims.append(im_conv_d)
				im_conv = np.stack(ims, axis=2).astype("float32")
			im_conv = np.sum(im_conv, axis = 2).reshape(1,h,w)
			out_conv.append(im_conv)
		out_conv = np.stack(out_conv, axis=2).astype("float32").reshape(self.o_channel,h,w)
		return out_conv
"""