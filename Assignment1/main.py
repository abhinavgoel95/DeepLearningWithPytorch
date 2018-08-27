import conv as c
import cv2
import numpy as np

#conv2d = c.Conv2D(in_channel, o_channel, kernel_size, stride, mode)
conv2d = c.Conv2D(1, 1, 3, 3, 1)
img = cv2.imread("test1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(1,img.shape[0],img.shape[1])
n, x = conv2d.forward(img)
cv2.imshow("",x[0])
cv2.waitKey()
print(x.shape)
