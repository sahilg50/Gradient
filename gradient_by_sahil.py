
from PIL import Image
import numpy as np
import cv2
from math import sqrt


x = cv2.imread("nf.tif")
img = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
grad_out = img.copy()

print(img.shape)
print(img)


img_pad = np.pad(img, ((1, 1), (1, 1)), mode='constant')
print(img_pad.shape)
print(img_pad)
M = img_pad.shape[0]
N = img_pad.shape[1]


cv2.imwrite('padded_image.tif', img_pad)

# Horizontal Edge detection
wx = np.array([[-1, -2, -1], 
              [0, 0, 0],
              [1, 2, 1]])

# Vertical Edge detection
wy = np.array([[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]])


for i in np.arange(0, M-2):
    for j in np.arange(N-2):
        sum1 = 0.0; sum2 = 0.0;
        for s in np.arange(0, 3):
            for t in np.arange(0, 3):

                p1 = img_pad.item(i+s, j+t)
                p2 = wx[s, t]
                p3 = wy[s, t]

                sum1 = sum1 + p1*p2

                sum2 = sum2 + p1*p3

        gx = sum1
        gy = sum2
        
# The penultimate step is to create a new image of the same dimensions as the original image and store for the pixel data, the magnitude of the two gradient values:

        b = sqrt((gx*gx)+(gy*gy))
 
        grad_out.itemset((i, j), b)


cv2.imwrite('gradient.tif', grad_out)
cv2.imshow('Gradient', grad_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(grad_out == img)