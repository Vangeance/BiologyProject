import numpy as np
from scipy import signal
from PIL import Image
from pylab import *

kernel_size = 10
color_threshold = 40

img = Image.open("1441.tif")
img_array = np.array(img.getdata()).reshape((1200,1600))

X, Y = list(np.where(img_array<40)[1]), list(np.where(img_array<40)[0])
imshow(img_array)
# plot(X, Y, 'r*')
title('highlight dark pixels')
show()


kernel = np.ones((kernel_size,kernel_size))
conv = signal.convolve2d(img, kernel, mode='valid')/kernel_size**2

X.clear()
Y.clear()
loc = []
for i in range(len(conv)):
    for j in range(len(conv[0])):
        if conv[i][j] < color_threshold:
            X.append(j)
            Y.append(i)
            loc.append((i,j))

print(loc)

imshow(img_array)
plot(X, Y, 'r*')
title('result')
show()


