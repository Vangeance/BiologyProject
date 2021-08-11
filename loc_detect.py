import matplotlib.pyplot
import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt


def loc_detect(img_arr, kernel_size=10, color_threshold=20, filter_width=100, rect_width=100):
    """
    detect locations of fruiting bodies and draw
    :param img_arr: path of image
    :param kernel_size: size of convolving kernel
    :param color_threshold: threshold for convolving result
    :param filter_width: distance used to filter out the other coordinates within SAME or CLOSE fruiting body
    :param rect_width: width of final image of each detected fruiting body
    :return:
    """
    # plt.imshow(img_arr, cmap='gray')
    # X, Y = list(np.where(img_arr<color_threshold)[1]), list(np.where(img_arr<color_threshold)[0])
    # plt.plot(X, Y, 'r*')
    # plt.title('highlight dark pixels')
    # plt.show()

    # detect locations of fruiting bodies
    kernel = np.ones((kernel_size, kernel_size))
    # binary_img_arr = np.where(img_arr < 40, 0, 255)  # To Binary
    conv = signal.convolve2d(img_arr, kernel, mode='valid') / kernel_size ** 2

    X = []
    Y = []

    # ignore fruiting bodies near the borders
    for i in range(int(rect_width/2), len(conv)-int(rect_width*2/3)):
        for j in range(int(rect_width/2), len(conv[0])-rect_width):
            # the current is eligible AND there are NO neighbors chosen yet
            if 0 < conv[i][j] < color_threshold:
                height, width = img_arr.shape[0], img_arr.shape[1]
                if (conv[max(0,i-int(filter_width/2)):min(height,i+filter_width), max(0,j-int(filter_width/2)):min(width,j+filter_width)]>0).all():
                    X.append(j)
                    Y.append(i)
                conv[i][j] -= 256  # mark the coordinate negative

    plt.imshow(img_arr, cmap='gray')
    plt.plot(X, Y, 'r*')
    rect1 = plt.Rectangle((55, 45), rect_width, rect_width, fill=False, edgecolor='black', linewidth=1)
    rect2 = plt.Rectangle((555, 100), rect_width, rect_width, fill=False, edgecolor='black', linewidth=1)
    plt.gca().add_patch(rect1)
    plt.gca().add_patch(rect2)
    plt.title("Result(kernel_size=" + str(kernel_size) + " color_threshold=" + str(color_threshold) + " filter_width=" + str(
            filter_width))
    plt.show()


img = Image.open("1441.tif")
img_arr = np.array(img.getdata()).reshape((1200, 1600))
loc_detect(img_arr)
