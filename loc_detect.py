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
    # imshow(img_arr, cmap='gray')
    # X, Y = list(np.where(img_arr<color_threshold)[1]), list(np.where(img_arr<color_threshold)[0])
    # plt.plot(X, Y, 'r*')
    # plt.title('highlight dark pixels')
    # plt.show()

    # detect locations of fruiting bodies
    kernel = np.ones((kernel_size, kernel_size))
    conv = signal.convolve2d(img_arr, kernel, mode='valid') / kernel_size ** 2

    X = []
    Y = []
    for i in range(len(conv)):
        for j in range(len(conv[0])):
            if 0 < conv[i][j] < color_threshold:
                conv[i][j] -= 256  # mark the coordinate negative
                # the current is eligible AND there're NO neighbors chosen yet
                if i < int(filter_width/2) and j < int(filter_width/2) and conv[0:i+filter_width,0:j+filter_width] or j > 0 and conv[i][j - 1] < 0:
                    X.append(j)
                    Y.append(i)

    # filter_size=2
    # i=1

    plt.imshow(img_arr, cmap='gray')
    plt.plot(X, Y, 'r*')
    rect1 = plt.Rectangle((55, 45), rect_width, rect_width, fill=False, edgecolor='black', linewidth=1)
    rect2 = plt.Rectangle((555, 100), rect_width, rect_width, fill=False, edgecolor='black', linewidth=1)
    plt.gca().add_patch(rect1)
    plt.gca().add_patch(rect2)
    plt.title(
        "Result(kernel_size=" + str(kernel_size) + " color_threshold=" + str(color_threshold) + " filter_width=" + str(
            filter_width))
    plt.show()


img = Image.open("1441.tif")
img_arr = np.array(img.getdata()).reshape((1200, 1600))
loc_detect(img_arr)
