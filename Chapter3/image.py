from sklearn import preprocessing
import numpy as np
X = np.array([
    [0., 0., 5., 13., 9., 1.],
    [0., 0., 13., 15., 10., 15.],
    [0., 3., 15., 2., 0., 11.]
])
print(preprocessing.scale(X))








# import mahotas as mh
# from mahotas.features import surf
#
# image = mh.imread('zipper.jpg', as_grey=True)
# print('The first SURF descriptor:\n', surf.surf(image)[0])
# print('Extracted %s descriptors' % len(surf.surf(image)))

# import numpy as np
# from skimage.feature import corner_harris, corner_peaks
# from skimage.color import rgb2gray
# import matplotlib.pyplot as plt
# import skimage.io as io
# from skimage.exposure import equalize_hist
#
# def show_corners(corners, image):
#     fig = plt.figure()
#     plt.gray()
#     plt.imshow(image)
#     y_corner, x_corner = zip(*corners)
#     plt.plot(x_corner, y_corner, 'or')
#     plt.xlim(0, image.shape[1])
#     plt.ylim(image.shape[0], 0)
#     fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
#     plt.show()
#
# mandrill = io.imread('./mandrill.png')
# mandrill = equalize_hist(rgb2gray(mandrill))
# corners = corner_peaks(corner_harris(mandrill), min_distance=2)
# show_corners(corners, mandrill)

# from sklearn import datasets
# digits = datasets.load_digits()
# print('Digit:', digits.target[0])
# print(digits.images[0])
# print('Feature vector:\n', digits.images[0].reshape(-1, 64))
