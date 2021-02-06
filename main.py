"""
@Author : Halil İbrahim Bestil
"""
import cv2
import math
from feature_matcher import FeatureMatcher
from random import randrange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import figure
from matplotlib.patches import Rectangle


def subimage(image, center, theta, width, height):
    '''
    Rotates OpenCV image around center with angle theta (in deg)
    then crops the image according to width and height.
    '''

    # Uncomment for theta in radians
    # theta *= 180/np.pi

    shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)

    matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

    x = int(center[0] - width / 2)
    y = int(center[1] - height / 2)

    image = image[y:y + height, x:x + width]

    return image


def draw_angled_rec(x0, y0, width, height, angle, img, color):
    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    img = cv2.line(img, pt0, pt1, color, 3)
    img = cv2.line(img, pt1, pt2, color, 3)
    img = cv2.line(img, pt2, pt3, color, 3)
    img = cv2.line(img, pt3, pt0, color, 3)
    return img



crop_h = 200
crop_w = 200
padding = 200

image_path = "images/StarMap.png"
base_im = cv2.imread(image_path)
base_im_h, base_im_w = base_im.shape[:2]

rand_h = randrange(base_im_h - crop_h - padding)
rand_w = randrange(base_im_w - crop_w - padding)
rand_theta = randrange(360)
test_image_start_x = rand_w + padding
test_image_start_y = rand_h + padding

random_base_coords = (test_image_start_x + crop_w / 2, test_image_start_y + crop_h / 2)
randomly_rotated_im = subimage(base_im, random_base_coords, rand_theta, crop_w, crop_h)

