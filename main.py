"""
@Author : Halil İbrahim Bestil
"""
import cv2
import math
from feature_matcher import FeatureMatcher
from random import randrange
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


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


def generate_random_sub_image(base_im, crop_h=200, crop_w=200, padding=200):
    base_im_h, base_im_w = base_im.shape[:2]

    rand_h = randrange(base_im_h - crop_h - padding)
    rand_w = randrange(base_im_w - crop_w - padding)
    rand_theta = randrange(360)
    test_image_start_x = rand_w + padding
    test_image_start_y = rand_h + padding

    random_base_coords = (test_image_start_x + crop_w / 2, test_image_start_y + crop_h / 2)
    randomly_rotated_im = subimage(base_im, random_base_coords, rand_theta, crop_w, crop_h)

    return randomly_rotated_im, rand_theta, random_base_coords


# Read the base image
image_path = "images/StarMap.png"
base_im = cv2.imread(image_path)

# Generate random rotated sub image for testing purposes.
crop_h = 200
crop_w = 200
padding = 200
randomly_rotated_im, rand_theta, random_base_coords = generate_random_sub_image(base_im, crop_h, crop_w, padding)

# Create figure
fig = figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()

# Create a FeatureMatcher Object
MIN_MATCH_COUNT = 10
matcher = FeatureMatcher(MIN_MATCH_COUNT)

# Predict coordinates of given sub image in base image
result_arr = matcher.match_features(base_im, randomly_rotated_im)
[sub, kp1, scene, kp2, good, matchesMask, estimated_poly_lines] = result_arr

# Check if match counts enough
if estimated_poly_lines is None:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
else:
    # Plot ground truth of sub-images rectangle
    red_color = (255, 0, 0,)
    scene_gt = draw_angled_rec(random_base_coords[0], random_base_coords[1], crop_w, crop_h, rand_theta, base_im, red_color)

    # Plot predicted sub-images coordinates
    blue_color = (0, 0, 255)
    scene_gt_pred = cv2.polylines(scene_gt, estimated_poly_lines, True, blue_color, 3, cv2.LINE_AA)  # Prediction

    # Plot feature matcher related stuff
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(sub, kp1, scene_gt_pred, kp2, good, None, **draw_params)

    # Show result
    plt.imshow(img3, 'gray'), plt.show()
