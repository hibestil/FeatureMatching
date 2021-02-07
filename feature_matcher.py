import cv2
import numpy as np


class FeatureMatcher():

    def __init__(self, min_match_count=10):
        self.MIN_MATCH_COUNT = min_match_count
        # FLANN parameters
        self.FLANN_INDEX_KDTREE = 1
        # Initiate SIFT detector
        self.sift = cv2.SIFT_create()
        self.number_of_trees = 5
        self.number_of_checks = 50
        self.goodness_ratio = 0.7

    def match_features(self, scene_rgb_img, sub_rgb_img, ):
        # Convert RGB images to Grayscale
        scene = cv2.cvtColor(scene_rgb_img, cv2.COLOR_BGR2GRAY)  # im2
        sub = cv2.cvtColor(sub_rgb_img, cv2.COLOR_BGR2GRAY)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.sift.detectAndCompute(sub, None)
        kp2, des2 = self.sift.detectAndCompute(scene, None)

        index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=self.number_of_trees)
        search_params = dict(checks=self.number_of_checks)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < self.goodness_ratio * n.distance:
                good.append(m)

        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            h, w = sub.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            estimated_poly_lines = [np.int32(dst)]

        else:
            matches_mask = None
            estimated_poly_lines = None

        return [sub, kp1, scene, kp2, good, matches_mask, estimated_poly_lines]
