#!/usr/bin/python

import os
import sys
import cv2
import math
import numpy as np

from numpy import linalg


class Stitch(object):

    def __init__(self):
        pass


    def filter_matches(self, matches, ratio=0.75):
        filtered_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                filtered_matches.append(m[0])

        return filtered_matches

    def imageDistance(self, matches):

        sumDistance = 0.0

        for match in matches:
            sumDistance += match.distance

        return sumDistance

    def findDimensions(self, image, homography):
        base_p1 = np.ones(3, np.float32)
        base_p2 = np.ones(3, np.float32)
        base_p3 = np.ones(3, np.float32)
        base_p4 = np.ones(3, np.float32)

        (y, x) = image.shape[:2]

        base_p1[:2] = [0, 0]
        base_p2[:2] = [x, 0]
        base_p3[:2] = [0, y]
        base_p4[:2] = [x, y]

        max_x = None
        max_y = None
        min_x = None
        min_y = None

        for pt in [base_p1, base_p2, base_p3, base_p4]:

            hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

            hp_arr = np.array(hp, np.float32)

            normal_pt = np.array([hp_arr[0] / hp_arr[2], hp_arr[1] / hp_arr[2]], np.float32)

            if (max_x == None or normal_pt[0, 0] > max_x):
                max_x = normal_pt[0, 0]

            if (max_y == None or normal_pt[1, 0] > max_y):
                max_y = normal_pt[1, 0]

            if (min_x == None or normal_pt[0, 0] < min_x):
                min_x = normal_pt[0, 0]

            if (min_y == None or normal_pt[1, 0] < min_y):
                min_y = normal_pt[1, 0]

        min_x = min(0, min_x)
        min_y = min(0, min_y)

        return (min_x, min_y, max_x, max_y)

    def stitch(self, base_img_rgb, next_img_rgb, H, status, round=0):
        base_img = cv2.GaussianBlur(cv2.cvtColor(base_img_rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        # base_img, next_img, H
        next_img_gray = cv2.GaussianBlur(cv2.cvtColor(next_img_rgb,cv2.COLOR_BGR2GRAY), (5,5), 0)

        closestImage = None

        inlierRatio = float(np.sum(status)) / float(len(status))

        if (closestImage == None or inlierRatio > closestImage['inliers']):
            closestImage = {}
            closestImage['h'] = H
            closestImage['inliers'] = inlierRatio
            closestImage['img'] = next_img_gray
            closestImage['rgb'] = next_img_rgb

        H = closestImage['h']
        H = H / H[2, 2]
        H_inv = linalg.pinv(H)

        if (closestImage['inliers'] > 0.1):  # and
            (min_x, min_y, max_x, max_y) = self.findDimensions(closestImage['img'], H_inv)
            self.closest_img = closestImage['img']
            self.min_x, self.min_y, self.max_x, self.max_y = min_x, min_y, max_x, max_y
            # Adjust max_x and max_y by base img size
            max_x = max(max_x, base_img.shape[1])
            max_y = max(max_y, base_img.shape[0])

            move_h = np.matrix(np.identity(3), np.float32)

            if (min_x < 0):
                move_h[0, 2] += -min_x
                max_x += -min_x

            if (min_y < 0):
                move_h[1, 2] += -min_y
                max_y += -min_y

            mod_inv_h = move_h * H_inv

            img_w = int(math.ceil(max_x))
            img_h = int(math.ceil(max_y))

            # crop edges
            base_h, base_w, base_d = base_img_rgb.shape
            next_h, next_w, next_d = closestImage['rgb'].shape

            base_img_rgb = base_img_rgb[5:(base_h - 5), 5:(base_w - 5)]
            closestImage['rgb'] = closestImage['rgb'][5:(next_h - 5), 5:(next_w - 5)]

            # Warp the new image given the homography from the old image
            base_img_warp = cv2.warpPerspective(base_img_rgb, move_h, (img_w, img_h))
            next_img_warp = cv2.warpPerspective(closestImage['rgb'], mod_inv_h, (img_w, img_h))
            self.base_img_warp, self.next_img_warp = base_img_warp, next_img_warp

            enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

            # enlarged_base_img[y:y+base_img_rgb.shape[0],x:x+base_img_rgb.shape[1]] = base_img_rgb
            # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp

            # Create masked composite
            (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY),
                                            0, 255, cv2.THRESH_BINARY)

            # add base image
            enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp,
                                        mask=np.bitwise_not(data_map),
                                        dtype=cv2.CV_8U)
            self.enlarged_base_img = enlarged_base_img

            # add next image
            final_img = cv2.add(enlarged_base_img, next_img_warp,
                                dtype=cv2.CV_8U)
            self.final_img = final_img

            # Crop black edge
            final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
            dino, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            max_area = 0
            best_rect = (0, 0, 0, 0)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                deltaHeight = h - y
                deltaWidth = w - x
                area = deltaHeight * deltaWidth
                if (area > max_area and deltaHeight > 0 and deltaWidth > 0):
                    max_area = area
                    best_rect = (x, y, w, h)
            if (max_area > 0):
                final_img_crop = final_img[best_rect[1]:best_rect[1] + best_rect[3],
                                 best_rect[0]:best_rect[0] + best_rect[2]]
                final_img = final_img_crop


            # contours = sorted(contours, key=lambda contour: len(contour), reverse=True)
            # roi = cv2.boundingRect(contours[0])
            # # use the roi to select into the original 'stitched' image
            # final_img = final_img[roi[1]:roi[3], roi[0]:roi[2]]

            return final_img

        else:
            return next_img_rgb






