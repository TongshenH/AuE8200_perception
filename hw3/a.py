import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from PIL import Image
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

img = np.array(Image.open('ParkingLot.jpg'))
# Implement Canny algorithm to get the binary edge
edges = canny(img, sigma=2, low_threshold=85, high_threshold=150)
identified_lines = np.array(probabilistic_hough_line(edges))
# Cluster the detected Line
line_cluster = []
for i in range(identified_lines.shape[0]):
    line_points = identified_lines[i, :, :].flatten()
    if len(line_cluster) == 0:
        line_cluster.append(line_points)
    else:
        found_cluster = False
        for j in range(len(line_cluster)):
            cluster_k = (line_cluster[j][3] - line_cluster[j][1]) / (line_cluster[j][2] - line_cluster[j][0])
            cluster_b = line_cluster[j][1] - (line_cluster[j][3] - line_cluster[j][1]) * line_cluster[j][0] / (line_cluster[j][2] - line_cluster[j][0])
            x0, y0, x1, y1 = line_points[0], line_points[1], line_points[2], line_points[3]
            # Calculate the k & b
            k = (y1 - y0) / (x1 - x0)
            b = y0 - (y1 - y0) * x0 / (x1 - x0)
            # if abs(k -cluster_k) < 0.1 or abs(b - cluster_b) < 10:
            #     break
            # if j == len(line_cluster)-1:
            #     if k < 0:
            #         line_points[0] = 0
            #         line_points[1] = b
            #         line_points[2] = 960
            #         line_points[3] = k * 960 + b
            #     line_cluster.append(line_points)
            if abs(k - cluster_k) < 0.1:
              if abs(b - cluster_b) < 10:


            if j == len(line_cluster)-1:
                if k < 0:
                    line_points[0] = 0
                    line_points[1] = b
                    line_points[2] = 960
                    line_points[3] = k * 960 + b
                line_cluster.append(line_points)



for i in range(len(line_cluster)):
    cv2.line(img, (line_cluster[i][0], line_cluster[i][1]), (line_cluster[i][2], line_cluster[i][3]), (0, 0, 220))
Image.fromarray(img).save('Parking.jpg')


# Find the intersection
intersections = []
for i in range(len(intersections)):
    for j in range(i+1, len(intersections)):
        for k in range(j+1, len(intersections)):
            for l in range(k+1, len(intersections)):
                p1, p2, p3, p4 = intersections[i], intersections[j], intersections[k], intersections[l]
                sides = [cv2.norm(np.array(p1) - np.array(p2)),
                         cv2.norm(np.array(p2) - np.array(p3)),
                         cv2.norm(np.array(p3) - np.array(p4)),
                         cv2.norm(np.array(p4) - np.array(p1))]
                if all(side > 30 for side in sides):
                    area = cv2.contourArea(np.array([p1, p2, p3, p4]))
                    if area > 1000:
                        polygons.append([p1, p2, p3, p4])


