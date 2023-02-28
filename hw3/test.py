import cv2
import numpy as np
from PIL import Image


# Load the image
img = cv2.imread('ParkingLot.jpg')
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# Apply Hough line detection
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)
# Cluster the detected lines
line_clusters = []
for line in lines:
    rho, theta = line[0]
    if len(line_clusters) == 0:
        line_clusters.append([(rho, theta)])
    else:
        found_cluster = False
        for cluster in line_clusters:
            if abs(cluster[0][0] - rho) < 50 and abs(cluster[0][1] - theta) < np.pi/36:
                cluster.append((rho, theta))
                found_cluster = True
                break
        if not found_cluster:
            line_clusters.append([(rho, theta)])
# Find intersection points
intersections = []
for cluster in line_clusters:
    for i in range(len(cluster)):
        rho1, theta1 = cluster[i]
        for j in range(i+1, len(cluster)):
            rho2, theta2 = cluster[j]
            A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
            b = np.array([rho1, rho2])
            x, y = np.linalg.solve(A, b)
            if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
                intersections.append((int(x), int(y)))
# Find parking space polygons
polygons = []
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

# Draw the polygons on the image
for i, parking_space in enumerate(polygons):
    pts = np.array(parking_space, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 255, 0), thickness=2)

# Display the image with polygons drawn
cv2.imshow('Parking Lot with Polygons', img)
cv2.waitKey(0)
cv2.destroyAllWindows()