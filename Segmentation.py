import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load an color image in grayscale (SPHERO POSITION)
#img = cv2.imread("3.jpg", 0)
#ret, thresh = cv2.threshold(img, 127, 200, cv2.THRESH_BINARY)

#cv2.imshow("Sphero", thresh)
'''ret1, thresh1 = cv2.threshold(img, 127, 220, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(img, 127, 230, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(img, 127, 250, cv2.THRESH_BINARY)'''
'''
titles = ['Original Image', 'BINARY', 'BINARY1', 'BINARY2', 'BINARY3']
images = [img, thresh, thresh1,  thresh2, thresh3]
cv2.imwrite('gray31.png', images[1])
'''
'''
for i in range(5):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()
'''
# Color segmentation

Colorimg = cv2.imread('brightimage.png', 1)
HSVimg = cv2.cvtColor(Colorimg, cv2.COLOR_BGR2HSV)

# Red color
'''
low_red = np.array([0, 150, 0])
high_red = np.array([255, 255, 255])
red_mask = cv2.inRange(HSVimg, low_red, high_red)
red = cv2.bitwise_and(Colorimg, Colorimg, mask = red_mask)
cv2.imshow("RedMask", red_mask)
cv2.imshow("Colorimg", Colorimg)
cv2.imshow("Red", red)
'''
low_green = np.array([40, 65, 25])
high_green = np.array([80, 220, 160])
green_mask = cv2.inRange(HSVimg, low_green, high_green)
green = cv2.bitwise_and(Colorimg, Colorimg, mask=green_mask)
cv2.imshow("Green Mask", green_mask)
green_mask = cv2.bitwise_not(green_mask)
cv2.imshow("Green Mask", green_mask)
cv2.imshow("Colorimg", Colorimg)
cv2.imshow("Green", green)




'''
# BLOB DETECTION
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = False
params.filterByCircularity = False

thresh = cv2.bitwise_not(thresh)
# Change thresholds
# params.minThreshold = 200
# params.maxThreshold = 255

#print(green_mask.dtype)
#print(cv2.__version__)

print(params.filterByColor)
print(params.filterByArea)
print(params.filterByCircularity)
print(params.filterByInertia)
print(params.filterByConvexity)
# Filter by Area.
params.filterByArea = True
params.minArea = 0
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(thresh)

print(keypoints)
#keypoints = detector.detect(frame) #list of blobs keypoints
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#im_with_keypoints = cv2.drawKeypoints(green_mask, keypoints, np.array([]), (0, 0, 255))
for keyPoint in keypoints:
    x = keyPoint.pt[0]
    y = keyPoint.pt[1]
    s = keyPoint.size
    print(x)
    print(y)
    print(s)
# Show keypoints

#cv2.imshow("Keypoints", im_with_keypoints)'''
cv2.waitKey(0)
cv2.destroyAllWindows()




