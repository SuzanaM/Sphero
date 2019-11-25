import numpy as np
import cv2

def get_coordinates(img):

    # Lists of coordinates
    X = []
    Y = []
    # Invert image
    img = cv2.bitwise_not(img)

    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False
    params.filterByCircularity = False

    # Filter by Area
    params.filterByArea = True
    params.minArea = 100# Podesiti ovaj parametar u skladu sa slikama

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)
    # Get coordinates
    for keyPoint in keypoints:
        x = int(keyPoint.pt[0])
        X.append(x)
        y = int(keyPoint.pt[1])
        Y.append(y)

        # Size
        # s = keyPoint.size
    Coordinates = list(zip(X, Y))

    return Coordinates

# Color segmentation

Colorimg = cv2.imread('1.jpg', 1)
HSVimg = cv2.cvtColor(Colorimg, cv2.COLOR_BGR2HSV)
low_green = np.array([40, 65, 25])
high_green = np.array([80, 220, 160])
green_mask = cv2.inRange(HSVimg, low_green, high_green)

# Load an color image in grayscale (SPHERO POSITION)
# img = cv2.imread("1.jpg", 0)
# ret, thresh = cv2.threshold(img, 127, 200, cv2.THRESH_BINARY)

cv2.imshow("Sphero", green_mask)
#cv2.imshow("Sero", green_mask)
Koordinate = get_coordinates(green_mask)
print(Koordinate)
cv2.waitKey(0)
cv2.destroyAllWindows()