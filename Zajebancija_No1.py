import cv2
import numpy as np
from matplotlib import pyplot as plt

cam = cv2.VideoCapture(0)

cv2.namedWindow("Pocetna_slika")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("Pocetna_slika", frame)
    print("Hit SPACE!")
    # Image not read
    if not ret:
        break
    k = cv2.waitKey(1)


    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k % 256 == 32:
        # SPACE pressed
        img_name = "Slike_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print(" Hit ESC!!!!!!!!")


        # print("{} written!".format(img_name))
        # img_counter += 1


        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

# Segmentacija pocetne slike
# Potrebno je ucitati sliku, prvo izdvojiti sfera, a potom izdvojiti cunjeve

# Load an color image in grayscale (SPHERO POSITION)
img = cv2.imread("Pocetna_slika.png", 0)
ret, thresh = cv2.threshold(img, 127, 200, cv2.THRESH_BINARY)

# ODREDITI KOORDINATE SFERA

# Color segmentation

Colorimg = cv2.imread('Pocetna_slika.png', 1)
HSVimg = cv2.cvtColor(Colorimg, cv2.COLOR_BGR2HSV)

# Red color

low_red = np.array([0, 150, 0])
high_red = np.array([255, 255, 255])

# Ova maska je crno bijela slika-iz nje odrediti koordinate crvenih cunjeva
red_mask = cv2.inRange(HSVimg, low_red, high_red)
# OVO DOLE NAM NIJE POTREBNO
# red = cv2.bitwise_and(Colorimg, Colorimg, mask = red_mask)
# cv2.imshow("Colorimg", Colorimg)
# cv2.imshow("Red", red)

# Green color

low_green = np.array([40, 65, 25])
high_green = np.array([80, 220, 160])
# Ova maska je crno bijela slika-iz nje odrediti koordinate zelenih cunjeva
green_mask = cv2.inRange(HSVimg, low_green, high_green)
# OVO DOLE NAM NIJE POTREBNO
# green = cv2.bitwise_and(Colorimg, Colorimg, mask=green_mask)
# cv2.imshow("Colorimg", Colorimg)
# cv2.imshow("Green", green)

