from pathlib import Path

import cv2
import numpy as np

if __name__ == '__main__':
    # Let's load a simple image with 3 black squares
    image = cv2.imread('images/1_21_s.bmp')

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("image thresholding", thresh1)
    cv2.waitKey(0)

    # Find Canny edges
    edged = cv2.Canny(thresh1, 30, 200)
    cv2.imshow("image edged", edged)
    cv2.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    real_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 200:
            rect = cv2.minAreaRect(contour)
            print(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    print("Number of Contours found = " + str(len(contours)))
    print("Number of Contours filteres = " + str(len(real_contours)))

    # Draw all contours
    # -1 signifies drawing all contours
    # cv2.drawContours(image, real_contours, -1, (0, 0, 255), -1)

    cv2.imshow('Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()