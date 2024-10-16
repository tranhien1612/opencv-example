import cv2
import numpy as np

image = cv2.imread("./images/tennis.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(image, (300, 300))
template = cv2.Canny(small_image, 175, 200)
contours, hierarchy = cv2.findContours(template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("template", template)
cv2.drawContours(small_image, contours, -1, (0, 255, 0), 3)
cv2.imshow("draw contours", small_image)
cv2.waitKey(0)