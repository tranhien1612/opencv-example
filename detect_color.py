import cv2
import numpy as np

def get_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = image[y, x]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
        print(f'BGR: {bgr}, HSV: {hsv[0][0]}')

def read_vaule(image):
    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', get_hsv)

lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255]) 
lower_red = np.array([0, 130, 200])
upper_red = np.array([5, 255, 255]) 

image = cv2.imread("./images/rubickscube.png")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

#Use bitwise AND operation to extract the colored regions from the original image
blue = cv2.bitwise_and(image, image, mask=blue_mask)
green = cv2.bitwise_and(image, image, mask=green_mask)
red = cv2.bitwise_and(image, image, mask=red_mask)

read_vaule(image)

cv2.imshow('Original Image', image)
cv2.imshow('hsv_image', hsv_image)
cv2.imshow('Blue', blue)
cv2.imshow('Green', green)
cv2.imshow('Red', red)

cv2.waitKey(0)
cv2.destroyAllWindows()


