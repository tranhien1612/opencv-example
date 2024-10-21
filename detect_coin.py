import cv2
import numpy as np

def nothing(x):
    pass

image = './images/coin.jpg'
img = cv2.imread(image)
img = cv2.resize(img, (640, 512))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (9,9), 0)

cv2.namedWindow('Image Processing')
cv2.createTrackbar('min', 'Image Processing', 0, 255, nothing)
cv2.createTrackbar('max', 'Image Processing', 0, 255, nothing)

while True:
    # min = cv2.getTrackbarPos('min', 'Image Processing')
    # max = cv2.getTrackbarPos('max', 'Image Processing')
    # img_canny = cv2.Canny(img_blur, int(min), int(max))

    img_canny = cv2.Canny(img_blur, 115, 250) #coin2: 115, 250
    (cnts,_) = cv2.findContours(img_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coins = img.copy()
    cv2.drawContours(coins, cnts, -1, (0,0,255), 2)
    
    cnt = 0
    for (i,c) in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c)
        sub_image = img[y:(y+h), x:(x+w)]
        title = f"coint_{cnt}"
        cv2.imshow(title, sub_image)

    cv2.imshow("image", img)
    cv2.imshow("Image Processing", img_canny)
    cv2.imshow("coins", coins)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()