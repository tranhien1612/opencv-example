'''
Loading of test image
Detect edges (gray scaling, Gaussian smoothing, Canny edge detection)
Define region of interest
Apply Hough transforms
Post-processing of lane lines (averaging, extrapolating)
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

def nothing(e):
    pass

def canny_edge_detector(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 5x5 gaussian blur to reduce noise 
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Canny edge detector with minVal of 50 and maxVal of 150
    min = cv2.getTrackbarPos('min', 'Image Processing')
    max = cv2.getTrackbarPos('max', 'Image Processing')
    canny = cv2.Canny(blur, int(min), int(max))
    # canny = cv2.Canny(blur, 50, 150)
    return canny  

def ROI_mask(image):
    
    height = image.shape[0]
    width = image.shape[1]

    
    # A triangular polygon to segment the lane area and discarded other irrelevant parts in the image
    # Defined by three (x, y) coordinates    
    polygons = np.array([ 
        [(0, height), (round(width/2), round(height/2)), (1000, height)] 
        ]) 
    
    mask = np.zeros_like(image) 
    cv2.fillPoly(mask, polygons, 255)  ## 255 is the mask color
    
    # Bitwise AND between canny image and mask image
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

if __name__ == "__main__":
    inputimage = cv2.imread("test1.jpg")
    inputimage = cv2.resize(inputimage, (640, 512))
    print(inputimage)
    cv2.imshow("image", inputimage)

    cv2.namedWindow('Image Processing')
    cv2.createTrackbar('min', 'Image Processing', 0, 255, nothing)
    cv2.createTrackbar('max', 'Image Processing', 0, 255, nothing)

    while True:
        canny_edges = canny_edge_detector(inputimage)
        # cv2.imshow("canny_edges", canny_edges)

        # cropped_image = ROI_mask(canny_edges)
        cv2.imshow("Image Processing", canny_edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
