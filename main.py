import numpy as np
import cv2, time, sys
import matplotlib.pyplot as plt

def open_display_img():
    img = cv2.imread('images/messi.jpg', cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError
    else:
        print ('Image Loaded')

    print (type(img))
    print (img.shape)
    print (img.dtype)
    print (img)
    
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_object():
    img2 = np.zeros((512,512,3), np.uint8)
    
    # Draw a line using cv2.line(image, startPoint, endPoint, rgb, thinkness)
    cv2.line(img2, (0,0), (511,511), (255,0,0), 5)

    # Draw a rectangle using cv2.rectangle(image, topLeft, bottomRight, rgb, thinkness)
    cv2.rectangle(img2, (384,0), (510,128), (0,255,0), 3)

    # Draw a circle using cv2.circle(image, center, radius, rgb, thinkness)
    cv2.circle(img2, (447,63), 63, (0,0,255), -1)

    # Draw a ellipse using cv2.ellipse(image, center, axes, angle, startAngle, endAngle, rgb, thinkness)
    cv2.ellipse(img2, (256,256), (100,50), -45, 0, 180, (255,255,0), -1)

    # Draw a line using cv2.polylines(image, points, isClosed, rgb, thinkness, lineType, shift)
    pts = np.array([[10,10],[150,200],[300,150],[200,50]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img2,[pts],True,(0,255,255),3)

    # Put some text using cv2.putText(image, text, bottomLeft, fontType, fontScale, rgb, thinkness, lineType)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img2, 'OpenCV', (10,500), font, 4, (255,255,255), 10, cv2.LINE_AA)

    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def modify_pixels_and_roi():
    img = cv2.imread('images/messi.jpg', cv2.IMREAD_COLOR)

## modify pixels
    # img[50, 235]
    # for i in range(30):
    #     for j in range(30):
    #         img[50+i, 235+j] = (0, 255, 0)
    
## Roi
    ball = img[280:340, 330:390]
    cv2.imshow('ball', ball)

    # Random sample five positions
    top_left_x = np.random.randint(0, 480, 5)
    top_left_y = np.random.randint(0, 280, 5)

    # Paste the ball to these positions
    for x, y in zip(top_left_x, top_left_y):
        # print(x,y)
        img[y:y+60, x:x+60] = ball
    
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def canny_edge_detection():
    input_image = cv2.imread('images/wheel.png', cv2.IMREAD_GRAYSCALE) 
    threshold1 = 100
    threshold2 = 200
    canny = cv2.Canny(input_image, threshold1, threshold2)
    
    canny1 = cv2.Canny(input_image, 50, 150)
    canny2 = cv2.Canny(input_image, 100, 200)
    canny3 = cv2.Canny(input_image, 150, 250)

    cv2.imshow('canny', canny)
    cv2.imshow('canny1', canny1)
    cv2.imshow('canny2', canny2)
    cv2.imshow('canny3', canny3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def feature_matching():
    beaver = cv2.imread('images/beaver.png')
    gray = cv2.cvtColor(beaver, cv2.COLOR_BGR2GRAY)

    print(str(beaver.shape) + " => " + str(gray.shape))
    cv2.imshow('beaver', beaver)
    cv2.imshow('gray', gray)

    # SIFT feature detector/descriptor
    sift = cv2.xfeatures2d.SIFT_create()
    # SIFT feature detection
    start_time = time.time()
    # kp = sift.detect(gray, None) # 2nd pos argument is a mask indicating a part of image to be searched in
    kp = sift.detect(beaver, None) # 2nd pos argument is a mask indicating a part of image to be searched in
    print('Elapsed time: %.6fs' % (time.time() - start_time))
    beaver_sift = cv2.drawKeypoints(beaver, kp, None)
    cv2.imshow('beaver_sift', beaver_sift)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def homography():
    img = cv2.imread('images/messi.jpg', cv2.IMREAD_GRAYSCALE)

# Translation
    rows,cols = img.shape
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))

# Rotation
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    print(M)
    dst1 = cv2.warpAffine(img,M,(cols,rows))
    dst2 = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_WRAP)
    dst3 = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REFLECT)

# Homography(Perspective Transformation)
    img1 = cv2.imread('images/track.jpg')
    img2 = cv2.imread('images/logo.jpg')
    rows1, cols1, ch1 = img1.shape
    rows2, cols2, ch2 = img2.shape

    pts1 = np.float32([(0,0),(cols2-1,0),(cols2-1,rows2-1),(0,rows2-1)])
    pts2 = np.float32([(671,314),(1084,546),(689,663),(386,361)])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    # print(M)
    img3 = np.copy(img1)
    cv2.warpPerspective(img2,M,(cols1,rows1),img3,borderMode=cv2.BORDER_TRANSPARENT)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)


    # cv2.imshow('img', img)
    # cv2.imshow('dst', dst)
    # cv2.imshow('dst1', dst1)
    # cv2.imshow('dst2', dst2)
    # cv2.imshow('dst3', dst3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def face_detection():
    cascPath = 'detect/haarcascade_frontalface_alt.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    img = cv2.imread('images/ioi2.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    print("Found %d faces!" % len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)
    
    # cv2.imshow('img', img)
    cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def open_save_video():
    input = cv2.VideoCapture("videos/vtest.mp4")
    print("Frame width: %d" % input.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Frame height: %d" % input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS: %d" % input.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter("output.avi", fourcc, 10.0, (768, 576))

    while(input.isOpened()):
        ret, frame = input.read()
        if ret==True:
            output.write(frame)
        else:
            break

    input.release()
    output.release()

def img_process():
    image = cv2.imread("./images/car_2.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("origin", image)
    cv2.imshow("gray", gray)
    blur = cv2.bilateralFilter(gray, 11,90, 90)
    # cv2.imshow("blur", blur)
    edges = cv2.Canny(blur, 30, 200)
    cv2.imshow("edges", edges)
    cnts, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Trả về danh sách tất cả các đường viền trong ảnh

    # image_copy = image.copy()
    # _ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)
    # cv2.imshow("contours", image_copy)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    image_copy = image.copy()
    _ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)
    cv2.imshow("contours", image_copy)
    
    for c in cnts:
        perimeter = cv2.arcLength(c, True) # tính độ dài cung
        #Khi 1 hình kín không đầy đủ, sử dụng hàm để lấy sấp xỉ hình kín 
        #0.02 * perimeter: Khoảng cách tối đa từ đường viền đến đường viền ước tính
        edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True) 
        if len(edges_count) == 4:
            x,y,w,h = cv2.boundingRect(c) #Tìm hình chữ nhật bao phủ hình đầu vào c
            plate = image[y:y+h, x:x+w]
            break

    # import pytesseract
    # text = pytesseract.image_to_string(plate, lang="eng")
    # print(text)

    cv2.imshow("plate", plate)
    cv2.waitKey(0)

if __name__ == "__main__":
    # open_display_img()
    # draw_object()
    # modify_pixels_and_roi()
    # canny_edge_detection()
    # feature_matching()
    # homography()
    # face_detection()
    # open_save_video()
    img_process()