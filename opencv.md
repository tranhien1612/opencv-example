# [OpenCV](https://www.geeksforgeeks.org/computer-vision/)

https://www.kaggle.com/learn/computer-vision

## Installation
```
  conda create -n <name_environment> python=3.10
  conda activate <name_environment>
  pip install opencv-python
```

## Getting Started with OpenCV

### Types of an image
1. BINARY IMAGE– The binary image as its name suggests, contain only two pixel elements i.e 0 & 1,where 0 refers to black and 1 refers to white. This image is also known as Monochrome.
2. BLACK AND WHITE IMAGE– The image which consist of only black and white color is called BLACK AND WHITE IMAGE.
3. 8 bit COLOR FORMAT– It is the most famous image format.It has 256 different shades of colors in it and commonly known as Grayscale Image. In this format, 0 stands for Black, and 255 stands for white, and 127 stands for gray.
4. 16 bit COLOR FORMAT– It is a color image format. It has 65,536 different colors in it.It is also known as High Color Format. In this format the distribution of color is not as same as Grayscale image.


### Reading in an image
Working with opencv we need to read in our image. To do this we just called cv2.imread() function. It returns the image in a numpy array format.
```
image = cv2.imread('dog.png')
```

So we have a simple image called dog.png. If we run print on the image we could see the results.
```
image = cv2.imread('dog.png')
print(image)
```

A sample of what a numpy array looks lie is shown below.

### Getting image shape
We can get the image shape by calling image.shape. Where image is the image we read in using ```cv2.imread()``` function.

Let see this in action.
```
import cv2

image = cv2.imread('./images/tennis.jpg')
print(image.shape)
```
The results you will see is
```
(547, 512, 3)
```
Where ```547``` is the height, ```512``` is the width and ```3``` is the number of channels.

### Viewing an image
After we read in the image we can display this image again using c```v2.imshow```. This takes two arguments The window name and the numpy array.
```
import cv2

image = cv2.imread('./images/tennis.jpg')
cv2.imshow("tennis", image)
cv2.waitKey(0)
```
Using ```cv2.waitKey(0)``` stops the dialog from closing before we get to see the image. So we can see the image in a dialog with the title we put in the first argument.

### Changing the color space
You can change the color space of an image quite easily. Lets see how.

Lets change our image to gray scale. To do this we call the ```cv2.cvtColor``` function and pass in the image and the ```cv2.COLOR_BGR2GRAY``` option. The code is shown below.
```
image = cv2.imread('dog.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("dog", gray_image)
cv2.waitKey(0)
```

#### From RGB to BGR
Remember when we used matplotlib. The image showed in the color space GBR. Lets try to get the same results. Below we add the option ```cv2.COLOR_RGB2BGR```
```
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
```

#### From BGR to RGB
So we can always convert our matplot image using
```
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```
Play with these options to see the different results you can get.

#### Converting to HSV
A color space to note is HSV. We can create comparisions between image with it using histograms. Lets convert our image to this
```
import cv2

image = cv2.imread('./images/tennis.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("tennis", gray_image)
cv2.waitKey(0)
```

### Resizing images
Lets see how we can resize image. To do this we need to run the ```cv2.resize()``` function.
```
import cv2

image = cv2.imread('./images/tennis.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(gray_image, (300, 300)) # resize image
cv2.imshow("tennis", small_image)
cv2.waitKey(0)
```
Above we call the ```cv2.resize``` function. We pass the image we want to resize then we pass a tuple ```(300, 300)``` with the width and height as parameters.

### Saving images
We can save an image by calling the ```cv2.imwrite()``` function. It takes the name of the image and the object to save. Lets see the code below
```
import cv2

image = cv2.imread('./images/tennis.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(gray_image, (300, 300))
cv2.imwrite("test.jpg", small_image)
```
So we called imwrite and named our file ```dogtest.png```. 

### Thresholding
This is an important part of working with images in Opencv. You can learn more about it here on wikipedia or here on opencv docs

But right now let me show you what it does.
```
import cv2

image = cv2.imread("./images/tennis.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(gray_image, (300, 300))
ret, thresh_image = cv2.threshold(small_image, 127, 255, cv2.THRESH_BINARY) # thresholding
cv2.imshow("tennis", thresh_image)
cv2.waitKey(0)
```
In the above code. We did things we already showed you. We read the image; We turned it gray. Then we resized it. THEN lets get into thresholding

### Drawing rectangles

Let see how we can draw rectangles on your image. To do this we use the cv2.rectangle function. An example is shown below
```
cv2.rectangle(image, (50, 100), (100, 200), (0, 255, 0))
```
So above we have the source image called image. Then we have a tuple for the start point (x, y) coordinates ```(50, 100)```. Then the end point with (x, y) coordinates ```(100, 200)```. Finally we have to set the color of the rectangle via a tuple in rgb format. So green is ```(0, 255, 0)```.
```
import cv2

image = cv2.imread("./images/tennis.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(image, (300, 300))
cv2.rectangle(small_image, (150, 80), (230, 150), (0, 255, 0))
cv2.imshow("tennis", small_image)
cv2.waitKey(0)
```

### Cropping Images
So you can now draw rectangles. Lets see how we can crop an image. To crop an image we only need to do this as shown below
```
cropped = orig[startY:endY, startX:endX]
```
Where ```cropped``` is the finally image. ```orig``` is the original image. ```startY``` and ```endY``` are y coordinates. ```startX``` and ```endX``` are x coordinates.

The full code can be seen here
```
import cv2

image = cv2.imread("./images/tennis.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(image, (300, 300))
cv2.rectangle(small_image, (150, 80), (230, 150), (0, 255, 0))
cropped = small_image[80:150, 150:230]
cv2.imshow("tennis", small_image)
cv2.imshow("cropped image", cropped)
cv2.waitKey(0)
```

### Rotating your images
Lets see how we can rotate images with opencv. We can use ```cv2.warpAffine()``` to rotate our images.
```
import cv2

image = cv2.imread("./images/tennis.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(image, (300, 300))
(h, w) = small_image.shape[:2] # get height and width
center = (w / 2, h / 2) # divide by 2
M = cv2.getRotationMatrix2D(center, 30, 1.0) # get matrix
rotated = cv2.warpAffine(small_image, M, (h, w)) # rotate image
cv2.imshow("Rotated Image", rotated)
cv2.waitKey(0)
```

### Smoothing and Blurring
When we are dealing with images at some points the images will be crisper and sharper which we need to smoothen or blur to get a clean image, or sometimes the image will be with a really bad edge which also we need to smooth it down to make the image usable.

#### Method 1: With 2D Convolution 
In this method of smoothing, we have complete flexibility over the filtering process because we will be using our custom-made kernel [a simple 2d matrix of NumPy array which helps us to process the image by convolving with the image pixel by pixel]. A kernel basically will give a specific weight for each pixel in an image and sum up the weighted neighbor pixels to form a pixel, with this method we will be able to compress the pixels in an image and thus we can reduce the clarity of an image, By this method, we can able to smoothen or blur an image easily.  

![image](https://media.geeksforgeeks.org/wp-content/uploads/20211031005735/kernelworking.png)

```
import cv2
import numpy as np

image = cv2.imread("./images/tennis.jpg")

# Creating the kernel with numpy 
kernel2 = np.ones((5, 5), np.float32)/25
# Applying the filter 
img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2) 
  
# showing the image 
cv2.imshow('Original', image) 
cv2.imshow('Kernel Blur', img) 
  
cv2.waitKey(0) 
cv2.destroyAllWindows() 
```

#### Method 2:  With pre-built functions
OpenCV comes with many prebuilt blurring and smoothing functions let us see them in brief,

##### 1. Averaging
```
Syntax: cv2.blur(image, shapeOfTheKernel)

Image– The image you need to smoothen
shapeOfTheKernel– The shape of the matrix-like 3 by 3 / 5 by 5
```
The averaging method is very similar to the 2d convolution method as it is following the same rules to smoothen or blur an image and uses the same type of kernel which will basically set the center pixel’s value to the average of the kernel weighted surrounding pixels. And by this, we can greatly reduce the noise of the image by reducing the clarity of an image by replacing the group of pixels with similar values which is basically similar color. We can greatly reduce the noise of the image and smoothen the image. The kernel we are using for this method is the desired shape of a matrix with all the values as “1”  and the whole matrix is divided by the number of values in the respective shape of the matrix [which is basically averaging the kernel weighted values in the pixel range]

![image](https://media.geeksforgeeks.org/wp-content/uploads/20211031010455/Averagekernel.png)

```
import cv2
import numpy as np

image = cv2.imread("./images/tennis.jpg")

# Applying the filter 
averageBlur = cv2.blur(image, (5, 5)) 
  
# Showing the image 
cv2.imshow('Original', image) 
cv2.imshow('Average blur', averageBlur) 
  
cv2.waitKey() 
cv2.destroyAllWindows() 
```

##### 2. Gaussian Blur
```
Syntax: cv2. GaussianBlur(image, shapeOfTheKernel, sigmaX )

Image– the image you need to blur
shapeOfTheKernel– The shape of the matrix-like 3 by 3 / 5 by 5
sigmaX– The Gaussian kernel standard deviation which is the default set to 0
```
In a gaussian blur, instead of using a box filter consisting of similar values inside the kernel which is a simple mean we are going to use a weighted mean. In this type of kernel, the values near the center pixel will have a higher weight. With this type of blurs, we will probably get a less blurred image but a natural blurred image which will look more natural because it handles the edge values very well. Instead of averaging the weighted sum of the pixels here, we will divide it with a specific value which is 16 in the case of a 3 by 3 shaped kernel which will look like this.

![image](https://media.geeksforgeeks.org/wp-content/uploads/20211031010456/Gaussiankernel.png)

```
import cv2
import numpy as np

image = cv2.imread("./images/tennis.jpg")

gaussian = cv2.GaussianBlur(image, (3, 3), 0) 
  
# Showing the image 
cv2.imshow('Original', image) 
cv2.imshow('Gaussian blur', gaussian) 
  
cv2.waitKey() 
cv2.destroyAllWindows() 
```

##### 3. Median blur
```
Syntax: cv. medianBlur(image, kernel size)

Image– The image we need to apply the smoothening
KernelSize– the size of the kernel as it always takes a square matrix the value must be a positive integer more than 2.
```
In this method of smoothing, we will simply take the median of all the pixels inside the kernel window and replace the center value with this value. The one positive of this method over the gaussian and box blur is in these two cases the replaced center value may contain a pixel value that is not even present in the image which will make the image’s color different and weird to look, but in case of a median blur though it takes the median of the values that are already present in the image it will look a lot more natural.

```
import cv2
import numpy as np

image = cv2.imread("./images/tennis.jpg")

# Applying the filter 
medianBlur = cv2.medianBlur(image, 9) 
  
# Showing the image 
cv2.imshow('Original', image) 
cv2.imshow('Median blur', medianBlur) 
  
cv2.waitKey() 
cv2.destroyAllWindows() 
```

##### 4. Bilateral blur
```
Syntax: cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)

- Image– The image we need to apply the smoothening
- Diameter– similar to the size of the kernel
- SigmaColor– The number of colors to be considered in the given range of pixels [the higher value represents the increase in the number of colors in the given area of pixels]—Should not keep very high
- SigmaSpace – the space between the biased pixel and the neighbor pixel higher value means the pixels far out from the pixel will manipulate in the pixel value
```
The smoothening methods we saw earlier are fast but we might end up losing the edges of the image which is not so good. But by using this method, this function concerns more about the edges and smoothens the image by preserving the images. This is achieved by performing two gaussian distributions. This might be very slow while comparing to the other methods we discussed so far.

```
import cv2
import numpy as np

image = cv2.imread("./images/tennis.jpg")

# Applying the filter 
bilateral = cv2.bilateralFilter(image, 9, 75, 75) 
  
# Showing the image 
cv2.imshow('Original', image) 
cv2.imshow('Bilateral blur', bilateral) 
  
cv2.waitKey() 
cv2.destroyAllWindows()
```

### Edge Detection

The detection of edges in an image enables us to identify the objects that are present. So, it is a significant use case in computer vision. The edges are formed by a significant variation in the adjacent pixel intensities of an image. Two popular edge detection algorithms are - Sobel filter and the Canny filter

#### 1. Sobel Edge Detection
The Sobel detector calculates the gradient of the pixels in an image. It identifies the rate at which the pixel intensity changes from light to dark and the direction of the change. This change will let us know the presence of an edge by judging how quickly or gradually the intensity changes. It contains two kernels, one for detecting horizontal edges and the other for vertical edges. 

![image](https://dezyre.gumlet.io/images/tutorial/computer-vision-tutorial-for-beginners/image_763647877381628842955434.png?w=1080&dpr=1.0)

#### 2. Canny Edge Detection
Canny edge detector follows a multi-stage edge detection algorithm. It is robust and highly efficient as it incorporates the Sobel filter method along with some post-processing steps. The significant stages are listed below: 

##### 1. Noise Reduction
This is a preprocessing step with Gaussian Blur to smoothen the image and reduce the noise around the edges for accurate results. 

##### 2. Sobel Filtering
This step calculates the intensity gradient of the image along with both the directions - X and Y. Similar to the Sobel filter; it outputs the direction and the rate of change of intensity gradient.

##### 3. Non Maximum Suppression
This technique is used to ignore the false edges generated. It checks whether the gradient value of the current pixel is greater than the neighbouring pixel or not. In the former case, the gradient value of the current pixel remains unchanged otherwise, the value is updated to zero. 

##### 4. Hysteresis Thresholding:
This kind of thresholding will categorise the edges into strong, suppressed and weak. Based on a threshold, stron, they,g edges are included in the final detection map, suppressed edges are removed, and if the weak edges are connected with the strong edges, then they are also included in the final detection map.

```
import cv2
import numpy as np

image = cv2.imread("./images/candy.jpg")
image = cv2.resize(image, (640, 512))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), sigmaX=0, sigmaY=0)

x_sobel = cv2.Sobel(blur, cv2.CV_64F, dx=1, dy=0, ksize=5)
y_sobel = cv2.Sobel(blur, cv2.CV_64F, dx=0, dy=1, ksize=5)
xy_sobel = cv2.Sobel(blur, cv2.CV_64F, dx=1, dy=1, ksize=5)
candy = cv2.Canny(blur, 100, 200)

cv2.imshow("x_sobel", x_sobel)
cv2.imshow("y_sobel", y_sobel)
cv2.imshow("xy_sobel", xy_sobel)
cv2.imshow("candy", candy)
cv2.waitKey(0)
```

#### Drawing contours
We can draw the contours we find using edge detection.

```
import cv2

image = cv2.imread("./images/tennis.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_image = cv2.resize(image, (300, 300))
template = cv2.Canny(small_image, 175, 200)
contours, hierarchy = cv2.findContours(template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("template", template)
cv2.drawContours(small_image, contours, -1, (0, 255, 0), 3)
cv2.imshow("draw contours", small_image)
cv2.waitKey(0)
```

### Morphological Operations
Morphological operations are also some basic operations that can be performed by convolution of a kernel with an image. The transformations are performed on a binary image. Different kinds of operations are erosion, dilation, opening, closing and few others. 

#### Erosion
Erosion is helpful for noise removal. As the name implies, this operation erodes the boundaries of the foreground objects hence making them thinner. The idea is when a kernel convolves with the image, a particular pixel value of the image is kept one only if all pixels under the kernel are 1. Else, the pixel value becomes zero. 

#### Dilation
Dilation is usually performed after erosion to increase the object area. Since erosion removes noise and makes the boundaries thin, dilation does just the opposite, i.e. it makes the object boundaries thicker. When the convolution occurs, if there is at least one pixel under the kernel whose value is one then the pixel value of the original image becomes 1. 

#### Opening
This operation is simply erosion followed by dilation. It is helpful in removing noises while retaining the object boundaries. 

#### Closing
This operation is just the opposite of opening, i.e. dilation followed by erosion. It helps in closing small gaps in the objects of an image. 

```
import cv2
import numpy as np

image = cv2.imread("./images/tennis.jpg")
image = cv2.resize(image, (640, 512))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), sigmaX=0, sigmaY=0)

binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 5)
morph_binary = 255 - binary

#kernel
kernel = np.ones((5, 5), np.uint8)

#Erosion
erode = cv2.erode(morph_binary, kernel, iterations=1)

#Dilation
dilate = cv2.dilate(morph_binary, kernel, iterations=1)

#Opening
opening = cv2.morphologyEx(morph_binary, cv2.MORPH_OPEN, kernel)

#Closing
closing = cv2.morphologyEx(morph_binary, cv2.MORPH_CLOSE, kernel)

cv2.imshow("gray", gray)
cv2.imshow("morph_binary", morph_binary)
cv2.imshow("erode", erode)
cv2.imshow("dilate", dilate)
cv2.imshow("opening", opening)
cv2.imshow("closing", closing)
cv2.waitKey(0)
```

###