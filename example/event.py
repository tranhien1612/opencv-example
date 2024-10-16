import cv2

refPt = []
def click_and_crop(event, x, y, flags, param):
	global refPt
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))
		cv2.rectangle(clone, refPt[0], refPt[1], (0, 255, 0), 2)
		
image = cv2.imread("../images/car_2.jpg")
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
	cv2.imshow("image", clone)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if cv2.waitKey(1) & 0xFF == ord('x'): #delete all rectangle
		clone = image.copy()

cv2.destroyAllWindows()
