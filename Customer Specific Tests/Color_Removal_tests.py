
from matplotlib import pyplot as plt
import cv2
import numpy as np
import re
import json

def showImg(image):
	cv2.imshow("img",image)
	cv2.waitKey(0) # waits until a key is pressed
	cv2.destroyAllWindows() # destroys the window showing image

img_path = "/Users/bren/Desktop/ME/IMGS_20191015102635/1_out.png"

img=cv2.imread(img_path)

img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#showImg(img_hsv)
# lower mask (0-10)
lower_blue = np.array([0,0,223])
upper_blue = np.array([250,300,265])
mask0 = cv2.inRange(img_hsv, lower_blue, upper_blue)

kernel = np.ones((2, 2), np.uint8)
erosion = cv2.erode(mask0, kernel, iterations=1)

#[ 63  31 233] [ 53  21 193] [ 73  41 273]

#blur_img = cv2.blur(mask0,(3,3))
#ret,processedImage = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#ret,processedImage = cv2.threshold(blur_img,127,255,cv2.THRESH_BINARY)
#img_hsv[:,:,2] = np.zeros([img_hsv.shape[0], img_hsv.shape[1]])
#img_hsv[:,:,1] = np.zeros([img_hsv.shape[0], img_hsv.shape[1]])
cv2.imwrite('/Users/bren/Desktop/ME/IMGS_20191015102635/1_out_mask_erosion.png',erosion) 
cv2.imwrite('/Users/bren/Desktop/ME/IMGS_20191015102635/1_out_mask.png',mask0) 
showImg(erosion)



# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
#showImg(mask1)


# join my masks
mask = mask0+mask1

# set my output img to zero everywhere except my mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0

# or your HSV image, which I *believe* is what you want
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0

