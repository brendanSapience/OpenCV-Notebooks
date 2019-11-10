
from matplotlib import pyplot as plt
import cv2
import pdf2image
import numpy as np
import re
import json

# Converts a PDF into images (png)
def convertPdfToImages(basePath, FileName):
    #oPath = createTempFolderForImages(basePath)
    #if(oPath != ""):
    images = pdf2image.convert_from_path(basePath+"/"+FileName, fmt='png')
    listofimages = []
    idx = 0
    for page in images:
        idx = idx + 1
        FileOut = basePath+"/"+str(idx)+'_out.png'
        page.save(FileOut, 'png')
        listofimages.append(FileOut)

    return listofimages

def showImg(image):
	cv2.imshow("img",image)
	cv2.waitKey(0) # waits until a key is pressed
	cv2.destroyAllWindows() # destroys the window showing image

pdf_path = "/Users/bren/Desktop/BOL LowQual.pdf"
img_path ="/Users/bren/Desktop/1_out.png"

#convertPdfToImages("/Users/bren/Desktop/", "BOL LowQual.pdf")
#exit()

img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

#img_gr=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
#ret,processedImage = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#processedImage = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
#processedImage = cv2.equalizeHist(img)
#ret,processedImage = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# Create kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

# Sharpen image
img2 = cv2.filter2D(img, -2, kernel)
processedImage = cv2.filter2D(img2, -2, kernel)
#blur = cv2.GaussianBlur(img,(5,5),0)
#processedImage = cv2.addWeighted(blur,1.5,img,-0.5,0)

#blur_img = cv2.blur(mask0,(3,3))
#ret,processedImage = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#ret,processedImage = cv2.threshold(blur_img,127,255,cv2.THRESH_BINARY)
#img_hsv[:,:,2] = np.zeros([img_hsv.shape[0], img_hsv.shape[1]])
#img_hsv[:,:,1] = np.zeros([img_hsv.shape[0], img_hsv.shape[1]])
cv2.imwrite('/Users/bren/Desktop/1_out_BIN.png',processedImage)
#cv2.imwrite('/Users/bren/Desktop/ME/IMGS_20191015102635/1_out_mask.png',mask0)
#showImg(processedImage)
