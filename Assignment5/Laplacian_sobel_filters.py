import cv2
import numpy as np
from matplotlib import pyplot as plt

read_image = cv2.imread('image2.jpg',)


grayscale = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)


filer_img = cv2.GaussianBlur(grayscale,(3,3),0)


laplacian = cv2.Laplacian(filer_img,cv2.CV_64F)
sobelx = cv2.Sobel(filer_img,cv2.CV_64F,1,0,ksize=7) 
sobely = cv2.Sobel(filer_img,cv2.CV_64F,0,1,ksize=7) 




plt.subplot(4,2,1)
plt.imshow(filer_img,cmap = 'gray')
plt.title('Original image')
plt.subplot(4,2,1)
plt.imshow(filer_img,cmap = 'gray')
plt.title('Original image')

plt.subplot(4,2,2)
plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian filter')

plt.subplot(4,2,5)
plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X filter')

plt.subplot(4,2,6)
plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y filter')



plt.savefig('./Laplacian_sobel_output.jpg')
plt.show()
