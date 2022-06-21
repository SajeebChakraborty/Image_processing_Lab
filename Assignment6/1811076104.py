import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

def salt_pepper_add(img):

    img_row, img_col = img.shape
    

    for i in range(random.randint(200, 8000)):

        cordinate_Y = random.randint(0, img_row-1)
        cordinate_x = random.randint(0, img_col-1)

        img[cordinate_Y][cordinate_x] = 255 

  
    for i in range(random.randint(200, 8000)):

        cordinate_Y = random.randint(0, img_row-1)
        cordinate_x = random.randint(0, img_col-1)

        img[cordinate_Y][cordinate_x] = 0

    return img 

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)


add_noice_img = salt_pepper_add(img)

plt.subplot(3, 2, 1)
plt.title('Salt Pepper added Noice Image')
plt.imshow(add_noice_img)

avg_kernel = np.ones((3, 3), np.float32)/9
    
processed_img = cv2.filter2D(src = add_noice_img, ddepth=-1, kernel = avg_kernel)

plt.subplot(3, 2, 2)
plt.title('Average Kernel')
plt.imshow(processed_img)

gaussian_filter = cv2.GaussianBlur(add_noice_img, (5,5), cv2.BORDER_DEFAULT )

plt.subplot(3, 2, 3)
plt.title('Gaussian kernel')
plt.imshow(gaussian_filter)



median_filter = cv2.medianBlur(add_noice_img, 3)

plt.subplot(3, 2, 4)
plt.title('Median Filter')
plt.imshow(median_filter)


plt.imshow(img)

plt.savefig('./Output Image.jpg')
plt.show()
