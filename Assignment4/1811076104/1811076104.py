import cv2
import numpy as np
import matplotlib.pyplot as plt
   
      

def polt_histrogram(image):
      
    amount_intensity =[]
    value_intensity_of_img = []
      
    for k in range(256):

        value_intensity_of_img.append(k)
        tem_cnt = 0

        for i in range(p):

            for j in range(q):

                if image[i, j]== k:

                    tem_cnt = tem_cnt + 1

        amount_intensity.append(tem_cnt)
          
    return (value_intensity_of_img, amount_intensity)

orginal_image = plt.imread('H:\\Assignment\\Assignment4\\1811076104\\image.jpg')
plt.subplot(3,2,1)
plt.title('orginal Image')
plt.imshow(orginal_image) 


plt.subplot(3,2,2)
plt.xlabel('intensity')
plt.ylabel('pixels')
plt.title('Built-in histogram function')
plt.hist(orginal_image.ravel(), 256, [0, 256])


image = cv2.imread('H:\\Assignment\\Assignment4\\1811076104\\image.jpg', 0)
p, q = image.shape
inten, tem_cnt = polt_histrogram(image)
  
plt.subplot(3,2,5)
plt.stem(inten, tem_cnt, markerfmt='')
plt.xlabel('intensity')
plt.ylabel('pixels')
plt.title('Without Built-in histogram function')

plt.subplot(3,2,6)
plt.title('Neighborhood Processing')
grayscale = cv2.cvtColor(orginal_image, cv2.COLOR_RGB2GRAY)


kernel1 = np.array([[-1, -1, -3],
                    [0, 11, -1],
                    [-1, -1, -1]], dtype=np.int8)
processed_img = cv2.filter2D(grayscale, -1, kernel1)
plt.imshow(processed_img)
plt.savefig('H:\\Assignment\\Assignment4\\1811076104\\Output Image.jpg')    
plt.show()



