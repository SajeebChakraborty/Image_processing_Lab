import matplotlib.pyplot as plt
import cv2 
import numpy as np


image = plt.imread('image.jpg')
convert2dimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

r,c = convert2dimage.shape
s = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

T1=70
T2=180

for i in range(r):
    for j in range(c):
        if convert2dimage[i][j]>=T1 and convert2dimage[i][j]<=T2:
            s[i][j]=100
        else:
            s[i][j]=10
        
plt.subplot(2,2,1)
plt.title('First-condition')
plt.imshow(s,cmap='gray')
s1 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

for i in range(r):
    for j in range(c):
        if convert2dimage[i][j]>=T1 and convert2dimage[i][j]<=T2:
            s1[i][j]=100
        


plt.subplot(2,2,2)
plt.title('Second-condition')
plt.imshow(s1,cmap='gray')


s2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
k = 2
# here c=k
for i in range(r):
    for j in range(c):
        s2[i][j]=k*np.log(1+convert2dimage[i][j])

plt.subplot(2,2,3)
plt.imshow(s2,cmap='gray')
plt.title('third-condition')

s3 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
ep = 0.0000001
p = 3

for i in range(r):
    for j in range(c):
        s2[i][j]=k*(convert2dimage[i][j]+ep)**p

plt.subplot(2,2,4)
plt.title('fourth-condition')
plt.imshow(s3,cmap='gray')
plt.savefig('Output Image.jpg')
plt.show()
