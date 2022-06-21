
from matplotlib.ft2font import FIXED_SIZES
import matplotlib.pyplot as plt
import cv2



def main():

    image_path = "H:\Assignment\Assignment1\image.jpg"

    print(image_path)

    rgb_image = plt.imread(image_path)

    print(rgb_image.shape)

    print(rgb_image)

    plt.figure(figsize=(25,25))

    plt.subplot(3,4,1)
    plt.title('RGB')
    plt.imshow(rgb_image)

    plt.subplot(3,4,2)
    plt.title('RGB Color Model Histogram')
    plt.hist(rgb_image.ravel(),256,[0,256])

    plt.subplot(3,4,3)
    plt.title('Grayscale_Channel')
    grayscale = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    plt.imshow(grayscale,cmap='gray')
    
    plt.subplot(3,4,4)
    plt.title('Grascale_Histogram')
    plt.hist(grayscale.ravel(),256,[0,256])
    
    plt.subplot(3,4,5)
    plt.title('Red _Channel')
    redChanel = rgb_image[:, :, 0]
    plt.imshow(redChanel, cmap='Reds')
    
    plt.subplot(3,4,6)
    plt.title('Red_Histogram')
    plt.hist(redChanel.ravel(),256,[0,256])
    
    plt.subplot(3,4,7)
    plt.title('Green_Channel')
    greenchanel = rgb_image[:, :, 1]
    plt.imshow(greenchanel,cmap='Greens')
    
    plt.subplot(3,4,8)
    plt.title('Green_Histogram')
    plt.hist(greenchanel.ravel(),256,[0,256]);
    
    plt.subplot(3,4,9)
    plt.title('Blue_Channel')
    bluechanel = rgb_image[:, :, 2]
    plt.imshow(bluechanel,cmap='Blues')
    
    plt.subplot(3,4,10)
    plt.title('Blue_Histogram')
    plt.hist(bluechanel.ravel(),256,[0,256])
    
    plt.subplot(3,4,11)
    plt.title('Binary_Channel')
    th, bin_image = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    plt.imshow(bin_image,cmap='binary')
    
    plt.subplot(3,4,12)
    plt.title('Binary_Histogram')
    plt.hist(bin_image.ravel(),256,[0,256]);

    
    plt.savefig('H:\Assignment\Assignment1\Output Image.jpg')
    plt.show()
    
 
    

if __name__ == '__main__':
	main()