import matplotlib.pyplot as plt
import cv2
import numpy as np


def main():

    image_path = 'sajeeb_image.jpg'
    rgb_image = plt.imread(image_path)
    print(rgb_image.shape)                                                                                                                                  

   

    plt.subplot(3, 3, 1)
    plt.title('RGB Image')
    plt.imshow(rgb_image)


    kernel1 = np.array([[-1, -1, -3],
                    [0, 11, -1],
                    [-1, -1, -1]], dtype=np.int8)
   
    print(kernel1)
    
    processed_img = cv2.filter2D(src =rgb_image, ddepth=-1, kernel = kernel1)

    kernel2 = np.ones((3, 3), np.float32)*0.15

    plt.subplot(3, 3, 2)
    plt.title('Kerne-1')
    plt.imshow(processed_img, cmap = 'gray')

    print(kernel2)
   
    processed_img2 = cv2.filter2D(src =rgb_image, ddepth=-1, kernel = kernel2)

    plt.subplot(3, 3, 3)
    plt.title('Kernel-2')
    plt.imshow(processed_img2)

    kernel3 = np.ones((3, 3), np.float32)/4

    print(kernel3)

    processed_img3 = cv2.filter2D(src =rgb_image, ddepth=-1, kernel = kernel3)

    plt.subplot(3, 3, 4)
    plt.title('Kernel-3')
    plt.imshow(processed_img3)

    kernel4 = np.array([
        [15, -1, 23],
        [-1, -23, -1],
        [0, -1, -5]
    ], dtype=np.int8)

    print(kernel4)
    
    processed_img4 = cv2.filter2D(src =rgb_image, ddepth=-1, kernel = kernel4)

    plt.subplot(3, 3, 5)
    plt.title('Kernel-4')
    plt.imshow(processed_img4)

    kernel5 = np.array([
        [0, -1, 0],
        [-3, 9, -3],
        [0, -1, 0]
    ], dtype=np.int8)
    
    print(kernel5)

    processed_img5 = cv2.filter2D(src =rgb_image, ddepth=-1, kernel = kernel5)

    plt.subplot(3, 3, 6)
    plt.title('Kernel-5')
    plt.imshow(processed_img5)

    kernel6 = np.array([
        [0, -1, 0],
        [-1, 9, -1],
        [-5, -1, 0]
    ], dtype=np.int8)

    print(kernel6)
    
    processed_img6 = cv2.filter2D(src =rgb_image, ddepth=-1, kernel = kernel6)

    plt.subplot(3, 3, 7)
    plt.title('Kernel-6')
    plt.imshow(processed_img6)
    plt.savefig('Output Image.jpg')
    plt.show()
 
    


if __name__ == '__main__':
    main()