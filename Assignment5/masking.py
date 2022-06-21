import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    image_path = 'image2.jpg'
    rgb = plt.imread(image_path)
    w,h = rgb.shape[:2]

    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


    mask_setup = np.zeros((w,h), dtype = np.uint8)
    
    for i in range(50,130):

        for j in range(50,130):

            mask_setup[i][j] = 255

    outputImage = cv2.bitwise_and(grayscale, grayscale, mask = mask_setup)


    plt.subplot(4, 3, 4)
    plt.title('rgb')
    plt.imshow(rgb)

    plt.subplot(4, 3, 5)
    plt.title('grayscale')
    plt.imshow(grayscale, cmap = 'gray')

    plt.subplot(4, 3, 6)
    plt.title('masksetup')
    plt.imshow(mask_setup, cmap = 'gray')


    plt.subplot(4, 3, 10)
    plt.title('output_image')
    plt.imshow(outputImage, cmap = 'gray')

    plt.savefig('./binary_masking_output.jpg')
    plt.show()




if __name__ == '__main__':
    main()
    