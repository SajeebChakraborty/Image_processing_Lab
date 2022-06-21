import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path = './image.jpg'
    rgb = plt.imread(img_path)

    w,h =rgb.shape[:2]

    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    th, binary = cv2.threshold(grayscale, 50, 255, cv2.THRESH_BINARY)

    bit1 = np.zeros((w,h), dtype = np.int8)
    bit2 = np.zeros((w,h), dtype = np.int8)
    bit3 = np.zeros((w,h), dtype = np.int8)
    bit4 = np.zeros((w,h), dtype = np.int8)
    bit5 = np.zeros((w,h), dtype = np.int8)
    bit6 = np.zeros((w,h), dtype = np.int8)
    bit7 = np.zeros((w,h), dtype = np.int8)
    bit8 = np.zeros((w,h), dtype = np.int8)
    bit_slice_image = [bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8]

    for_bit_operation =[1,2,4,8,16,32,64,128]

    for i in range(for_bit_operation.__len__()):
        for j in range(w):
            for k in range(h):
                if (grayscale[j][k] &for_bit_operation[i]):
                   bit_slice_image[i][j][k] = 255
                else:
                   bit_slice_image[i][j][k] = 0
    
    
    plt.subplot(6, 2, 1)
    plt.title('rgb')
    plt.imshow(rgb, cmap = 'gray')

    plt.subplot(6, 2, 2)
    plt.title('grayscale')
    plt.imshow(grayscale, cmap = 'gray')

    plt.subplot(6, 2, 3)
    plt.title('binary')
    plt.imshow(binary, cmap = 'gray')
    
    plt.subplot(6, 2, 4)
    plt.title('Eight Bit slicing')
    plt.imshow(bit8, cmap = 'gray')
    
    plt.subplot(6, 2, 5)
    plt.title('Seven Bit slicing')
    plt.imshow(bit7, cmap = 'gray')

    plt.subplot(6, 2, 6)
    plt.title('Six Bit slicing')
    plt.imshow(bit6, cmap = 'gray')

    plt.subplot(6, 2, 7)
    plt.title('Fifth Bit slicing')
    plt.imshow(bit5, cmap = 'gray')

    plt.subplot(6, 2, 8)
    plt.title('Four Bit slicing')
    plt.imshow(bit4, cmap = 'gray')

    plt.subplot(6, 2, 9)
    plt.title('Three Bit slicing')
    plt.imshow(bit3, cmap = 'gray')


    plt.subplot(6, 2, 10)
    plt.title('Two Bit slicing')
    plt.imshow(bit2, cmap = 'gray')

    plt.subplot(6, 2, 11)
    plt.title('One Bit slicing')
    plt.imshow(bit1, cmap = 'gray')
    plt.savefig('./bit_slicing_output.jpg')
    plt.show()


if __name__ == '__main__':
    main()
    