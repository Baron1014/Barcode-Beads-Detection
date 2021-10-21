import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def plot(fig, figname):
    cv2.imwrite(f'./report/{figname}.jpg', fig)

# 建立conv mask大小
def get_mask():
    return np.array([[1,2,1], [2,4,2], [1,2,1]])
    
def myCONV(data):
    mask = get_mask()
    img_width = data.shape[0]
    img_length= data.shape[1]
    mask_width = mask.shape[0]
    mask_length= mask.shape[1]

    # init output array 
    output_array = np.zeros((img_width-math.ceil(mask_width/2), img_length-math.ceil(mask_length/2)))
    # 從影像列第二個位置開始
    for ii in range(math.ceil(mask_width/2), img_width-math.floor(mask_width/2)):
        # 從影像行第二個位置開始
        for ij in range(math.ceil(mask_length/2), img_length-math.floor(mask_length/2)):
            # 進行muti-add
            for mi in range(mask_width):
                for mj in range(mask_length):
                    output_array[ii-math.ceil(mask_width/2), ij-math.ceil(mask_length/2)] += (mask[mi, mj] * data[ii+mi-1, ij+mj-1]) * (1/16)

    return output_array

def clear(dir_name):
    pwd = os.path.dirname(os.path.abspath(__file__))
    delete_dir = pwd + '/' + dir_name
    print(f"start clear {delete_dir}")
    
    ds = list(os.listdir(delete_dir))
    for file in ds:
        if file.endswith('.jpg'):
            os.remove(delete_dir + '/' + file)
    
def main(data_name):
    # 清除圖檔
    clear('report')

    print(f"start execute convolution on {data_name}")
    # 讀取圖檔
    img = cv2.imread(f'data/{data_name}.jpg', cv2.IMREAD_GRAYSCALE)

    # Erosion
    iter = 20
    kernel = np.ones((3,3), np.uint8) 
    erosion = cv2.erode(img, kernel, iterations = iter)
    plot(erosion, "Erosion")

    # Dilation
    dilation = cv2.dilate(erosion, kernel, iterations = iter)
    plot(dilation, "Dilation")

    # 模糊化
    blur = cv2.GaussianBlur(dilation,(5,5), 0)
    plot(blur, "smoothing")

    # Thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    plot(thresh, "GAUSSIAN Adaptive Thresholding After Blur")

    # convolution
    print(f"img shape: {thresh.shape}")
#    myconv = myCONV(thresh)
#    print(f"after conv img shape: {myconv.shape}")
#    plot(myconv, "After convolution")

if __name__ == "__main__":
    main('W_A2_0_3')
