import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def plot(fig, figname, report_path):
    cv2.imwrite(f'{report_path}/{figname}.jpg', fig)

# create convolution mask
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
    # Start at the second position in the image column
    for ii in range(math.ceil(mask_width/2), img_width-math.floor(mask_width/2)):
        # Start at the second position in the image row
        for ij in range(math.ceil(mask_length/2), img_length-math.floor(mask_length/2)):
            # muti-add
            for mi in range(mask_width):
                for mj in range(mask_length):
                    output_array[ii-math.ceil(mask_width/2), ij-math.ceil(mask_length/2)] += (mask[mi, mj] * data[ii+mi-1, ij+mj-1]) * (1/16)

    return output_array

def clear(delete_dir):
    print(f"start clear {delete_dir}")
    
    ds = list(os.listdir(delete_dir))
    for file in ds:
        if file.endswith('.jpg'):
            os.remove(delete_dir + '/' + file)
    
def main(data_name):
    report_dir = os.path.dirname(__file__) + f"/report/{data_name}"
    if os.path.exists(report_dir):
        # clear image
        clear(report_dir)
    else:
        os.mkdir(report_dir)

    print(f"start execute convolution on {data_name}")
    # read image
    img = cv2.imread(f'data/{data_name}.jpg', cv2.IMREAD_GRAYSCALE)

    # GaussianBlur
    blur = cv2.GaussianBlur(img,(5,5), 5)
    plot(blur, "smoothing", report_dir)

    # Thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 63, 14)
    plot(thresh, "GAUSSIAN_Adaptive_Thresholding_After_Blur", report_dir)

    # # Erosion
    # iter = 3
    # kernel = np.ones((3,3), np.uint8) 
    # erosion = cv2.erode(thresh, kernel, iterations = iter)
    # plot(erosion, "Erosion", report_dir)

    # # Dilation
    # dilation = cv2.dilate(erosion, kernel, iterations = iter)
    # plot(dilation, "Dilation", report_dir)

    # convolution
    print(f"img shape: {thresh.shape}")
    myconv = myCONV(thresh)
    print(f"after conv img shape: {myconv.shape}")
    plot(myconv, f"{data_name}_after_convolution", report_dir)

if __name__ == "__main__":
    main('W_A1_0_3')
    main('W_A2_0_3')
    main('W_A3_0_3')
    main('W_B2_3_3')
    main('W_B2_6_3')
    main('W_B4_0_3')
    
