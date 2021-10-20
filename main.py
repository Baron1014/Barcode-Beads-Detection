import cv2
import matplotlib.pyplot as plt

def plot(fig, figname):
    cv2.imwrite(f'./report/{figname}.jpg', fig)

def main(data_name):
    # 讀取圖檔
    img = cv2.imread(f'data/{data_name}.jpg', cv2.IMREAD_GRAYSCALE)

    # 模糊化
    blur = cv2.GaussianBlur(img,(5,5), 0)
    plot(blur, "smoothing")

    # Thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    plot(thresh, "GAUSSIAN Adaptive Thresholding After Blur")
    print(thresh.shape)


if __name__ == "__main__":
    main('W_A2_0_3_Binary-72')
