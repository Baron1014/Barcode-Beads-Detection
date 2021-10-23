# 黑色定位點偵測 (Barcode Bead Detection)

## 讀取資料
- 利用OpenCV將檔案讀入
```
img = cv2.imread(f'data/{data_name}.jpg', cv2.IMREAD_GRAYSCALE)
```

## 資料處理
- Open CV Gaussian Blur
    - kernel size=5*5
```
blur = cv2.GaussianBlur(img,(5,5), 5)
```
    example:

- Open CV Gaussian Adaptive Threshold
```
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 63, 14)
```
    example:
    

## Convolution 設計
- mask設計
```
def get_mask():
    # 設定為Gaussian filter
    return np.array([[1,2,1], [2,4,2], [1,2,1]])
```
- convolution
```
def myCONV(data):
    # 取得 Gaussian filter
    mask = get_mask()
    # 影像寬及長
    img_width = data.shape[0]
    img_length= data.shape[1]
    # mask寬及長
    mask_width = mask.shape[0]
    mask_length= mask.shape[1]

    # init output array 
    output_array = np.zeros((img_width-math.ceil(mask_width/2), img_length-math.ceil(mask_length/2)))
    # 從影像列第二個位置開始
    for ii in range(math.ceil(mask_width/2), img_width-math.floor(mask_width/2)):
        # 從影像行第二個位置開始
        for ij in range(math.ceil(mask_length/2), img_length-math.floor(mask_length/2)):
            # 進行mask的乘加運算
            for mi in range(mask_width):
                for mj in range(mask_length):
                    # 最後乘以1/16的原因是因為，需使mask數值總和為1，達到守恆
                    output_array[ii-math.ceil(mask_width/2), ij-math.ceil(mask_length/2)] += (mask[mi, mj] * data[ii+mi-1, ij+mj-1]) * (1/16)

    return output_array
```
- result example