## Content
- [Project Description](#project-description)
- [Read Data](#read-data)
- [Data Preprocessing](#data-preprocessing)
- [Convolution Design](#convolution-design)

## Project Description
The goal is to segment the Barcode Bead and segment the black positioning bar used in the Barcode Bead.

## Read Data
- Read the file using OpenCV
```
img = cv2.imread(f'data/{data_name}.jpg', cv2.IMREAD_GRAYSCALE)
```

## Data Preprocessing
- Open CV Gaussian Blur
    - kernel size=5*5
```
blur = cv2.GaussianBlur(img,(5,5), 5)
```
example:
![](./report/W_B4_0_3/smoothing.jpg?raw=true)

- Open CV Gaussian Adaptive Threshold
```
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 63, 14)
```
example:
![](./report/W_B4_0_3/GAUSSIAN_Adaptive_Thresholding_After_Blur.jpg?raw=true)

## Convolution Design
- mask design
```
def get_mask():
    # setting aussian filter
    return np.array([[1,2,1], [2,4,2], [1,2,1]])
```
- convolution
```
def myCONV(data):
    # get Gaussian filter
    mask = get_mask()
    # Image width and length
    img_width = data.shape[0]
    img_length= data.shape[1]
    # Mask width and length
    mask_width = mask.shape[0]
    mask_length= mask.shape[1]

    # init output array 
    output_array = np.zeros((img_width-math.ceil(mask_width/2), img_length-math.ceil(mask_length/2)))
    # Start at the second position in the image row
    for ii in range(math.ceil(mask_width/2), img_width-math.floor(mask_width/2)):
        # Start at the second position in the image column
        for ij in range(math.ceil(mask_length/2), img_length-math.floor(mask_length/2)):
            # multi-add
            for mi in range(mask_width):
                for mj in range(mask_length):
                    # The reason for multiplying by 1/16 at the end is because the sum of the mask values ​​needs to be 1 to achieve conservation
                    output_array[ii-math.ceil(mask_width/2), ij-math.ceil(mask_length/2)] += (mask[mi, mj] * data[ii+mi-1, ij+mj-1]) * (1/16)

    return output_array
```
- result example
![](./report/W_B4_0_3/W_B4_0_3_after_convolution.jpg?raw=true)
