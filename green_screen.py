import cv2
import numpy as np
import skimage.exposure

# load image
img = cv2.imread('GreenScreen.jpg')

# convert to LAB
lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

# extract A channel
A = lab[:,:,1]

# threshold A channel
thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# blur threshold image to remove noise
blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT)

# stretch so that 255 -> 255 and 127.5 -> 0
mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255)).astype(np.uint8)

# morphological closing to remove noise (nvidia sticker)
kernel = np.ones((11,11),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# invert the mask to keep the green color and remove everything else
mask_inv = np.logical_not(mask).astype(np.uint8)

# apply the mask
result = cv2.bitwise_and(img, img, mask=mask_inv)

# display the result
cv2.imshow('result', result)

#save the result
cv2.imwrite('result.png',result)

cv2.waitKey(0)
cv2.destroyAllWindows()
