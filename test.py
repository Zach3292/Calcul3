import numpy as np
import skimage.io as ski

img = ski.imread("dataset/jayz.jpg")

m = 2

img_output = np.zeros_like(img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):

        imin = max(0,i-m)
        imax = min(img.shape[0], i+m+1)
        jmin = max(0,j-m)
        jmax = min(img.shape[1], j+m+1)

        A = img[imin:imax+1, jmin:jmax+1]

        B = A.flatten()
        B.sort()
        position = (B.shape[0] - 1) / 2
        d = B[int(position)]
        img_output[i,j] = d

ski.imshow(img_output.astype(np.uint8))
ski.show()

