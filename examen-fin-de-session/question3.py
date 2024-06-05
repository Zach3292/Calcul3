import numpy as np
import skimage as ski
import scipy.signal as scpsig

img = ski.io.imread("./dataset/fatbike.png")

ker1 = 0.25 * np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

ker2 = 0.25 * np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

result1 = scpsig.convolve2d(img, ker1)
result2 = scpsig.convolve2d(img,ker2)

results = np.sqrt(result1**2 + result2**2)

ski.io.imshow(results)
ski.io.show()