import numpy as np
import matplotlib.pyplot as plt
import skimage.io as ski
from scipy.signal import fftconvolve, convolve
import scipy.fft as scpfft
import time

filt = 1/16*np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

img = ski.imread('dataset/fatbike.png')

img_freq = scpfft.fft2(img)
filt_freq = scpfft.fft2(filt, s = img.shape)

img_filt_freq = img_freq * filt_freq
img_filt = scpfft.ifft2(img_filt_freq).real


plt.imshow(img, cmap = 'gray')
plt.axis('off')
plt.show()

plt.imshow(img_filt, cmap = 'gray')
plt.axis('off')
plt.show()