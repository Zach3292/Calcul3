import numpy as np
import skimage as ski
import os

folder = "dataset/highway"

items = os.listdir(folder)
items = sorted(items)
for i in range(len(items)):
    if items[i] != "" and items[i][0] == ".":
        items[i] = "SYSTEMFILE"
items = list(filter("SYSTEMFILE".__ne__, items))

first = ski.io.imread(folder + "/" + items[0])

video = np.zeros((first.shape[0], first.shape[1], len(items)))

for k in range(len(items)):
    video[:,:,k] = ski.io.imread(folder + "/" + items[k])
    

background = np.zeros_like(first)

# Estimation de l'arrière-plan
for i in range(video.shape[0]):
    for j in range(video.shape[1]):
        background[i,j] = np.median(video[i,j,:])


ski.io.imsave("dataset/background.png", background)

seuil = 50

for k in range(len(items)):
    soustraction = np.zeros_like(first)
    soustraction = np.abs(video[:,:,k] - background)

    soustraction = soustraction > seuil
    soustraction = soustraction * 255
    
    ski.io.imsave("dataset/output/soustraction" + str(k) + ".png", soustraction.astype(np.uint8))