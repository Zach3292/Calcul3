import numpy as np
import skimage.io as ski
import matplotlib.pyplot as plt

def gradient_image(img):
    G = np.zeros((img.shape[0], img.shape[1], 2))

    # Ajouter du code ici
    dadx = np.gradient(img, axis = 1)
    dady = np.gradient(img, axis = 0)
    G[:, :, 0] = dadx
    G[:, :, 1] = dady

    return G

def norme_gradient(grad):
    N = np.zeros(grad.shape[0:2])

    # Ajouter du code ici
    N = np.sqrt(grad[:,:,0]**2 + grad[:,:,1]**2)

    return N

def image_contour(N, s):
    C = np.zeros_like(N)

    # Ajouter du code ici
    C = N > s

    return C

def moyenne_image(img, m):
    Im = np.zeros_like(img)

    # Ajouter du code ici
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            debuti = max(0, i - m)
            fini = min(img.shape[0]-1, i + m)
            debutj = max(0, j - m)
            finj = min(img.shape[1]-1, j + m)
            Im[i, j] = np.mean(img[debuti:fini+1, debutj:finj+1])
    

    return Im

def mediane_image(img, m):
    Im = np.zeros_like(img)

    # Ajouter du code ici
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            debuti = max(0, i - m)
            fini = min(img.shape[0]-1, i + m)
            debutj = max(0, j - m)
            finj = min(img.shape[1]-1, j + m)
            Im[i, j] = np.median(img[debuti:fini+1, debutj:finj+1])

    return Im

image = ski.imread("dataset/fatbike.png")
ski.imshow(image)
ski.show()

# Paramètres
moyenne_m = 3
mediane_m = 3
seuil_s = 15

# Détection de contours sur l'image originale

print("Calcul du gradient...")
G = gradient_image(image)

ski.imshow(G[:,:,0].astype(np.uint8) + 128, cmap=plt.cm.gray)
ski.show()
ski.imshow(G[:,:,1].astype(np.uint8) + 128, cmap=plt.cm.gray)
ski.show()

print("Calcul de la norme du gradient...")
N = norme_gradient(G)

ski.imshow((255*N[:,:]/N.max()).astype(np.uint8), cmap=plt.cm.gray)
ski.show()

print("Application du seuil...")
C = image_contour(N, seuil_s)

ski.imshow(C.astype(np.uint8), cmap=plt.cm.gray)
ski.show()

# Détection de contours sur l'image moyenne

print("Calcul de l'image moyenne...")
image_moy = moyenne_image(image, moyenne_m)

ski.imshow(image_moy.astype(np.uint8), cmap=plt.cm.gray)
ski.show()

print("Calcul du gradient...")
G = gradient_image(image_moy)

ski.imshow(G[:,:,0].astype(np.uint8) + 128, cmap=plt.cm.gray)
ski.show()
ski.imshow(G[:,:,1].astype(np.uint8) + 128, cmap=plt.cm.gray)
ski.show()

print("Calcul de la norme du gradient...")
N = norme_gradient(G)

ski.imshow((255*N[:,:]/N.max()).astype(np.uint8), cmap=plt.cm.gray)
ski.show()

print("Application du seuil...")
C = image_contour(N, seuil_s)

ski.imshow(C.astype(np.uint8), cmap=plt.cm.gray)
ski.show()

# Détection de contours sur l'image médiane

print("Calcul de l'image médiane...")
image_med = mediane_image(image, mediane_m)

ski.imshow(image_moy.astype(np.uint8), cmap=plt.cm.gray)
ski.show()

print("Calcul du gradient...")
G = gradient_image(image_med)

ski.imshow(G[:,:,0].astype(np.uint8) + 128, cmap=plt.cm.gray)
ski.show()
ski.imshow(G[:,:,1].astype(np.uint8) + 128, cmap=plt.cm.gray)
ski.show()

print("Calcul de la norme du gradient...")
N = norme_gradient(G)

ski.imshow((255*N[:,:]/N.max()).astype(np.uint8), cmap=plt.cm.gray)
ski.show()

print("Application du seuil...")
C = image_contour(N, seuil_s)

ski.imshow(C.astype(np.uint8), cmap=plt.cm.gray)
ski.show()