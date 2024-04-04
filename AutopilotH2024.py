import numpy as np
import skimage as ski
import skvideo.io
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.stats import norm
from scipy.signal import fftconvolve

np.float = np.float64
np.int = np.int_

def gradient_image(img):
    G = np.zeros((img.shape[0], img.shape[1], 2))
    
    # Méthode avec Sobel filters
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    G[:,:,0] = fftconvolve(img, kernel_x, mode='same')
    G[:,:,1] = fftconvolve(img, kernel_y, mode='same')

    N = np.hypot(G[:,:,0], G[:,:,1])
    N = N / N.max() * 255
    theta = np.arctan2(G[:,:,1], G[:,:,0])
    return N, theta

def non_maximum_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            
            q = 255
            r = 255

            # Split en 8 directions
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = G[i, j+1]
                r = G[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = G[i+1, j]
                r = G[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]
            
            if (G[i,j] >= q) and (G[i,j] >= r):
                Z[i,j] = G[i,j]
            else:
                Z[i,j] = 0
    return Z

def hysteresis_double_thresholding(img, T1, T2):
    M, N = img.shape
    resulant = np.zeros((M,N), dtype=np.int32)

    fort = np.int32(255)
    faible = np.int32(25)

    fort_i, fort_j = np.where(img >= T2)
    faible_i, faible_j = np.where((img <= T2) & (img >= T1))

    resulant[fort_i, fort_j] = fort
    resulant[faible_i, faible_j] = faible

    for i in range(1, M-1):
        for j in range(1, N-1):
            if resulant[i,j] == faible:
                if ((resulant[i+1, j-1] == fort) or (resulant[i+1, j] == fort) or (resulant[i+1, j+1] == fort)
                    or (resulant[i, j-1] == fort) or (resulant[i, j+1] == fort)
                    or (resulant[i-1, j-1] == fort) or (resulant[i-1, j] == fort) or (resulant[i-1, j+1] == fort)
                    ):
                    resulant[i, j] = fort
                else:
                    resulant[i, j] = 0

    return resulant

def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = norm.pdf(kernel_1D[i], loc=0, scale=sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D

def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=np.sqrt(kernel_size))
    return fftconvolve(image, kernel, mode='same')

def region(image, polygon):

    mask = np.zeros_like(image)

    mask = cv.fillPoly(mask, polygon, 255)
    mask = cv.bitwise_and(image, mask)
    return mask

xleft = np.linspace(300, 600, 1000)
yleft = xleft
xright = np.linspace(680, 980, 1000)
yright = xright

# Chargement de la vidéo en mémoire
video = skvideo.io.vread("dataset/autopilot/stecatherine.mp4", as_grey=True)

# Traitement de chaque image composant la vidéo
for it in range(400, 500):
    fr = video[it,:,:,:]
    img = np.array(fr[:,:,0])

    
    
    # Affichage du numéro l'image traitée
    print(it)
    polygon1 = np.array([
                        [(100, 900), (100, 500), (1280, 500), (1280, 900)]
                        ])
    # Application d'un masque pour réduire le temps de compilation
    img_crop = region(img, polygon1)
    # Application d'un filtre gaussian pour réduire le bruit
    gaussian = gaussian_blur(img_crop, 10)
    # Calcul du module du gradient et de sa norme
    grad, theta = gradient_image(gaussian)
    # Suppression des non-maxima locaux
    max = non_maximum_suppression(grad, theta)
    # Détection des contours avec un double seuil
    contour = hysteresis_double_thresholding(max, 9, 15)

    polygon2 = np.array([
                        [(150, 750), (500, 520), (660, 500), (1000, 700), (1280, 800)]
                        ])
    # Application d'un masque pour ne garder que les lignes de la route dans la région voulue
    C = region(contour, polygon2)

    mask = C

    # Pour débugger, décommentez les lignes suivantes
    # B = np.copy(contour)
    # mask = cv.fillPoly(B, polygon2, 255)

    """

    Pourquoi mon code produit-il un meilleur résultat que la référence?
    
    1. J'ai utilisé un premier masque pour réduire le temps de compilation
    2. J'ai utilisé un filtre gaussien pour réduire le bruit au lieu d'un filtre moyenneur ce qui a permis de mieux détecter les lignes
    3. J'ai implémemté l'agorithme de détection de contour Canny qui implique les étapes suivantes:
        - Calcul du gradient de l'image
            - Utilise la convolution avec un filtre Sobel pour calculer le gradient de l'image réduisant le temps de calcul
        - Suppression des non-maxima locaux
            - Cette étape permet de réduire le nombre de points de contours en ne gardant que les points de contours les plus forts créant un coutour plus précis
        - Détection des contours avec un double seuil
            - Cette étape permet de ne garder que les points de contours les plus forts et d'éliminer les points de contours isolés des autres réduisant ainsi le bruit
    4. J'ai utilisé un deuxième masque pour ne garder que les lignes de la route dans la région voulue éliminant ainsi les lignes parasites lors du calcul avec l'algorithme RANSAC
    
    Le résultat est une détection de lignes plus épurée et plus précise que la référence
    
    """

    
    # Données de gauche (modifiées pour améliorer la performance de l'algorithme)
    Cleft = np.copy(C)
    Cleft[0:960, 575:1280] = 0

    # Prédiction à gauche (Vous ne devriez pas modifier cette section)
    leftdata = np.fliplr(np.argwhere(Cleft > 0))
    model_robust, inliers = ski.measure.ransac(leftdata, ski.measure.LineModelND, min_samples=50, residual_threshold=2, max_trials=1000)
    leftpos = model_robust.predict_x([750])
    yleft = model_robust.predict_y(xleft)
    
    # Données de droite (modifiées pour améliorer la performance de l'algorithme)
    Cright = np.copy(C)
    Cright[0:960, 0:575] = 0

    # Prédiction à droite (Vous ne devriez pas modifier cette section)
    rightdata = np.fliplr(np.argwhere(Cright > 0))
    model_robust, inliers = ski.measure.ransac(rightdata, ski.measure.LineModelND, min_samples=50, residual_threshold=2, max_trials=1000)
    rightpos = model_robust.predict_x([750])
    yright = model_robust.predict_y(xright)  

    # Affichage (Vous ne devriez pas modifier cette section)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img, cmap="gray")
    ax[0].set_title('Image')

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title('Lignes')

    ax[2].imshow(C * 0)
    ax[2].scatter(leftdata[:,0], leftdata[:,1], s=1, c='b')
    ax[2].scatter(rightdata[:,0], rightdata[:,1], s=1, c='g')
    
    ax[2].plot(xleft, yleft, c='r')
    ax[2].plot(xright, yright, c='r')

    ax[2].plot(np.mean([leftpos, rightpos]), 750, 'co')
    ax[2].plot(rightpos, 750, 'yo')
    ax[2].plot(leftpos, 750, 'yo')

    ax[2].set_xlim((0, img.shape[1]))
    ax[2].set_ylim((img.shape[0], 0))
    ax[2].set_title('Autopilot')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.savefig('dataset/autopilot/output/Frame%i.png' % it)
    plt.close()