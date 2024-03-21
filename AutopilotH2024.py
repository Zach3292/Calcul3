from aem import con
import numpy as np
import skimage as ski
import skvideo.io
import matplotlib.pyplot as plt
import cv2 as cv

np.float = np.float64
np.int = np.int_

def region(image):
    height, width = image.shape
    triangle = np.array([
                        [(0, 900), (575, 450), (width, 875)]
                        ])
    
    mask = np.zeros_like(image)
    
    mask = cv.fillPoly(mask, triangle, 255)
    mask = cv.bitwise_and(image, mask)
    return mask

xleft = np.linspace(300, 600, 1000)
yleft = xleft
xright = np.linspace(680, 980, 1000)
yright = xright

# Chargement de la vidéo en mémoire
video = skvideo.io.vread("dataset/autopilot/stecatherine.mp4", as_grey=True)

# Traitement de chaque image composant la vidéo
for it in range(250, 500):
    fr = video[it,:,:,:]
    img = np.array(fr[:,:,0])
    
    # Affichage du numéro l'image traitée
    print(it)

    # Application d'un filtre gaussien
    gaussian = cv.GaussianBlur(img,(5,5),0)
    
    # Détection des contours avec OpenCV
    contour = cv.Canny(gaussian, 10, 150, L2gradient=True)
    
    # Application d'un masque pour ne garder que les lignes de la route dans la région voulue
    C = region(contour)


    # Pour débugger, décommentez les lignes suivantes

    # triangle = np.array([
    #                     [(0, 900), (575, 450), (1280, 875)]
    #                     ])
    # B = np.copy(contour)
    # mask = cv.fillPoly(B, triangle, 255)

    mask = C

    # Données de gauche
    Cleft = np.copy(C)
    Cleft[0:960, 640:1280] = 0

    # Prédiction à gauche (Vous ne devriez pas modifier cette section)
    leftdata = np.fliplr(np.argwhere(Cleft > 0))
    model_robust, inliers = ski.measure.ransac(leftdata, ski.measure.LineModelND, min_samples=50, residual_threshold=2, max_trials=1000)
    leftpos = model_robust.predict_x([750])
    yleft = model_robust.predict_y(xleft)
    
    # Données de droite
    Cright = np.copy(C)
    Cright[0:960, 0:640] = 0

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
    