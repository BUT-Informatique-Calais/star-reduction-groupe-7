from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# ----------------------------
# Chargement de l'image FITS
# ----------------------------
fits_file = './examples/HorseHead.fits'
hdul = fits.open(fits_file)
data = hdul[0].data.astype(np.float32)
hdul.close()

# Normalisation [0,255]
img_norm = cv.normalize(data, None, 0, 255, cv.NORM_MINMAX)
img_uint8 = img_norm.astype(np.uint8)

# Image originale float
Ioriginal = img_uint8.astype(np.float32)

# ----------------------------
# ÉTAPE A : Détection des étoiles (DAOStarFinder)
# ----------------------------

# Statistiques du fond de ciel
mean, median, std = sigma_clipped_stats(data, sigma=3.0)

# Détecteur d'étoiles
daofind = DAOStarFinder(
    fwhm=3.0,           # taille moyenne des étoiles
    threshold=1. * std # seuil de détection
)

sources = daofind(data - median)

# Création du masque vide
mask = np.zeros(img_uint8.shape, dtype=np.uint8)

# Dessin des étoiles dans le masque
if sources is not None:
    for star in sources:
        x = int(star['xcentroid'])
        y = int(star['ycentroid'])
        cv.circle(mask, (x, y), 5, 255, -1)

# Nettoyage léger
kernel = np.ones((3, 3), np.uint8)
mask = cv.dilate(mask, kernel, iterations=1)

# Flou pour transitions douces
mask_blur = cv.GaussianBlur(mask, (15, 15), 0)

# Masque normalisé [0,1]
M = mask_blur.astype(np.float32) / 255.0

# ----------------------------
# ÉTAPE B : Réduction localisée
# ----------------------------

kernel_erode = np.ones((3, 3), np.uint8)
Ierode = cv.erode(img_uint8, kernel_erode, iterations=3).astype(np.float32)

# Interpolation officielle
Ifinal = (M * Ierode) + ((1 - M) * Ioriginal)
Ifinal = np.clip(Ifinal, 0, 255).astype(np.uint8)

# ----------------------------
# Sauvegardes
# ----------------------------
cv.imwrite('./results/mask_stars_daofinder.png', mask)
cv.imwrite('./results/final_star_reduction_daofinder.png', Ifinal)

# ----------------------------
# Affichage
# ----------------------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img_uint8, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Masque étoiles (DAOStarFinder)")
plt.imshow(M, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Image finale")
plt.imshow(Ifinal, cmap='gray')
plt.axis('off')

plt.show()
