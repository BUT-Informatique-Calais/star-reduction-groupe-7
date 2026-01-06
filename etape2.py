from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

fits_file = './examples/HorseHead.fits'
hdul = fits.open(fits_file)
data = hdul[0].data.astype(np.float32)
hdul.close()

# Normalisation [0,255]
img_norm = cv.normalize(data, None, 0, 255, cv.NORM_MINMAX)
img_uint8 = img_norm.astype(np.uint8)

# Image originale float
Ioriginal = img_uint8.astype(np.float32)

# Statistiques du fond de ciel
mean, median, std = sigma_clipped_stats(data, sigma=3.0)

# Détecteur d'étoiles
daofind = DAOStarFinder(
    fwhm=3.0,           # taille moyenne des étoiles
    threshold=1.0 * std # seuil de détection
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

# Image érodée (réduction des étoiles)
kernel_erode = np.ones((3, 3), np.uint8)
Ierode = cv.erode(img_uint8, kernel_erode, iterations=3).astype(np.float32)

# Image recomposée finale
Ifinal = (M * Ierode) + ((1 - M) * Ioriginal)
Ifinal = np.clip(Ifinal, 0, 255).astype(np.uint8)

# (a) Étoiles fortement visibles
stars_only = (M * Ioriginal)
stars_only = np.clip(stars_only, 0, 255).astype(np.uint8)

# (b) Image star-less
starless = Ierode
starless = np.clip(starless, 0, 255).astype(np.uint8)

# (c) Masque d’étoiles
mask_out = mask_blur

# (d) Image recomposée
final_out = Ifinal

# Sauvegardes 
cv.imwrite('./results/original.png', img_uint8)
cv.imwrite('./results/b_starless.png', starless)
cv.imwrite('./results/c_star_mask.png', mask_out)
cv.imwrite('./results/d_final_recomposed.png', final_out)

# Affichage
plt.figure(figsize=(18, 10))

plt.subplot(2, 2, 1)
plt.title("(a) Image d'origine")
plt.imshow(img_uint8, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("(b) Image star-less")
plt.imshow(starless, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("(c) Masque d’étoiles")
plt.imshow(M, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("(d) Image recomposée finale")
plt.imshow(final_out, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
