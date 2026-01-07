from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# ===============================
# 1. Chargement FITS
# ===============================
fits_file = './examples/HorseHead.fits'
hdul = fits.open(fits_file)
data = hdul[0].data.astype(np.float32)
hdul.close()

# ===============================
# 2. Normalisation
# ===============================
img_norm = cv.normalize(data, None, 0, 255, cv.NORM_MINMAX)
img_uint8 = img_norm.astype(np.uint8)
Ioriginal = img_uint8.astype(np.float32)

# ===============================
# 3. Statistiques fond de ciel
# ===============================
mean, median, std = sigma_clipped_stats(data, sigma=3.0)

# ===============================
# 4. MASQUE DAO (petites étoiles)
# ===============================
daofind = DAOStarFinder(
    fwhm=3.0,
    threshold=1.0 * std
)

sources = daofind(data - median)

mask_dao = np.zeros(img_uint8.shape, dtype=np.uint8)

if sources is not None:
    for star in sources:
        x = int(star['xcentroid'])
        y = int(star['ycentroid'])
        cv.circle(mask_dao, (x, y), 5, 255, -1)

mask_dao = cv.dilate(mask_dao, np.ones((3,3),np.uint8), iterations=1)

# ===============================
# 5. MASQUE ADAPTATIF (grosses étoiles)
# ===============================
mask_adapt = cv.adaptiveThreshold(
    img_uint8, 255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    31, -5
)

mask_adapt = cv.morphologyEx(
    mask_adapt,
    cv.MORPH_OPEN,
    np.ones((5,5),np.uint8)
)

mask_adapt = cv.dilate(mask_adapt, np.ones((5,5),np.uint8), iterations=2)

# ===============================
# 6. FUSION DES MASQUES
# ===============================
mask_combined = cv.bitwise_or(mask_dao, mask_adapt)

# Flou des masques
mask_dao_blur = cv.GaussianBlur(mask_dao, (21,21), 0)
mask_adapt_blur = cv.GaussianBlur(mask_adapt, (21,21), 0)
mask_combined_blur = cv.GaussianBlur(mask_combined, (21,21), 0)

M_dao = mask_dao_blur.astype(np.float32) / 255.0
M_adapt = mask_adapt_blur.astype(np.float32) / 255.0
M_combined = mask_combined_blur.astype(np.float32) / 255.0

# ===============================
# 7. IMAGE ÉRODÉE (réduction étoiles)
# ===============================
kernel_erode = np.ones((3,3), np.uint8)
Ierode = cv.erode(img_uint8, kernel_erode, iterations=6).astype(np.float32)

# ===============================
# 8. IMAGES FINALES AVANT CORRECTION
# ===============================
Ifinal_dao = (M_dao * Ierode) + ((1 - M_dao) * Ioriginal)
Ifinal_dao = np.clip(Ifinal_dao, 0, 255).astype(np.uint8)

Ifinal_adapt = (M_adapt * Ierode) + ((1 - M_adapt) * Ioriginal)
Ifinal_adapt = np.clip(Ifinal_adapt, 0, 255).astype(np.uint8)

Ifinal_combined = (M_combined * Ierode) + ((1 - M_combined) * Ioriginal)
Ifinal_combined = np.clip(Ifinal_combined, 0, 255).astype(np.uint8)

# ===============================
# 9. CORRECTION DES TACHES NOIRES (SOLUTION 1)
# ===============================
# Lissage très léger
blur_final = cv.GaussianBlur(Ifinal_combined, (3,3), 0)

# Application UNIQUEMENT sous le masque combiné
Ifinal_corrected = (M_combined * blur_final.astype(np.float32)) + \
                   ((1 - M_combined) * Ifinal_combined.astype(np.float32))

Ifinal_corrected = np.clip(Ifinal_corrected, 0, 255).astype(np.uint8)

# ===============================
# 10. SAUVEGARDE
# ===============================
cv.imwrite('./results/a_original.png', img_uint8)
cv.imwrite('./results/b_dao_only.png', Ifinal_dao)
cv.imwrite('./results/c_adaptive_only.png', Ifinal_adapt)
cv.imwrite('./results/d_starless_final.png', Ifinal_corrected)

# ===============================
# 11. AFFICHAGE FINAL
# ===============================
plt.figure(figsize=(18, 10))

plt.subplot(2, 2, 1)
plt.title("(a) Image d'origine")
plt.imshow(img_uint8, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("(b) Réduction étoiles – DAO seul")
plt.imshow(Ifinal_dao, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("(c) Réduction étoiles – Adaptive threshold seul")
plt.imshow(Ifinal_adapt, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("(d) Image star-less finale (corrigée)")
plt.imshow(Ifinal_corrected, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
