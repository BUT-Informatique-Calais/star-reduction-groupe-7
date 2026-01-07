import argparse
import os
import cv2 as cv
import numpy as np

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


# ======================================================
# TRAITEMENT D'UNE IMAGE FITS (MÊME ALGO QUE L'APP)
# ======================================================
def process_fits_array(data, fwhm, sigma_factor):
    img = cv.normalize(data, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    original = img.astype(np.float32)

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # DAO – petites étoiles
    finder = DAOStarFinder(fwhm=fwhm, threshold=sigma_factor * std)
    sources = finder(data - median)

    mask_dao = np.zeros(img.shape, np.uint8)
    if sources is not None:
        for s in sources:
            x, y = int(s["xcentroid"]), int(s["ycentroid"])
            cv.circle(mask_dao, (x, y), 5, 255, -1)

    mask_dao = cv.dilate(mask_dao, np.ones((3, 3), np.uint8), iterations=1)

    # Adaptive – grosses étoiles
    mask_adapt = cv.adaptiveThreshold(
        img, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        31, -5
    )
    mask_adapt = cv.morphologyEx(mask_adapt, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask_adapt = cv.dilate(mask_adapt, np.ones((5, 5), np.uint8), iterations=2)

    # Masque combiné
    mask = cv.bitwise_or(mask_dao, mask_adapt)
    mask_blur = cv.GaussianBlur(mask, (15, 15), 0)
    M = mask_blur.astype(np.float32) / 255.0

    # Image starless brute
    starless = cv.erode(img, np.ones((3, 3), np.uint8), iterations=3).astype(np.float32)

    # Fusion
    fused = M * starless + (1 - M) * original
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    # Correction des taches noires
    blur_fix = cv.GaussianBlur(fused, (3, 3), 0)
    final = np.clip(
        M * blur_fix.astype(np.float32) +
        (1 - M) * fused.astype(np.float32),
        0, 255
    ).astype(np.uint8)

    return final


# ======================================================
# MODE BATCH
# ======================================================
def batch_process(input_dir, output_dir, fwhm, sigma):
    os.makedirs(output_dir, exist_ok=True)

    fits_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".fits")
    ])

    print(f"📂 {len(fits_files)} fichiers FITS détectés")

    for i, filename in enumerate(fits_files, start=1):
        path = os.path.join(input_dir, filename)

        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)

        final = process_fits_array(data, fwhm, sigma)

        out_name = filename.replace(".fits", "_starless.png")
        out_path = os.path.join(output_dir, out_name)

        cv.imwrite(out_path, final)

        print(f"[{i}/{len(fits_files)}] ✔ {out_name}")

    print("✅ Traitement batch terminé")


# ======================================================
# MAIN CLI
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch Star Reduction – Catalogue FITS"
    )

    parser.add_argument("--input", required=True, help="Dossier contenant les FITS")
    parser.add_argument("--output", required=True, help="Dossier de sortie")
    parser.add_argument("--fwhm", type=float, default=3.0, help="FWHM DAO")
    parser.add_argument("--sigma", type=float, default=1.0, help="Seuil sigma")

    args = parser.parse_args()

    batch_process(
        args.input,
        args.output,
        args.fwhm,
        args.sigma
    )
