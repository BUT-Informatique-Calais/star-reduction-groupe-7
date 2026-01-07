import tkinter as tk
from tkinter import filedialog, messagebox

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Traitement de l'image FITS
def process_fits_image(path, fwhm, sigma_factor):
    """Charge une image FITS et applique une réduction d'étoiles"""

    # Lecture FITS
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)

    # Normalisation pour affichage
    img_norm = cv.normalize(data, None, 0, 255, cv.NORM_MINMAX)
    img_uint8 = img_norm.astype(np.uint8)

    original = img_uint8.astype(np.float32)

    # Statistiques de fond
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # Détection des étoiles
    finder = DAOStarFinder(
        fwhm=fwhm,
        threshold=sigma_factor * std
    )

    sources = finder(data - median)

    # Création du masque
    mask = np.zeros(img_uint8.shape, dtype=np.uint8)

    if sources is not None:
        for star in sources:
            x = int(star["xcentroid"])
            y = int(star["ycentroid"])
            cv.circle(mask, (x, y), 5, 255, -1)

    # Adoucissement du masque
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=1)
    mask = cv.GaussianBlur(mask, (15, 15), 0)

    mask_float = mask.astype(np.float32) / 255.0

    # Image sans étoiles (érosion)
    starless = cv.erode(img_uint8, kernel, iterations=3).astype(np.float32)

    # Fusion finale
    final = (mask_float * starless) + ((1 - mask_float) * original)
    final = np.clip(final, 0, 255).astype(np.uint8)

    return img_uint8, starless.astype(np.uint8), mask_float, final

# Interface graphique
class StarReductionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Star Reduction - Astrophotography")
        self.root.geometry("1200x850")

        self.fits_path = None

        self.build_controls()
        self.build_figure()

    # UI
    def build_controls(self):
        controls = tk.Frame(self.root)
        controls.pack(pady=10, fill="x")

        tk.Button(
            controls,
            text="Charger FITS",
            command=self.load_fits
        ).pack(side="left", padx=10)

        self.fwhm_var = tk.DoubleVar(value=3.0)
        self.slider(
            controls, "FWHM",
            self.fwhm_var, 1.0, 10.0, 0.1
        )

        self.sigma_var = tk.DoubleVar(value=1.0)
        self.slider(
            controls, "Seuil (σ)",
            self.sigma_var, 0.0, 10.0, 0.1
        )

        tk.Button(
            controls,
            text="Lancer",
            command=self.run
        ).pack(side="left", padx=20)

    def slider(self, parent, label, variable, vmin, vmax, step):
        frame = tk.Frame(parent)
        frame.pack(side="left", padx=20)

        tk.Label(frame, text=label).pack()

        tk.Entry(
            frame,
            textvariable=variable,
            width=6,
            justify="center"
        ).pack(pady=2)

        tk.Scale(
            frame,
            variable=variable,
            from_=vmin,
            to=vmax,
            resolution=step,
            orient="horizontal",
            length=200
        ).pack()

    def build_figure(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(expand=True)

    # Actions
    def load_fits(self):
        self.fits_path = filedialog.askopenfilename(
            filetypes=[("FITS files", "*.fits")]
        )

        if self.fits_path:
            messagebox.showinfo(
                "FITS chargé",
                os.path.basename(self.fits_path)
            )

    def run(self):
        if not self.fits_path:
            messagebox.showerror(
                "Erreur",
                "Aucun fichier FITS chargé"
            )
            return

        img, starless, mask, final = process_fits_image(
            self.fits_path,
            self.fwhm_var.get(),
            self.sigma_var.get()
        )

        titles = [
            "Original",
            "Starless",
            "Masque étoiles",
            "Final"
        ]

        images = [img, starless, mask, final]

        for ax, im, title in zip(self.axes.flat, images, titles):
            ax.clear()
            ax.imshow(im, cmap="gray")
            ax.set_title(title)
            ax.axis("off")

        self.canvas.draw()

        os.makedirs("results", exist_ok=True)
        cv.imwrite("results/original.png", img)
        cv.imwrite("results/starless.png", starless)
        cv.imwrite("results/final.png", final)

        messagebox.showinfo(
            "Terminé",
            "Images sauvegardées dans le dossier /results"
        )

# main
if __name__ == "__main__":
    root = tk.Tk()
    app = StarReductionApp(root)
    root.mainloop()
