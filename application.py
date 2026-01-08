import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import ttkbootstrap as tb
from ttkbootstrap.constants import *

from tkinter import filedialog, messagebox

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# UTILS
def get_base_dir():
    """Compatible exe / script"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)


BASE_DIR = get_base_dir()
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# traitement image 

def process_fits_image(path, fwhm, sigma_factor):
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)

    # Normalisation
    img_norm = cv.normalize(data, None, 0, 255, cv.NORM_MINMAX)
    img = img_norm.astype(np.uint8)
    original = img.astype(np.float32)

    # Statistiques
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # Détection étoiles
    finder = DAOStarFinder(fwhm=fwhm, threshold=sigma_factor * std)
    sources = finder(data - median)

    # Masque
    mask = np.zeros(img.shape, dtype=np.uint8)

    if sources is not None:
        for star in sources:
            x, y = int(star["xcentroid"]), int(star["ycentroid"])
            cv.circle(mask, (x, y), 5, 255, -1)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=1)
    mask = cv.GaussianBlur(mask, (15, 15), 0)

    mask_f = mask.astype(np.float32) / 255.0
    starless = cv.erode(img, kernel, iterations=3).astype(np.float32)

    final = (mask_f * starless) + ((1 - mask_f) * original)
    final = np.clip(final, 0, 255).astype(np.uint8)

    return img, starless.astype(np.uint8), mask_f, final


# interface utilisateur 
class StarReductionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Star Reduction – Astrophotography")
        self.root.geometry("1300x900")

        self.fits_path = None
        self.images = {}
        self.blinking = False
        self.blink_index = 0
        self.blink_images = []

        self.update_timer = None
        self.update_delay = 300

        self.build_controls()
        self.build_figure()

    # ---------- UI ----------
    def build_controls(self):
        controls = tb.Frame(self.root, padding=10)
        controls.pack(fill=X)

        tb.Button(
            controls, text="Charger FITS",
            command=self.load_fits
        ).pack(side=LEFT, padx=10)

        self.fwhm_var = tb.DoubleVar(value=3.0)
        self.create_slider(controls, "FWHM", self.fwhm_var, 1, 10, 0.1)

        self.sigma_var = tb.DoubleVar(value=1.0)
        self.create_slider(controls, "Seuil σ", self.sigma_var, 0, 10, 0.1)

        tb.Button(
            controls, text="Lancer",
            command=self.run
        ).pack(side=LEFT, padx=15)

        tb.Button(
            controls, text="Avant / Après",
            command=self.toggle_blink
        ).pack(side=LEFT, padx=10)

        self.fwhm_var.trace_add("write", lambda *_: self.schedule_update())
        self.sigma_var.trace_add("write", lambda *_: self.schedule_update())

    def create_slider(self, parent, label, var, vmin, vmax, step):
        frame = tb.Frame(parent)
        frame.pack(side=LEFT, padx=20)

        tb.Label(frame, text=label).pack()
        tb.Entry(frame, textvariable=var, width=6, justify=CENTER).pack(pady=2)
        tb.Scale(
            frame, variable=var,
            from_=vmin, to=vmax,
            orient=HORIZONTAL,
            length=180
        ).pack()

    def build_figure(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(expand=True, fill=BOTH)

        self.ax_images = []
        for ax in self.axes.flat:
            im = ax.imshow(np.zeros((100, 100)), cmap="gray")
            ax.axis("off")
            self.ax_images.append(im)

    # ---------- ACTIONS ----------
    def load_fits(self):
        self.fits_path = filedialog.askopenfilename(
            filetypes=[("FITS files", "*.fits")]
        )
        if self.fits_path:
            messagebox.showinfo("FITS chargé", os.path.basename(self.fits_path))
            self.update_image()

    def run(self):
        if not self.fits_path:
            messagebox.showerror("Erreur", "Aucun fichier FITS chargé")
            return
        self.update_image()
        messagebox.showinfo("Terminé", f"Résultats enregistrés dans {RESULTS_DIR}")

    def schedule_update(self):
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(self.update_delay, self.update_image)

    def update_image(self):
        if not self.fits_path:
            return

        img, starless, mask, final = process_fits_image(
            self.fits_path,
            self.fwhm_var.get(),
            self.sigma_var.get()
        )

        self.images = {
            "img": img,
            "starless": starless,
            "mask": mask,
            "final": final
        }

        self.blink_images = [img, final]
        self.blink_index = 0

        titles = ["Original", "Starless", "Masque", "Final"]
        imgs = [img, starless, mask, final]

        for ax, im_ax, data, title in zip(self.axes.flat, self.ax_images, imgs, titles):
            im_ax.set_data(data)
            im_ax.set_clim(data.min(), data.max())
            ax.set_title(title)

        self.canvas.draw_idle()

        cv.imwrite(os.path.join(RESULTS_DIR, "original.png"), img)
        cv.imwrite(os.path.join(RESULTS_DIR, "starless.png"), starless)
        cv.imwrite(os.path.join(RESULTS_DIR, "final.png"), final)

    # BLINK 
    def toggle_blink(self):
        if not self.blink_images:
            return
        self.blinking = not self.blinking
        if self.blinking:
            self.blink_step()

    def blink_step(self):
        if not self.blinking:
            return
        self.ax_images[0].set_data(self.blink_images[self.blink_index])
        self.axes[0, 0].set_title("Avant / Après")
        self.canvas.draw_idle()

        self.blink_index = 1 - self.blink_index
        self.root.after(500, self.blink_step)


# Main 
if __name__ == "__main__":
    app = tb.Window(themename="darkly")
    StarReductionApp(app)
    app.mainloop()