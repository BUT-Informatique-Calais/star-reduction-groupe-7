import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import threading

import ttkbootstrap as tb
from ttkbootstrap.constants import *

from tkinter import filedialog, messagebox, END, NORMAL, DISABLED
from tkinter import scrolledtext

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

    # Détection si image couleur ou monochrome
    is_color = len(data.shape) == 3
    
    if is_color:
        # Image couleur (3 canaux)
        # Convertir en luminance pour la détection d'étoiles
        if data.shape[0] == 3:  # Format (C, H, W)
            data_gray = np.mean(data, axis=0)
            data = np.transpose(data, (1, 2, 0))  # Convertir en (H, W, C)
        else:  # Format (H, W, C)
            data_gray = np.mean(data, axis=2)
    else:
        # Image monochrome
        data_gray = data

    # Normalisation pour affichage
    if is_color:
        img_norm = np.zeros_like(data, dtype=np.uint8)
        for i in range(3):
            img_norm[:, :, i] = cv.normalize(data[:, :, i], None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        img = img_norm
        original = img.astype(np.float32)
    else:
        img_norm = cv.normalize(data, None, 0, 255, cv.NORM_MINMAX)
        img = img_norm.astype(np.uint8)
        original = img.astype(np.float32)

    # Statistiques sur l'image en niveaux de gris
    mean, median, std = sigma_clipped_stats(data_gray, sigma=3.0)

    # Détection étoiles sur l'image en niveaux de gris
    finder = DAOStarFinder(fwhm=fwhm, threshold=sigma_factor * std)
    sources = finder(data_gray - median)

    # Masque
    if is_color:
        mask = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)
    else:
        mask = np.zeros(img.shape, dtype=np.uint8)

    if sources is not None:
        for star in sources:
            x, y = int(star["xcentroid"]), int(star["ycentroid"])
            cv.circle(mask, (x, y), 5, 255, -1)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=2)
    mask = cv.GaussianBlur(mask, (25, 25), 0)

    mask_f = mask.astype(np.float32) / 255.0
    
    # Élargir encore plus le masque pour une transition douce
    mask_f = cv.GaussianBlur(mask_f, (31, 31), 0)

    # Traitement différent selon couleur ou monochrome
    if is_color:
        starless = np.zeros_like(img, dtype=np.float32)
        final = np.zeros_like(img, dtype=np.float32)
        
        for i in range(3):
            channel = img[:, :, i].astype(np.float32)
            # Utiliser un flou au lieu d'une érosion pour éviter les artefacts
            starless_channel = cv.GaussianBlur(img[:, :, i], (7, 7), 0).astype(np.float32)
            # Inpainting pour remplir les zones d'étoiles de manière plus naturelle
            temp_mask = (mask > 127).astype(np.uint8) * 255
            starless_channel = cv.inpaint(img[:, :, i], temp_mask, 3, cv.INPAINT_TELEA).astype(np.float32)
            
            starless[:, :, i] = starless_channel
            final[:, :, i] = (mask_f * starless_channel) + ((1 - mask_f) * channel)
        
        # Appliquer un léger flou sur le résultat final pour lisser
        final = cv.GaussianBlur(final, (3, 3), 0)
        final = np.clip(final, 0, 255).astype(np.uint8)
        starless = starless.astype(np.uint8)
    else:
        # Utiliser inpainting au lieu d'érosion
        temp_mask = (mask > 127).astype(np.uint8) * 255
        starless = cv.inpaint(img, temp_mask, 3, cv.INPAINT_TELEA).astype(np.float32)
        final = (mask_f * starless) + ((1 - mask_f) * original)
        final = cv.GaussianBlur(final, (3, 3), 0)
        final = np.clip(final, 0, 255).astype(np.uint8)
        starless = starless.astype(np.uint8)

    return img, starless, mask_f, final, is_color


# interface utilisateur 
class StarReductionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Star Reduction – Astrophotography")
        self.root.geometry("1300x900")

        self.fits_path = None
        self.images = {}
        self.is_color = False
        self.blinking = False
        self.blink_index = 0
        self.blink_images = []

        self.update_timer = None
        self.update_delay = 300

        # Variables batch
        self.batch_running = False
        self.batch_terminal = None
        self.batch_progress = None

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

        # AJOUT : Bouton Batch
        tb.Button(
            controls, text="🚀 Mode Batch",
            command=self.open_batch_window,
            bootstyle="success"
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

        img, starless, mask, final, self.is_color = process_fits_image(
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
            # Gérer l'affichage couleur vs monochrome
            if title == "Masque":
                im_ax.set_data(data)
                im_ax.set_clim(data.min(), data.max())
                im_ax.set_cmap("gray")
            elif self.is_color and len(data.shape) == 3:
                im_ax.set_data(data)
                im_ax.set_cmap(None)
            else:
                im_ax.set_data(data)
                im_ax.set_clim(data.min(), data.max())
                im_ax.set_cmap("gray")
            
            ax.set_title(title)

        self.canvas.draw_idle()

        # Sauvegarder en BGR pour OpenCV
        if self.is_color:
            cv.imwrite(os.path.join(RESULTS_DIR, "original.png"), cv.cvtColor(img, cv.COLOR_RGB2BGR))
            cv.imwrite(os.path.join(RESULTS_DIR, "starless.png"), cv.cvtColor(starless, cv.COLOR_RGB2BGR))
            cv.imwrite(os.path.join(RESULTS_DIR, "final.png"), cv.cvtColor(final, cv.COLOR_RGB2BGR))
        else:
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
        
        data = self.blink_images[self.blink_index]
        self.ax_images[0].set_data(data)
        
        if self.is_color and len(data.shape) == 3:
            self.ax_images[0].set_cmap(None)
        else:
            self.ax_images[0].set_cmap("gray")
        
        self.axes[0, 0].set_title("Avant / Après")
        self.canvas.draw_idle()

        self.blink_index = 1 - self.blink_index
        self.root.after(500, self.blink_step)

    # ---------- MODE BATCH (NOUVEAU) ----------
    def open_batch_window(self):
        """Ouvre une fenêtre pour le traitement batch"""
        batch_win = tb.Toplevel(self.root)
        batch_win.title("🚀 Mode Batch Processing")
        batch_win.geometry("800x600")

        # Frame pour les contrôles
        ctrl_frame = tb.Frame(batch_win, padding=10)
        ctrl_frame.pack(fill=X)

        # Sélection dossier input
        tb.Label(ctrl_frame, text="Dossier FITS:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        self.batch_input_var = tb.StringVar()
        tb.Entry(ctrl_frame, textvariable=self.batch_input_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        tb.Button(ctrl_frame, text="📂", command=self.select_batch_input).grid(row=0, column=2, padx=5, pady=5)

        # Sélection dossier output
        tb.Label(ctrl_frame, text="Dossier sortie:").grid(row=1, column=0, padx=5, pady=5, sticky=W)
        self.batch_output_var = tb.StringVar()
        tb.Entry(ctrl_frame, textvariable=self.batch_output_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        tb.Button(ctrl_frame, text="📁", command=self.select_batch_output).grid(row=1, column=2, padx=5, pady=5)

        # Paramètres
        param_frame = tb.Frame(ctrl_frame)
        param_frame.grid(row=2, column=0, columnspan=3, pady=10)

        tb.Label(param_frame, text="FWHM:").pack(side=LEFT, padx=5)
        self.batch_fwhm_var = tb.DoubleVar(value=self.fwhm_var.get())
        tb.Entry(param_frame, textvariable=self.batch_fwhm_var, width=8).pack(side=LEFT, padx=5)

        tb.Label(param_frame, text="Sigma:").pack(side=LEFT, padx=5)
        self.batch_sigma_var = tb.DoubleVar(value=self.sigma_var.get())
        tb.Entry(param_frame, textvariable=self.batch_sigma_var, width=8).pack(side=LEFT, padx=5)

        # Bouton de lancement
        self.batch_btn = tb.Button(
            ctrl_frame,
            text="▶ Lancer le traitement batch",
            command=lambda: self.run_batch(batch_win),
            bootstyle="success"
        )
        self.batch_btn.grid(row=3, column=0, columnspan=3, pady=10)

        # Terminal (zone de texte avec scrollbar)
        term_frame = tb.LabelFrame(batch_win, text="📟 Terminal")
        term_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.batch_terminal = scrolledtext.ScrolledText(
            term_frame,
            wrap="word",
            height=20,
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Consolas", 10)
        )
        self.batch_terminal.pack(fill=BOTH, expand=True, padx=5, pady=5)
        self.batch_terminal.config(state=DISABLED)

        # Progress bar
        self.batch_progress = tb.Progressbar(
            batch_win,
            bootstyle="success-striped",
            mode="determinate"
        )
        self.batch_progress.pack(fill=X, padx=10, pady=5)

    def select_batch_input(self):
        folder = filedialog.askdirectory(title="Sélectionner le dossier FITS")
        if folder:
            self.batch_input_var.set(folder)

    def select_batch_output(self):
        folder = filedialog.askdirectory(title="Sélectionner le dossier de sortie")
        if folder:
            self.batch_output_var.set(folder)

    def log_to_terminal(self, message):
        """Ajoute un message au terminal"""
        if self.batch_terminal is None:
            return
        try:
            self.batch_terminal.config(state=NORMAL)
            self.batch_terminal.insert(END, message + "\n")
            self.batch_terminal.see(END)
            self.batch_terminal.config(state=DISABLED)
            self.batch_terminal.update()
        except:
            pass

    def run_batch(self, window):
        """Lance le traitement batch dans un thread séparé"""
        if self.batch_running:
            messagebox.showwarning("En cours", "Un traitement batch est déjà en cours")
            return

        input_dir = self.batch_input_var.get()
        output_dir = self.batch_output_var.get()

        if not input_dir or not output_dir:
            messagebox.showerror("Erreur", "Veuillez sélectionner les dossiers d'entrée et de sortie")
            return

        if not os.path.isdir(input_dir):
            messagebox.showerror("Erreur", "Le dossier d'entrée n'existe pas")
            return

        # Désactiver le bouton
        self.batch_btn.config(state=DISABLED, text="⏳ Traitement en cours...")

        # Lancer dans un thread
        thread = threading.Thread(
            target=self.batch_process_thread,
            args=(input_dir, output_dir, self.batch_fwhm_var.get(), self.batch_sigma_var.get())
        )
        thread.daemon = True
        thread.start()

    def batch_process_thread(self, input_dir, output_dir, fwhm, sigma):
        """Traitement batch dans un thread séparé"""
        self.batch_running = True
        os.makedirs(output_dir, exist_ok=True)

        # Récupérer les fichiers
        fits_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".fits", ".fit"))
        ])

        if not fits_files:
            self.log_to_terminal("❌ Aucun fichier FITS trouvé")
            self.batch_btn.config(state=NORMAL, text="▶ Lancer le traitement batch")
            self.batch_running = False
            return

        self.log_to_terminal("="*60)
        self.log_to_terminal("🚀 BATCH STAR REDUCTION")
        self.log_to_terminal("="*60)
        self.log_to_terminal(f"📂 Entrée  : {input_dir}")
        self.log_to_terminal(f"📁 Sortie  : {output_dir}")
        self.log_to_terminal(f"🔧 FWHM={fwhm}, Sigma={sigma}")
        self.log_to_terminal(f"📊 Fichiers: {len(fits_files)}")
        self.log_to_terminal("="*60)
        self.log_to_terminal("")

        success = 0
        errors = 0

        # Configurer la barre de progression
        if self.batch_progress:
            self.batch_progress["maximum"] = len(fits_files)
            self.batch_progress["value"] = 0

        for i, filename in enumerate(fits_files, start=1):
            path = os.path.join(input_dir, filename)

            try:
                # Lire et traiter
                _, _, _, final, is_color = process_fits_image(path, fwhm, sigma)

                # Sauvegarder
                base_name = Path(filename).stem
                out_name = f"{base_name}_starless.png"
                out_path = os.path.join(output_dir, out_name)

                if is_color:
                    cv.imwrite(out_path, cv.cvtColor(final, cv.COLOR_RGB2BGR))
                else:
                    cv.imwrite(out_path, final)

                success += 1
                self.log_to_terminal(f"[{i}/{len(fits_files)}] ✔ {filename}")

            except Exception as e:
                errors += 1
                self.log_to_terminal(f"[{i}/{len(fits_files)}] ✖ {filename} - ERREUR: {str(e)}")

            # Mettre à jour la progress bar
            if self.batch_progress:
                self.batch_progress["value"] = i

        # Résumé
        self.log_to_terminal("")
        self.log_to_terminal("="*60)
        self.log_to_terminal(f"✅ Traitement terminé")
        self.log_to_terminal(f"   Succès : {success}/{len(fits_files)}")
        if errors > 0:
            self.log_to_terminal(f"   Erreurs: {errors}/{len(fits_files)}")
        self.log_to_terminal("="*60)

        # Réactiver le bouton
        self.batch_btn.config(state=NORMAL, text="▶ Lancer le traitement batch")
        self.batch_running = False

        messagebox.showinfo("Terminé", f"Traitement terminé!\n{success} succès, {errors} erreurs")
              

# Main 
if __name__ == "__main__":
    app = tb.Window(themename="darkly")
    StarReductionApp(app)
    app.mainloop()