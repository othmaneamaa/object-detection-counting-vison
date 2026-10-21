import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def _hist_data(objets):
    """Retourne (aires, aires_boites) a partir de [(x, y, w, h, aire), ...]."""
    aires = np.array([o[4] for o in objets], dtype=float)
    boites = np.array([o[2] * o[3] for o in objets], dtype=float)
    return aires, boites


def montrer_histogramme(objets, parent=None, titre="Histogramme des tailles d'objets"):
    if objets is None or len(objets) == 0:
        if parent:
            messagebox.showinfo("Histogramme", "Aucun objet a afficher.")
        return None
    aires, boites = _hist_data(objets)
    fig = Figure(figsize=(7.2, 4.6), dpi=100, facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    bins = min(30, max(8, len(aires) // 3))
    if len(aires) >= 1:
        ax.hist(aires, bins=bins, color="#1e40af", alpha=0.85,
                label="Aires (px)")
    ax.set_xlabel("Aire (px)")
    ax.set_ylabel("Nombre d'objets")
    ax.set_title(titre, fontsize=12, color="#0f172a")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    # Statistiques
    if len(aires) >= 1:
        txt = (f"n = {len(aires)}\n"
               f"moy = {np.mean(aires):.0f} px\n"
               f"med = {np.median(aires):.0f} px\n"
               f"ecart-type = {np.std(aires):.0f} px")
        ax.text(0.98, 0.96, txt, transform=ax.transAxes,
                fontsize=9, va="top", ha="right", color="#334155",
                bbox=dict(boxstyle="round", fc="white", ec="#cbd5e1", alpha=0.9))
    fig.tight_layout(pad=2.0)

    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title(titre)
    win.configure(bg="white")
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    canvas.draw()
    return win
