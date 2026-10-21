import csv
import os
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np

from .config import EXTENSIONS_IMAGES


def exporter(image_bgr, objets, methode, n, parent=None):
    """Exporte l'image annotée + un fichier CSV des comptages."""
    # Demande dossier
    dossier = filedialog.askdirectory(title="Choisir dossier d'exportation",
                                      parent=parent)
    if not dossier:
        return False

    # Timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Exporter l'image annotée
    # On reconstruit une image avec bandeau (meme principe que dans drawing.py)
    h, w = image_bgr.shape[:2]
    img_out = image_bgr.copy()
    bandeau = cv2.copyMakeBorder(img_out, 42, 0, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(15, 23, 42))
    texte_bandeau = f"Objet(s) detec : {n}   |   {methode}   |   {ts}"
    cv2.putText(bandeau, texte_bandeau, (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Chemin image
    img_name = f"resultat_{methode}_{ts}.png"
    img_path = os.path.join(dossier, img_name)
    cv2.imwrite(img_path, bandeau)

    # 2) Exporter CSV
    csv_name = f"comptes_{methode}_{ts}.csv"
    csv_path = os.path.join(dossier, csv_name)

    try:
        lignes = []
        ligne_totale = [methode, n, ts]
        lignes.append(ligne_totale)

        # Lignes objets si disponibles
        for o in objets:
            if len(o) >= 5:
                x, y, w, h, aire = o[:5]
                ligne_obj = [methode, str(n), str(aire), str(x), str(y), str(w), str(h)]
            else:
                ligne_obj = [methode, "0", "0", "0", "0", "0", "0"]
            lignes.append(ligne_obj)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["methode", "n_total", "date"] + ["aire", "x", "y", "largeur", "hauteur"])
            writer.writerows(lignes)
    except Exception as e:
        messagebox.showerror("Erreur export CSV", str(e), parent=parent)
        return False

    messagebox.showinfo("Export termine",
                        f"Image : {img_name}\nCSV : {csv_name}\nDossier : {dossier}",
                        parent=parent)
    return True


def compter_lot(dossier_parent=None, *, progress_cb=None, cancel_cb=None):
    """
    Compte toutes les images d'un dossier (sous-dossiers inclus).
    Retourne liste de dicts [{fichier, methode, n, params?, ...}, ...].
    """
    from .auto_engine import recherche_auto

    if dossier_parent is None:
        dossier_parent = tk.filedialog.askdirectory(title="Choisir dossier d'images")

    if not dossier_parent:
        return []

    resultats = []

    for racine, _, fichiers in os.walk(dossier_parent):
        if cancel_cb and cancel_cb():
            break
        for fn in sorted(fichiers):
            if cancel_cb and cancel_cb():
                break
            ext = fn.lower()
            if not any(ext.endswith(e) for e in EXTENSIONS_IMAGES):
                continue
            chemin = os.path.join(racine, fn)
            try:
                import cv2
                bgr = cv2.imread(chemin)
                if bgr is None:
                    continue
                # Si annuler, retourne ce qu'on a
                if progress_cb:
                    progress_cb(f"Lecture : {fn}", 0)
                gagnant = recherche_auto(bgr, progress=None, cancel=cancel_cb)
                if gagnant is None:
                    resultats.append({"fichier": fn, "dossier": racine, "methode": "inconnu", "n": 0})
                else:
                    resultats.append({
                        "fichier": fn,
                        "dossier": racine,
                        "methode": gagnant["methode"],
                        "n": gagnant["n"],
                        "score": gagnant.get("score", 0),
                        "params": gagnant.get("params", {}),
                    })
            except Exception as e:
                resultats.append({"fichier": fn, "dossier": racine, "methode": "erreur", "n": 0, "error": str(e)})

    return resultats