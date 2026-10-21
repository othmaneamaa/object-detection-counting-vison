import os
import csv
import datetime
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import (SEUIL_AIRE_MIN, SEUIL_AIRE_MAX, EXTENSIONS_IMAGES)
from .video import VideoWindow
from .drawing import ajouter_bandeau, dessiner_numero
from .auto_engine import recherche_auto
from .detectors import detect_cercles, detect_watershed, detect_visages
from .stats import montrer_histogramme
from .export import exporter


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection and Counting - Version Finale (AUTO)")
        self.root.geometry("1240x860")
        self.root.configure(bg="#f1f5f9")

        self.image_originale = None
        self._chemin_image = None
        self.dernier_auto = None
        self.derniers_objets = []      # [(x, y, w, h, aire), ...]
        self.derniere_methode = None
        self.dernier_finale = None     # image annotée, sans bandeau
        self._img2 = self._img3 = None
        self._auto_en_cours = False
        self._auto_cancel = False

        # Correction manuelle
        self._correction_on = tk.BooleanVar(value=False)
        self._objets_courants = []     # [(x, y, w, h)] modifiables

        self._construire_ui()

    # ------------------------------------------------------------------ UI
    def _construire_ui(self):
        header = tk.Frame(self.root, bg="#0f172a", height=56)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="  Object Detection and Counting",
                 font=("Segoe UI", 14, "bold"), bg="#0f172a", fg="white").pack(side=tk.LEFT, pady=13)
        tk.Label(header, text="Methodes classiques - OpenCV  |  AUTO recommande",
                 font=("Segoe UI", 9), bg="#0f172a", fg="#94a3b8").pack(side=tk.LEFT, padx=12)

        body = tk.Frame(self.root, bg="#f1f5f9")
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # --- Panneau gauche defilant ---
        left_outer = tk.Frame(body, bg="white", width=350)
        left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left_outer.pack_propagate(False)
        self.canvas_scroll = tk.Canvas(left_outer, bg="white", highlightthickness=0)
        scrollbar = tk.Scrollbar(left_outer, orient="vertical",
                                 command=self.canvas_scroll.yview)
        self.canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left = tk.Frame(self.canvas_scroll, bg="white", padx=14, pady=12)
        self._win = self.canvas_scroll.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>",
                  lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all")))
        self.canvas_scroll.bind("<Configure>",
                                lambda e: self.canvas_scroll.itemconfig(self._win, width=e.width))
        left.bind("<Enter>", lambda e: self.canvas_scroll.bind_all("<MouseWheel>", self._molette))
        left.bind("<Leave>", lambda e: self.canvas_scroll.unbind_all("<MouseWheel>"))

        tk.Label(left, text="1. Choisissez une image", font=("Segoe UI", 11, "bold"),
                 bg="white", fg="#0f172a").pack(anchor="w")
        tk.Label(left, text="Billes, pieces ou formes sur fond uni",
                 font=("Segoe UI", 8), bg="white", fg="#64748b").pack(anchor="w", pady=(0, 8))
        self.btn_image = tk.Button(left, text="Choisir une image ...", command=self.choisir_image,
                                   bg="#2563eb", fg="white", font=("Segoe UI", 11, "bold"),
                                   relief=tk.FLAT, padx=10, pady=10, cursor="hand2",
                                   activebackground="#1d4ed8")
        self.btn_image.pack(fill=tk.X)
        self.btn_dossier = tk.Button(left, text="... ou tout un dossier (mode lot)",
                                     command=self.compter_lot,
                                     bg="#475569", fg="white", font=("Segoe UI", 9),
                                     relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
                                     activebackground="#334155")
        self.btn_dossier.pack(fill=tk.X, pady=(6, 0))

        self.btn_video = tk.Button(left, text="Webcam / Vidéo",
                                   command=self._ouvrir_video,
                                   bg="#6c5ce7", fg="white", font=("Segoe UI", 10, "bold"),
                                   relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
                                   activebackground="#5a4ad9")
        self.btn_video.pack(fill=tk.X, pady=4)
        self.lbl_fichier = tk.Label(left, text="Aucune image chargee",
                                    font=("Segoe UI", 8), bg="white", fg="#94a3b8",
                                    wraplength=290, justify=tk.CENTER)
        self.lbl_fichier.pack(pady=6)
        tk.Frame(left, bg="#e2e8f0", height=1).pack(fill=tk.X, pady=8)

        tk.Label(left, text="2. Methode de detection", font=("Segoe UI", 10, "bold"),
                 bg="white", fg="#0f172a").pack(anchor="w")
        self.mode_detection = tk.StringVar(value="auto")
        for txt, val, coul in [
                ("AUTO - trouve le meilleur reglage", "auto", "#059669"),
                ("Billes rondes (Hough)", "cercles", "#1e40af"),
                ("Formes quelconques (Watershed)", "watershed", None),
                ("Visages (Haar)", "visages", "#7c3aed")]:
            tk.Radiobutton(left, text=txt, variable=self.mode_detection, value=val,
                           bg="white", font=("Segoe UI", 9, "bold" if val == "auto" else "normal"),
                           fg=coul or "black").pack(anchor="w")

        # AUTO + progression + annulation
        self.btn_auto = tk.Button(left, text="AUTO : analyser et compter",
                                  command=self.compter,
                                  bg="#059669", fg="white", font=("Segoe UI", 11, "bold"),
                                  relief=tk.FLAT, padx=10, pady=11, cursor="hand2",
                                  activebackground="#047857")
        self.btn_auto.pack(fill=tk.X, pady=(10, 6))
        self.lbl_auto = tk.Label(left, text="Astuce : laissez AUTO coche et cliquez ci-dessus.\nLe programme choisit tout seul.",
                                 font=("Segoe UI", 8), bg="white", fg="#64748b", justify=tk.CENTER)
        self.lbl_auto.pack()
        self.progress = ttk.Progressbar(left, orient="horizontal", length=290, mode="determinate")
        self.btn_annuler = tk.Button(left, text="Annuler l'analyse", command=self._annuler_auto,
                                     bg="#dc2626", fg="white", font=("Segoe UI", 9, "bold"),
                                     relief=tk.FLAT, state=tk.DISABLED, cursor="hand2")
        self._pack_progress_barre()

        tk.Frame(left, bg="#e2e8f0", height=1).pack(fill=tk.X, pady=10)

        tk.Label(left, text="Reglages manuels (si AUTO decoche)", font=("Segoe UI", 9, "bold"),
                 bg="white", fg="#334155").pack(anchor="w")
        tk.Label(left, text="Sensibilite billes (bas = detecte plus)",
                 font=("Segoe UI", 9, "bold"), bg="white", fg="#334155").pack(anchor="w", pady=(8, 0))
        self.slider_sensibilite = tk.Scale(left, from_=10, to=60, orient=tk.HORIZONTAL,
                                           bg="white", highlightthickness=0, length=290)
        self.slider_sensibilite.set(28)
        self.slider_sensibilite.pack(fill=tk.X)

        tk.Label(left, text="Seuil contours Canny (bas = bords faibles gardes)",
                 font=("Segoe UI", 9, "bold"), bg="white", fg="#334155").pack(anchor="w")
        self.slider_canny = tk.Scale(left, from_=30, to=150, orient=tk.HORIZONTAL,
                                     bg="white", highlightthickness=0, length=290)
        self.slider_canny.set(70)
        self.slider_canny.pack(fill=tk.X)

        tk.Label(left, text="Distance min entre billes (px)",
                 font=("Segoe UI", 9, "bold"), bg="white", fg="#334155").pack(anchor="w")
        self.slider_dist_billes = tk.Scale(left, from_=10, to=200, orient=tk.HORIZONTAL,
                                           bg="white", highlightthickness=0, length=290)
        self.slider_dist_billes.set(40)
        self.slider_dist_billes.pack(fill=tk.X)

        tk.Frame(left, bg="#e2e8f0", height=1).pack(fill=tk.X, pady=8)
        tk.Label(left, text="Options (mode Watershed)", font=("Segoe UI", 9, "bold"),
                 bg="white", fg="#334155").pack(anchor="w")
        self.methode = tk.StringVar(value="otsu")
        f1 = tk.Frame(left, bg="white")
        f1.pack(fill=tk.X)
        tk.Radiobutton(f1, text="Otsu (fond uni)", variable=self.methode, value="otsu",
                       bg="white", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Radiobutton(f1, text="Adaptatif", variable=self.methode, value="adaptatif",
                       bg="white", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.var_inverser = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Objets sombres sur fond clair",
                       variable=self.var_inverser, bg="white",
                       font=("Segoe UI", 9)).pack(anchor="w", pady=3)
        self.var_watershed = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Separer objets colles (watershed)",
                       variable=self.var_watershed, bg="white",
                       font=("Segoe UI", 9), fg="#1e40af").pack(anchor="w")
        tk.Label(left, text="Aire minimale (px)", font=("Segoe UI", 9, "bold"),
                 bg="white", fg="#334155").pack(anchor="w", pady=(8, 0))
        self.slider_min = tk.Scale(left, from_=10, to=5000, orient=tk.HORIZONTAL,
                                   bg="white", highlightthickness=0, length=290)
        self.slider_min.set(SEUIL_AIRE_MIN)
        self.slider_min.pack(fill=tk.X)
        tk.Label(left, text="Aire maximale (px)", font=("Segoe UI", 9, "bold"),
                 bg="white", fg="#334155").pack(anchor="w")
        self.slider_max = tk.Scale(left, from_=5000, to=500000, orient=tk.HORIZONTAL,
                                   bg="white", highlightthickness=0, length=290)
        self.slider_max.set(SEUIL_AIRE_MAX)
        self.slider_max.pack(fill=tk.X)

        tk.Label(left, text="3. Lancez le comptage", font=("Segoe UI", 10, "bold"),
                 bg="white", fg="#0f172a").pack(anchor="w", pady=(10, 0))
        self.btn_manuel = tk.Button(left, text="Appliquer les reglages manuels", command=self.compter,
                                    bg="#ea580c", fg="white", font=("Segoe UI", 11, "bold"),
                                    relief=tk.FLAT, padx=10, pady=11, cursor="hand2",
                                    activebackground="#c2410c")
        self.btn_manuel.pack(fill=tk.X, pady=(6, 0))
        tk.Label(left, text="Modifiez les reglages puis cliquez :\nle resultat se met a jour a droite.",
                 font=("Segoe UI", 8), bg="white", fg="#94a3b8", justify=tk.CENTER).pack(pady=4)
        tk.Button(left, text="Reinitialiser les reglages", command=self.reinitialiser,
                  bg="white", fg="#475569", font=("Segoe UI", 9),
                  relief=tk.FLAT, cursor="hand2").pack(fill=tk.X)
        self.lbl_params = tk.Label(left, text="", font=("Segoe UI", 8),
                                   bg="white", fg="#64748b", justify=tk.LEFT,
                                   wraplength=290)
        self.lbl_params.pack(anchor="w", pady=(4, 0))

        self.card = tk.Frame(left, bg="#f0fdf4", highlightbackground="#bbf7d0",
                             highlightthickness=1, padx=12, pady=12)
        self.card.pack(fill=tk.X, pady=12)
        self.lbl_resultat = tk.Label(self.card, text="--",
                                     font=("Segoe UI", 34, "bold"),
                                     bg="#f0fdf4", fg="#15803d")
        self.lbl_resultat.pack()
        self.lbl_detail = tk.Label(self.card, text="Choisissez une image\npour commencer",
                                   font=("Segoe UI", 9), bg="#f0fdf4", fg="#64748b",
                                   justify=tk.CENTER)
        self.lbl_detail.pack()

        # --- Outils sur le resultat ---
        tk.Frame(left, bg="#e2e8f0", height=1).pack(fill=tk.X, pady=8)
        tk.Label(left, text="4. Outils sur le resultat", font=("Segoe UI", 10, "bold"),
                 bg="white", fg="#0f172a").pack(anchor="w")
        self.btn_hist = tk.Button(left, text="Histogramme des tailles",
                                  command=self.histogramme_tailles,
                                  bg="#7c3aed", fg="white", font=("Segoe UI", 10, "bold"),
                                  relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
                                  activebackground="#6b21a8")
        self.btn_hist.pack(fill=tk.X, pady=(6, 4))
        self.btn_export = tk.Button(left, text="Exporter (image + CSV)",
                                    command=self.exporter,
                                    bg="#0891b2", fg="white", font=("Segoe UI", 10, "bold"),
                                    relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
                                    activebackground="#155e75")
        self.btn_export.pack(fill=tk.X, pady=4)
        self.btn_corriger = tk.Button(left, text="Correction manuelle : OFF",
                                      command=self._toggle_correction,
                                      bg="#b45309", fg="white", font=("Segoe UI", 9, "bold"),
                                      relief=tk.FLAT, padx=10, pady=5, cursor="hand2",
                                      activebackground="#92400e")
        self.btn_corriger.pack(fill=tk.X, pady=4)
        tk.Label(left, text="Quand la correction est ON :\nclic sur un objet = le retirer,\nclic sur le vide = ajouter un objet.",
                 font=("Segoe UI", 8), bg="white", fg="#94a3b8", justify=tk.CENTER).pack()

        # --- Zone droite : images en 2x2 ---
        right = tk.Frame(body, bg="white", padx=8, pady=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(8.4, 6.4), dpi=100, facecolor="white")
        self.axes = [self.figure.add_subplot(2, 2, i + 1) for i in range(4)]
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click_image)
        self.placeholder()

    def _pack_progress_barre(self):
        """Affiche/masque la barre de progression + bouton Annuler."""
        try:
            self.progress.pack_forget()
            self.btn_annuler.pack_forget()
        except Exception:
            pass

    def _montrer_progress(self):
        self.progress["value"] = 0
        self.progress.pack(fill=tk.X, pady=4)
        self.btn_annuler.config(state=tk.NORMAL)
        self.btn_annuler.pack(fill=tk.X, pady=(0, 4))

    def _masquer_progress(self):
        self.progress.pack_forget()
        self.btn_annuler.config(state=tk.DISABLED)
        self.btn_annuler.pack_forget()

    def _molette(self, event):
        if event.num in (4, 5):
            d = -1 if event.num == 4 else 1
            self.canvas_scroll.yview_scroll(d, "units")
        else:
            self.canvas_scroll.yview_scroll(int(-event.delta / 120), "units")

    # ------------------------------------------------------------ actions
    def reinitialiser(self):
        self.mode_detection.set("auto")
        self.slider_sensibilite.set(28)
        self.slider_canny.set(70)
        self.slider_dist_billes.set(40)
        self.methode.set("otsu")
        self.var_inverser.set(True)
        self.var_watershed.set(True)
        self.slider_min.set(SEUIL_AIRE_MIN)
        self.slider_max.set(SEUIL_AIRE_MAX)
        self.lbl_auto.config(text="Reglages reinitialises. Choisissez une image puis AUTO.",
                             fg="#64748b")
        if self.image_originale is not None:
            self.compter()

    def placeholder(self):
        for ax, t in zip(self.axes, ["1. Originale", "2. Pretraitement",
                                     "3. Contours / Masque", "4. Resultat"]):
            ax.clear()
            ax.set_title(t, fontsize=10, color="#334155")
            ax.axis("off")
            ax.text(0.5, 0.5, "En attente\nd'image", ha="center", va="center",
                    fontsize=11, color="#94a3b8")
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def choisir_image(self):
        chemin = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", " ".join(EXTENSIONS_IMAGES)), ("Tous", "*.*")])
        if not chemin:
            return
        if self._auto_en_cours:
            self.lbl_fichier.config(text="Changement annule\n(AUTO en cours)", fg="#b42318")
            return
        img = cv2.imread(chemin)
        if img is None:
            messagebox.showerror("Erreur", f"Impossible de lire :\n{chemin}")
            return
        self.image_originale = img
        self._chemin_image = chemin
        nom = os.path.basename(chemin)
        self.lbl_fichier.config(text=nom, fg="#334155")
        self.compter()

    # -------------------------------------------------------- pipelines
    def pipeline_cercles(self, bgr):
        sens = int(self.slider_sensibilite.get())
        seuil_canny = int(self.slider_canny.get())
        dist_min = int(self.slider_dist_billes.get())
        r = detect_cercles(bgr, sens, seuil_canny, dist_min)
        finale = ajouter_bandeau(r["finale_brute"], f"Objets : {r['n']}")
        self.lbl_params.config(
            text=f"Hough : sens={sens}, canny={seuil_canny}, dist={dist_min}\n"
                 f"rayons {r['r_min']}-{r['r_max']}px, support moy={r['avg_support']:.2f}")
        return r["egalisee"], r["canny"], r["finale_brute"], r["n"], r["objets"]

    def pipeline_watershed(self, bgr):
        aire_min = int(self.slider_min.get())
        aire_max = int(self.slider_max.get())
        r = detect_watershed(bgr, self.methode.get(), bool(self.var_inverser.get()),
                             bool(self.var_watershed.get()), aire_min, aire_max)
        finale = ajouter_bandeau(r["finale_brute"], f"Objets : {r['n']}")
        self.lbl_params.config(text=f"Watershed : {self.methode.get()}, boites={r['n']}, qual={r['qual']:.2f}")
        return r["egalisee"], r["morpho"], r["finale_brute"], r["n"], r["objets"]

    def pipeline_visages(self, bgr):
        aire_min = int(self.slider_min.get())
        aire_max = int(self.slider_max.get())
        r = detect_visages(bgr, 5, aire_min, aire_max)
        finale = ajouter_bandeau(r["finale_brute"], f"Objets : {r['n']}")
        self.lbl_params.config(
            text=f"Haar visages : detectees={r['raw_n']}, gardees={r['n']}\n"
                 f"taille_min={r['taille_min']}, aire {aire_min}-{aire_max}")
        return r["egalisee"], r["masque"], r["finale_brute"], r["n"], r["objets"]

    # ------------------------------------------------------------- AUTO
    def _annuler_auto(self):
        self._auto_cancel = True
        self.btn_annuler.config(state=tk.DISABLED)
        self.lbl_auto.config(text="Annulation demandee ...", fg="#b42318")

    def _maj_progress(self, idx, total):
        if not self._auto_en_cours:
            return
        self.progress["value"] = 100.0 * idx / max(total, 1)
        self.lbl_auto.config(text=f"Analyse AUTO : {idx}/{total} reglages ...", fg="#059669")

    def compter_auto(self):
        """Lance la recherche AUTO dans un thread (UI fluide) avec progression."""
        if self._auto_en_cours:
            return
        self._auto_en_cours = True
        self._auto_cancel = False
        for b in (self.btn_auto, self.btn_manuel, self.btn_image, self.btn_dossier,
                  self.btn_hist, self.btn_export, self.btn_corriger):
            b.config(state=tk.DISABLED)
        self._montrer_progress()
        self.lbl_resultat.config(text="...")
        self.lbl_detail.config(text="test de plusieurs\nreglages ...")
        self.root.update_idletasks()

        image = self.image_originale

        def travail():
            def progress(idx, total):
                self.root.after(0, lambda i=idx, t=total: self._maj_progress(i, t))

            def cancel():
                return self._auto_cancel

            try:
                resultat = recherche_auto(image, progress=progress, cancel=cancel)
            except Exception as e:
                resultat = e
            self.root.after(0, lambda: self._auto_termine(resultat))

        threading.Thread(target=travail, daemon=True).start()

    def _auto_termine(self, gagnant):
        self._auto_en_cours = False
        self._masquer_progress()
        for b in (self.btn_auto, self.btn_manuel, self.btn_image, self.btn_dossier,
                  self.btn_hist, self.btn_export, self.btn_corriger):
            b.config(state=tk.NORMAL)

        if self._auto_cancel and (gagnant is None or not isinstance(gagnant, Exception)):
            self.lbl_auto.config(text="Analyse AUTO annulee.", fg="#b42318")
            return
        if isinstance(gagnant, Exception):
            messagebox.showerror("AUTO", f"Erreur de traitement :\n{gagnant}")
            return
        if gagnant is None:
            messagebox.showerror("AUTO", "Aucun resultat - image illisible ?")
            return

        self.dernier_auto = gagnant
        n = gagnant["n"]
        score = gagnant["score"]
        methode = gagnant["methode"]
        params = gagnant["params"]
        self.derniers_objets = gagnant.get("objets", [])
        self.derniere_methode = methode
        self.dernier_finale = gagnant["images"][2]
        self._objets_courants = [tuple(o[:4]) for o in self.derniers_objets]

        # cale les controles sur le gagnant
        try:
            if methode == "cercles":
                self.slider_sensibilite.set(params["sens"])
                self.slider_canny.set(params["canny"])
                self.slider_dist_billes.set(params["dist"])
            elif methode == "watershed":
                self.methode.set(params["methode"])
                self.var_inverser.set(params["inverser"])
                self.slider_min.set(params["aire_min"])
                self.slider_max.set(params["aire_max"])
            elif methode == "visages":
                self.slider_min.set(params["aire_min"])
                self.slider_max.set(params["aire_max"])
        except Exception:
            pass

        img2, img3, finale_brute = gagnant["images"]
        self._img2, self._img3 = img2, img3
        if methode == "cercles":
            t2, t3 = "2. Contraste (CLAHE)", "3. Contours (Canny)"
            detail_auto = (f"AUTO -> billes (Hough)  score {score:.2f}\n"
                           f" sens={params['sens']} canny={params['canny']} dist={params['dist']}")
        elif methode == "watershed":
            t2 = f"2. Seuillee ({params['methode']})"
            t3 = "3. Apres morphologie"
            inv_txt = "sombres/fond clair" if params["inverser"] else "clairs/fond sombre"
            detail_auto = (f"AUTO -> formes (Watershed)  score {score:.2f}\n"
                           f" {params['methode']}, {inv_txt}, aire>={params['aire_min']}")
        else:
            t2, t3 = "2. Contraste (CLAHE)", "3. Masque visages"
            detail_auto = (f"AUTO -> visages (Haar)  score {score:.2f}\n"
                           f" minNeighbors={params['min_neighbors']}")

        top = sorted(gagnant.get("candidats", []), key=lambda c: c["score"], reverse=True)[:3]
        top_txt = "  |  ".join(f"{c['methode']}:{c['n']} (sc {c['score']:.2f})" for c in top)
        self.lbl_params.config(text=detail_auto + f"\nTop3  {top_txt}")
        self.lbl_auto.config(
            text=f"AUTO a choisi : {methode}  ->  {n} objet(s)  (confiance {score:.0%})",
            fg="#065f46")

        finale = ajouter_bandeau(finale_brute, f"Objets : {n}  (AUTO {methode})")
        self._afficher_resultat(img2, img3, finale, n, t2, t3)

    def _afficher_resultat(self, img2, img3, finale, n, t2, t3):
        cmaps = [None, "gray", "gray", None]
        self.lbl_resultat.config(text=str(n))
        self.lbl_detail.config(text=f"objet(s) detecte(s)\n{n} au total")
        orig_rgb = cv2.cvtColor(self.image_originale, cv2.COLOR_BGR2RGB)
        fin_rgb = cv2.cvtColor(finale, cv2.COLOR_BGR2RGB)
        for ax, img, titre, cmap in zip(
                self.axes, [orig_rgb, img2, img3, fin_rgb],
                ["1. Originale", t2, t3, f"4. Resultat : {n} objet(s)"], cmaps):
            ax.clear()
            ax.set_title(titre, fontsize=10, color="#0f172a")
            ax.axis("off")
            ax.imshow(img, cmap=cmap) if cmap else ax.imshow(img)
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def _ouvrir_video(self):
        """Ouvre la fenêtre webcam/vidéo avec la méthode et réglages courants."""
        methode = self.mode_detection.get()  # "auto" | "cercles" | "watershed" | "visages"
        if methode == "auto":
            methode = None  # VideoWindow realise sa propre recherche AUTO (methode == "auto")
        # Quand AUTO est choisi, premiere frame tune + stream partiel
        params = {
            "sens": int(self.slider_sensibilite.get()),
            "canny": int(self.slider_canny.get()),
            "dist": int(self.slider_dist_billes.get()),
            "methode": self.methode.get(),
            "inverser": bool(self.var_inverser.get()),
            "separer": bool(self.var_watershed.get()),
            "aire_min": int(self.slider_min.get()),
            "aire_max": int(self.slider_max.get()),
        }
        VideoWindow(self.root, methode, params)

    def compter(self):
        if self.image_originale is None:
            messagebox.showwarning("Aucune image", "Choisissez d'abord une image.")
            return
        try:
            mode = self.mode_detection.get()
            if mode == "auto":
                self.compter_auto()
                return
            if mode == "cercles":
                img2, img3, finale_brute, n, objets = self.pipeline_cercles(self.image_originale)
                t2, t3 = "2. Contraste (CLAHE)", "3. Contours (Canny)"
            elif mode == "visages":
                img2, img3, finale_brute, n, objets = self.pipeline_visages(self.image_originale)
                t2, t3 = "2. Contraste (CLAHE)", "3. Masque visages"
            else:
                img2, img3, finale_brute, n, objets = self.pipeline_watershed(self.image_originale)
                t2, t3 = f"2. Seuillee ({self.methode.get()})", "3. Apres morphologie"
        except Exception as e:
            messagebox.showerror("Erreur de traitement", str(e))
            return
        self._img2, self._img3 = img2, img3
        self.derniers_objets = objets
        self.derniere_methode = mode
        self.dernier_finale = finale_brute
        self._objets_courants = [tuple(o[:4]) for o in objets]
        finale = ajouter_bandeau(finale_brute, f"Objets : {n}  ({mode})")
        self._afficher_resultat(img2, img3, finale, n, t2, t3)

    # ------------------------------------------------------- outils
    def histogramme_tailles(self):
        if not self.derniers_objets:
            messagebox.showinfo("Histogramme", "Lancez d'abord un comptage.")
            return
        montrer_histogramme(self.derniers_objets, parent=self.root,
                            titre=f"Histogramme des tailles - {self.derniere_methode}")

    def exporter(self):
        if self.image_originale is None or self.dernier_finale is None:
            messagebox.showinfo("Export", "Lancez d'abord un comptage.")
            return
        exporter(self.dernier_finale, self.derniers_objets,
                 self.derniere_methode or "inconnu",
                 len(self._objets_courants) if self._correction_on.get() else
                 len(self.derniers_objets),
                 parent=self.root)

    # ------------------------------------------- correction manuelle
    def _toggle_correction(self):
        etat = not self._correction_on.get()
        self._correction_on.set(etat)
        self.btn_corriger.config(
            text=f"Correction manuelle : {'ON' if etat else 'OFF'}",
            bg="#16a34a" if etat else "#b45309")
        if etat and self.dernier_finale is None:
            messagebox.showinfo("Correction", "Lancez d'abord un comptage.")
            self._correction_on.set(False)
            self.btn_corriger.config(text="Correction manuelle : OFF", bg="#b45309")

    def _on_click_image(self, event):
        """Clic sur l'axe 4 (Resultat) quand la correction est active."""
        if not self._correction_on.get():
            return
        if event.inaxes is not self.axes[3]:
            return
        if self.derniere_methode is None:
            return
        # Coordonnees image (l'image affichee porte un bandeau de 42 px en haut)
        BANDEAU = 42
        px, py = int(round(event.xdata)), int(round(event.ydata)) - BANDEAU
        h, w = self.image_originale.shape[:2]
        if not (0 <= px < w and 0 <= py < h):
            return
        # Retire si on clique sur un objet existant
        for i, (x, y, ww, hh) in enumerate(self._objets_courants):
            if x <= px <= x + ww and y <= py <= y + hh:
                del self._objets_courants[i]
                self._redessiner_correction()
                return
        # Sinon ajoute un objet de taille mediane centre sur le clic
        if self._objets_courants:
            med = int(np.median([o[2] * o[3] for o in self._objets_courants]))
            cote = max(20, int(np.sqrt(med)))
        else:
            cote = max(30, min(h, w) // 20)
        x = int(np.clip(px - cote // 2, 0, w - cote))
        y = int(np.clip(py - cote // 2, 0, h - cote))
        self._objets_courants.append((x, y, cote, cote))
        self._redessiner_correction()

    def _redessiner_correction(self):
        """Reconstruit l'image annotée a partir de self._objets_courants."""
        img = self.image_originale.copy()
        h, w = img.shape[:2]
        ep = max(2, w // 350)
        for i, (x, y, ww, hh) in enumerate(self._objets_courants, start=1):
            if self.derniere_methode == "cercles":
                cx, cy, r = x + ww // 2, y + hh // 2, ww // 2
                cv2.circle(img, (cx, cy), r, (0, 200, 0), ep)
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
                dessiner_numero(img, str(i), cx, cy)
            else:
                cv2.rectangle(img, (x, y), (x + ww, y + hh), (0, 200, 0), ep)
                dessiner_numero(img, str(i), x + ww // 2, y + hh // 2)
        self.dernier_finale = img
        # Met a jour le bandeau + l'axe resultat + le compteur
        finale = ajouter_bandeau(img, f"Objets : {len(self._objets_courants)}  (corrige)")
        fin_rgb = cv2.cvtColor(finale, cv2.COLOR_BGR2RGB)
        ax = self.axes[3]
        ax.clear()
        ax.set_title(f"4. Resultat : {len(self._objets_courants)} objet(s) (corrige)",
                     fontsize=10, color="#0f172a")
        ax.axis("off")
        ax.imshow(fin_rgb)
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()
        self.lbl_resultat.config(text=str(len(self._objets_courants)))
        self.lbl_detail.config(text=f"objet(s) detecte(s)\n"
                                    f"{len(self._objets_courants)} au total (corrige)")

    # ---------------------------------------------------------- mode lot
    def compter_lot(self):
        dossier = filedialog.askdirectory(title="Choisir un dossier d'images")
        if not dossier:
            return
        win = tk.Toplevel(self.root)
        win.title("Mode lot - en cours")
        win.configure(bg="white")
        win.geometry("520x180")
        tk.Label(win, text="Analyse du dossier :", bg="white",
                 font=("Segoe UI", 10, "bold")).pack(pady=(12, 0))
        tk.Label(win, text=dossier, bg="white", fg="#475569",
                 font=("Segoe UI", 9)).pack()
        bar = ttk.Progressbar(win, orient="horizontal", length=440, mode="indeterminate")
        bar.pack(pady=12)
        lbl = tk.Label(win, text="", bg="white", font=("Segoe UI", 9))
        lbl.pack()
        btn = tk.Button(win, text="Annuler", command=lambda: annuler.set(True),
                        bg="#dc2626", fg="white", font=("Segoe UI", 9, "bold"),
                        relief=tk.FLAT)
        btn.pack(pady=6)
        annuler = {"flag": False}
        btn.config(command=lambda: annuler.__setitem__("flag", True))
        bar.start(10)

        fichier_courant = {"nom": ""}

        def progress_cb(nom, _):
            fichier_courant["nom"] = nom
            self.root.after(0, lambda n=nom: lbl.config(text=f"Fichier : {n}"))

        def cancel_cb():
            return annuler["flag"]

        def travail():
            from .export import compter_lot as _lot
            resultats = _lot(dossier, progress_cb=progress_cb, cancel_cb=cancel_cb)
            self.root.after(0, lambda: self._lot_termine(win, resultats, dossier))

        threading.Thread(target=travail, daemon=True).start()

    def _lot_termine(self, win, resultats, dossier):
        win.destroy()
        if not resultats:
            messagebox.showinfo("Mode lot", "Aucune image trouvee ou analyse annulee.")
            return
        total = sum(r.get("n", 0) for r in resultats)
        win = tk.Toplevel(self.root)
        win.title(f"Mode lot - {len(resultats)} images - {total} objets")
        win.geometry("820x420")
        win.configure(bg="white")
        tk.Label(win, text=f"{len(resultats)} images analysees  |  "
                           f"{total} objets au total",
                 bg="white", font=("Segoe UI", 12, "bold"), fg="#0f172a").pack(pady=10)

        cols = ("fichier", "methode", "n", "score")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=16)
        tree.heading("fichier", text="Fichier")
        tree.heading("methode", text="Methode")
        tree.heading("n", text="Objets")
        tree.heading("score", text="Score")
        tree.column("fichier", width=380)
        tree.column("methode", width=120)
        tree.column("n", width=80, anchor="center")
        tree.column("score", width=100, anchor="center")
        tree.pack(fill=tk.BOTH, expand=True, padx=10)
        for r in resultats:
            tree.insert("", "end",
                        values=(os.path.relpath(os.path.join(r.get("dossier", ""), r["fichier"]),
                                                  dossier) or r["fichier"],
                                r.get("methode", "?"),
                                r.get("n", 0),
                                f"{r.get('score', 0):.2f}" if r.get("score") is not None else "-"))

        f1 = tk.Frame(win, bg="white")
        f1.pack(fill=tk.X, pady=8)
        tk.Button(f1, text="Exporter le CSV du lot",
                  command=lambda: self._exporter_lot_csv(resultats, dossier),
                  bg="#0891b2", fg="white", font=("Segoe UI", 10, "bold"),
                  relief=tk.FLAT, padx=12, pady=6, cursor="hand2").pack(side=tk.LEFT, padx=10)
        tk.Button(f1, text="Fermer", command=win.destroy,
                  bg="white", fg="#475569", font=("Segoe UI", 9),
                  relief=tk.FLAT, padx=12, pady=6).pack(side=tk.RIGHT, padx=10)

    @staticmethod
    def _exporter_lot_csv(resultats, dossier):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        chemin = filedialog.asksaveasfilename(
            title="Enregistrer le CSV du lot",
            initialfile=f"comptage_lot_{ts}.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if not chemin:
            return
        try:
            with open(chemin, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["fichier", "methode", "n_objets", "score"])
                for r in resultats:
                    writer.writerow([r.get("fichier", ""), r.get("methode", ""),
                                     r.get("n", 0), f"{r.get('score', 0):.3f}"])
            messagebox.showinfo("Export lot", f"CSV enregistre :\n{chemin}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
