import os
import threading
import io
import tkinter as tk
from tkinter import messagebox, filedialog
import cv2

from .detectors import detect_cercles, detect_watershed, detect_visages


class VideoWindow:
    """Fenetre dediee : comptage en direct (webcam) ou fichier video."""

    def __init__(self, parent, methode, params):
        self.win = tk.Toplevel(parent)
        self.win.title("Webcam / Video - comptage en direct")
        self.win.configure(bg="white")
        self.win.geometry("900x620")

        self.methode = methode  # None si AUTO: etalonne sur la 1ere frame
        self.params = params or {}
        self.cap = None
        self.running = False
        self.thread = None
        self.n_last = 0
        self.fps = 0.0
        self._calibre = False  # etalonne sur la 1ere frame si methode None

        # --- Barre du haut : source + controles ---
        bar = tk.Frame(self.win, bg="white")
        bar.pack(fill=tk.X, padx=10, pady=8)
        tk.Label(bar, text="Source :", bg="white",
                 font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.btn_webcam = tk.Button(bar, text="Webcam", command=self._ouvrir_webcam,
                                    bg="#2563eb", fg="white", font=("Segoe UI", 10, "bold"),
                                    relief=tk.FLAT, padx=12, pady=6, cursor="hand2")
        self.btn_webcam.pack(side=tk.LEFT, padx=8)
        self.btn_video = tk.Button(bar, text="Fichier video ...", command=self._ouvrir_video,
                                   bg="#475569", fg="white", font=("Segoe UI", 10, "bold"),
                                   relief=tk.FLAT, padx=12, pady=6, cursor="hand2")
        self.btn_video.pack(side=tk.LEFT, padx=8)
        self.btn_stop = tk.Button(bar, text="Arreter", command=self._stop,
                                  bg="#dc2626", fg="white", font=("Segoe UI", 10, "bold"),
                                  relief=tk.FLAT, padx=12, pady=6, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=8)

        self.lbl_infos = tk.Label(bar, text=f"Detection : {methode}",
                                  bg="white", fg="#475569", font=("Segoe UI", 9))
        self.lbl_infos.pack(side=tk.RIGHT, padx=8)

        # --- Zone video ---
        self.canvas = tk.Canvas(self.win, bg="#0f172a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # --- Barre du bas : compteur + FPS ---
        bottom = tk.Frame(self.win, bg="white")
        bottom.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.lbl_count = tk.Label(bottom, text="Objets : --", bg="white",
                                  font=("Segoe UI", 18, "bold"), fg="#15803d")
        self.lbl_count.pack(side=tk.LEFT)
        self.lbl_fps = tk.Label(bottom, text="", bg="white",
                                font=("Segoe UI", 9), fg="#64748b")
        self.lbl_fps.pack(side=tk.RIGHT)

        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------- source
    def _ouvrir_webcam(self):
        self._demarrer(cv2.VideoCapture(0, cv2.CAP_DSHOW), "webcam")

    def _ouvrir_video(self):
        chemin = filedialog.askopenfilename(
            title="Choisir une video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("Tous", "*.*")],
            parent=self.win)
        if not chemin:
            return
        cap = cv2.VideoCapture(chemin)
        if not cap.isOpened():
            messagebox.showerror("Erreur", f"Impossible d'ouvrir :\n{chemin}", parent=self.win)
            return
        self._demarrer(cap, os.path.basename(chemin))

    def _demarrer(self, cap, source):
        self._stop_capture()
        self.cap = cap
        if not self.cap.isOpened():
            messagebox.showerror("Erreur", "Camera / video non disponible.", parent=self.win)
            return
        self.running = True
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_infos.config(text=f"Source : {source}  |  Detection : {self.methode}")
        self.thread = threading.Thread(target=self._boucle, daemon=True)
        self.thread.start()

    def _stop(self):
        self.running = False
        self.btn_stop.config(state=tk.DISABLED)
        self._stop_capture()
        self.lbl_count.config(text="Objets : --")
        self.lbl_fps.config(text="")

    def _stop_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    # --------------------------------------------------------------- boucle
    def _boucle(self):
        while self.running and self.cap is not None and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.win.after(0, lambda: self.lbl_infos.config(text="Fin de video."))
                self.win.after(0, self._stop)
                return
            # Recadre pour la vitesse (largeur max 640)
            h, w = frame.shape[:2]
            if w > 640:
                k = 640.0 / w
                frame = cv2.resize(frame, (640, int(h * k)), interpolation=cv2.INTER_AREA)
            try:
                frame_traite, n = self._traiter(frame)
            except Exception:
                frame_traite, n = frame, self.n_last
            self.n_last = n
            # Affichage via PhotoImage (PNG encode)
            rgb = cv2.cvtColor(frame_traite, cv2.COLOR_BGR2RGB)
            ok2, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not ok2:
                continue
            data = buf.tobytes()
            self.win.after(0, lambda d=data, cnt=n: self._afficher(d, cnt))

    def _afficher(self, png_bytes, n):
        try:
            from PIL import Image, ImageTk  # Pillow si dispo
            img = Image.open(io.BytesIO(png_bytes))
            photo = ImageTk.PhotoImage(image=img)
        except Exception:
            photo = tk.PhotoImage(data=png_bytes)
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.canvas.create_image(cw // 2, ch // 2, image=photo)
        self._photo_ref = photo  # evite GC
        self.lbl_count.config(text=f"Objets : {n}")

    # ----------------------------------------------------------- detection
    def _traiter(self, bgr):
        # etalonnage auto sur les premieres frames si methode = AUTO (None)
        if self.methode is None and not self._calibre:
            try:
                from .auto_engine import recherche_auto
                r = recherche_auto(bgr)
                if r is not None and r.get("n", 0) > 0:
                    self.methode = r["methode"]
                    p = r["params"]
                    self.params = dict(self.params)
                    for k in ("sens", "canny", "dist", "aire_min", "aire_max"):
                        if k in p:
                            self.params[k] = int(p[k])
                    self.params["methode"] = p.get("methode", "otsu")
                    self.params["inverser"] = p.get("inverser", True)
                else:
                    self.methode = "watershed"
            except Exception:
                self.methode = "watershed"
            self._calibre = True
        m = self.methode or "watershed"

        # --- Cercles ---
        if m == "cercles":
            sens = int(self.params.get("sens", 28))
            canny = int(self.params.get("canny", 70))
            dist = int(self.params.get("dist", 40))
            r = detect_cercles(bgr, sens, canny, dist)
            if r["n"] == 0:
                # fallback: essayer avec plus de sensibilite
                r = detect_cercles(bgr, 45, 50, 30)
            return r["finale_brute"], r["n"]

        # --- Visages ---
        if m == "visages":
            aire_min = int(self.params.get("aire_min", 100))
            aire_max = int(self.params.get("aire_max", 200000))
            r = detect_visages(bgr, 5, aire_min, aire_max)
            return r["finale_brute"], r["n"]

        # --- Watershed (par defaut) : essayer les deux sens, puis baisser aire_min si vide ---
        aire_min = int(self.params.get("aire_min", 100))
        aire_max = int(self.params.get("aire_max", 200000))
        methode_ws = self.params.get("methode", "otsu")
        inverser = bool(self.params.get("inverser", True))
        r = detect_watershed(bgr, methode_ws, inverser, True, aire_min, aire_max)
        if r["n"] == 0:
            # essayer l'inverse (objets clairs sur fond sombre ou inversement)
            r = detect_watershed(bgr, methode_ws, not inverser, True, aire_min, aire_max)
        if r["n"] == 0:
            # baisser fortement le seuil d'aire (frames redimensionnes a 640 px de large)
            r = detect_watershed(bgr, methode_ws, inverser, True, max(20, aire_min // 4), aire_max)
        if r["n"] == 0:
            r = detect_watershed(bgr, methode_ws, not inverser, True, max(20, aire_min // 4), aire_max)
        return r["finale_brute"], r["n"]

    def _on_close(self):
        self._stop()
        self.win.destroy()
