"""
Object Detection and Counting - Version Finale V5 (modulaire)

Lancement : py -I app.py

Structure :
  app.py              -> lanceur (ce fichier)
  src/config.py       -> constantes
  src/drawing.py      -> utilitaires d'affichage OpenCV
  src/detectors/      -> methodes de detection (cercles, watershed, visages)
  src/auto_engine.py  -> moteur AUTO (grille + score + stabilite, rapide)
  src/stats.py        -> histogramme des tailles d'objets
  src/export.py       -> export image + CSV, mode lot
  src/ui.py           -> interface graphique Tkinter

Nouveautes V5 :
  - interface non bloquante (AUTO en thread + barre de progression + Annuler)
  - AUTO rapide (recherche sur image reduite, re-execution gagnante en pleine resolution)
  - histogramme des tailles
  - export image + CSV
  - mode lot (dossier entier) avec tableau recapitulatif
  - correction manuelle par clic (ajouter / retirer un objet)
"""

import os
import sys

# Limite les threads OpenCV / FFmpeg : evite "fctx->async_lock failed"
# (conflit du decodeur FFmpeg multithread sur webcam ET fichiers video).
# Doit etre pose AVANT l'import de cv2.
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"
os.environ["OPENCV_OPENMP_NUM_THREADS"] = "1"

# py -I isole sys.path : on ajoute le dossier du projet manuellement
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui import main

if __name__ == "__main__":
    main()
