import cv2
import numpy as np

from ..config import SEUIL_DISTANCE
from ..drawing import supprimer_boites_imbriquees


def _aire(boite):
    x, y, w, h = boite
    return w * h


def detect_watershed(bgr, methode, inverser, separer, aire_min, aire_max):
    """Retourne dict {n, qual, boites, egalisee, morpho, finale_brute, objets}."""
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flou = cv2.GaussianBlur(gris, (5, 5), 0)

    if methode == "adaptatif":
        mode = cv2.THRESH_BINARY_INV if inverser else cv2.THRESH_BINARY
        seuillee = cv2.adaptiveThreshold(flou, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         mode, 51, 5)
    else:
        mode = cv2.THRESH_BINARY_INV if inverser else cv2.THRESH_OTSU
        _, seuillee = cv2.threshold(flou, 0, 255, mode + cv2.THRESH_OTSU)

    noyau = np.ones((5, 5), np.uint8)
    ouverte = cv2.morphologyEx(seuillee, cv2.MORPH_OPEN, noyau)
    morpho = cv2.morphologyEx(ouverte, cv2.MORPH_CLOSE, noyau)

    finale = bgr.copy()
    boites = []
    if separer:
        fond_sur = cv2.dilate(morpho, noyau, iterations=3)
        dist = cv2.distanceTransform(morpho, cv2.DIST_L2, 5)
        if dist.max() > 0:
            _, avant = cv2.threshold(dist, SEUIL_DISTANCE * dist.max(), 255, 0)
            avant = np.uint8(avant)
            inconnue = cv2.subtract(fond_sur, avant)
            _, marqueurs = cv2.connectedComponents(avant)
            marqueurs = marqueurs + 1
            marqueurs[inconnue == 255] = 0
            marqueurs = cv2.watershed(bgr, marqueurs)
            for label in range(2, int(marqueurs.max()) + 1):
                masque = np.uint8(marqueurs == label) * 255
                aire = cv2.countNonZero(masque)
                if aire < aire_min or aire > aire_max:
                    continue
                x, y, w2, h2 = cv2.boundingRect(masque)
                boites.append((x, y, w2, h2))
        else:
            boites = []
    else:
        contours, _ = cv2.findContours(morpho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            aire = cv2.contourArea(c)
            if aire < aire_min or aire > aire_max:
                continue
            x, y, w2, h2 = cv2.boundingRect(c)
            boites.append((x, y, w2, h2))

    boites = supprimer_boites_imbriquees(boites)
    # Dessin
    for i, (x, y, w2, h2) in enumerate(boites, start=1):
        cv2.rectangle(finale, (x, y), (x + w2, y + h2), (0, 200, 0), 2)
        # Pas de numeros ici par defaut

    # Liste d'objets pour stats/histogramme/export
    objets = [(x, y, w2, h2, _aire((x, y, w2, h2))) for x, y, w2, h2 in boites]
    n = len(boites)
    # qualite : regularite des aires
    if n >= 2:
        aires = np.array([_aire(b) for b in boites], dtype=float)
        cv_aires = float(np.std(aires) / (np.mean(aires) + 1e-9))
        qual = 1.0 - min(cv_aires, 1.0)
    elif n == 1:
        qual = 0.45
    else:
        qual = 0.0

    return dict(n=n, qual=qual, boites=boites,
                egalisee=seuillee, morpho=morpho, finale_brute=finale,
                objets=objets)