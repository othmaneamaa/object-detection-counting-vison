import cv2
import numpy as np

from ..config import SUPPORT_MIN
from ..drawing import dessiner_numero, support_cercle


def detect_cercles(bgr, sens, seuil_canny, dist_min):
    """Retourne dict {n, avg_support, gardes, egalisee, canny, finale_brute, r_min, r_max}."""
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    egalisee = clahe.apply(gris)
    flou = cv2.medianBlur(egalisee, 5)
    flou = cv2.GaussianBlur(flou, (9, 9), 2)

    h, w = gris.shape
    petit = min(h, w)
    r_min = max(10, petit // 60)
    r_max = max(50, petit // 4)

    cercles = cv2.HoughCircles(
        flou, cv2.HOUGH_GRADIENT, dp=1.0, minDist=dist_min,
        param1=seuil_canny, param2=sens,
        minRadius=r_min, maxRadius=r_max)

    canny = cv2.Canny(flou, max(20, seuil_canny // 2), seuil_canny)

    candidats = []
    if cercles is not None:
        for x, y, r in np.uint16(np.around(cercles[0])):
            s = support_cercle(canny, int(x), int(y), int(r))
            if s >= SUPPORT_MIN:
                candidats.append((int(x), int(y), int(r), s))

    candidats.sort(key=lambda c: (c[3], c[2]), reverse=True)
    gardes = []
    for x, y, r, s in candidats:
        if all(np.hypot(x - gx, y - gy) >= 0.55 * (r + gr)
               for gx, gy, gr, _ in gardes):
            gardes.append((x, y, r, s))

    finale = bgr.copy()
    ep = max(2, w // 350)
    objets = []  # liste homogene (x, y, w, h, aire) pour stats / histogramme / export
    for i, (x, y, r, _) in enumerate(gardes, start=1):
        cv2.circle(finale, (x, y), r, (0, 200, 0), ep)
        cv2.circle(finale, (x, y), 3, (0, 0, 255), -1)
        dessiner_numero(finale, str(i), x, y)
        aire = int(np.pi * r * r)
        objets.append((x - r, y - r, 2 * r, 2 * r, aire))

    n = len(gardes)
    avg_support = float(np.mean([s for _, _, _, s in gardes])) if gardes else 0.0
    return dict(n=n, avg_support=avg_support, gardes=gardes, objets=objets,
                egalisee=egalisee, canny=canny, finale_brute=finale,
                r_min=r_min, r_max=r_max)

