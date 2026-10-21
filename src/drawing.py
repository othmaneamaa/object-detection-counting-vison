import cv2
import numpy as np


def ajouter_bandeau(image, texte):
    """Ajoute un bandeau sombre en haut avec le compteur."""
    bandeau = cv2.copyMakeBorder(image, 42, 0, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(15, 23, 42))
    cv2.putText(bandeau, texte, (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return bandeau


def dessiner_numero(image, texte, cx, cy):
    font = cv2.FONT_HERSHEY_SIMPLEX
    echelle, ep = 0.8, 2
    (tw, th), _ = cv2.getTextSize(texte, font, echelle, ep)
    ox, oy = cx - tw // 2, cy + th // 2
    cv2.putText(image, texte, (ox, oy), font, echelle, (0, 0, 0), ep + 2)
    cv2.putText(image, texte, (ox, oy), font, echelle, (255, 255, 255), ep)


def support_cercle(canny, x, y, r, nb_points=36):
    """Part du perimetre presente sur les contours Canny (36 pts = rapide)."""
    h, w = canny.shape
    ok = total = 0
    for k in range(nb_points):
        a = 2.0 * np.pi * k / nb_points
        cx = int(round(x + r * np.cos(a)))
        cy = int(round(y + r * np.sin(a)))
        if 0 <= cx < w and 0 <= cy < h:
            total += 1
            if np.any(canny[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] > 0):
                ok += 1
    return ok / max(total, 1)


def supprimer_boites_imbriquees(boites):
    triees = sorted(boites, key=lambda b: b[2] * b[3], reverse=True)
    gardees = []
    for x, y, w, h in triees:
        cx, cy = x + w / 2, y + h / 2
        imbriquee = any((gx - 5 <= cx <= gx + gw + 5) and
                        (gy - 5 <= cy <= gy + gh + 5)
                        for gx, gy, gw, gh in gardees)
        if not imbriquee:
            gardees.append((x, y, w, h))
    return gardees


def cercles_vers_boites(gardes):
    """[(x, y, r, s)] -> [(x, y, w, h)]."""
    return [(x - r, y - r, 2 * r, 2 * r) for x, y, r, _ in gardes]
