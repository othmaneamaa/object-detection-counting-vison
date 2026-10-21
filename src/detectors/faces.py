import cv2
import numpy as np


def detect_visages(bgr, min_neighbors, aire_min, aire_max):
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    egalisee = clahe.apply(gris)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return dict(n=0, qual=0.0, boites=[], egalisee=egalisee, masque=np.zeros_like(gris),
                    finale_brute=bgr.copy(), objets=[], raw_n=0, taille_min=(30, 30))

    h, w = gris.shape
    taille_min = (max(30, w // 12), max(30, h // 12))
    faces = face_cascade.detectMultiScale(
        egalisee, scaleFactor=1.08, minNeighbors=min_neighbors,
        minSize=taille_min, flags=cv2.CASCADE_SCALE_IMAGE)
    raw_n = len(faces)

    boites = []
    for (x, y, w2, h2) in faces:
        aire = w2 * h2
        if aire < aire_min or aire > aire_max:
            continue
        boites.append((int(x), int(y), int(w2), int(h2)))

    # Suppression recouvrements
    boites = sorted(boites, key=lambda b: b[2] * b[3], reverse=True)
    gardees = []
    for x, y, w2, h2 in boites:
        cx, cy = x + w2 / 2, y + h2 / 2
        recouvrement = False
        for gx, gy, gw, gh in gardees:
            ix1, iy1 = max(x, gx), max(y, gy)
            ix2, iy2 = min(x + w2, gx + gw), min(y + h2, gy + gh)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            petite = min(w2 * h2, gw * gh)
            if petite > 0 and inter / petite > 0.4:
                recouvrement = True
                break
            if gx <= cx <= gx + gw and gy <= cy <= gy + gh:
                recouvrement = True
                break
        if not recouvrement:
            gardees.append((x, y, w2, h2))

    finale = bgr.copy()
    ep = max(2, w // 350)
    for i, (x, y, w2, h2) in enumerate(gardees, start=1):
        cv2.rectangle(finale, (x, y), (x + w2, y + h2), (0, 200, 0), ep)

    masque = np.zeros_like(gris)
    for (x, y, w2, h2) in gardees:
        cv2.rectangle(masque, (x, y), (x + w2, y + h2), 255, -1)

    objets = [(x, y, w2, h2, w2 * h2) for x, y, w2, h2 in gardees]
    n = len(gardees)
    if n > 0:
        qual = 0.55 + 0.35 * (n / max(raw_n, 1))
        if n >= 3:
            qual = min(qual + 0.07, 0.95)
    elif raw_n > 0:
        qual = 0.15
    else:
        qual = 0.0

    return dict(n=n, qual=qual, boites=gardees, egalisee=egalisee,
                masque=masque, finale_brute=finale, objets=objets,
                raw_n=raw_n, taille_min=taille_min)