import itertools
import cv2
import numpy as np

from .config import SEUIL_AIRE_MAX, AUTO_MAX_DIM
from .detectors import detect_cercles, detect_watershed, detect_visages


def _stabilite_voisins(n_ref, voisins):
    if n_ref == 0 and all(v == 0 for v in voisins):
        return 0.6
    if not voisins:
        return 0.5
    diffs = [abs(v - n_ref) for v in voisins]
    ecart = float(np.mean(diffs)) / max(n_ref, 1)
    return float(1.0 / (1.0 + 3.0 * ecart))


def _resize_pour_auto(bgr):
    h, w = bgr.shape[:2]
    k = min(1.0, AUTO_MAX_DIM / max(h, w))
    if k >= 0.99:
        return bgr, 1.0
    return cv2.resize(bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA), k


def recherche_auto(bgr, *, progress=None, cancel=None):
    """
    Retourne le dict gagnant {methode, params, n, score, ...} ou None.
    - progress(idx, total) : appele apres chaque combinaison.
    - cancel() -> bool     : si vrai, abandonne et retourne None.
    Reexecute le gagnant en pleine resolution pour les images finales.
    """
    bgr_small, k = _resize_pour_auto(bgr)
    h, w = bgr_small.shape[:2]
    aire_img = h * w
    aire_min_auto = max(60, aire_img // 8000)
    aire_max_auto = min(SEUIL_AIRE_MAX, aire_img // 2)

    # Nombre total pour la barre de progression
    total = 18 + 8 + 3
    idx = 0

    candidats = []

    # ---- Cercles : 18 combos ----
    for sens, canny_thr, dist_min in itertools.product(
            [22, 30, 40], [50, 70, 95], [28, 50]):
        if cancel and cancel():
            return None
        try:
            r = detect_cercles(bgr_small, sens, canny_thr, dist_min)
        except Exception:
            if progress:
                progress((idx := idx + 1), total)
            continue
        n, avg_s = r["n"], r["avg_support"]
        if n == 0:
            score_base = 0.05
        else:
            densite_ok = 1.0 if 1 <= n <= 80 else max(0.3, 1.0 - (n - 80) * 0.015)
            score_base = (0.70 * avg_s + 0.15 * min(n / 10, 1.0) + 0.15 * densite_ok)
            if r["r_min"] < 12 and n > 15:
                score_base *= 0.9
        voisins_n = []
        for ds, dc in [(4, 0), (-4, 0), (0, 12), (0, -12)]:
            try:
                rv = detect_cercles(bgr_small, max(10, min(60, sens + ds)),
                                     max(30, min(150, canny_thr + dc)), dist_min)
                voisins_n.append(rv["n"])
            except Exception:
                pass
        stab = _stabilite_voisins(n, voisins_n)
        score = 0.72 * score_base + 0.28 * stab
        if n > 0 and stab > 0.85:
            score += 0.04
        candidats.append(dict(
            methode="cercles", params=dict(sens=sens, canny=canny_thr, dist=dist_min),
            n=n, score=float(np.clip(score, 0, 1)), avg_support=avg_s, stab=stab,
            images=(r["egalisee"], r["canny"], r["finale_brute"]), r_min=r["r_min"], r_max=r["r_max"]))
        if progress:
            progress((idx := idx + 1), total)

    # ---- Watershed : 8 combos ----
    for methode, inverser in itertools.product(["otsu", "adaptatif"], [True, False]):
        for aire_min in [aire_min_auto, max(aire_min_auto * 3, 250)]:
            if cancel and cancel():
                return None
            try:
                r = detect_watershed(bgr_small, methode, inverser, True, aire_min, aire_max_auto)
            except Exception:
                if progress:
                    progress((idx := idx + 1), total)
                continue
            n, qual = r["n"], r["qual"]
            if n == 0:
                score_base = 0.02
            elif n == 1:
                score_base = qual * 0.6
            elif n > 80:
                score_base = qual * 0.6
            else:
                score_base = qual
            voisins_n = []
            for da in [int(aire_min * 0.5), int(aire_min * 1.6)]:
                try:
                    rv = detect_watershed(bgr_small, methode, inverser, True, da, aire_max_auto)
                    voisins_n.append(rv["n"])
                except Exception:
                    pass
            stab = _stabilite_voisins(n, voisins_n)
            bonus = 0.03 if (methode == "otsu" and 2 <= n <= 40) else 0
            score = 0.70 * score_base + 0.30 * stab + bonus
            candidats.append(dict(
                methode="watershed",
                params=dict(methode=methode, inverser=inverser, aire_min=aire_min, aire_max=aire_max_auto),
                n=n, score=float(np.clip(score, 0, 1)), qual=qual, stab=stab,
                images=(r["egalisee"], r["morpho"], r["finale_brute"])))
            if progress:
                progress((idx := idx + 1), total)

    # ---- Visages : 3 combos ----
    for mn in [3, 5, 7]:
        if cancel and cancel():
            return None
        try:
            r = detect_visages(bgr_small, mn, aire_min_auto, aire_max_auto)
        except Exception:
            if progress:
                progress((idx := idx + 1), total)
            continue
        n, qual = r["n"], r["qual"]
        voisins_n = []
        for mn2 in [3, 5, 7]:
            if mn2 == mn:
                continue
            try:
                rv = detect_visages(bgr_small, mn2, aire_min_auto, aire_max_auto)
                voisins_n.append(rv["n"])
            except Exception:
                pass
        stab = _stabilite_voisins(n, voisins_n) if voisins_n else 0.5
        score_base = qual if n else 0.01
        score = 0.65 * score_base + 0.35 * stab
        candidats.append(dict(
            methode="visages",
            params=dict(min_neighbors=mn, aire_min=aire_min_auto, aire_max=aire_max_auto),
            n=n, score=float(np.clip(score, 0, 1)), qual=qual, stab=stab,
            images=(r["egalisee"], r["masque"], r["finale_brute"]),
            raw_n=r["raw_n"], taille_min=r["taille_min"]))
        if progress:
            progress((idx := idx + 1), total)

    if not candidats:
        return None

    candidats.sort(key=lambda c: (c["score"], c["n"]), reverse=True)
    gagnant = candidats[0]
    if gagnant["methode"] == "visages" and gagnant["n"] == 0:
        non_vis = [c for c in candidats if c["methode"] != "visages" and c["n"] > 0]
        if non_vis:
            non_vis.sort(key=lambda c: (c["score"], c["n"]), reverse=True)
            gagnant = non_vis[0]
    gagnant["candidats"] = candidats

    # Reexecute le gagnant en pleine resolution pour les images finales
    try:
        if gagnant["methode"] == "cercles":
            p = gagnant["params"]
            r = detect_cercles(bgr, p["sens"], p["canny"], p["dist"])
            gagnant["images"] = (r["egalisee"], r["canny"], r["finale_brute"])
            gagnant["n"] = r["n"]
            gagnant["objets"] = r["objets"]
        elif gagnant["methode"] == "watershed":
            p = gagnant["params"]
            # remettre a l'echelle les seuils d'aire
            h0, w0 = bgr.shape[:2]
            scale = (h0 * w0) / max(1, aire_img)
            amn = int(p["aire_min"] * scale)
            amx = int(p["aire_max"] * scale)
            r = detect_watershed(bgr, p["methode"], p["inverser"], True, amn, amx)
            gagnant["images"] = (r["egalisee"], r["morpho"], r["finale_brute"])
            gagnant["n"] = r["n"]
            gagnant["objets"] = r["objets"]
        else:
            p = gagnant["params"]
            h0, w0 = bgr.shape[:2]
            scale = (h0 * w0) / max(1, aire_img)
            amn = int(p["aire_min"] * scale)
            amx = int(p["aire_max"] * scale)
            r = detect_visages(bgr, p["min_neighbors"], amn, amx)
            gagnant["images"] = (r["egalisee"], r["masque"], r["finale_brute"])
            gagnant["n"] = r["n"]
            gagnant["objets"] = r["objets"]
    except Exception:
        pass

    gagnant.setdefault("objets", [])
    return gagnant
