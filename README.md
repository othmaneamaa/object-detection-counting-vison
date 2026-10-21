# Object Detection and Counting

Comptage d'objets avec des **methodes classiques OpenCV** (aucun deep learning).

## Lancement

```bash
pip install -r requirements.txt
py -I app.py
```

## Methodes

| Methode | Cas d'usage |
|---------|-------------|
| **AUTO** | Teste plusieurs methodes/parametres et retient le meilleur |
| **Billes rondes (HoughCircles + CLAHE)** | Billes, pieces, objects circulaires |
| **Formes quelconques (Otsu + watershed)** | N'importe quelle forme, separe les objets colles |
| **Visages (Haar Cascade)** | Visages frontaux |

## Fonctionnalites

- **AUTO rapide** : la recherche se fait sur une version reduite de l'image, le gagnant est re-execute en pleine resolution. Barre de progression + bouton **Annuler** pendant l'analyse (l'interface reste fluide).
- **Histogramme des tailles** : visualise la distribution des aires des objets detectes (fenetre separee avec moyenne, mediane, ecart-type).
- **Export image + CSV** : enregistre l'image annotee (PNG) et un fichier CSV avec la methode, le score et les coordonnees de chaque objet.
- **Mode lot** : analyse tout un dossier d'images d'un coup, affiche un tableau recapitulatif et exporte le CSV global.
- **Correction manuelle** : activez "Correction manuelle", puis cliquez sur l'image resultats — clic sur un objet = le retirer, clic sur le vide = ajouter un objet.

## Structure du code

```
app.py                  lanceur
src/
  config.py             constantes (seuils, extensions)
  drawing.py            bandeau, numeros, support perimetrique
  detectors/
    circles.py          HoughCircles
    watershed.py        Otsu + morpho + watershed
    faces.py            Haar Cascade
  auto_engine.py        moteur AUTO (grille + score + stabilite)
  stats.py              histogramme des tailles
  export.py             export image/CSV, mode lot
  ui.py                 interface Tkinter
```

## Aide rapide

1. **Choisir une image** (ou un dossier pour le mode lot)
2. Laisser **AUTO** coche et cliquer sur **"AUTO : analyser et compter"**
3. Resultat : image avec objets encadres / cerclés + compteur
4. Outils : histogramme, export, correction manuelle
