# pivot-ai-handball

Pipeline computer vision pour scouting handball : detection, tracking, stats par joueur, decoupage automatique en actions.

## Architecture

```
pivot-ai-handball/
├── pivot_ai/                # Module principal
│   ├── detection.py         # Detection YOLO locale (inference GPU)
│   ├── tracking.py          # ByteTrack + stabilisation classes
│   ├── homographie.py       # Projection pixels -> coordonnees terrain
│   ├── equipes.py           # Classification equipes par KMeans
│   ├── radar.py             # Rendu radar 2D
│   ├── stats.py             # Stats joueur (distance, vitesse, heatmap)
│   ├── decoupage.py         # Decoupage automatique en actions
│   ├── pipeline.py          # Orchestrateur end-to-end
│   ├── config.py            # Constantes et schemas
│   └── cli.py               # CLI `pivot-ai`
├── notebooks/
│   ├── pivot_ai.ipynb      # Inférence locale
│   └── train_handball_yolo.ipynb  # Entraînement local
├── scripts/
│   └── build_notebook.py    # Generation du notebook depuis Python
├── tests/                   # Tests unitaires
├── docs/
│   └── correspondances_example.json  # Modele homographie 10 points
├── legacy/                  # Code monolithique d'origine (reference)
├── data/                    # Videos, caches, outputs (ignored par git)
└── pyproject.toml
```

## Modele de detection : COCO generique vs fine-tune handball

Par defaut, le pipeline utilise **YOLOv8m pre-entraine COCO** : il ne connait que la classe
generique "person". Consequence : pas de distinction joueur / gardien / arbitre / ballon, et un
tracking fragmente (beaucoup d'IDs pour peu de joueurs reels) car les detections sont bruitees.

Pour un vrai saut de qualite, **entraine un modele specialise handball** :

1. Ouvre `notebooks/train_handball_yolo.ipynb` avec le noyau Python du projet sous WSL et un GPU CUDA.
2. Choisis un export parmi ceux recensés dans [la provenance des datasets](data/datasets/README.md).
   Le notebook le cherche dans `data/datasets/`. Pour télécharger un export absent, utilise
   une clé Roboflow dans la variable d’environnement `ROBOFLOW_API_KEY`. Installe au besoin le SDK avec
   `python -m pip install roboflow`.
3. Le notebook entraîne YOLOv8m, affiche les métriques et exporte les meilleurs poids vers
   `models/handball_yolov8m.pt`.
4. Utilise le chemin affiché à l’export dans la cellule « Modele de detection » du notebook
   d’inférence.

Le pipeline remappe les classes du modele vers `CLASSES_HANDBALL` **par nom** (pas par ordre), donc
peu importe l'ordre des classes dans le dataset d'entrainement.

Depuis Python, indique le modèle explicitement :

```python
from pivot_ai.config import ModeleConfig
from pivot_ai.pipeline import traiter_match_complet

config = ModeleConfig.pour_handball("models/handball_yolov8m.pt")
traiter_match_complet(..., modele_config=config)
```

### Extraire des images à annoter avec FFmpeg

Depuis la racine du dépôt, crée le dossier de destination par défaut.

```bash
mkdir -p data/annotation/images
```

**Échantillonner une vidéo toutes les deux secondes**

```bash
start_index=1
ffmpeg -n -i data/raw_clips/mon_match.mp4 -vf "fps=1/2" -q:v 2 \
  -start_number "$start_index" \
  data/annotation/images/mon_match_%05d.jpg
```

`fps=1/2` extrait une image toutes les deux secondes. Utilise `fps=1/5` pour
espacer les images de cinq secondes. `%05d` numérote les images sur cinq chiffres.
Pour reprendre après `mon_match_00060.jpg`, règle `start_index=61`. Cette valeur
est transmise à l’option FFmpeg `-start_number` et règle le numéro du premier JPEG.

**Extraire des images sur un passage de 30 secondes**

```bash
start_index=1
ffmpeg -n -ss 00:12:30 -i data/raw_clips/mon_match.mp4 -t 30 \
  -vf "fps=2" -q:v 2 -start_number "$start_index" \
  data/annotation/images/mon_match_12m30_%05d.jpg
```

**Extraire une seule image à un instant précis**

```bash
ffmpeg -n -ss 00:12:42.500 -i data/raw_clips/mon_match.mp4 -frames:v 1 \
  -q:v 2 data/annotation/images/mon_match_12m42s500.jpg
```

Les images gardent la résolution de la vidéo. `-q:v 2` produit des JPEG de haute
qualité et `-n` protège les fichiers existants. Change le préfixe pour chaque
vidéo ou passage afin de garder des noms distincts.

### Découper une vidéo de match en extraits

**Extraire 30 secondes à partir de 12 min 30 s**

```bash
ffmpeg -n -ss 00:12:30 -i data/raw_clips/mon_match_complet.mp4 -t 30 \
  -map 0:v:0 -map '0:a:0?' -c:v libx264 -crf 18 -c:a aac \
  data/raw_clips/mon_match_12m30_30s.mp4
```

Remplace `00:12:30` par l’instant de départ et `30` par la durée souhaitée en
secondes. La résolution est conservée. L’audio est inclus si la vidéo en contient.

**Découper le match entier en extraits successifs de 30 secondes**

```bash
duree=30
dossier="data/raw_clips/mon_match_${duree}s"
mkdir "$dossier" && ffmpeg -i data/raw_clips/mon_match_complet.mp4 \
  -map 0:v:0 -map '0:a:0?' -c:v libx264 -crf 18 -c:a aac \
  -force_key_frames "expr:gte(t,n_forced*${duree})" \
  -f segment -segment_time "$duree" -reset_timestamps 1 \
  "$dossier/extrait_%03d.mp4"
```

Modifie `duree` pour choisir la longueur des extraits. Le dernier peut être plus
court. Le découpage démarre uniquement si le dossier de destination est créé,
ce qui évite de remplacer des extraits existants.

## Installation locale sous WSL

Dans Ubuntu (WSL), depuis la racine du dépôt, utilise Python 3.11 et FFmpeg.
Avec Python 3.11 déjà installé, crée l’environnement avec `python3.11 -m venv .venv`.
Avec [uv](https://docs.astral.sh/uv/getting-started/installation/), tu peux installer
Python 3.11 et créer l’environnement en une commande.

```bash
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cu132 -c .devcontainer/constraints.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

La dernière commande doit afficher `True` pour utiliser le GPU NVIDIA.

### Lancer les notebooks

Lance `jupyter notebook` depuis le dépôt avec le `venv` activé. Dans VS Code,
ouvre le dossier depuis un terminal WSL avec `code .` et sélectionne
`.venv/bin/python` comme interpréteur et comme noyau Jupyter.
Pour enregistrer un noyau nommé, lance depuis l’environnement activé
`python -m ipykernel install --user --name pivot-ai-handball-wsl --display-name "Python (Pivot AI Handball WSL)"`.

Dans `notebooks/pivot_ai.ipynb`, indique le nom du clip déposé dans
`data/raw_clips/`. Le notebook charge le JSON de calibration à côté de la vidéo.
Les résultats sont écrits dans `data/outputs/`.

### Lancer le pipeline via CLI

```bash
pivot-ai traiter \
  --video data/raw_clips/mon_clip.mp4 \
  --output data/outputs/ \
  --subsample 2
```

Flags optionnels :
- `--debut HH:MM:SS:cc` indique où commencer dans la vidéo, avec `cc` en centièmes de seconde (`00:00:00:00` par défaut). La notation FFmpeg `HH:MM:SS.cc` est aussi acceptée.
- `--duree SECONDES` limite le traitement à cette durée à partir du début choisi. Sans cette option, le traitement va jusqu'à la fin de la vidéo.
- `--modele models/handball_yolov8m.pt` : utilise les poids YOLO entraînés sur le handball. Par défaut, la CLI utilise YOLOv8m COCO.
- `--homographie CHEMIN` : remplace le chemin de calibration déduit de la vidéo.
- `--no-video-radar` désactive la vidéo de sortie. Par défaut, elle empile la source annotée et la projection. Chaque panneau mesure 1920 × 540 (32:9).
- `--no-fisheye` désactive la correction radiale du radar. Par défaut, si A à F sont présents, elle utilise A–E–B et C–F–D et recale les quatre coins sur un rectangle de 40 × 20 m. La vidéo source reste intacte.
- `--equipes` active la classification par couleur de maillot.
- `--stats` active le calcul et l'export des statistiques. Combinez les deux options pour exporter aussi les largeurs des blocs défensifs.
- `--no-decoupage` : ne decoupe pas les actions detectees
- `--tracker {bytetrack,botsort}` : backend de tracking. `bytetrack` (defaut,
  rapide, mouvement seul) ou `botsort` (ReID par apparence : maintient les IDs
  au contact/occlusion, plus lent). `botsort` necessite un vrai modele YOLO.
- `--min-frames-track N` : ecarte les trackers vus sur moins de N frames (fragments)

Par exemple, ajoutez `--debut 00:01:00:25 --duree 30` pour analyser 30 secondes à partir de 1 min et 25 centièmes. Le pipeline se positionne sur l'image la plus proche dans la vidéo source et traite directement la séquence. Les horodatages des résultats repartent de zéro.

Codes de sortie :
- `0` : succes
- `1` : erreur runtime (video ou homographie invalide, exception pipeline)
- `2` : argument obligatoire manquant

### Créer la calibration à la souris

Depuis la racine du dépôt, dans un environnement avec affichage graphique et Tkinter
(`python3-tk` sous Debian ou Ubuntu), lance :

```bash
python -m scripts.calibrer_terrain data/raw_clips/mon_clip.mp4
```

Le script affiche la frame du milieu. Utilise `--frame 250` pour choisir une autre
frame. Pour calibrer à partir d’une image fixe, passe l’image en entrée et ajoute
`--video-reference chemin/du/match.mp4`. Clique sur un repère du schéma sous l’image,
puis clique sur son emplacement dans l’image ou dans la marge grise. La molette
zoome autour du pointeur. Fais glisser avec le bouton droit pour déplacer la vue,
ou clique sur **Ajuster l’image** pour revenir à la vue d’ensemble.
Par défaut, la marge ajoute 10 %
de la largeur à gauche et à droite, et 10 % de la hauteur en haut et en bas.
Utilise `--marge 150` pour régler les quatre côtés à 150 pixels. Les coordonnées
du JSON restent relatives à l’image d’origine et peuvent donc dépasser son cadre.
Place au moins quatre repères répartis sur le terrain,
puis clique sur **Enregistrer**. Pour `data/raw_clips/mon_clip.mp4`, le script écrit
`data/raw_clips/mon_clip.mp4.correspondances.json`. Le notebook et la CLI retrouvent
ce fichier automatiquement.
S’il existe déjà, indique une autre destination avec `--sortie` ou utilise `--force`
pour le remplacer.

### Format du fichier homographie

JSON, minimum 4 points (>= 5 active RANSAC). Voir `docs/correspondances_example.json` pour un modele.

```json
{
  "A": {"pixel": [120, 540], "terrain_m": [0.0, 0.0]},
  "B": {"pixel": [1800, 540], "terrain_m": [40.0, 0.0]},
  "C": {"pixel": [1700, 300], "terrain_m": [40.0, 20.0]},
  "D": {"pixel": [220, 300], "terrain_m": [0.0, 20.0]}
}
```

Les cles commencant par `_` sont traitees comme commentaires (utile pour documenter le JSON inline).

## Tests

```bash
pytest tests/ -v --cov=pivot_ai
```

Tests qui requierent `ffmpeg` (3 tests de decoupage video) : auto-skip si `ffmpeg` n'est pas dans le PATH.

## Conventions de code

- Python 3.11+, type hints obligatoires sur les signatures publiques
- Variables et fonctions en **snake_case francais** : `traiter_clip`, `calculer_distance_parcourue`, `dets_par_frame`
- Docstrings courtes en francais, style Google
- Logging au lieu de print
- Pas de bare except
- Tests pytest pour toute fonction critique
- Ruff active avec `select = ["E", "F", "I", "N", "W", "UP", "B", "SIM"]`

## Roadmap

- [x] Squelette repo
- [x] Refactor du code legacy en modules
- [x] Migration Roboflow Cloud -> inference locale GPU (`DetecteurLocal`)
- [x] Module stats joueur (distance, vitesse, heatmap, bloc defensif)
- [x] Decoupage actions par heuristique simple
- [x] Pipeline match complet (avec interpolation et vidéo radar)
- [x] CLI `pivot-ai traiter`
- [x] Notebook d’inférence local
- [x] Fine-tune modele YOLO handball (4 classes) + notebook d'entrainement
- [x] Tracker BoT-SORT + ReID (maintient les IDs au contact) en option
- [ ] Detection ballon -> possession et passes
- [ ] Detection d'evenements (tirs, buts) par heuristique
- [ ] Migration `sv.ByteTrack` vers le nouveau tracker supervision (`>=0.30`)
- [ ] Determination automatique de l'equipe en attaque (sens de jeu)
- [ ] Detection automatique lignes terrain (homographie auto)
- [ ] Identification individuelle joueurs (numeros maillots)
- [ ] Classification d'evenements ML

## Licence

Proprietaire. Tous droits reserves.
