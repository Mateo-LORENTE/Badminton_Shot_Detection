# 🏸 BadTrackNet - Analyse vidéo intelligente de matchs de badminton

Ce projet est une pipeline complète pour l’analyse automatique de matchs de **badminton** à partir de vidéos, en utilisant l’intelligence artificielle pour :
- détecter le volant image par image,
- analyser le spectre sonore,
- détecter les frappes (strikes),
- prédire le type de frappe (top ou bottom),
- annoter automatiquement la vidéo.

Il combine le modèle **TrackNetV2** pour le tracking du volant avec des modèles de **computer vision** créer et entrainer sur mesure pour l'analyse des spectres audio, ainsi qu’un pipeline de traitement audio avancé.

## 📁 Structure du projet

```
FFBADMINTON

ffbadminton/
│── README.md                # présentation du projet
│── requirements.txt         # dépendances Python
│── .gitignore               # fichiers à ignorer par git
│── .gitlab-ci.yml           # CI/CD
│
├── 0_ARCHIVE/               # Travail des stagiaires précendents 
│
├── 1_FFBAD/                 # code source principal
│   ├── __init__.py
│   └── inference/           # scripts pour l’inférence
│       ├── complete_tracking_TrackNetV2.py     # Pipeline principale 
│       ├── extract_sons_pipeline.py
│       ├── camera_change.py
│       ├── extract_trajectoire.py
│       ├── modele_que_volant_CNN0.h5
│       ├── strike_cnn_feat_new.pt
│       └── TrackNetV2/
│           ├── 3_in_1_out
│           └── three_in_three_out
│               └── predict3.py
│   
│   
├── 2_vidIN/                # dir de input video
│   
├── 2_vidOUT/               # dir de output

```

## ⚙️ Technologies utilisées

- Python 3.9+
- PyTorch & TensorFlow (modèles deep learning)
- OpenCV, librosa, scikit-learn, pandas, matplotlib
- TrackNetV2 (détection du volant)
- moviepy (extraction audio)
- CNN + RNN pour la classification des frappes

## 🔧 Installation et création environnement virtuel python

Création de l'environnement virtuel python

```bash
python -m venv .venv
```

Activation de l'environnement 

```bash
.venv\Scripts\Activate.ps1
```

Installe les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

> **Note :** Certains packages comme `librosa`, `tensorflow`, `torch`, `opencv-python` sont essentiels.

## 🚀 Exécution du script principal

### Depuis une vidéo locale :

```bash
python complete_tracking_TrackNetV2.py \
  --inputs_path /chemin/2_vidIN/nomVID.mp4 \
  --outputs_path /chemin/2_vidOUT
```


Le script :
1. Télécharge la vidéo si besoin
2. Génére une image de référence
3. Applique TrackNetV2 sur les scènes valides
4. Extrait les signaux audio
5. Génère les features et spectrogrammes
6. Applique les modèles CNN et GRU
7. Sauvegarde les résultats et une vidéo annotée

## 📦 Composants principaux

- `complete_tracking_TrackNetV2.py` :
  - Point d’entrée du script
  - Gestion des vidéos, du modèle TrackNet
  - prédictions via trajectoires du volant, et via classification des pics sonores.
  - fusion des prédictions
  - Annotation finale sur la vidéo

- `extract_sons_pipeline.py` :
  - Extraction audio, traitement du signal, détection des pics sonores
  - Extraction des features audio (ZCR, MFCC, flux spectral, spectrogramme, etc..)

- `extract_trajectoire.py` :
  - Prétraitement et interpolation des trajectoires du volant
  - Formatage pour le modèle GRU

## ✅ Résultats générés

- `*_predict.mp4` : vidéo avec les points de tracking du volant
- `*_annotated.mp4` : vidéo annotée avec les frappes (top / bottom) + trajectoires du volant
- `*_df_predict.pkl` : DataFrame des prédiction de frappe.

## 📊 Modèles utilisés

| Composant        | Modèle                        | Emplacement                             |
|------------------|-------------------------------|-----------------------------------------|
| Tracking volant  | TrackNetV2 (Keras)            | `TrackNet/TrackNetv2-master/model906_30.h5` |
| Détection frappe | CNN + features audio (PyTorch)| `strike_cnn_feat_new.pt`                |
| Type de frappe   | GRU + trajectoires (Keras)    | `modele_que_volant_CNN0.h5`             |

## 🖼️ Schéma de la pipeline

Voici un aperçu visuel de la pipeline complète :

![Pipeline globale](docs/img/pipeline_global.png)

Si nécessaire, voici aussi les sous-modules détaillés :

- **Bloc audio**  
  ![Pipeline audio](docs/img/pipeline_audio.png)

- **Bloc trajectoire**  
  ![Pipeline trajectoires](docs/img/pipeline_traj.png)

## 📎 Remarques

- Le système est sensible à l’angle de caméra : il détecte automatiquement les scènes avec un angle classique.
- La vidéo doit comporter une piste audio.
- Les modèles sont à placer aux chemins spécifiés ou à adapter dans le code.
- La video doit faire au moins 3 min de temps de jeu effectif.

## 🙌 Crédits

Développé par Mateo LORENTE et Rehyann BOUTEILLER
Utilise TrackNetV2 : https://github.com/cehsan/TrackNetV2
