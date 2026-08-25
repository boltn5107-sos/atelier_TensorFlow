# Atelier TensorFlow – Prédiction de la consommation énergétique d'un bâtiment

## Contexte et objectif

Une entreprise dispose de données issues de capteurs installés dans plusieurs bâtiments :
température moyenne, humidité, nombre d'occupants, etc.

L'objectif de cet atelier est de construire avec **TensorFlow/Keras** un réseau de neurones
capable de **prédire la consommation énergétique** d'un bâtiment à partir de ces caractéristiques.

Le dataset est généré synthétiquement à partir d'une formule « physique » bruitée :

```
consommation = 50 + 5 × température + 1,5 × humidité + 4 × occupants + ε    (ε ~ N(0, 10²))
```

## Structure du projet

```
atelier_tensorflow_iot/
│
├── notebooks/
│   └── atelier_tensorflow_iot.ipynb     # notebook complet (exécuté, avec sorties)
│
└── models/
    └── modele_consommation.keras        # modèle entraîné sauvegardé au format Keras
```

## Environnement et installation

- Python 3.12
- Dépendances utilisées : TensorFlow 2.21, NumPy 2.5, Matplotlib 3.11, Jupyter

```bash
pip install tensorflow matplotlib numpy jupyter
```

## Utilisation

Lancer Jupyter puis ouvrir le notebook :

```bash
jupyter notebook notebooks/atelier_tensorflow_iot.ipynb
```

Le notebook est fourni déjà exécuté (sorties, graphiques et modèle inclus). Il peut aussi être
ré-exécuté intégralement : les graines aléatoires sont fixées pour garantir la reproductibilité.
Les chemins étant relatifs au dossier `notebooks/`, exécuter le notebook depuis son propre
répertoire (comportement par défaut de Jupyter).

## Contenu du notebook

| Partie | Contenu |
|--------|---------|
| 0 | Mise en place de l'environnement (imports, graines aléatoires) |
| 1 | Génération du dataset : 1000 observations (température ~ N(25, 4), humidité ~ U(30, 80), occupants ~ U{1..49}), matrice X (1000×3) et cible y en float32 |
| 2 | Découpage Train/Test reproductible : 80 % / 20 % |
| 3 | Création du modèle séquentiel : Dense(16, relu) → Dense(8, relu) → Dense(1) |
| 4 | Compilation : optimiseur `adam`, perte `mse`, métrique `mae` |
| 5 | Entraînement : `validation_split=0.2`, `epochs=50`, `batch_size=16`, courbes d'apprentissage |
| 6 | Évaluation sur le test (MSE et MAE) et interprétation |
| 7 | Prédictions sur le test, comparaison aux valeurs réelles (tableau + nuage de points) |
| 8 | Sauvegarde du modèle : `models/modele_consommation.keras` |
| 9 | Chargement du modèle, résumé, nouvelle prédiction |
| 10 | Fonction d'inférence `predire_consommation(temperature, humidite, occupants)` |
| 11 | Bonus : comparaison à la formule théorique + analyse de sensibilité à la température |

## Résultats principaux

- **MAE sur le test ≈ 10 unités**, soit environ 3 % de la consommation typique (≈ 360 unités) :
  l'erreur résiduelle correspond essentiellement au bruit gaussien injecté dans les données (σ = 10),
  qui constitue un plancher irréductible.
- **Pas de sur-apprentissage** : les courbes de validation suivent celles de l'entraînement.
- **Bonus** : l'écart moyen modèle/formule théorique (≈ 8,7) est du même ordre que l'écart
  données/formule (≈ 8,8) — le réseau reproduit fidèlement la relation physique en lissant le bruit.
- Si le modèle était parfait, tous les points du graphique « réel vs prédit » seraient exactement
  sur la diagonale y = x ; ici ils s'en écartent uniquement à hauteur du bruit des données.
