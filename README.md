

# Airbus AI Hackathon 2026 - Pipeline de Détection d'Obstacles Lidar

**Équipe :** Loïc Brangier, Titouan Desnoyer, François Martinez, Alejandro Morera Navarro

---

## Contexte et Objectifs du Projet
Voler à basse altitude en hélicoptère comporte des risques majeurs, le principal danger résidant dans les collisions avec des obstacles difficiles à percevoir à l'œil nu ou à grande vitesse, tels que les lignes électriques, les pylônes, les antennes ou les éoliennes.

Organisé par Airbus Helicopters, ce hackathon visait à développer une solution capable d'analyser les nuages de points fournis par un capteur Lidar embarqué. L'objectif était triple :

Détecter la présence des obstacles dans l'espace environnant.

Classifier chaque objet parmi quatre catégories cibles (antennes, câbles, pylônes électriques, éoliennes).

Reconstruire les dimensions 3D (boîtes englobantes / bounding boxes) de chaque obstacle pour permettre l'anticipation des trajectoires et renforcer la sécurité du vol.

### Le défi des données
Les données d'entraînement fournies représentaient jusqu'à 575 000 points bruts par seconde, sans aucune boîte 3D préexistante pour superviser l'apprentissage. Il a donc fallu concevoir un algorithme autonome capable de reconstruire ces formes géométriques à partir des labels attribués point par point, tout en garantissant une grande robustesse face aux variations de densité de points (de 100% à 25%).


## Notre création : 

### 1. Philosophie du Pipeline : L'Intelligence Géométrique

Plutôt que de nous tourner vers des modèles de Deep Learning 3D très lourds (type VoxelNet), souvent inadaptés aux contraintes d'informatique embarquée, nous avons fait le choix d'une approche déterministe basée sur les signatures morphologiques.

En exploitant directement les propriétés géométriques fondamentales de chaque classe — telles que la densité, la verticalité ou la linéarité —, notre pipeline parvient à segmenter le nuage de points et à reconstruire des boîtes englobantes 3D précises, le tout avec un coût en calcul minimal.

---

### 2. Détection et Reconstruction Multi-Classes

Le jeu de données initial ne fournissant pas directement les boîtes 3D, notre algorithme reconstruit dynamiquement leurs dimensions (`x`, `y`, `z`, `l`, `w`, `h`) en analysant la structure topologique des points :

* **Éoliennes (Classe 3) :**
  Ces structures génèrent des regroupements de points massifs, denses et bien isolés du sol. Nous appliquons un algorithme de clustering spatial (type DBSCAN) ajusté pour les grands volumes. La boîte 3D est construite à partir du centre du mât (forte densité verticale), puis élargie pour intégrer l'espace de rotation des pales.

* **Pylônes Électriques (Classe 2) :**
  Les pylônes se distinguent par une très forte hauteur relative par rapport à leur surface au sol. Nous utilisons un filtre de verticalité ciblant l'axe Z. De plus, notre algorithme croise la position des pylônes avec les extrémités des câbles détectés pour valider la classification de manière cohérente.

* **Antennes (Classe 0) :**
  Proches des pylônes mais beaucoup plus fines, les antennes se caractérisent par un ratio hauteur / emprise au sol extrêmement élevé. La boîte englobante est définie de façon très ajustée autour du mât principal avec une hauteur maximale.

* **Câbles (Classe 1) :**
  Représentant le défi principal en raison de la faible densité de points, les câbles sont identifiés via une recherche d'orientations majoritaires (transformée directionnelle). Une ligne est confirmée si elle conserve une orientation constante sur une longue distance, couplée à une densité minimale de points.

---

### 3. Alignement des Données & Validation

Pour garantir la fiabilité théorique et pratique de notre système, deux choix techniques clés ont été intégrés :

* **Resynchronisation Lidar sur-mesure :** Une fonction d'alignement (`get_frame_id_from_prediction`) synchronise la position de l'hélicoptère (`ego_yaw`, `ego_x`, `ego_y`) avec une tolérance très stricte (1e-4), assurant que chaque détection est attribuée à la bonne image de balayage.
* **Validation isolée :** L'évaluation est réalisée image par image (`count_objects_in_scene_frame`) pour mesurer les performances réelles sans biais d'agrégation globale.

---

### 4. Spécifications & Atouts pour l'Aéronautique

* **Nombre de paramètres :** 0
* **Explicabilité totale :** Dans le secteur aéronautique, l'absence d'effet « boîte noire » est un avantage déterminant : notre algorithme est 100% compréhensible et auditables.
* **Performance embarquée :** Empreinte mémoire (VRAM) nulle et temps d'exécution très rapide sur simple processeur central (CPU), compatible avec des contraintes temps réel à bord d'un appareil.

> **Note d'expérimentation :** Nous avons également testé des approches par réseaux de neurones 3D (PointNet, PointPillars), mais la nature spécifique des données a confirmé la supériorité et la fiabilité de notre démarche géométrique.
