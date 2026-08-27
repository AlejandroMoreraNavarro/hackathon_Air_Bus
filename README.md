# 🚁 Airbus AI Hackathon 2026 - Lidar Obstacle Detection Pipeline

**Équipe :** Brangier Loïc, Desnoyer Titouan, Martinez François, Morera Navarro Alejandro

---

## 1. Philosophie du Pipeline : L'intelligence géométrique

Plutôt que d'utiliser un modèle Deep Learning 3D lourd (type **VoxelNet**) inadapté aux contraintes embarquées, notre pipeline est une heuristique déterministe basée sur les signatures morphologiques. Nous exploitons les propriétés géométriques (densité, verticalité, linéarité) de chaque classe pour segmenter et reconstruire les bounding boxes 3D avec une grande précision et un coût calculatoire minimal.

## 2. Détection et Reconstruction Multi-Classes

Le jeu de données ne fournissant pas les boîtes 3D, notre algorithme reconstruit dynamiquement les dimensions (`x`, `y`, `z`, `l`, `w`, `h`) en analysant la topologie des points pour chaque classe :

* **🌬️ Éoliennes (Classe 3) - Clustering Spatial Densifié :**
    * **Méthode :** Les éoliennes génèrent des clusters massifs et denses, isolés du terrain. Nous appliquons un algorithme de type **DBSCAN** (ou un clustering euclidien) paramétré pour les objets de grand volume.
    * **Bounding Box :** La boîte est construite en extrayant le centre géométrique du mât (forte densité verticale) et en étendant la largeur/longueur (`w`, `l`) pour englober la rotation maximale des pales (qui forment un disque de points plus épars en hauteur).

* **⚡ Pylônes Électriques (Classe 2) - Analyse de Verticalité & Graphes :**
    * **Méthode :** Les pylônes ont une variance en Z (hauteur) très forte par rapport à leur emprise au sol (X, Y). Nous utilisons un filtre de verticalité (histogramme de l'axe Z).
    * **Synergie :** Les pylônes servent souvent de "nœuds" pour les câbles. Notre algorithme vérifie la proximité spatiale entre un cluster vertical et les extrémités des lignes détectées (Classe 1) pour confirmer la classe "Pylône".

* **📡 Antennes (Classe 0) - Profil Altitudinal Extrême :**
    * **Méthode :** Similaires aux pylônes mais généralement plus fines et plus hautes, parfois accompagnées de haubans (fils de tension). La différenciation se fait par le ratio Hauteur / Emprise au sol extrêmement élevé.
    * **Bounding Box :** Une boîte très étroite (`w`, `l` minimaux) maximisée sur l'axe Z (`h`) englobant le mât principal.

* **🧵 Câbles (Classe 1) - Transformée Directionnelle (Hough 3D) :**
    * **Méthode :** Le défi majeur (points très épars). Nous filtrons les "lignes fantômes" par une analyse des directions majoritaires. Une ligne est validée si elle respecte une cohérence angulaire stricte (ex: alignement parfait à 102.5°) sur une longue distance, couplée à un seuil de densité minimum.

## 3. Choix Techniques & Alignement des Données

* **Resynchronisation Lidar (Custom) :** Le script implémente une fonction d'alignement (`get_frame_id_from_prediction`) utilisant la pose de l'hélicoptère (`ego_yaw`, `ego_x`, `ego_y`) avec une tolérance stricte (1e-4) pour garantir que chaque détection est assignée à la bonne frame de balayage.
* **Isolation de la validation :** Évaluation des performances frame par frame via `count_objects_in_scene_frame`, évitant les biais d'une évaluation globale sur la scène complète.

## 4. Spécifications de l'Architecture

* **Architecture :** Pipeline de post-traitement géométrique (Clustering + Analyse de covariance + Heuristiques directionnelles).
* **Avantage compétitif :** Dans un contexte aéronautique (Airbus), l'absence de réseau de neurones "boîte noire" offre une explicabilité totale des décisions, une consommation VRAM nulle, et un temps d'inférence CPU ultra-rapide (compatible temps réel).

---

### 📊 Modèles & Paramètres

* **Nombre de paramètres :** 0
* **Modèles utilisés :** Nous avons tenté d'utiliser des approches Deep Learning telles que **PointNet** ou **PointPillars**, mais avec des résultats peu satisfaisants face à la nature des données. Cela a motivé et validé notre approche purement géométrique.