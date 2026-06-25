# Optimisation de Tournées de Véhicules (VRP)

Ce projet académique traite de la résolution du **Problème de Tournées de Véhicules (VRP - *Vehicle Routing Problem*)**, un problème d'optimisation combinatoire classique de la recherche opérationnelle et de la logistique. Le projet compare une approche de résolution exacte par programmation linéaire et une approche heuristique approximative.

---

## Sommaire et Concepts Clés

Le problème consiste à planifier des tournées optimales pour une flotte de véhicules de capacité limitée, partant d'un dépôt central, afin de livrer un ensemble de clients répartis géographiquement, tout en minimisant la distance ou le coût total de transport.

### 1. Résolution Exacte avec CPLEX
* **Méthodologie** : Formulation sous forme de programme linéaire en nombres entiers (PLNE) résolu à l'aide de la bibliothèque d'optimisation **IBM ILOG CPLEX** en Python.
* **Variables de décision** :
  * $x_{ijk}$ : Variable binaire indiquant si le véhicule $k$ se déplace directement de la ville $i$ à la ville $j$.
  * $y_{ik}$ : Variable binaire indiquant si le véhicule $k$ dessert le client $i$.
  * $u_{i}$ : Variable continue représentant l'ordre de passage, utilisée pour éliminer les sous-tours.
* **Contraintes principales** :
  * Visite unique de chaque client.
  * Capacité maximale des camions ($Q$) non dépassée par la somme des demandes des clients de la tournée.
  * Équilibre des flux entrants et sortants pour chaque ville.
  * Retour obligatoire au dépôt (départ et arrivée).
  * **Contraintes de Miller-Tucker-Zemlin (MTZ)** : Élimination des cycles et sous-tours indépendants.
* **Résultat** : Garantie d'obtenir la solution globalement optimale (coût obtenu dans l'exemple : 545 €).

### 2. Résolution Heuristique (Voisin le plus proche)
* **Méthodologie** : Implémentation d'une heuristique gloutonne du **voisin le plus proche** (*Nearest Neighbor*).
* **Fonctionnement** : Le véhicule démarre du dépôt, se rend chez le client non visité le plus proche, remplit sa capacité, puis retourne au dépôt pour recharger lorsque sa capacité maximale est atteinte. Le processus se répète jusqu'à ce que tous les clients soient livrés.
* **Visualisation cartographique** : Génération d'une carte interactive HTML (`map.html`) traçant les tournées sur fond géographique réel grâce aux bibliothèques **Geopy** et **Folium**.
* **Résultat** : Résolution quasi instantanée mais sous-optimale (coût obtenu dans l'exemple : 615 €).

### 3. Comparaison Méthodes Exactes vs Heuristiques
* **CPLEX (Exact)** : Garantit l'optimum global mais présente une complexité de calcul exponentielle (NP-difficile), rendant la résolution lente ou impossible pour de très grands réseaux de clients.
* **Heuristique** : Rapide et extensible aux problèmes industriels de grande échelle, mais sans garantie d'optimalité.

## Structure des Fichiers
* `Optimisation_Combinatoire.ipynb` : Le notebook Jupyter contenant le code d'optimisation CPLEX, l'algorithme heuristique et la génération de la carte interactive.
* `Optimisation_Combinatoire.pdf` : Le document explicatif comparant les deux méthodes, explicitant les équations mathématiques des contraintes et affichant les résultats textuels et cartographiques.

---
*Projet réalisé par Yasser Houssein Hassan*
