# -*- coding: utf-8 -*-
"""
============================================================================
Problème de Tournées de Véhicules (CVRP) — Exact & Heuristique
============================================================================

Résolution du Capacitated Vehicle Routing Problem (CVRP) : planifier les
tournées d'une flotte de véhicules de capacité Q, partant d'un dépôt unique,
pour livrer N clients de demandes d_i, en minimisant la distance totale.

Deux approches sont fournies :
    1. HEURISTIQUE — du plus proche voisin (Nearest Neighbour), gloutonne,
       quasi-instantanée mais sans garantie d'optimalité.
    2. EXACTE      — formulation PLNE (Miller-Tucker-Zemlin) résoluble par
       un solveur (CPLEX / PuLP). La fonction écrit le modèle ; la résolution
       nécessite un solveur installé.

Formulation PLNE (variables x_ijk, y_ik, contraintes MTZ) :
    min  Σ_ijk c_ij x_ijk
    s.c. visite unique, conservation du flux, capacité Q, élimination des
         sous-tours (MTZ).

Auteur : Yasser Houssein Hassan
Dépendances : numpy (obligatoire) ; pulp (optionnel, résolution exacte)
============================================================================
"""

import sys
import numpy as np

# Sortie console en UTF-8 (symboles : ->, é, è...)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# matplotlib optionnel (sauvegarde des figures en PNG)
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


# ---------------------------------------------------------------------------
# Matrice des distances (euclidienne ; substituable par une distance routière)
# ---------------------------------------------------------------------------
def matrice_distances(coords: np.ndarray) -> np.ndarray:
    r"""c_ij = || p_i - p_j ||_2 entre chaque paire de sites (dépôt = indice 0)."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=2))


# ---------------------------------------------------------------------------
# 1. Heuristique du plus proche voisin (Nearest Neighbour) avec capacité
# ---------------------------------------------------------------------------
def heuristique_plus_proche_voisin(D, demandes, Q):
    r"""Construit des tournées gloutonnes respectant la capacité Q.

    Le véhicule part du dépôt (0), rejoint le client non visité le plus
    proche tant que sa capacité résiduelle le permet, puis retourne au
    dépôt pour recharger. Le processus se répète jusqu'à livrer tous les
    clients.

    Retourne (tournées, distance_totale).
    """
    n = len(demandes)
    non_visites = set(range(1, n))   # le dépôt (0) n'est pas un client
    tournees, distance_totale = [], 0.0

    while non_visites:
        tournee = [0]                # départ du dépôt
        charge, position = 0.0, 0
        while True:
            # clients atteignables sans dépasser la capacité
            candidats = [j for j in non_visites if charge + demandes[j] <= Q]
            if not candidats:
                break
            j = min(candidats, key=lambda c: D[position, c])  # plus proche voisin
            distance_totale += D[position, j]
            tournee.append(j)
            charge += demandes[j]
            position = j
            non_visites.remove(j)
        distance_totale += D[position, 0]  # retour au dépôt
        tournee.append(0)
        tournees.append(tournee)

    return tournees, distance_totale


# ---------------------------------------------------------------------------
# 2. Formulation exacte PLNE (Miller-Tucker-Zemlin) via PuLP
# ---------------------------------------------------------------------------
def resoudre_exact_mtz(D, demandes, Q, n_vehicules):
    r"""Construit et résout le CVRP par programmation linéaire en nombres
    entiers avec élimination des sous-tours de Miller-Tucker-Zemlin.

    Variables :
        x_ij ∈ {0,1} : arc (i -> j) emprunté.
        u_i  ≥ 0     : charge cumulée à la sortie du client i (ordre MTZ).

    Nécessite PuLP : pip install pulp  (utilise le solveur CBC par défaut,
    ou CPLEX si disponible).
    """
    try:
        import pulp
    except ImportError as e:
        raise ImportError("pip install pulp pour la résolution exacte") from e

    n = len(demandes)
    clients = range(1, n)
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]

    prob = pulp.LpProblem("CVRP", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", arcs, cat="Binary")
    u = pulp.LpVariable.dicts("u", clients, lowBound=0, upBound=Q)

    # Objectif : distance totale
    prob += pulp.lpSum(D[i, j] * x[(i, j)] for (i, j) in arcs)

    # Chaque client a exactement un arc entrant et un sortant
    for j in clients:
        prob += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1
        prob += pulp.lpSum(x[(j, k)] for k in range(n) if k != j) == 1

    # Le dépôt émet et reçoit n_vehicules tournées
    prob += pulp.lpSum(x[(0, j)] for j in clients) == n_vehicules
    prob += pulp.lpSum(x[(i, 0)] for i in clients) == n_vehicules

    # Contraintes MTZ : élimination des sous-tours + capacité
    for (i, j) in arcs:
        if i != 0 and j != 0:
            prob += u[i] - u[j] + Q * x[(i, j)] <= Q - demandes[j]
    for i in clients:
        prob += u[i] >= demandes[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    statut = pulp.LpStatus[prob.status]
    cout = pulp.value(prob.objective)
    arcs_actifs = [(i, j) for (i, j) in arcs if x[(i, j)].value() > 0.5]
    return statut, cout, arcs_actifs


# ---------------------------------------------------------------------------
# Reconstruction des tournées à partir des arcs actifs
# ---------------------------------------------------------------------------
def reconstruire_tournees(arcs_actifs):
    suivant = {}
    for i, j in arcs_actifs:
        suivant.setdefault(i, []).append(j)
    tournees = []
    for premier in list(suivant.get(0, [])):
        tournee, courant = [0, premier], premier
        while courant != 0:
            courant = suivant[courant].pop(0) if courant in suivant else 0
            tournee.append(courant)
        tournees.append(tournee)
    return tournees


# ---------------------------------------------------------------------------
# Visualisation : tracé géographique des tournées
# ---------------------------------------------------------------------------
def tracer_tournees(coords, tournees, titre="Tournées de véhicules (CVRP)",
                    fichier="figure_tournees_vrp.png"):
    if not _HAS_PLT:
        print("  (matplotlib absent : graphique ignoré)")
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    couleurs = plt.cm.tab10(np.linspace(0, 1, max(len(tournees), 1)))

    # Clients et dépôt
    ax.scatter(coords[1:, 0], coords[1:, 1], c="black", s=60, zorder=3)
    ax.scatter(coords[0, 0], coords[0, 1], c="red", marker="s", s=200,
               zorder=4, label="Dépôt")
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x, y), fontsize=9, xytext=(4, 4),
                    textcoords="offset points")

    # Arcs de chaque tournée
    for k, (tournee, c) in enumerate(zip(tournees, couleurs), 1):
        pts = coords[tournee]
        ax.plot(pts[:, 0], pts[:, 1], "-", color=c, lw=2, alpha=0.8,
                label=f"Véhicule {k}")
        # flèches de sens de parcours
        for a, b in zip(tournee[:-1], tournee[1:]):
            ax.annotate("", xy=coords[b], xytext=coords[a],
                        arrowprops=dict(arrowstyle="->", color=c, alpha=0.6))

    ax.set_title(titre)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fichier, dpi=130)
    print(f"\n  Figure enregistrée : {fichier}")


# ---------------------------------------------------------------------------
# Démonstration
# ---------------------------------------------------------------------------
def _demonstration():
    rng = np.random.default_rng(3)
    n_clients = 8
    coords = np.vstack([[50, 50],                          # dépôt
                        rng.uniform(0, 100, (n_clients, 2))])
    demandes = np.array([0] + list(rng.integers(5, 20, n_clients)))
    Q = 40

    D = matrice_distances(coords)

    print("=" * 64)
    print("PROBLÈME DE TOURNÉES DE VÉHICULES (CVRP)")
    print(f"  {n_clients} clients | capacité véhicule Q = {Q}")
    print(f"  demande totale = {demandes.sum()} (=> {int(np.ceil(demandes.sum()/Q))} véhicules min.)")
    print("=" * 64)

    tournees, cout = heuristique_plus_proche_voisin(D, demandes, Q)
    print("\n--- HEURISTIQUE (plus proche voisin) ---")
    for k, t in enumerate(tournees, 1):
        charge = sum(demandes[i] for i in t)
        print(f"  Véhicule {k} : {t}  (charge = {charge}/{Q})")
    print(f"  Distance totale = {cout:.1f}")

    tracer_tournees(coords, tournees,
                    titre=f"Tournées heuristiques (coût = {cout:.1f})")

    print("\n--- EXACT (PLNE / MTZ) ---")
    try:
        statut, cout_ex, arcs = resoudre_exact_mtz(D, demandes, Q,
                                                   n_vehicules=len(tournees))
        print(f"  Statut = {statut} | Distance optimale = {cout_ex:.1f}")
        for k, t in enumerate(reconstruire_tournees(arcs), 1):
            print(f"  Véhicule {k} : {t}")
        ecart = (cout - cout_ex) / cout_ex * 100
        print(f"\n  Écart heuristique / optimum = {ecart:+.1f} %")
    except ImportError as e:
        print(f"  (résolution exacte ignorée : {e})")


if __name__ == "__main__":
    _demonstration()
