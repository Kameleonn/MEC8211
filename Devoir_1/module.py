import numpy as np
from scipy.linalg import solve


def solution_analytique(r, R=0.5, S=2e-8, Deff=1e-10, Ce=20.0):
    """
    Calcule la solution analytique exacte selon l'équation (2) du devoir.
    """
    term = (1/4) * (S / Deff) * (R**2) * ((r**2 / R**2) - 1)
    return term + Ce

def solve_finite_difference(N, R=0.5, S=2e-8, Deff=1e-10, Ce=20.0, schema='D'):
    """
    Résout l'équation de diffusion 1D radiale par différences finies.
    
    Paramètres:
    -----------
    N : int
        Nombre total de nœuds.
    R, S, Deff, Ce : float
        Paramètres physiques du problème.
    schema : str
        'D' pour le schéma décentré (Question D)
        'E' pour le schéma centré (Question E)
        
    Retourne:
    ---------
    r : array
        Vecteur des positions radiales.
    C : array
        Vecteur des concentrations calculées.
    """
    # Création du maillage
    dr = R / (N - 1)
    r = np.linspace(0, R, N)
    
    # Initialisation matricielle (Systeme A*C = b)
    A = np.zeros((N, N))
    b = np.zeros(N)
    const_source = S / Deff

    # --- 1. CONDITION FRONTIÈRE AU CENTRE (r=0, i=0) ---
    if schema == 'D':
        # Question D : 
        # Schéma décentré avant (Forward) 
        A[0, 0] = -1.0
        A[0, 1] = 1.0
        b[0] = 0
    elif schema == 'E':
        # Question E : 
        # Intègre la symétrie (flux nul) 
        A[0, 0] = -3.0
        A[0, 1] = 4.0
        A[0, 2] = -1.0
        b[0] = 0.0

    # --- 2. NŒUDS INTÉRIEURS (0 < r < R) ---
    for i in range(1, N - 1):
        ri = r[i]
        
        # Facteurs communs
        inv_dr2 = 1.0 / (dr**2)
        inv_rdr = 1.0 / (ri * dr)
        
        if schema == 'D':
            # Question D : Dérivée première décentrée avant (Forward) [cite: 55]
            
            # C_{i-1}
            A[i, i-1] = inv_dr2
            # C_{i}
            A[i, i]   = -2.0 * inv_dr2 - inv_rdr
            # C_{i+1}
            A[i, i+1] = inv_dr2 + inv_rdr
            
        elif schema == 'E':
            # Question E : Dérivée première centrée [cite: 67]
            
            # C_{i-1}
            A[i, i-1] =  (inv_dr2 - 0.5 * inv_rdr)
            # C_{i}
            A[i, i]   = -2.0 * inv_dr2 
            # C_{i+1}
            A[i, i+1] =  (inv_dr2 + 0.5 * inv_rdr)
            
        b[i] = S/Deff

    # --- 3. CONDITION FRONTIÈRE SURFACE (r=R, i=N-1) ---
    # Condition de Dirichlet : C = Ce 
    A[N-1, N-1] = 1.0
    b[N-1] = Ce
    
    # Résolution
    C = solve(A, b)
    
    return r, C
