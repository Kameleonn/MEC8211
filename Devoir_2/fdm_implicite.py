import numpy as np
from scipy.linalg import solve

########################
# Constantes physiques #
########################

R = 0.5 # Rayon du pilier (m)
D_EFF = 1e-10 # Diffusivité effective (m^2/s)
C_E = 20.0 # Concentration à la surface (mol/m^3)
K = 4e-9 # (1/s)

def solve_fdm_implicite(N, dt):
    """
    Résout l'équation de diffusion 1D radiale instationnaire par différences finies implicites.

    Args:
        N (int): Nombre total de nœuds.
        dt (float): Pas de temps.

    Returns:
        tuple: Vecteur des concentrations calculées.
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
    
    return C
