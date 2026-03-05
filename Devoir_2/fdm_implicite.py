import numpy as np
from scipy.linalg import solve

########################
# Constantes physiques #
########################

R = 0.5 # Rayon du pilier (m)
D_EFF = 1e-10 # Diffusivité effective (m^2/s)
C_E = 20.0 # Concentration à la surface (mol/m^3)
K = 4e-9 # (1/s)

def solve_fdm_implicite(N, T, N_t):
    """
    Résout l'équation de diffusion 1D radiale instationnaire par différences finies implicites.

    Args:
        N (int): Nombre total de nœuds.
        T (float): Temps total de simulation.
        N_t (int): Nombre de pas de temps.

    Returns:
        tuple: Vecteur des concentrations calculées.
    """
    # Création du maillage
    r = np.linspace(0, R, N)
    t = np.linspace(0, T, N_t)
    dr = r[1] - r[0]
    dt = t[1] - t[0]
    
    # Initialisation matricielle (Systeme A*C = b)
    A = np.zeros((N, N))
    b = np.zeros((N))
    C = np.zeros((N, N_t)) # Sert aussi pour la condition initiale nulle
    
    # Loop temporel
    for i in range(N_t - 1):
        # Condition frontière au centre (r=0, i=0)
        A[0, 0] = -3
        A[0, 1] = 4
        A[0, 2] = -1
        b[0] = 2 * dr *np.sin(t[i+1])
        
        # Condition frontière à la surface (r=R, i=N-1)
        A[-1, -1] = 1
        b[-1] = (t[i+1]) * np.cos(R) + np.exp(R) * np.sin(t[i+1]) # Nouvelle condition de Dirichlet à la surface avec MMS
        
        # Nœuds intérieurs
        for j in range(1, N - 1):
            terme_source_MMS = -1 * D_EFF * (-(t[i+1])*np.cos(r[j]) + np.exp(r[j])*np.sin(t[i+1]) + (-(t[i+1])*np.sin(r[j]) + np.exp(r[j])*np.sin(t[i+1]))/r[j]) + K * ((t[i+1])*np.cos(r[j]) + np.exp(r[j])*np.sin(t[i+1])) + np.exp(r[j]) * np.cos(t[i+1]) + np.cos(r[j])
            A[j, j-1] = -1 *dt * D_EFF * (1 - dr/(2*r[j]))
            A[j, j] = dr**2 + 2 * dt * D_EFF + K * dr**2 * dt
            A[j, j+1] = -1 * dt * D_EFF * (1 + dr/(2*r[j]))
            b[j] = dr**2 * C[j, i] + dr**2 * dt *terme_source_MMS
        
        # Résolution du système linéaire pour le pas de temps ti+1  
        C[:, i+1] = solve(A, b)
    
    return C

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 10
    T = 4e9
    N_t = 100

    # Domaines 
    r_dom = np.linspace(0, R, N)
    t_dom = np.linspace(0, T, N_t)
    ti, ri = np.meshgrid(t_dom, r_dom)

    # Evaluation de la solution numérique sur le domaine
    C = solve_fdm_implicite(N, T, N_t)

    # Tracage de la solution numérique
    plt.figure()
    plt.contourf(ri, ti, C, levels=20)
    plt.colorbar()
    plt.title('Solution Numerique')
    plt.xlabel('Rayon r (m)')
    plt.ylabel('Temps t (s)')
    plt.show()