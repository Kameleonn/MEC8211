import numpy as np
from scipy.linalg import solve
import sympy as sp
 
def get_symbolic_functions(C_MMS):
    """
    Calcule les fonctions symboliques nécessaires pour la méthode MMS.

    Args:
        C_MMS (sympy.Expr): Fonction de référence pour la méthode MMS.

    Returns:
        tuple: Fonctions symboliques pour le terme source, la condition de Dirichlet à la surface et la condition de Neumann au centre.
    """
    
    # Symboles
    t, r, d_eff_symbm, k_symb, R_symb = sp.symbols('t r d_eff_symb k_symb R_symb')
    
    # Fonction de C_MMS
    f_C_MMS = sp.lambdify([t, r], C_MMS, 'numpy')
    
    # Calcul des derivées symboliques pour la méthode MMS
    C_t = sp.diff(C_MMS, t)
    C_r = sp.diff(C_MMS, r)
    C_rr = sp.diff(C_r, r)

    # Calcul du terme source
    source = C_t - d_eff_symbm * (C_rr + C_r / r) + k_symb * C_MMS
    f_source = sp.lambdify([t, r, d_eff_symbm, k_symb, R_symb], source, 'numpy')
    
    # Calcul de la condition de Dirichlet à la surface
    C_surface = C_MMS.subs(r, R_symb)
    f_C_surface = sp.lambdify([t, r, d_eff_symbm, k_symb, R_symb], C_surface, 'numpy')

    # Calcul de la condition de Neumann au centre
    C_r_center = C_r.subs(r, 0)
    f_C_r_centre = sp.lambdify([t, r, d_eff_symbm, k_symb, R_symb], C_r_center, 'numpy')

    return f_source, f_C_surface, f_C_r_centre, f_C_MMS


def solve_fdm_implicite(N, T, N_t, D_EFF, K, R = 0.5, C_e = None, C_MMS = None):
    """
    Résout l'équation de diffusion 1D radiale instationnaire par différences finies implicites.

    Args:
        N (int): Nombre total de nœuds.
        T (float): Temps total de simulation.
        N_t (int): Nombre de pas de temps.
        D_EFF (float): Diffusivité effective.
        K (float): Constante de réaction.
        C_MMS (sympy.Expr, optional): Fonction de référence pour la méthode MMS.

    Returns:
        tuple: Vecteur des concentrations calculées.
    """
    
    # Création du maillage
    rdom = np.linspace(0, R, N)
    tdom = np.linspace(0, T, N_t)
    dr = rdom[1] - rdom[0]
    dt = tdom[1] - tdom[0]
    
    # Get les fonctions symboliques pour la methode MMS, si elle est fournie
    if C_MMS is not None:
        f_source, f_C_surface, f_C_r_centre, f_C = get_symbolic_functions(C_MMS)
    else:
        # Si aucune fonction MMS n'est fournie, on instancie les fonctions à des fonctions lambda retournant 0, pour la source, la condition de Neumann et la condition
        # initiale. Pour la condition de Dirichlet, on retourne la valeur de C_e à la surface. Et ce, pour n'importe quelle valeur d'arguments
        f_source = lambda t, r, d_eff_symbm, k_symb, R_symb: 0
        f_C_surface = lambda t, r, d_eff_symbm, k_symb, R_symb: C_e
        f_C_r_centre = lambda t, r, d_eff_symbm, k_symb, R_symb: 0
        f_C = lambda t, r: np.zeros(len(rdom))
    
    # Initialisation matricielle (Systeme A*C = b)
    A = np.zeros((N, N))
    b = np.zeros((N))
    C = np.zeros((N, N_t)) # Sert aussi pour la condition initiale nulle
    
    # Condition intiale a t=0
    C[:, 0] = f_C(0, rdom) # Condition initiale avec MMS
    
    # Loop temporel
    for i in range(N_t - 1):
        # Condition frontière au centre (r=0, i=0)
        A[0, 0] = -3
        A[0, 1] = 4
        A[0, 2] = -1
        b[0] = 2 * dr * f_C_r_centre(tdom[i+1], 0, D_EFF, K, R) # Nouvelle condition de Neumann au centre avec MMS
        
        # Condition frontière à la surface (r=R, i=N-1)
        A[-1, -1] = 1
        b[-1] = f_C_surface(tdom[i+1], R, D_EFF, K, R) # Nouvelle condition de Dirichlet à la surface avec MMS
        
        # Nœuds intérieurs
        for j in range(1, N - 1):
            A[j, j-1] = -1 *dt * D_EFF * (1 - dr/(2*rdom[j]))
            A[j, j] = dr**2 + 2 * dt * D_EFF + K * dr**2 * dt
            A[j, j+1] = -1 * dt * D_EFF * (1 + dr/(2*rdom[j]))
            b[j] = dr**2 * C[j, i] + dr**2 * dt * f_source(tdom[i+1], rdom[j], D_EFF, K, R)
        
        # Résolution du système linéaire pour le pas de temps ti+1  
        C[:, i+1] = solve(A, b)
    
    return C, rdom, tdom


##############################
# Pour le script automatique #
##############################
if __name__ == "__main__":
    
    variable = VVVV
    
    # Set constantes
    if variable == "r":
        T = 4e9
        N = YYYY
        N_t = 100
    elif variable == "t":
        T = 1
        N = 700
        N_t = ZZZZ
    
    R = 0.5
    D_EFF = 1
    K = 4
    # MMS
    # Symboles
    t, r = sp.symbols('t r')
    C_MMS = t *sp.cos(r) + sp.exp(r) * sp.sin(t)
    C_MMS_func = sp.lambdify([t, r], C_MMS, 'numpy')
    
    # Rouler la simulation pour un pas specifique avec la MMS
    C_num, r_, t_ = solve_fdm_implicite(N, T, N_t, D_EFF, K, R, None,  C_MMS)
    
    # Calculer le delta necessaire
    if variable == "r":
        delta = r_[1]-r_[0]
    elif variable == "t":
        delta = t_[1] - t_[0]

    # Trouver l'erreur a l aide de la solution exacte
    ti, ri = np.meshgrid(t_, r_)
    C_exact = C_MMS_func(ti, ri)
    erreur = np.abs(C_num - C_exact)
    
    # Calculer les erreurs 
    L1 = np.mean(erreur)
    L2 = np.sqrt(np.mean(erreur**2))
    Linf = np.max(erreur)
    
    # Print resultat
    print(f"{delta} {L1} {L2} {Linf}")