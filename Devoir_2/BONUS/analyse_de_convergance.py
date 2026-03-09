import numpy as np
import matplotlib.pyplot as plt

# Lecture de la deuxieme ligne pour savoir quelle variable
with open("donnees_erreurs.txt", "r") as file:
    lines = file.readlines()
    variable_interet = lines[1].strip()

# Lecture des données générées par Bash (on ignore la ligne d'en-tête et la deuxieme qui contient la variable d interet)
donnees = np.loadtxt("donnees_erreurs.txt", skiprows=2)

delta_list = donnees[:, 0]
L1_err = donnees[:, 1]
L2_err = donnees[:, 2]
Linf_err = donnees[:, 3]

# Calcul des pentes (sur les 3 derniers points pour la précision asymptotique)
pente_L1_temps, ord_L1_temps = np.polyfit(np.log(delta_list[-3:]), np.log(L1_err[-3:]), 1)
pente_L2_temps, ord_L2_temps = np.polyfit(np.log(delta_list[-3:]), np.log(L2_err[-3:]), 1)
pente_Linf_temps, ord_Linf_temps = np.polyfit(np.log(delta_list[-3:]), np.log(Linf_err[-3:]), 1)

#Droites de regression pour visualiser les ordres de convergence
delta_array = np.array(delta_list)
droite_L1 = np.exp(ord_L1_temps) * (delta_array ** pente_L1_temps)
droite_L2 = np.exp(ord_L2_temps) * (delta_array ** pente_L2_temps)
droite_Linf = np.exp(ord_Linf_temps) * (delta_array ** pente_Linf_temps)

plt.figure(figsize=(8, 6))
plt.loglog(delta_list, L1_err, 'bo', label='Erreur $L_1$')
plt.loglog(delta_list, L2_err, 'gs', label='Erreur $L_2$')
plt.loglog(delta_list, Linf_err, 'r^', label='Erreur $L_\\infty$')

plt.loglog(delta_array, droite_L1, 'b--', alpha=0.6, label=f'Régression L1 (pente={pente_L1_temps:.2f})')
plt.loglog(delta_array, droite_L2, 'g--', alpha=0.6, label=f'Régression L2 (pente={pente_L2_temps:.2f})')
plt.loglog(delta_array, droite_Linf, 'r--', alpha=0.6, label=f'Régression Linf (pente={pente_Linf_temps:.2f})')

if variable_interet == "t":
    plt.xlabel('$\\Delta t$ (s)')
    dimension = "temps"

elif variable_interet == "r":
    plt.xlabel('$\\Delta r$ (m)')
    dimension = "espace"
    

plt.title(f"Graphique : Convergence d'ordre {pente_L1_temps:.2f} en {dimension} des erreurs L1, L2 et Linf")
plt.ylabel("Erreur (mol/m³)")
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.show()