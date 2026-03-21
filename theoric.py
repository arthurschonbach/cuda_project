import numpy as np
from scipy.integrate import quad
import cmath
import subprocess
import re

def heston_characteristic_function(phi, kappa, theta, sigma, rho, v0, r, T, s0, j):
    """
    Calcule la fonction caractéristique f_j(phi) pour le modèle de Heston.
    Utilise la formulation "Little Trap" d'Albrecher et al. pour éviter 
    les instabilités numériques (problèmes de coupure de branche complexe).
    """
    i = 1j
    
    # Paramètres spécifiques selon l'intégrale (P1 ou P2)
    if j == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    x = np.log(s0)
    
    # Calcul des variables intermédiaires complexes
    d = cmath.sqrt((rho * sigma * i * phi - b)**2 - sigma**2 * (2 * u * i * phi - phi**2))
    g = (b - rho * sigma * i * phi + d) / (b - rho * sigma * i * phi - d)
    
    # Formulation stable (d remplacé par -d dans le numérateur de C pour la stabilité)
    g_stable = (b - rho * sigma * i * phi - d) / (b - rho * sigma * i * phi + d)
    
    C = r * i * phi * T + (a / sigma**2) * ((b - rho * sigma * i * phi - d) * T - 2 * cmath.log((1 - g_stable * cmath.exp(-d * T)) / (1 - g_stable)))
    D = ((b - rho * sigma * i * phi - d) / sigma**2) * ((1 - cmath.exp(-d * T)) / (1 - g_stable * cmath.exp(-d * T)))
    
    return cmath.exp(C + D * v0 + i * phi * x)

def heston_integrand(phi, kappa, theta, sigma, rho, v0, r, T, s0, K, j):
    """
    La fonction à intégrer : Re( e^{-i * phi * ln(K)} * f_j(phi) / (i * phi) )
    """
    i = 1j
    f = heston_characteristic_function(phi, kappa, theta, sigma, rho, v0, r, T, s0, j)
    numerator = cmath.exp(-i * phi * np.log(K)) * f
    denominator = i * phi
    
    return (numerator / denominator).real

def heston_call_price(kappa, theta, sigma, rho, v0, r, T, s0, K):
    """
    Calcule le prix exact du Call via l'intégration de Fourier.
    """
    limit = 100
    
    # Correction ici : ajout de K avant le 1 (pour j=1)
    integral1, _ = quad(heston_integrand, 1e-5, limit, args=(kappa, theta, sigma, rho, v0, r, T, s0, K, 1))
    P1 = 0.5 + (1 / np.pi) * integral1
    
    # Correction ici : ajout de K avant le 2 (pour j=2)
    integral2, _ = quad(heston_integrand, 1e-5, limit, args=(kappa, theta, sigma, rho, v0, r, T, s0, K, 2))
    P2 = 0.5 + (1 / np.pi) * integral2
    
    call_price = s0 * P1 - K * np.exp(-r * T) * P2
    return call_price


def get_cuda_price():
    """Lance l'exécutable CUDA et récupère le prix affiché sur la console."""
    try:
        # Exécute le binaire et capture la sortie
        result = subprocess.run(['./Q1_shared'], capture_output=True, text=True, check=True)
        
        # Utilise un regex pour trouver le chiffre après "Call Heston ="
        print("Sortie CUDA :")
        print(result.stdout)
        match = re.search(r"Call Heston\s*=\s*([\d\.]+)", result.stdout)
        if match:
            return float(match.group(1))
        else:
            print("Erreur : Impossible de trouver le prix dans la sortie CUDA.")
            return None
    except Exception as e:
        print(f"Erreur lors de l'exécution du binaire CUDA : {e}")
        return None

if __name__ == "__main__":
    # 1. Calcul du prix exact (Python)
    S0 = 1.0
    K = 1.0       # At-the-money
    v0 = 0.1
    kappa = 0.5
    theta = 0.1
    sigma = 0.3
    rho = -0.5     # Paramètre manquant de l'énoncé
    r = 0.0
    T = 1.0  
    exact_price = heston_call_price(kappa, theta, sigma, rho, v0, r, T, S0, K)
    
    # 2. Récupération du prix approché (CUDA)
    cuda_price = get_cuda_price()
    
    if cuda_price is not None:
        error_abs = abs(exact_price - cuda_price)
        error_rel = (error_abs / exact_price) * 100
        
        print("\n" + "="*40)
        print(f"{'MÉTHODE':<20} | {'PRIX':<15}")
        print("-" * 40)
        print(f"{'Heston Théorique':<20} | {exact_price:.6f}")
        print(f"{'CUDA Monte Carlo':<20} | {cuda_price:.6f}")
        print("-" * 40)
        print(f"Erreur Absolue : {error_abs:.6f}")
        print(f"Erreur Relative : {error_rel:.4f} %")
        print("="*40)
        
        # Interprétation de la substance
        if error_rel < 0.1:
            print("Résultat : EXCELLENT. Ton code CUDA est très précis.")
        elif error_rel < 1.0:
            print("Résultat : CORRECT. Augmente N_sims pour réduire le bruit statistique.")
        else:
            print("Résultat : ÉCART SIGNIFICATIF. Vérifie ton dt ou tes paramètres (rho ?).")