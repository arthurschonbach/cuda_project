import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paramètres globaux
N_SIMS = 1000000
RHO = -0.5  # Corrélation usuelle (non spécifiée dans Q1, on fixe à -0.5)

def compile_cuda():
    print("Compilation du code CUDA...")
    res = subprocess.run(["nvcc", "-O3", "heston.cu", "-o", "heston"], capture_output=True, text=True)
    if res.returncode != 0:
        print("Erreur de compilation:\n", res.stderr)
        exit(1)
    print("Compilation réussie.\n")

def run_heston(method, kappa, theta, sigma, rho, n_steps, n_sims=N_SIMS):
    # method: 0=Euler, 1=Exact, 2=Almost Exact
    cmd = f"./heston {method} {kappa} {theta} {sigma} {rho} {n_steps} {n_sims}"
    res = subprocess.run(cmd.split(), capture_output=True, text=True)
    if res.returncode != 0:
        return None, None
    try:
        price, time_ms = map(float, res.stdout.strip().split(','))
        return price, time_ms
    except:
        return None, None

def run_q1_q2():
    print("="*50)
    print("QUESTIONS 1 & 2: Test unitaire (kappa=0.5, theta=0.1, sigma=0.3)")
    print("="*50)
    kappa, theta, sigma = 0.5, 0.1, 0.3
    
    p_euler, t_euler = run_heston(0, kappa, theta, sigma, RHO, 1000)
    p_exact, t_exact = run_heston(1, kappa, theta, sigma, RHO, 1000)
    p_almost, t_almost = run_heston(2, kappa, theta, sigma, RHO, 1000)

    print(f"Euler (dt=1/1000)       : Prix = {p_euler:.6f} | Temps = {t_euler:.2f} ms")
    print(f"Exact (dt=1/1000)       : Prix = {p_exact:.6f} | Temps = {t_exact:.2f} ms")
    print(f"Almost Exact (dt=1/1000): Prix = {p_almost:.6f} | Temps = {t_almost:.2f} ms\n")

def run_q3():
    print("="*50)
    print("QUESTION 3: Analyse sur Grille de Paramètres")
    print("="*50)
    
    # Génération des paramètres respectant 20 * kappa * theta > sigma^2
    np.random.seed(42)
    test_cases =[]
    while len(test_cases) < 15: # On teste 15 configurations
        k = np.random.uniform(0.1, 10)
        th = np.random.uniform(0.01, 0.5)
        sig = np.random.uniform(0.1, 1.0)
        if 20 * k * th > sig**2:
            test_cases.append((k, th, sig))
            
    results =[]
    
    for i, (k, th, sig) in enumerate(test_cases):
        print(f"Simulation {i+1}/15 (k={k:.2f}, th={th:.2f}, sig={sig:.2f})...")
        
        # Ground Truth = Exact dt=1/1000
        p_exact, t_exact = run_heston(1, k, th, sig, RHO, 1000)
        
        # Euler dt=1/1000
        p_euler, t_euler = run_heston(0, k, th, sig, RHO, 1000)
        
        # Almost dt=1/1000
        p_alm1, t_alm1 = run_heston(2, k, th, sig, RHO, 1000)
        
        # Almost dt=1/30 (Question 3 spécifique)
        p_alm30, t_alm30 = run_heston(2, k, th, sig, RHO, 30)
        
        results.append({
            'k': k, 'th': th, 'sig': sig,
            'Price_Exact': p_exact,
            'Price_Euler': p_euler, 'Time_Euler': t_euler,
            'Price_Alm1000': p_alm1, 'Time_Alm1000': t_alm1,
            'Price_Alm30': p_alm30, 'Time_Alm30': t_alm30
        })

    df = pd.DataFrame(results)
    
    # Calcul des erreurs absolues par rapport au schéma exact
    df['Err_Euler'] = np.abs(df['Price_Euler'] - df['Price_Exact'])
    df['Err_Alm1000'] = np.abs(df['Price_Alm1000'] - df['Price_Exact'])
    df['Err_Alm30'] = np.abs(df['Price_Alm30'] - df['Price_Exact'])

    # Sauvegarde des données
    df.to_csv('heston_results.csv', index=False)
    
    # === PLOT 1 : TEMPS D'EXECUTION ===
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Time_Euler'], label='Euler dt=1/1000', marker='o')
    plt.plot(df.index, df['Time_Alm1000'], label='Almost Exact dt=1/1000', marker='s')
    plt.plot(df.index, df['Time_Alm30'], label='Almost Exact dt=1/30', marker='^')
    plt.title('Comparaison des Temps d\'Execution (1 Million de trajectoires)')
    plt.xlabel('Configuration de Test Index')
    plt.ylabel('Temps (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig('q3_execution_times.png')
    plt.close()

    # === PLOT 2 : ERREUR DE PRICING ===
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Err_Euler'], label='Euler dt=1/1000', marker='o')
    plt.plot(df.index, df['Err_Alm1000'], label='Almost Exact dt=1/1000', marker='s')
    plt.plot(df.index, df['Err_Alm30'], label='Almost Exact dt=1/30', marker='^')
    plt.title('Erreur Absolue de Pricing (Reference: Exact Scheme dt=1/1000)')
    plt.xlabel('Configuration de Test Index')
    plt.ylabel('Erreur Absolue')
    plt.yscale('log') # Echelle log car les erreurs sont petites
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('q3_pricing_errors.png')
    plt.close()

    print("\nSimulations terminées !")
    print("=> Les graphiques ont été sauvegardés sous 'q3_execution_times.png' et 'q3_pricing_errors.png'")
    print("=> Les données brutes sont dans 'heston_results.csv'")

    print("\n--- Analyse de l'impact de dt=1/30 sur Almost Exact ---")
    avg_speedup = df['Time_Alm1000'].mean() / df['Time_Alm30'].mean()
    avg_err_euler = df['Err_Euler'].mean()
    avg_err_alm30 = df['Err_Alm30'].mean()
    
    print(f"1. Accélération : Le schéma Almost Exact avec dt=1/30 est en moyenne {avg_speedup:.1f}x plus rapide que dt=1/1000.")
    print(f"2. Précision    : L'erreur moyenne de Euler(1000) est {avg_err_euler:.5f}.")
    print(f"                  L'erreur moyenne de Almost(30) est {avg_err_alm30:.5f}.")
    print("Conclusion      : Passer à dt=1/30 pour le schéma Broadie-Kaya réduit drastiquement le temps d'exécution")
    print("                  tout en gardant une précision excellente (car la variance reste simulée exactement).")

if __name__ == "__main__":
    compile_cuda()
    run_q1_q2()
    run_q3()