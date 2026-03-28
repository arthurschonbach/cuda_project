import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

RHO = -0.5
plt.style.use('ggplot')

def compile_codes():
    print("1. Compilation du code principal heston.cu...")
    res = subprocess.run(["nvcc", "-O3", "heston.cu", "-o", "heston"], capture_output=True, text=True)
    if res.returncode != 0:
        print("Erreur de compilation heston.cu:\n", res.stderr)
        exit(1)

    print("2. Génération et compilation de l'utilitaire gamma_test.cu...")
    gamma_cu = r"""
    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include <curand_kernel.h>

    __global__ void init_curand(curandState *state, unsigned long seed) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, id, 0, &state[id]);
    }

    __device__ float generate_gamma(curandState *state, float alpha) {
        float a = alpha;
        bool less_than_one = false;
        if (a < 1.0f) {
            a += 1.0f;
            less_than_one = true;
        }
        float d = a - 1.0f / 3.0f;
        float c = 1.0f / sqrtf(9.0f * d);
        float Z, U, V, X;

        while (true) {
            Z = curand_normal(state);
            if (Z > -1.0f / c) {
                V = 1.0f + c * Z;
                V = V * V * V;
                U = curand_uniform(state); 
                if (logf(U) < 0.5f * Z * Z + d - d * V + d * logf(V)) {
                    X = d * V;
                    break;
                }
            }
        }
        if (less_than_one) {
            float U2 = curand_uniform(state);
            X = X * powf(U2, 1.0f / alpha);
        }
        return X;
    }

    __global__ void get_gamma(curandState *state, float alpha, float *out, int n) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < n) out[id] = generate_gamma(&state[id], alpha);
    }

    int main(int argc, char **argv) {
        if(argc!=2) return 1;
        float alpha = atof(argv[1]);
        int n = 50000;
        float *d_out; cudaMallocManaged(&d_out, n * sizeof(float));
        curandState *d_state; cudaMalloc(&d_state, n * sizeof(curandState));
        
        init_curand<<<(n+255)/256, 256>>>(d_state, 1234ULL);
        cudaDeviceSynchronize();
        get_gamma<<<(n+255)/256, 256>>>(d_state, alpha, d_out, n);
        cudaDeviceSynchronize();
        
        FILE *f = fopen("gamma_samples.txt", "w");
        for(int i=0; i<n; i++) fprintf(f, "%f\n", d_out[i]);
        fclose(f);
        
        cudaFree(d_out); cudaFree(d_state);
        return 0;
    }
    """
    with open("gamma_test.cu", "w") as f:
        f.write(gamma_cu)
    
    res = subprocess.run(["nvcc", "-O3", "gamma_test.cu", "-o", "gamma_test"], capture_output=True, text=True)
    if res.returncode != 0:
        print("Erreur compilation gamma_test.cu:\n", res.stderr)
        exit(1)
    print("Compilation réussie.\n")

def run_heston(method, kappa, theta, sigma, rho, n_steps, n_sims):
    cmd = f"./heston {method} {kappa} {theta} {sigma} {rho} {n_steps} {n_sims}"
    res = subprocess.run(cmd.split(), capture_output=True, text=True)
    if res.returncode != 0:
        return np.nan, np.nan, np.nan
    try:
        output = res.stdout.strip().split(',')
        if len(output) != 3: # On attend maintenant 3 valeurs
            return np.nan, np.nan, np.nan
        return float(output[0]), float(output[1]), float(output[2])
    except:
        return np.nan, np.nan, np.nan

# ==========================================
# FIGURES GÉNÉRATION
# ==========================================

def fig1_euler_convergence():
    print("Génération Figure 1 : Convergence Euler...")
    n_sims_list =[1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    prices_mean = []
    prices_ci = []
    
    # Remplacement de la boucle de 10 runs par une utilisation propre de l'erreur standard
    for n in n_sims_list:
        p_mean, std_err, _ = run_heston(0, 0.5, 0.1, 0.3, RHO, 1000, n)
        
        prices_mean.append(p_mean)
        # Intervalle à 95% = 1.96 * Erreur Standard
        if not np.isnan(std_err):
            prices_ci.append(1.96 * std_err)
        else:
            prices_ci.append(np.nan)

    prices_mean = np.array(prices_mean)
    ci = np.array(prices_ci)

    plt.figure(figsize=(8, 5))
    plt.plot(n_sims_list, prices_mean, 'b-', marker='o', label="Prix estimé (Euler dt=1/1000)")
    plt.fill_between(n_sims_list, prices_mean - ci, prices_mean + ci, color='blue', alpha=0.2, label="Intervalle de confiance 95%")
    plt.xscale('log')
    plt.title("Convergence du schéma d'Euler en fonction de N")
    plt.xlabel("Nombre de trajectoires N (échelle log)")
    plt.ylabel("Prix estimé du Call")
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig1_euler_convergence.png', dpi=300)
    plt.close()

def fig2_gamma_distribution():
    print("Génération Figure 2 : Validation de la loi Gamma sur GPU...")
    alphas =[0.5, 2.5]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, alpha in enumerate(alphas):
        subprocess.run(["./gamma_test", str(alpha)], capture_output=True)
        samples = np.loadtxt("gamma_samples.txt")
        
        axes[idx].hist(samples, bins=100, density=True, alpha=0.6, color='skyblue', edgecolor='black', label="Samples GPU")
        
        x = np.linspace(0.01, max(samples), 500)
        y = stats.gamma.pdf(x, a=alpha, scale=1.0)
        axes[idx].plot(x, y, 'r-', lw=2, label="Théorique $\mathcal{G}(\\alpha)$")
        
        axes[idx].set_title(f"Génération Gamma pour $\\alpha = {alpha}$")
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig('fig2_gamma_validation.png', dpi=300)
    plt.close()
    if os.path.exists("gamma_samples.txt"):
        os.remove("gamma_samples.txt")

def fig3_and_4_tradeoff_and_time():
    print("Génération Figures 3 & 4 : Performances et Trade-off...")
    np.random.seed(42)
    configs = []
    
    while len(configs) < 10:
        k = np.random.uniform(0.1, 10.0)
        th = np.random.uniform(0.01, 0.5)
        sig = np.random.uniform(0.1, 1.0)
        if 20 * k * th > sig**2:
            configs.append((k, th, sig))
            
    results =[]
    for i, (k, th, sig) in enumerate(configs):
        # Adaptation pour récupérer les 3 variables et ignorer l'erreur standard (_)
        p_ref, _, _ = run_heston(1, k, th, sig, RHO, 1000, 1000000)
        p_eul, _, t_eul = run_heston(0, k, th, sig, RHO, 1000, 1000000)
        p_alm1, _, t_alm1 = run_heston(2, k, th, sig, RHO, 1000, 1000000)
        p_alm30, _, t_alm30 = run_heston(2, k, th, sig, RHO, 30, 1000000)
        
        if not np.isnan(p_ref):
            results.append({
                'k': k, 'th': th, 'sig': sig,
                'err_eul': abs(p_eul - p_ref), 't_eul': t_eul,
                'err_alm1000': abs(p_alm1 - p_ref), 't_alm1000': t_alm1,
                'err_alm30': abs(p_alm30 - p_ref), 't_alm30': t_alm30
            })
    df = pd.DataFrame(results)

    # --- FIGURE 3 : Bar Chart ---
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, df['t_eul'], width, label='Euler (dt=1/1000)', color='navy')
    plt.bar(x + width/2, df['t_alm1000'], width, label='Presque Exact (dt=1/1000)', color='darkorange')
    plt.ylabel('Temps d\'exécution (ms)')
    plt.xlabel('Configurations (Paramètres aléatoires)')
    plt.title("Temps d'exécution à pas fin (N = 1 000 000)")
    plt.xticks(x, [f"Conf {i+1}" for i in x])
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig3_execution_time.png', dpi=300)
    plt.close()

    # --- FIGURE 4 : Scatter Plot Trade-off ---
    plt.figure(figsize=(8, 6))
    plt.scatter(df['t_eul'], df['err_eul'], c='navy', marker='o', s=100, label='Euler (1/1000)')
    plt.scatter(df['t_alm1000'], df['err_alm1000'], c='darkorange', marker='s', s=100, label='Presque Exact (1/1000)')
    plt.scatter(df['t_alm30'], df['err_alm30'], c='green', marker='^', s=150, label='Presque Exact (1/30)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Temps d'exécution total (ms) - Échelle log")
    plt.ylabel("Erreur absolue vs Schéma exact - Échelle log")
    plt.title("Trade-off Précision vs Temps de Calcul")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig4_tradeoff.png', dpi=300)
    plt.close()

def fig5_heatmap():
    print("Génération Figure 5 : Heatmap des erreurs (Presque Exact dt=1/30 vs Exact)...")
    
    kappas = np.linspace(0.1, 10.0, 10)
    sigmas = np.linspace(0.1, 1.0, 10)
    theta_heatmap = 0.1 
    
    error_matrix = np.zeros((len(kappas), len(sigmas)))
    
    for i, k in enumerate(kappas):
        for j, sig in enumerate(sigmas):
            if 20 * k * theta_heatmap <= sig**2:
                error_matrix[i, j] = np.nan 
            else:
                # Adaptation pour récupérer les 3 variables
                p_ref, _, _ = run_heston(1, k, theta_heatmap, sig, RHO, 1000, 100000) 
                p_alm30, _, _ = run_heston(2, k, theta_heatmap, sig, RHO, 30, 100000)
                if not np.isnan(p_ref) and not np.isnan(p_alm30):
                    error_matrix[i, j] = abs(p_alm30 - p_ref)
                else:
                    error_matrix[i, j] = np.nan

    plt.figure(figsize=(8, 6))
    cmap = plt.colormaps.get_cmap('YlOrRd').copy()
    cmap.set_bad(color='lightgrey')
    
    ext =[sigmas.min(), sigmas.max(), kappas.min(), kappas.max()]
    plt.imshow(error_matrix, origin='lower', extent=ext, aspect='auto', cmap=cmap)
    
    plt.colorbar(label="Erreur Absolue")
    plt.xlabel("Volatilité de la volatilité ($\sigma$)")
    plt.ylabel("Vitesse de retour à la moyenne ($\kappa$)")
    plt.title(f"Heatmap Erreurs (Presque Exact dt=1/30, $\\theta={theta_heatmap}$)\nZones grises = Feller non respectée")
    plt.tight_layout()
    plt.savefig('fig5_heatmap.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    compile_codes()
    fig1_euler_convergence()
    fig2_gamma_distribution()
    fig3_and_4_tradeoff_and_time()
    fig5_heatmap()
    print("\nTERMINE ! Toutes les figures (fig1_*.png à fig5_*.png) ont été générées avec succès.")