#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

// Kernel d'initialisation des états aléatoires
__global__ void init_curand_states(curandState *state, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

// Kernel du Modèle de Heston avec Schéma d'Euler
__global__ void heston_euler_kernel(
    float S0, float v0, float kappa, float theta, float sigma, float rho, 
    float dt, int N_steps, int N_sims, 
    curandState *state, float *sum_payoff) 
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    extern __shared__ float cache[];
    float local_sum = 0.0f;

    // Pré-calcul des constantes
    float sqrt_dt = sqrtf(dt);
    float sqrt_one_minus_rho_sq = sqrtf(1.0f - rho * rho);

    // Chargement de l'état aléatoire
    curandState localState;
    if (gid < stride) localState = state[gid];

    // GRID-STRIDE LOOP
    for (int i = gid; i < N_sims; i += stride) {
        float S = S0;
        float v = v0;

        for (int step = 0; step < N_steps; step++) {
            float2 G = curand_normal2(&localState);
            
            float v_trunc = fmaxf(v, 0.0f);
            float sqrt_v = sqrtf(v_trunc);

            float v_next = v + kappa * (theta - v_trunc) * dt + sigma * sqrt_v * sqrt_dt * G.x;
            v_next = fmaxf(v_next, 0.0f); 

            float Z_S = rho * G.x + sqrt_one_minus_rho_sq * G.y;
            float S_next = S + sqrt_v * S * sqrt_dt * Z_S;

            S = S_next;
            v = v_next;
        }

        local_sum += fmaxf(S - 1.0f, 0.0f);
    }

    if (gid < stride) state[gid] = localState;

    // TRANSFERT EN SHARED MEMORY ET RÉDUCTION
    cache[tid] = local_sum;
    __syncthreads();

    int i_reduce = blockDim.x / 2;
    while (i_reduce != 0) {
        if (tid < i_reduce) {
            cache[tid] += cache[tid + i_reduce];
        }
        __syncthreads();
        i_reduce /= 2;
    }

    // ÉCRITURE FINALE
    if (tid == 0) {
        atomicAdd(sum_payoff, cache[0]);
    }
}

int main() {
    float S0 = 1.0f;
    float v0 = 0.1f;
    float kappa = 0.5f;
    float theta = 0.1f;
    float sigma = 0.3f;
    float rho = -0.5f; // À VÉRIFIER
    
    int N_steps = 1000;
    float dt = 1.0f / N_steps;
    int N_sims = 1000000; 

    int NTPB = 256;
    int NB = 30 * 4; 
    int total_threads = NB * NTPB;

    float *d_sum_payoff;
    cudaMallocManaged(&d_sum_payoff, sizeof(float));
    *d_sum_payoff = 0.0f;

    curandState *d_states;
    cudaMalloc(&d_states, total_threads * sizeof(curandState));

    init_curand_states<<<NB, NTPB>>>(d_states, 1234ULL);
    
    // ==========================================
    // INSTRUCTIONS DE CHRONOMÉTRAGE GPU (DÉBUT)
    // ==========================================
    float Tim;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // ==========================================

    size_t shared_mem_size = NTPB * sizeof(float);
    heston_euler_kernel<<<NB, NTPB, shared_mem_size>>>(
        S0, v0, kappa, theta, sigma, rho, dt, N_steps, N_sims, d_states, d_sum_payoff
    );

    cudaDeviceSynchronize();

    // ==========================================
    // INSTRUCTIONS DE CHRONOMÉTRAGE GPU (FIN)
    // ==========================================
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Tim, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // ==========================================

    float call_price = (*d_sum_payoff) / N_sims;
    printf("Prix estime du Call Heston = %f\n", call_price);
    printf("Execution time %f ms\n", Tim);

    cudaFree(d_states);
    cudaFree(d_sum_payoff);
    return 0;
}