#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

// Function that catches the error 

void testCUDA(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("There is an error in file %s at line %d\n", file, line);
        printf("Error description: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(val) testCUDA((val), __FILE__, __LINE__)

// INITIALISATION DU HASARD
__global__ void init_curand_states(curandState *state, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

// GÉNÉRATION GAMMA (Marsaglia & Tsang)
__device__ float generate_gamma(curandState *state, float alpha) {
    float a = alpha;
    bool less_than_one = false;

    // Astuce pour alpha < 1 afin d'atténuer la divergence de warp
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

// KERNEL 1 : HESTON (SCHÉMA D'EULER)
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

    float sqrt_dt = sqrtf(dt);
    float sqrt_one_minus_rho_sq = sqrtf(1.0f - rho * rho);

    curandState localState;
    if (gid < stride) localState = state[gid];

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

    cache[tid] = local_sum;
    __syncthreads();

    int i_reduce = blockDim.x / 2;
    while (i_reduce != 0) {
        if (tid < i_reduce) cache[tid] += cache[tid + i_reduce];
        __syncthreads();
        i_reduce /= 2;
    }

    if (tid == 0) atomicAdd(sum_payoff, cache[0]);
}

//  KERNEL 2 : HESTON (EXACT SIMULATION - BROADIE-KAYA / EULER)
__global__ void heston_exact_kernel(
    float S0, float v0, float kappa, float theta, float sigma, float rho, 
    float dt, int N_steps, int N_sims, 
    curandState *state, float *sum_payoff) 
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    extern __shared__ float cache[];
    float local_sum = 0.0f;

    curandState localState;
    if (gid < stride) localState = state[gid];

    // Constantes BK pour éviter les calculs redondants
    float exp_kdt = expf(-kappa * dt);
    float d = 2.0f * kappa * theta / (sigma * sigma);
    float c_lambda = (2.0f * kappa * exp_kdt) / (sigma * sigma * (1.0f - exp_kdt));
    float c_v_next = (sigma * sigma * (1.0f - exp_kdt)) / (2.0f * kappa);
    
    for (int i = gid; i < N_sims; i += stride) {
        float v_t = v0;
        float vI = 0.0f; 

        for (int step = 0; step < N_steps; step++) {
            float lambda = c_lambda * v_t;
            unsigned int N = curand_poisson(&localState, lambda);
            
            float alpha = d + (float)N;
            float v_next = c_v_next * generate_gamma(&localState, alpha);
            
            vI += 0.5f * (v_t + v_next) * dt;
            v_t = v_next;
        }

        float v_1 = v_t; 
        
        float int_sqrt_v_dW = (1.0f / sigma) * (v_1 - v0 - kappa * theta + kappa * vI);
        float m = -0.5f * vI + rho * int_sqrt_v_dW;
        float Sigma_sq = (1.0f - rho * rho) * vI;
        float Sigma = sqrtf(Sigma_sq);
        
        float G = curand_normal(&localState);
        float S_1 = S0 * expf(m + Sigma * G);
        
        local_sum += fmaxf(S_1 - S0, 0.0f); // Strike K = S0 = 1.0f
    }

    if (gid < stride) state[gid] = localState;

    cache[tid] = local_sum;
    __syncthreads();

    int i_reduce = blockDim.x / 2;
    while (i_reduce != 0) {
        if (tid < i_reduce) cache[tid] += cache[tid + i_reduce];
        __syncthreads();
        i_reduce /= 2;
    }

    if (tid == 0) atomicAdd(sum_payoff, cache[0]);
}

int main() {
    // Paramètres Heston
    float S0 = 1.0f;
    float v0 = 0.1f;
    float kappa = 0.5f;
    float theta = 0.1f;
    float sigma = 0.3f;
    float rho = -0.5f; // Standard pour les marchés financiers
    
    int N_steps = 1000;
    float dt = 1.0f / N_steps;
    int N_sims = 5000000; 

    // Configuration Grille (pour ma RTX 3060 : 30 SMs)
    int NTPB = 256;
    int NB = 120; // 30*4 blocs
    int total_threads = NB * NTPB;
    size_t shared_mem_size = NTPB * sizeof(float);

    // Allocation Unified Memory
    float *d_sum_euler, *d_sum_exact;
    CUDA_CHECK(cudaMallocManaged(&d_sum_euler, sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_sum_exact, sizeof(float)));
    *d_sum_euler = 0.0f;
    *d_sum_exact = 0.0f;

    // Allocation CurandStates
    curandState *d_states;
    CUDA_CHECK(cudaMalloc(&d_states, total_threads * sizeof(curandState)));

    // Timers
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float time_euler, time_exact;

    printf("Initialisation des etats Curand (Seed: 1234)...\n");
    init_curand_states<<<NB, NTPB>>>(d_states, 1235ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- TEST 1 : EULER ---
    printf("Lancement Kernel Euler (%d trajectoires)...\n", N_sims);
    CUDA_CHECK(cudaEventRecord(start, 0));
    
    heston_euler_kernel<<<NB, NTPB, shared_mem_size>>>(
        S0, v0, kappa, theta, sigma, rho, dt, N_steps, N_sims, d_states, d_sum_euler
    );
    CUDA_CHECK(cudaPeekAtLastError());
    
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_euler, start, stop));

    // --- TEST 2 : EXACT ---
    printf("Lancement Kernel Exact (%d trajectoires)...\n", N_sims);
    CUDA_CHECK(cudaEventRecord(start, 0));
    
    heston_exact_kernel<<<NB, NTPB, shared_mem_size>>>(
        S0, v0, kappa, theta, sigma, rho, dt, N_steps, N_sims, d_states, d_sum_exact
    );
    CUDA_CHECK(cudaPeekAtLastError());
    
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_exact, start, stop));

    // --- RÉSULTATS ---
    float price_euler = (*d_sum_euler) / N_sims;
    float price_exact = (*d_sum_exact) / N_sims;

    printf("\n================ RESULTATS ================\n");
    printf("%-20s | %-10s | %-10s\n", "Methode", "Prix", "Temps (ms)");
    printf("-------------------------------------------\n");
    printf("%-20s | %.6f   | %.2f\n", "Euler", price_euler, time_euler);
    printf("%-20s | %.6f   | %.2f\n", "Exact (Broadie-Kaya)", price_exact, time_exact);
    printf("===========================================\n");

    // Nettoyage
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_sum_euler));
    CUDA_CHECK(cudaFree(d_sum_exact));

    return 0;
}