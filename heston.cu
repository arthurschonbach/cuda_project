#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ==========================================
// KERNELS UTILITAIRES
// ==========================================
__global__ void init_curand_states(curandState *state, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

// Génération Gamma (Marsaglia & Tsang) gérant alpha < 1 et alpha >= 1
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

// ==========================================
// KERNELS DE PRICING
// ==========================================

// KERNEL 0 : EULER
__global__ void heston_euler(float S0, float v0, float kappa, float theta, float sigma, float rho, float K, float dt, int N_steps, int N_sims, curandState *state, float *sum_payoff) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    extern __shared__ float cache[];
    float local_sum = 0.0f;
    
    float sqrt_dt = sqrtf(dt);
    float sqrt_1_rho2 = sqrtf(1.0f - rho * rho);
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
            float S_next = S + sqrt_v * S * sqrt_dt * (rho * G.x + sqrt_1_rho2 * G.y);
            S = S_next;
            v = v_next;
        }
        local_sum += fmaxf(S - K, 0.0f);
    }
    if (gid < stride) state[gid] = localState;

    cache[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) cache[tid] += cache[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(sum_payoff, cache[0]);
}

// KERNEL 1 : EXACT (Broadie-Kaya)
__global__ void heston_exact(float S0, float v0, float kappa, float theta, float sigma, float rho, float K, float dt, int N_steps, int N_sims, curandState *state, float *sum_payoff) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    extern __shared__ float cache[];
    float local_sum = 0.0f;
    curandState localState;
    if (gid < stride) localState = state[gid];

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
        float int_sqrt_v_dW = (1.0f / sigma) * (v_t - v0 - kappa * theta + kappa * vI);
        float m = -0.5f * vI + rho * int_sqrt_v_dW;
        float Sigma_sq = (1.0f - rho * rho) * vI;
        float G = curand_normal(&localState);
        float S_final = S0 * expf(m + sqrtf(Sigma_sq) * G);
        local_sum += fmaxf(S_final - K, 0.0f); 
    }
    if (gid < stride) state[gid] = localState;

    cache[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) cache[tid] += cache[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(sum_payoff, cache[0]);
}

// KERNEL 2 : ALMOST EXACT
__global__ void heston_almost(float S0, float v0, float kappa, float theta, float sigma, float rho, float K, float dt, int N_steps, int N_sims, curandState *state, float *sum_payoff) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    extern __shared__ float cache[];
    float local_sum = 0.0f;
    curandState localState;
    if (gid < stride) localState = state[gid];

    float exp_kdt = expf(-kappa * dt);
    float d = 2.0f * kappa * theta / (sigma * sigma);
    float c_lambda = (2.0f * kappa * exp_kdt) / (sigma * sigma * (1.0f - exp_kdt));
    float c_v_next = (sigma * sigma * (1.0f - exp_kdt)) / (2.0f * kappa);

    float k_0 = -(rho / sigma) * kappa * theta * dt;
    float k_1 = ((rho * kappa) / sigma - 0.5f) * dt - (rho / sigma);
    float k_2 = rho / sigma;
    float var_coef = 1.0f - rho * rho;
    
    for (int i = gid; i < N_sims; i += stride) {
        float v_t = v0;
        float logS_t = logf(S0);
        for (int step = 0; step < N_steps; step++) {
            float lambda = c_lambda * v_t;
            unsigned int N = curand_poisson(&localState, lambda);
            float alpha = d + (float)N;
            float v_next = c_v_next * generate_gamma(&localState, alpha);
            
            float G_z = curand_normal(&localState); // Représente la combinaison linéaire (rho G1 + sqrt G2)
            logS_t += k_0 + k_1 * v_t + k_2 * v_next + sqrtf(var_coef * v_t * dt) * G_z;
            v_t = v_next;
        }        
        local_sum += fmaxf(expf(logS_t) - K, 0.0f);
    }
    if (gid < stride) state[gid] = localState;

    cache[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) cache[tid] += cache[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(sum_payoff, cache[0]);
}

// ==========================================
// MAIN (CLI Interface)
// ==========================================
int main(int argc, char **argv) {
    if (argc != 8) return 1;

    int method = atoi(argv[1]);
    float kappa = atof(argv[2]);
    float theta = atof(argv[3]);
    float sigma = atof(argv[4]);
    float rho = atof(argv[5]);
    int N_steps = atoi(argv[6]);
    int N_sims = atoi(argv[7]);

    float S0 = 1.0f, K = 1.0f, v0 = 0.1f;
    float dt = 1.0f / N_steps;

    int NTPB = 256;
    int NB = 120;
    int total_threads = NB * NTPB;
    size_t shared_mem_size = NTPB * sizeof(float);

    float *d_sum;
    curandState *d_states;
    CUDA_CHECK(cudaMallocManaged(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_states, total_threads * sizeof(curandState)));
    *d_sum = 0.0f;

    init_curand_states<<<NB, NTPB>>>(d_states, 1234ULL);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (method == 0) heston_euler<<<NB, NTPB, shared_mem_size>>>(S0, v0, kappa, theta, sigma, rho, K, dt, N_steps, N_sims, d_states, d_sum);
    else if (method == 1) heston_exact<<<NB, NTPB, shared_mem_size>>>(S0, v0, kappa, theta, sigma, rho, K, dt, N_steps, N_sims, d_states, d_sum);
    else if (method == 2) heston_almost<<<NB, NTPB, shared_mem_size>>>(S0, v0, kappa, theta, sigma, rho, K, dt, N_steps, N_sims, d_states, d_sum);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    float price = (*d_sum) / N_sims;
    printf("%.6f,%.3f\n", price, time_ms);

    cudaFree(d_states); cudaFree(d_sum);
    return 0;
}