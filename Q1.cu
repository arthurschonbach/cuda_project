#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

#define NUM_PATHS 4096 

#define KAPPA 0.5f
#define THETA 0.1f
#define SIGMA 0.3f
#define RHO -0.5f

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Erreur CUDA à la ligne %d : %s\n", __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0) 

__global__ void simulate_heston_euler(float *d_payoffs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > NUM_PATHS) return;

    curandState state;
    curand_init(1234, idx, 0, &state);

    float S = 1.0f;
    float v = 0.1f;

    int timesteps = 1000;
    for (int i=0; i<timesteps; i++) {
        float G1 = curand_normal(&state);
        float G2 = curand_normal(&state); 

        S = S * (1 + sqrtf(v / timesteps) * (RHO * G1 + sqrtf(1-pow(RHO, 2)) * G2));
        v = fmaxf(v + KAPPA * (THETA - v) / timesteps + SIGMA * sqrtf(v / timesteps) * G1, 0.0f);
    }

    d_payoffs[idx] = fmaxf(S - 1, 0.0f);
}

int main() {
    size_t size = NUM_PATHS * sizeof(float);

    float *h_payoffs = (float*)calloc(NUM_PATHS, sizeof(float));
    float *d_payoffs;

    CUDA_CHECK( cudaMalloc((void **) &d_payoffs, size) );

    simulate_heston_euler<<<4, 1024>>>(d_payoffs);
    CUDA_CHECK( cudaPeekAtLastError() ) ;
    CUDA_CHECK( cudaDeviceSynchronize() );

    CUDA_CHECK( cudaMemcpy(h_payoffs, d_payoffs, size, cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaFree(d_payoffs) );

    float avg_payoff;
    for (int i=0; i<NUM_PATHS; i++) {
        avg_payoff += h_payoffs[i];
    }
    avg_payoff /= NUM_PATHS;
    printf("Average payoff: %.2f\n", avg_payoff);

    free(h_payoffs);

    return 0;
}

