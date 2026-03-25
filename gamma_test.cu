
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
        
        // FIX: Ecriture dans un fichier plutot que stdout
        FILE *f = fopen("gamma_samples.txt", "w");
        for(int i=0; i<n; i++) fprintf(f, "%f\n", d_out[i]);
        fclose(f);
        
        cudaFree(d_out); cudaFree(d_state);
        return 0;
    }
    