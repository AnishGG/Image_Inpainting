#include<stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include<stdlib.h>

__global__ void fxn(double *W, double *X, double *Y, double *b1, double *b2, double *h, double *Z, double *loss){
    int N = gridDim.x, m = blockDim.x, n = m/2, T = 10;
    int td = threadIdx.x, bd = blockIdx.x;
    double lambda = 0.01;
    if(bd == 0){
        curandState state;
        for(int i = 0;i < n; i++){
            curand_init(clock64(), i, 0, &state);
            W[m*i + td] = curand_uniform(&state);
            //W[m*i + td] = 0.1; 
        }
        b1[td] = curand_uniform(&state);
        //b1[td] = 0.1;       // RANDOM VALUES HERE
        if(td < n){
            //b2[td] = 0.1;
            b2[td] = curand_uniform(&state);
        }
    }
    // PUT CUDA BARRIER
    __syncthreads();

    for(int t = 0; t < T; t++){
        // Initialize loss
        if(bd == 0 && td == 0){
            loss[t] = 0;
     //       printf("t, loss[t]: %d, %lf\n", t, loss[t]);
        }
        // Initializing h
        h[bd*m + td] = b1[td];
        for(int j = 0;j < n; j++)
            h[bd*m + td] += X[bd*n + j] * W[m*j + td];
        // Sigmoid
        double e = exp(h[bd*m + td]);
        h[bd*m + td] = e/(1 + e);

        // MAY BE BARRIER HERE 
        __syncthreads();

        // calculating Z
        if(td < n){
            Z[bd*n + td] = b2[td];
            for(int j = 0;j < m; j++)
                Z[bd*n + td] += h[bd*m + j] * W[n*td + j];
            e = exp(Z[bd*n + td]);
            Z[bd*n + td] = e/(1 + e);
        }
        __syncthreads();
        
        double d = 0;
        if(td < n){
            //printf("%d %lf %lf\n", bd*n + td, Z[bd*n + td], Y[bd*n + td]);
            printf(" ");
            d = Z[bd*n + td] - Y[bd*n + td]; 
            d = d * d;
        }
        if(bd == 0){
            for(int i = 0; i < n; i++){
                d += lambda * W[m*i + td] * W[m*i + td]; 
                //printf("debug: %d, %lf\n", m*i + td,W[m*i + td]);
            }
        }

        // ATOMIC OPERATION REQUIRED HERE
        atomicAdd(&loss[t], d);
        __syncthreads();

        // eta needs to be declared and grad needs to be found here
/*        if(bd == 0){
            for(int i = 0;i < n; i++){
                W[m*i + td] -= eta * grad
            }
            b1[td] -= eta * grad;
            if(td < n)
                b2[td] -= eta * grad;
        }*/
        // ITERATION COMPLETE
    }
}

int main(){
    int N, m, n, T;
    N = 3, m = 4, n = m/2, T = 10;
    double *X, *Y, *h, *Z, *loss;
    X = (double*)malloc(N*n*sizeof(double));
    Y = (double*)malloc(N*n*sizeof(double));
    h = (double*)malloc(N*m*sizeof(double));
    Z = (double*)malloc(N*n*sizeof(double));
    loss = (double*)malloc(T * sizeof(double));

    // device memory
    double *d_X, *d_Y, *d_W, *d_b1, *d_b2, *d_h, *d_loss, *d_Z;
    cudaMalloc((void**)&d_W, sizeof(double) * n * m);
    cudaMalloc((void**)&d_X, sizeof(double) * N * n);
    cudaMalloc((void**)&d_Y, sizeof(double) * N * n);
    cudaMalloc((void**)&d_Z, sizeof(double) * N * n);
    cudaMalloc((void**)&d_b1, sizeof(double) * m);
    cudaMalloc((void**)&d_b2, sizeof(double) * n);
    cudaMalloc((void**)&d_h, sizeof(double) * N * m);
    cudaMalloc((void**)&d_loss, sizeof(double) * T);
    for(int i = 0; i < N*n; i++){
        scanf("%lf", &X[i]);
    }
    for(int i = 0; i < N*n; i++){
        scanf("%lf", &Y[i]);
    }
    cudaMemcpy(d_X, X, sizeof(double) * N * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, sizeof(double) * N * n, cudaMemcpyHostToDevice);
    fxn<<<N, m>>>(d_W, d_X, d_Y, d_b1, d_b2, d_h, d_Z, d_loss);
    cudaMemcpy(h, d_h, sizeof(double) * N * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(Z, d_Z, sizeof(double) * N * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(loss, d_loss, sizeof(double) * T, cudaMemcpyDeviceToHost);
    for(int i = 0;i < N*n; i++)
        printf("%lf ", Z[i]);
    printf("\n");
    printf("LOSS\n");
    for(int i = 0;i < T; i++)
        printf("%lf ", loss[i]);
    printf("\n");
    return 0;

}
