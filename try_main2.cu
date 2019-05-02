#include<stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include<stdlib.h>

__global__ void fxn(double *W1, double *W2, double *X, double *Y, double *b1, double *b2, double *h, double *Z, double *loss){
    int m = blockDim.x, n = blockDim.y, T = 10;
    int mx = threadIdx.x, Nx = blockIdx.x, nx = threadIdx.y;
    double lambda = 0.01;

    // Initialization
    if(Nx == 0){
/*        curandState state;
        curand_init(clock64(), nx, 0, &state);
        W1[m*nx + mx] = curand_uniform(&state);
        curand_init(clock64(), nx, 0, &state);
        W2[n*mx + nx] = curand_uniform(&state);
        printf("%lf %lf\n", W1[m*nx + mx], W2[n*mx + nx]);
        */
        W1[m*nx + mx] = 0.1;
        W2[n*mx + nx] = 0.1;
        if(nx == 0){
            //b1[mx] = curand_uniform(&state);
            b1[mx] = 0.1;
        }
        if(mx == 0){
            //b2[nx] = curand_uniform(&state);
            b2[nx] = 0.1;
        }
    }
    __syncthreads();
    // PUT CUDA BARRIER

    for(int t = 0; t < T; t++){
        // Initialize loss
        if(Nx == 0 && mx == 0 && nx == 0){
            printf("***********\n\nTHIS IS ITERATION NUMBER %d\n\n****************\n\n", t);
            loss[t] = 0;
        }
        // Initializing h
        if(nx == 0){
            h[Nx*m + mx] = b1[mx];
        }

        atomicAdd(&h[Nx*m + mx], X[Nx*n + nx] * W1[m*nx + mx]);
        __syncthreads();
        if(nx == 0)
            printf("H values: %lf\n", h[Nx*m + mx]);

        double e;
        // Sigmoid
        if(nx == 0){
            e = exp(h[Nx*m + mx]);
            h[Nx*m + mx] = e/(1 + e);
        }
        __syncthreads();


        // calculating Z
        if(mx == 0){
            Z[Nx*n + nx] = b2[nx];
        }
        atomicAdd(&Z[Nx*n + nx], h[Nx*m + mx] * W2[n*mx + nx]); // CHECK SWAP
        __syncthreads();

        if(mx == 0){
            e = exp(Z[Nx*n + nx]);
            Z[Nx*n + nx] = e/(1 + e);
            printf("Z values: %lf\n", Z[Nx*n + nx]);
        }
        __syncthreads();
        printf("%d %d %d\n", Nx, mx, nx);
        
        double d = 0;
        if(mx == 0){
            //printf("%d %lf %lf\n", bd*n + td, Z[bd*n + td], Y[bd*n + td]);
            d = Z[Nx*n + nx] - Y[Nx*n + nx]; 
            d = d * d;
        }
        if(Nx == 0){
            d += lambda * (W1[m*nx + mx] * W1[m*nx + mx] + W2[n*mx + nx] * W2[n*mx + nx]); 
            //atomicAdd(&d, dx);
            printf("aya");
        }
        if(mx == 0 || Nx == 0)
            printf("d value here: %d %d %lf\n", mx, Nx, d);

        // ATOMIC OPERATION REQUIRED HERE
        if(d != 0){
            atomicAdd(&loss[t], d);
        }
        printf("down: %d %d %d\n", Nx, mx, nx);
        if(Nx + mx + nx == 0)
            printf("loss: %lf\n", loss[t]);

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
    double *d_X, *d_Y, *d_W1, *d_W2, *d_b1, *d_b2, *d_h, *d_loss, *d_Z;
    cudaMalloc((void**)&d_W1, sizeof(double) * n * m);
    cudaMalloc((void**)&d_W2, sizeof(double) * m * n);
    cudaMalloc((void**)&d_X, sizeof(double) * N * n);
    cudaMalloc((void**)&d_Y, sizeof(double) * N * n);
    cudaMalloc((void**)&d_Z, sizeof(double) * N * n);
    cudaMalloc((void**)&d_b1, sizeof(double) * m);
    cudaMalloc((void**)&d_b2, sizeof(double) * n);
    cudaMalloc((void**)&d_h, sizeof(double) * N * m);
    cudaMalloc((void**)&d_loss, sizeof(double) * T);
    for(int i = 0; i < N*n; i++){
        scanf("%lf", &X[i]);
        printf("%lf\n", X[i]);
    }
    for(int i = 0; i < N*n; i++){
        scanf("%lf", &Y[i]);
        printf("%lf\n", Y[i]);
    }
    cudaMemcpy(d_X, X, sizeof(double) * N * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, sizeof(double) * N * n, cudaMemcpyHostToDevice);

    dim3 threads(m, n);
    fxn<<<N, threads>>>(d_W1, d_W2, d_X, d_Y, d_b1, d_b2, d_h, d_Z, d_loss);
    cudaMemcpy(h, d_h, sizeof(double) * N * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(Z, d_Z, sizeof(double) * N * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(loss, d_loss, sizeof(double) * T, cudaMemcpyDeviceToHost);
    printf("h\n");
    for(int i = 0;i < N*m; i++)
        printf("%lf ", h[i]);
    printf("\nZ");
    for(int i = 0;i < N*n; i++)
        printf("%lf ", Z[i]);
    printf("\n");
    printf("LOSS\n");
    for(int i = 0;i < T; i++)
        printf("%lf ", loss[i]);
    printf("\n");
    return 0;

}
