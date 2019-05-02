#include<stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include<stdlib.h>

/* Thread structure: 1, m*n
   */
__global__ void init(double *W1, double *W2, double *b1, double *b2){
    int mx = threadIdx.x, nx = threadIdx.y;
    int m = blockDim.x, n = blockDim.y;
    W1[m*nx + mx] = 0.1;
    W2[n*mx + nx] = 0.1;
    if(nx == 0){ 
        b1[mx] = 0.1;
    }   
    if(mx == 0){ 
        b2[nx] = 0.1;
    }   
}

/* Thread structure: N, m*n
   */
__global__ void next_layer(double *X, double *h, double *W, double *b){
    int mx = threadIdx.x, nx = threadIdx.y;
    int m = blockDim.x, n = blockDim.y, Nx = blockIdx.x;
    // Initializing h
    if(nx == 0){            // First thread to reach here should initialize 
        h[Nx*m + mx] = b[mx];
    }
    __syncthreads();
    atomicAdd(&h[Nx*m + mx], X[Nx*n + nx] * W[m*nx + mx]);
    double e;
    if(nx == 0){
        e = exp(h[Nx*m + mx]);
        h[Nx*m + mx] = e/(1 + e);
    }
    //printf("device: %d                     %lf\n", Nx*m + mx, h[Nx*m + mx]);
}

/*  Thread structure: N, m*n
   */
__global__ void calc_loss(double *Z, double *Y, double *W1, double *W2, double *loss){
    int mx = threadIdx.x, nx = threadIdx.y;
    int m = blockDim.x, n = blockDim.y, Nx = blockIdx.x;
    double d = 0, lambda = 0.01;
    if(Nx + mx + nx == 0)   // Only one should equate to zero
        loss = 0;
    __syncthreads();

    if(mx == 0){ 
        d = Z[Nx*n + nx] - Y[Nx*n + nx]; 
        d = d * d;
        printf("calc_loss Z: %d    %lf\n", Nx*n+nx, Z[Nx*n+nx]);
    }
    if(Nx == 0){ 
        d += lambda * (W1[m*nx + mx] * W1[m*nx + mx] + W2[n*mx + nx] * W2[n*mx + nx]); 
        printf("Index: W1: %d, W2: %d\n", m*nx + mx, m*nx + mx);
    }

    // ATOMIC OPERATION REQUIRED HERE
    if(d != 0){ 
        printf("calc_loss d: %lf\n", d);
        atomicAdd(loss, d); 
        printf("calc_loss loss: %lf\n", *loss);
    }
}

void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void pre_train(int m, int n, double *d_X, double *d_Y, double *d_Z, double *d_W1, double *d_W2, double *d_b1, double *d_b2, double *d_h, double *d_loss){
    int T = 10, N = 3;
    dim3 threads(m, n), threads_T(n, m); 
    init<<<1, threads>>>(d_W1, d_W2, d_b1, d_b2);
    cudaDeviceSynchronize();

    for(int t = 0;t < T; t++){
        printf("Iteration number: %d\n", t);
        next_layer<<<N, threads>>>(d_X, d_h, d_W1, d_b1);
        checkCUDAError("memory copy in next_layer");
        cudaDeviceSynchronize();
        next_layer<<<N, threads_T>>>(d_h, d_Z, d_W2, d_b2);
        cudaDeviceSynchronize();
        checkCUDAError("memory copy in second next_layer");

        calc_loss<<<N, threads>>>(d_Z, d_Y, d_W1, d_W2, d_loss + t);
        cudaDeviceSynchronize();
        checkCUDAError("memory copy in calc_loss");
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
    double *d_hY, *d_hZ, *d_W3, *d_W4, *d_b3, *d_b4;
    double *d_hhY, *d_hhZ;
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

    pre_train(m, n, d_X, d_Y, d_Z, d_W1, d_W2, d_b1, d_b2, d_h, d_loss);
    //pre_train(n, m, d

    cudaMemcpy(h, d_h, sizeof(double) * N * m, cudaMemcpyDeviceToHost);
    checkCUDAError("memory copy in h");
    cudaMemcpy(loss, d_loss, sizeof(double) * T, cudaMemcpyDeviceToHost);
    checkCUDAError("memory copy in loss");
    cudaMemcpy(Z, d_Z, sizeof(double) * N * n, cudaMemcpyDeviceToHost);
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
    printf("bye\n");
    cudaFree(d_W1);cudaFree(d_W2); cudaFree(d_b1); cudaFree(d_b2);
    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_h); cudaFree(d_Z);
    cudaFree(d_loss);
    free(X); free(Y); free(Z); free(h); free(loss);
    return 0;
}
