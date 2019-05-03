#include<stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include<stdlib.h>

/* Thread structure: 1, m*n
   This function initializes random values into an array according to the above thread structure
   */
__global__ void init(double *W){
    int mx = threadIdx.x;
    int nx = threadIdx.y;
    int m = blockDim.x;/*, n = blockDim.y;*/

    // Random state from cuda device
    curandState state;
    curand_init(clock64(), mx, nx, &state);

    // Feeding random values according mx and nx values each time
    W[m*nx + mx] = curand_uniform(&state);
}

/* Thread structure: N, m*n
   This function calculates the essentials of the next layer 
   provided the inputs
   */
__global__ void next_layer(double *X, double *h, double *W, double *b){
    int mx = threadIdx.x;
    int nx = threadIdx.y;
    int m = blockDim.x, n = blockDim.y;
    int Nx = blockIdx.x;

    // Initializing h
    if(nx == 0){            // First thread to reach here should initialize 
        h[Nx*m + mx] = b[mx];
    }
    __syncthreads();

    // Need to have a lock on the variable before adding, as many threads could access this variable at the same time
    atomicAdd(&h[Nx*m + mx], X[Nx*n + nx] * W[m*nx + mx]);

    // Applying sigmoid function here for the loss value
    double e;
    if(nx == 0){
        e = exp(h[Nx*m + mx]);
        h[Nx*m + mx] = e/(1 + e);
    }
    //printf("device: %d                     %lf\n", Nx*m + mx, h[Nx*m + mx]);
}

/*  Thread structure: N, m*n
    This function calculates the loss suffered by the layer
    at any point in time
   */
__global__ void calc_loss(double *Z, double *Y, double *W1, double *W2, double *loss){
    int mx = threadIdx.x;
    int nx = threadIdx.y;
    int m = blockDim.x, n = blockDim.y;
    int Nx = blockIdx.x;

    double d = 0, lambda = 0.01;

    // Only one thread should equate it to zero
    if(Nx + mx + nx == 0)
        loss = 0;
    __syncthreads();

    // Squared loss
    if(mx == 0){ 
        d = Z[Nx*n + nx] - Y[Nx*n + nx]; 
        d = d * d;
        printf("calc_loss Z: %d    %lf\n", Nx*n+nx, Z[Nx*n+nx]);
    }

    // Loss functions second term
    if(Nx == 0){ 
        d += lambda * (W1[m*nx + mx] * W1[m*nx + mx] + W2[n*mx + nx] * W2[n*mx + nx]); 
        printf("Index: W1: %d, W2: %d\n", m*nx + mx, m*nx + mx);
    }

    // ATOMIC OPERATION REQUIRED HERE
    if(d != 0){ 
        *loss += d;
        //atomicAdd(loss, d); 
    }
    __syncthreads();

    if(Nx + mx + nx == 0)
        printf("calc_loss loss: %lf\n", *loss);
}

/* Function to check wrong memory 
   access and other errors
   */
void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Thread structure: N, m*n
   This function deals with the backpropogation and updating the values
   according to the learning rate as set by the user
   */
__global__ void update(double *X, double *Y, double *Z, double *W1, double *W2, double *b1, double *b2, double *h){
    int mx = threadIdx.x;
    int nx = threadIdx.y;
    int m = blockDim.x, n = blockDim.y;
    int Nx = blockIdx.x;
    double eta = 10, lambda = 0.1; 
    double temp;
     
    // Frob norm
    if(Nx == 0){
        atomicAdd(&W2[n*mx + nx], -lambda*W2[n*mx + nx]); 
        atomicAdd(&W1[m*nx + mx], -lambda*W1[m*nx + mx]);
    }
    __syncthreads();

    // Loss for b2
    if(mx == 0){        // for single m
        temp = (Z[n*Nx + nx] - Y[n*Nx + nx]) * Z[n*Nx + nx] * (1 - Z[n*Nx + nx]) * (-eta);
        printf("Temp: %.12lf\n", temp);
        atomicAdd(&b2[nx], temp);
    }

    // Loss for W2
    temp = (Z[n*Nx + nx] - Y[n*Nx + nx]) * Z[n*Nx + nx] * (1 - Z[n*Nx + nx]) * h[m*Nx + mx] * (-eta);
    atomicAdd(&W2[n*mx + nx], temp);

    // Loss for b1
    temp = (Z[n*Nx + nx] - Y[n*Nx + nx]) * Z[n*Nx + nx] * (1 - Z[n*Nx + nx]) * h[m*Nx + mx] * (1 - h[m*Nx + mx]) * W2[n*mx + nx] * (-eta);
    atomicAdd(&b1[mx], temp);

    // Loss for W1
    for(int it = 0; it < n; it++){
        temp = (Z[n*Nx + nx] - Y[n*Nx + nx]) * Z[n*Nx + nx] * (1 - Z[n*Nx + nx]) * h[m*Nx + mx] * (1 - h[m*Nx + mx]) * W2[n*mx + nx] * X[Nx*n + it]* (-eta);
        atomicAdd(&W1[m*it + mx], temp);
    } 

}

/* Pre training the model
   */
void pre_train(int N, int m, int n, double *d_X, double *d_Y, double *d_Z, double *d_W1, double *d_W2, double *d_b1, double *d_b2, double *d_h, double *d_loss, int which_itr){
    int T = 10;
    dim3 threads(m, n), threads_T(n, m); 

    // Only first iteration needs initialization of these vectors
    // Random initialization of vectors required only in the first iteration
    if(which_itr == 0){
        init<<<1, threads>>>(d_W1);
        init<<<1, m>>>(d_b1);
    }
    
    // Random initialization of the vectors of the second layer required each time this function is called
    init<<<1, threads_T>>>(d_W2);
    init<<<1, n>>>(d_b2);

    // Barrier to avoid running beyond this point
    cudaDeviceSynchronize();

    for(int t = 0;t < T; t++){
        printf("Iteration number: %d\n", t);

        // Calculating first hidden layer
        next_layer<<<N, threads>>>(d_X, d_h, d_W1, d_b1);
        cudaDeviceSynchronize();
        //checkCUDAError("memory copy in next_layer");

        // Calculating the layer next to hidden layer
        next_layer<<<N, threads_T>>>(d_h, d_Z, d_W2, d_b2);
        cudaDeviceSynchronize();
        //checkCUDAError("memory copy in second next_layer");

        // Updating the vectors according the gradient of those vectors
        update<<<N, threads>>>(d_X, d_Y, d_Z, d_W1, d_W2, d_b1, d_b2, d_h);
        cudaDeviceSynchronize();
        
        // Loss
        /*calc_loss<<<N, threads>>>(d_Z, d_Y, d_W1, d_W2, d_loss + t);
        cudaDeviceSynchronize();
        checkCUDAError("memory copy in calc_loss");
        */
    }
}

/* thread structure: N, m*n
    Calculates the loss for the test image
    as supplied by the user
   */
__global__ void calc_loss2(double *Z, double *Y, double *W1, double *W2, double *W3, double *W4, double *loss){
    int mx = threadIdx.x;
    int nx = threadIdx.y;
    int m = blockDim.x, n = blockDim.y;
    int Nx = blockIdx.x;
    double d = 0, lambda = 0.01;

    // Only one thread should equate the loss to zero
    if(Nx + mx + nx == 0)   
        loss = 0;
    __syncthreads();

    if(mx == 0){ 
        d = Z[Nx*n + nx] - Y[Nx*n + nx]; 
        d = d * d;
        printf("calc_loss Z: %d    %lf\n", Nx*n+nx, Z[Nx*n+nx]);
    }

    if(Nx == 0){ 
        d += lambda * (W1[m*nx + mx] * W1[m*nx + mx] + W2[n*mx + nx] * W2[n*mx + nx]); 
        d += lambda * (W3[m*nx + mx] * W3[m*nx + mx] + W4[n*mx + nx] * W4[n*mx + nx]); 
        printf("Index: W1: %d, W2: %d\n", m*nx + mx, m*nx + mx);
    }

    // ATOMIC OPERATION REQUIRED HERE
    if(d != 0){ 
        *loss += d;
    }
    __syncthreads();
    
    if(Nx + mx + nx == 0)
        printf("calc_loss loss: %lf\n", *loss);
}

/* Training the full sda after the pre training
   */
void train(int N, int m, int n, double *d_X, double *d_Y, double *d_Z, double *d_W1, double *d_W2, double *d_b1, double *d_b2, double *d_h, double *d_loss, double *d_W3, double *d_b3, double *d_hh, double *d_W4, double *d_b4, double *d_hhh){
    int T = 10;
    dim3 threads(m, n), threads_T(n, m);

    for(int t = 0; t < T; t++){

        /* calculate first layer */
        next_layer<<<N, threads>>>(d_X, d_h, d_W1, d_b1); 
        cudaDeviceSynchronize();

        /* calculate second layer */
        next_layer<<<N, threads_T>>>(d_h, d_hh, d_W2, d_b2);
        cudaDeviceSynchronize();

        /* calculate third layer */
        next_layer<<<N, threads>>>(d_hh, d_hhh, d_W3, d_b3);
        cudaDeviceSynchronize();

        /* calculate fourth layer */
        next_layer<<<N, threads_T>>>(d_hhh, d_Z, d_W4, d_b4);
        cudaDeviceSynchronize();

        // update rule here
        // UPDATE2
    } 
}

/* Function to calculate the result of the test image as 
   supplied by the user in order to test it
   */
void test(int N, int m, int n, double *d_X, double *d_Z, double *d_W1, double *d_W2, double *d_b1, double *d_b2, double *d_h, double *d_loss, double *d_W3, double *d_b3, double *d_hh, double *d_W4, double *d_b4, double *d_hhh){
    dim3 threads(m, n), threads_T(n, m);

    /* calculate first layer */
    next_layer<<<N, threads>>>(d_X, d_h, d_W1, d_b1); 
    cudaDeviceSynchronize();

    /* calculate second layer */
    next_layer<<<N, threads_T>>>(d_h, d_hh, d_W2, d_b2);
    cudaDeviceSynchronize();

    /* calculate third layer */
    next_layer<<<N, threads>>>(d_hh, d_hhh, d_W3, d_b3);
    cudaDeviceSynchronize();

    /* calculate fourth layer */
    next_layer<<<N, threads_T>>>(d_hhh, d_Z, d_W4, d_b4);
    cudaDeviceSynchronize();

    // calculating loss of test
    // Uncomment for checking what your loss is
    // calc_loss2<<<N, threads>>>(d_Z, d_Y, d_W1, d_W2, d_W3, d_W4, d_loss); 
}

int main(){
    int N, m, n, T;
    scanf("%d %d %d", &N, &n, &m);
    T = 10;
    double *X, *Y, *h, *Z, *loss;
    double *test_img;

    // CPU memory utilized here
    X = (double*)malloc(N*n*sizeof(double));
    Y = (double*)malloc(N*n*sizeof(double));
    h = (double*)malloc(N*m*sizeof(double));
    Z = (double*)malloc(N*n*sizeof(double));
    loss = (double*)malloc(T * sizeof(double));
    test_img = (double*)malloc(N*n*sizeof(double));

    // device memory
    // first pre train variables
    double *d_X, *d_Y, *d_W1, *d_W2, *d_b1, *d_b2, *d_h, *d_loss, *d_Z;

    // second pre train variables
    double *d_hY, *d_hZ, *d_W3, *d_W4, *d_b3, *d_b4, *d_hh;

    // third pre train variables
    double *d_hhY, *d_hhZ, *d_hhh;

    // first pre train
    cudaMalloc((void**)&d_W1, sizeof(double) * n * m);
    cudaMalloc((void**)&d_W2, sizeof(double) * m * n);
    cudaMalloc((void**)&d_X, sizeof(double) * N * n);
    cudaMalloc((void**)&d_Y, sizeof(double) * N * n);
    cudaMalloc((void**)&d_Z, sizeof(double) * N * n);
    cudaMalloc((void**)&d_b1, sizeof(double) * m);
    cudaMalloc((void**)&d_b2, sizeof(double) * n);
    cudaMalloc((void**)&d_h, sizeof(double) * N * m);
    cudaMalloc((void**)&d_loss, sizeof(double) * T);

    // second pre train
    cudaMalloc((void**)&d_hY, sizeof(double) * N * m);
    cudaMalloc((void**)&d_hZ, sizeof(double) * N * m);
    cudaMalloc((void**)&d_W3, sizeof(double) * n * m);
    cudaMalloc((void**)&d_b3, sizeof(double) * m);
    cudaMalloc((void**)&d_hh, sizeof(double) * N * n);

    // third pre train
    cudaMalloc((void**)&d_hhY, sizeof(double) * N * n);
    cudaMalloc((void**)&d_hhZ, sizeof(double) * N * n);
    cudaMalloc((void**)&d_W4, sizeof(double) * m * n);
    cudaMalloc((void**)&d_b4, sizeof(double) * n);
    cudaMalloc((void**)&d_hhh, sizeof(double) * N * m);
     
    // Scanning the input curropted image
    for(int i = 0; i < N*n; i++){
        scanf("%lf", &X[i]);
        printf("%lf\n", X[i]);
    }

    // Scanning the original image
    for(int i = 0; i < N*n; i++){
        scanf("%lf", &Y[i]);
        printf("%lf\n", Y[i]);
    }

    // Scanning the test image
    for(int i = 0;i < N*n; i++){
        scanf("%lf", &test_img[i]);
        printf("%lf\n", test_img[i]);
    }

    // Copy X and Y vectors to device memory
    cudaMemcpy(d_X, X, sizeof(double) * N * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, sizeof(double) * N * n, cudaMemcpyHostToDevice);

    // Allocate 2D threads
    dim3 threads_T(n, m), threads(m, n);

    // first pre train
    pre_train(N, m, n, d_X, d_Y, d_Z, d_W1, d_W2, d_b1, d_b2, d_h, d_loss, 0);

    // calculate hY
    next_layer<<<N, threads_T>>>(d_Y, d_hY, d_W1, d_b1);
    // second pre train
    pre_train(N, n, m, d_h, d_hY, d_hZ, d_W2, d_W3, d_b2, d_b3, d_hh, d_loss, 1);

    // calculate hhY
    next_layer<<<N, threads>>>(d_hY, d_hhY, d_W2, d_b2);

    // third pre train
    pre_train(N, m, n, d_hh, d_hhY, d_hhZ, d_W3, d_W4, d_b3, d_b4, d_hhh, d_loss, 2);

    // Copy h back from device memory to CPU memory
    cudaMemcpy(h, d_h, sizeof(double) * N * m, cudaMemcpyDeviceToHost);

    // Copy loss back from device memory to CPU memory
    cudaMemcpy(loss, d_loss, sizeof(double) * T, cudaMemcpyDeviceToHost);

    // Print the h vector
    /*printf("h\n");
    for(int i = 0;i < N*m; i++)
        printf("%lf ", h[i]);*/


    /******* TESTING IMAGE ***********/
    cudaMemcpy(d_X, test_img, sizeof(double) * N * n, cudaMemcpyHostToDevice);
    test(N, m, n, d_X, d_Z, d_W1, d_W2, d_b1, d_b2, d_h, d_loss, d_W3, d_b3, d_hh, d_W4, d_b4, d_hhh); 
    /********************************/

    // Copy Z vecotr back from device memory to CPU memory
    cudaMemcpy(Z, d_Z, sizeof(double) * N * n, cudaMemcpyDeviceToHost);

    // Print the Z vector
    printf("\nZ\n");
    for(int i = 0;i < N*n; i++)
        printf("%0.12lf ", Z[i]);
    printf("\n");

    // Free device memory
    cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_b1); cudaFree(d_b2);
    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_h); cudaFree(d_Z);
    cudaFree(d_loss);

    // Free memory of second pre train vectors
    cudaFree(d_hY); cudaFree(d_hZ); cudaFree(d_W3); cudaFree(d_b3);

    // Free memory of third pre train vectors
    cudaFree(d_hh); cudaFree(d_hhY); cudaFree(d_hhZ); cudaFree(d_W4);
    cudaFree(d_b4); cudaFree(d_hhh);

    // Free CPU memory
    free(X); free(Y); free(Z); free(h); free(loss);
    return 0;
}
