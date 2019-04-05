#include <iostream>
#include <vector>
#define PB push_back
#define MP make_pair
#define vvd vector<vector<double> >
#define vd vector<double>
typedef long long int ll;
using namespace std;

#define rep(i, begin, end) for (__typeof(end) i = (begin) - ((begin) > (end)); i != (end) - ((begin) > (end)); i += 1 - 2 * ((begin) > (end)))
#define sz(a) (int)(a).size()
#define pii pair<int, int>
#define pll pair<ll, ll>

// Device version of pre-training
// Can use thrust library
// OR use a flattened array here
//__global__ void pretrain_fwd(vvd *W1, vvd *b1, vvd *W2, vvd *b2, vvd *X, vvd *Z, vd *rho, vd *err){

__global__ void pretrain_fwd(double *W1){
    int bid = blockIdx.x, tid = threadIdx.x;
    int idx = tid + bid*blockDim.x;
}

int main(){
    vvd W1, W2, x, y, h_x, y_x;
    vd b1, b2;
    //int H = sz(W1), W = sz(W1[0]);
    int H = 2, W = 3;
    double *dW1, *db1, *dW2, *db2, *dX, *dZ, *drho, *derr;
    cudaMalloc(&dW1, H * W * sizeof(double));
    double *dst = dW1;
    for(vvd::iterator it = W1.begin(); it != W1.end(); it++){
        double *src = &((*it)[0]);
        int sz = it->size();
        cudaMemcpy(dst, src, sizeof(double)*sz, cudaMemcpyHostToDevice);
        dst += sz;
    }
    pretrain_fwd<<<1, 1>>>(dW1);


}
