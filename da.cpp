#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
#define PB push_back
#define MP make_pair
typedef long long int ll;
using namespace std;
using namespace __gnu_pbds;

#define rep(i, begin, end) for (__typeof(end) i = (begin) - ((begin) > (end)); i != (end) - ((begin) > (end)); i += 1 - 2 * ((begin) > (end)))
#define BITCOUNT(n) __builtin_popcount(n)
#define BITCOUNTLL(n) __builtin_popcountll(n)
#define sz(a) (int)(a).size()
#define pii pair<int, int>
#define pll pair<ll, ll>
#define x first
#define y second

#define TRACE
#ifdef TRACE
#define trace(...) __f(#__VA_ARGS__, __VA_ARGS__)
template <typename Arg1>
void __f(const char* name, Arg1&& arg1){
	cerr << name << " : " << arg1 << std::endl;
}
template <typename Arg1, typename... Args>
void __f(const char* names, Arg1&& arg1, Args&&... args){
	const char* comma = strchr(names + 1, ',');cerr.write(names, comma - names) << " : " << arg1<<" | ";__f(comma+1, args...);
}
#else
#define trace(...)
#endif

const int N = 1e3;
const double lambda = 1e-4, rho = 0.05, beta = 0.01;

double sigmoid(double x){
    double e = exp((double) x);
    double ret = e / (1 + e);
    return ret;
}

void get(vector<double> &res, vector<vector<double> > &W, vector<double> &x, vector<double> &b){
    int num_r = W.size(), num_c = W[0].size();
    assert(num_c == x.size());
    for(int row = 0;row < num_r; row++){
        res[row] = b[row];
        for(int col = 0;col < num_c; col++){
            res[row] += W[row][col]*x[col];
        }
        res[row] = sigmoid(res[row]);
    }
}

double norm(vector<double> &x){
    double ret = 0;
    for(auto i: x){
        ret += i*i;
    }
    return ret;
}

double frob_norm(vector<vector<double> > &W){
    double ret = 0;
    for(auto i: W){
        ret += norm(i);
    }
    return ret;
}

double ms_error(vector<double> &u, vector<double> &v){
    int siz = sz(u); 
    assert(siz == sz(v));
    vector<double> err(siz);
    rep(i, 0, siz)
        err[i] = u[i] - v[i];
    double ret = norm(err) / 2;
    return ret;
}

double recon_loss(vector<vector<double> > &h){
    int N = sz(h), M = sz(h[0]);
    vector<double> rho_cap(M, 0);
    for(auto x: h){
        rep(j, 0, M){
            rho_cap[j] += x[j];
        }
    }
    rep(j, 0, M)
        rho_cap[j] /= (double)N;

    double ret = 0;
    for(auto x: rho_cap){
        ret += (rho * log(rho/x) + (1 - rho) * log((1 - rho) / (1 - x))); 
    }
    return ret;
}

double fin_error(vector<vector<double> > &y, vector<vector<double> > &y_x, vector<vector<vector<double> > > &W_list){
    assert(sz(y) == sz(y_x));
    double N = sz(y);
    double ms_err = 0;
    rep(i, 0, sz(y)){
        ms_err += ms_error(y[i], y_x[i]);
    }
    ms_err /= N;
    double frob_err = 0;
    for(auto W: W_list){
        frob_err += frob_norm(W); 
    }
    frob_err *= (lambda / 2);
    return ms_err + frob_err;
}

int main(){
    int m = 4, n = 2;
    vector<vector<double>> W1(m), W2(n);
    vector<double> x, y, h_x(m), y_x(n), b1, b2;
    srand(time(NULL));
    cout << "W1\n" << endl;
    rep(i, 0, m){
        rep(j, 0, n){
            W1[i].PB(rand() % 5);
            cout << W1[i][j] << ", ";
        }
        cout << ";" << endl;
    }
    cout << "W2\n";
    rep(i, 0, n){
        rep(j, 0, m){
            W2[i].PB(rand() % 5);
            cout << W2[i][j] << ", ";
        }
        cout << ";" << endl;
    }
    cout << "x\n";
    rep(i, 0, n){
        x.PB((double) rand() / (RAND_MAX));
        cout << x[i] << endl;
    }
    cout << "b1\n";
    rep(i, 0, m){
        b1.PB(rand() % 5);
        cout << b1[i] << endl;
    }
    cout << "b2\n";
    rep(i, 0, n){
        b2.PB(rand() % 5);
        cout << b2[i] << endl;
    }
    get(h_x, W1, x, b1);
    get(y_x, W2, h_x, b2);
    cout << "h_x\n";
    rep(i, 0, m){
        cout << std::setprecision(20) << h_x[i] << endl;
    }
    cout << "y_x\n";
    rep(i, 0, n){
        cout << std::setprecision(20) << y_x[i] << endl;
    }
    trace(lambda);
    return 0;
}
