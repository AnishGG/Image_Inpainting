#include <bits/stdc++.h>
#define PB push_back
#define MP make_pair
#define vvvd vector<vector<vector<double> > >
#define vvd vector<vector<double> >
#define vd vector<double>
typedef long long int ll;
using namespace std;

#define rep(i, begin, end) for (__typeof(end) i = (begin) - ((begin) > (end)); i != (end) - ((begin) > (end)); i += 1 - 2 * ((begin) > (end)))
#define sz(a) (int)(a).size()
#define pii pair<int, int>
#define pll pair<ll, ll>
//#define x first
//#define y second

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

int N, m, n;
const double lambda = 1e-4, rho = 0.05, beta = 0.01, epsilon = 1e-10;
const double inv_epsilon = 1e10;

double sigmoid(double x){
    double e = exp((double) x);
    double ret = e / (1 + e);
    return ret;
}

void get(vvd &res, vvd &W, vvd &x, vector<double> &b){
    int num_r = W.size(), num_c = W[0].size();
    assert(num_c == sz(x[0]));
    rep(i, 0, N){
        res.PB(vector<double>(0));
        for(int row = 0;row < num_r; row++){
            res[i].PB(b[row]);
            for(int col = 0;col < num_c; col++){
                res[i][row] += W[row][col]*x[i][col];
            }
            res[i][row] = sigmoid(res[i][row]);
        }
    }
}

double norm(vector<double> &x){
    double ret = 0;
    for(auto i: x){
        ret += i*i;
    }
    return ret;
}

double frob_norm(vvvd &W_list){
    double ret = 0;
    for(auto W: W_list){
        for(auto i: W){
            ret += norm(i);
        }
    }
    ret *= (lambda / 2);
    return ret;
}

double ms_error(vvd &u, vvd &v){
    assert(sz(u) == sz(v));
    double ret = 0;
    rep(i, 0, sz(u)){
        int siz = sz(u[i]); 
        assert(siz == sz(v[i]));
        vd err(siz);
        rep(j, 0, siz)
            err[j] = u[i][j] - v[i][j];
        ret += norm(err) / 2;
    }
    ret /= (double)N;
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
    rep(j, 0, M){
        rho_cap[j] /= (double)N;
    }

    double ret = 0;
    for(auto x: rho_cap){
        ret += (rho * log(rho/x) + (1 - rho) * log((1 - rho) / (1 - x))); 
    }
    return ret;
}

/*
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
*/
void input(vector<vector<double> > &W1, vector<vector<double> > &W2, vector<double> &b1, vector<double> &b2, vector<vector<double> > &x, vector<vector<double> > &y){
    double temp;
    rep(i, 0, N){
        x.PB(vector<double>(0));
        rep(j, 0, n){
            cin >> temp;
            x[i].PB(temp);
        }
    }
    rep(i, 0, N){
        y.PB(vector<double>(0));
        rep(j, 0, n){
            cin >> temp;
            y[i].PB(temp);
        }
    }
    rep(i, 0, m){
        W1.PB(vector<double>(0));
        rep(j, 0, n){
            cin >> temp;
            W1[i].PB(temp);
        }
    }
    rep(i, 0, n){
        W2.PB(vector<double>(0));
        rep(j, 0, m){
            cin >> temp;
            W2[i].PB(temp);
        }
    }
    rep(i, 0, m){
        cin >> temp;
        b1.PB(temp);
    }
    rep(i, 0, n){
        cin >> temp;
        b2.PB(temp);
    }
}

void feature(vd &U, vvvd &W_list, vvd &b_list){
    U.clear();
    for(auto W: W_list){
        for(auto row: W){
            U.insert(U.end(), row.begin(), row.end()); 
        }
    }
    for(auto b: b_list){
        U.insert(U.end(), b.begin(), b.end());
    }
}

double error(vvvd W_list, vvd &b_list, vvd &y, vvd &x, bool sda){
    double loss = 0; 
    vvd *v, h[2];
    v = &x;
    int cnt = 0;
    for(int i = 0;i < sz(W_list); i++){
        get(h[cnt], W_list[i], *v, b_list[i]);
        v = &h[cnt];
        cnt = (cnt + 1) % 2;
        if(sda) h[cnt].clear();
    }
    loss += ms_error(y, *v);
    if(!sda)
        loss += recon_loss(h[(cnt+1)%2]);
    return loss;
}

void gradient(vd &gd, vvvd &W_list, vvd &b_list, vvd &y, vvd &x, double loss, bool sda){
    for(auto &W:W_list){
        for(auto &row: W){
            for(auto &elem: row){
                double delta = elem * epsilon;
                elem += delta; 
                double new_loss = error(W_list, b_list, y, x, sda);
                elem -= delta;
                double grad = (inv_epsilon / elem) * (new_loss - loss) + 2*elem;
                gd.PB(grad); 
            }
        }
    }
    for(auto &row: b_list){
        for(auto &elem: row){
            double delta = elem * epsilon;
                elem += delta; 
                double new_loss = error(W_list, b_list, y, x, sda);
                elem -= delta;
                double grad = (inv_epsilon / elem) * (new_loss - loss);
                gd.PB(grad); 
        }
    }
}

int main(){
    cin >> m >> n;
    cin >> N;
    vvd W1, W2, x, y, h_x, y_x, b_list;
    vd b1, b2, U, gd;
    input(W1, W2, b1, b2, x, y);

    get(h_x, W1, x, b1);
    get(y_x, W2, h_x, b2);
    vvvd W_list;
    W_list.PB(W1), W_list.PB(W2);
    b_list.PB(b1), b_list.PB(b2);
    feature(U, W_list, b_list);

    // Printing 
    trace(ms_error(y, y_x));
    trace(frob_norm(W_list));
    trace(recon_loss(h_x));
    trace("U");
    for(auto i: U)
        trace(i);
    trace(sz(U));
    gradient(gd, W_list, b_list, y, x, error(W_list, b_list, y, x, 0), 0);
    trace("gradient");
    for(auto i: gd)
        trace(i);
    return 0;
}
