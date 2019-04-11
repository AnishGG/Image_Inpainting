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
const double lambda = 1e-4, rho = 0.5, beta = 0.01, epsilon = 1e-10;
const double inv_epsilon = 1e10;
const int T = 20, MAX_BFGS = 10;

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
        assert(sz(u[i]) == sz(v[i]));
        int siz = sz(u[i]); 
        assert(siz == sz(v[i]));
        vd err(siz);
        rep(j, 0, siz){
            err[j] = u[i][j] - v[i][j];
        }
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

void inverse_feature(vd &U, vvvd &W_list, vvd &b_list){
    int cnt = 0;
    for(auto &W: W_list){
        for(auto &row: W){
            for(auto &elem: row){
                elem = U[cnt++];
            } 
        }
    }
    for(auto &b: b_list){
        for(auto &elem: b){
            elem = U[cnt++];
        }
    }
}

double error(vvvd W_list, vvd &b_list, vvd &y, vvd &x, bool sda){
    double loss = 0; 
    vvd *v, h[2];
    v = &x;
    bool cnt = 0;
    for(int i = 0;i < sz(W_list); i++){
        get(h[cnt], W_list[i], *v, b_list[i]);
        v = &h[cnt];
        cnt = 1 - cnt;
        if(sda) h[cnt].clear();
    }
    loss += ms_error(y, *v);
    /*trace("ms_error", loss);
    trace("X h[1-cnt] Y[x]");
    rep(i, 0, sz(x)){
        rep(j, 0, sz(x[i]))
            trace(i, j, x[i][j]);
        rep(j, 0, sz(h[1-cnt][i]))
            trace(i, j, h[1-cnt][i][j]);
        rep(j, 0, sz(h[cnt][i]))
            trace(i, j, h[cnt][i][j]);
    }*/
    if(!sda)
        loss += recon_loss(h[cnt])/*, trace("recon_loss:", loss)*/;
    return loss;
}

double try_error(vvvd W_list, vvd &b_list, vvd &y, vvd &x, bool sda){
    double loss = 0; 
    vvd h_x, y_x;
    get(h_x, W_list[0], x, b_list[0]);
    get(y_x, W_list[1], h_x, b_list[1]);
    loss += ms_error(y, y_x);
    trace("ms_error", loss);
    trace("X h[1-cnt] Y[x]");
    rep(i, 0, sz(x)){
        rep(j, 0, sz(x[i]))
            trace(i, j, x[i][j]);
        rep(j, 0, sz(h_x[i]))
            trace(i, j, h_x[i][j]);
        rep(j, 0, sz(y_x[i]))
            trace(i, j, y_x[i][j]);
    }
    loss += recon_loss(h_x);
    return loss;
}

void gradient(vd &gd, vvvd &W_list, vvd &b_list, vvd &y, vvd &x, double loss, bool sda){
    bool push = 0;
    int cnt = 0;
    if(sz(gd) == 0)
        push = 1;
    for(auto &W:W_list){
        for(auto &row: W){
            for(auto &elem: row){
                double delta = elem * epsilon;
                elem += delta; 
                double new_loss = error(W_list, b_list, y, x, sda);
                elem -= delta;
                double grad = (inv_epsilon / elem) * (new_loss - loss) + 2*elem;    // TO CHANGE TO 2
                if(push)
                    gd.PB(0);
                //trace(cnt, elem, new_loss, loss, inv_epsilon, grad);
                gd[cnt++] = grad;
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
                if(push)
                    gd.PB(0);
                //trace(cnt, elem, new_loss, loss, inv_epsilon, grad);
                gd[cnt++] = grad;
        }
    }
}

void vec_sum(vd &ret, double a, vd &x, double b, vd &y){
    assert(sz(x) == sz(y));
    bool push = 0;
    if(sz(ret) == 0)
        push = 1;
    rep(i, 0, sz(x)){
        if(push)
            ret.PB(0);
        ret[i] = a * x[i] + b * y[i];
    }
}

double dot_product(vd &x, vd &y){
    double ret = 0;
    rep(i, 0, sz(x))
        ret += x[i]*y[i];
    return ret;
}

void point_wise(vd &ret, vd &x, vd &y, double c){
    assert(sz(x) == sz(y));
    rep(i, 0, sz(x)){
        ret.PB(x[i]*y[i]*c);
    }
}

void bfgsMultiply(vvd &S, vvd &Y, vd &d){
    int num_itr = sz(S);
    vd alpha;
    double gamma = 0;
    rep(i, num_itr, 1){
        double alpha_i, rho_i;
        rho_i = 1 / dot_product(S[i], Y[i]);
        alpha_i = rho_i * dot_product(S[i], d); 
        alpha.PB(alpha_i);
        vec_sum(d, 1, d, -alpha_i, Y[i]);
    }
    gamma = dot_product(S[num_itr - 1], Y[num_itr - 1]) / dot_product(Y[num_itr - 1], Y[num_itr - 1]);
    vec_sum(d, -gamma, d, 0, d);
    rep(i, 1, num_itr){
        double beta, rho_i; 
        rho_i = 1 / dot_product(S[i], Y[i]);
        beta = rho_i * dot_product(Y[i], d);
        vec_sum(d, 1, d, (alpha[i] - beta), S[i]);
    }
}

void lbfgs(vvvd &W_list, vvd &b_list, vvd &y, vvd &x, bool sda){
    vvd S, Y;
    vd d, U[2], gd[2];
    double loss = 0, alpha = 0.000001;
    bool converged = 0, use = 0;
    S.PB(vector<double>(0)), Y.PB(vector<double>(0));
    feature(U[use], W_list, b_list);

    loss = error(W_list, b_list, y, x, sda);
    gradient(gd[use], W_list, b_list, y, x, loss, sda);

    for(auto i: gd[use])  d.PB(i);

    for(int itr = 0, idx = -1; itr != T; itr++){
        vec_sum(U[1 - use], 1, U[use], -alpha, d);
        inverse_feature(U[1 - use], W_list, b_list);


        trace("MY ITERATION", itr, loss, frob_norm(W_list), loss + frob_norm(W_list));
        /*rep(i, 0, sz(U[1-use]))
            trace(U[1-use][i], U[use][i], d[i]);*/
        loss = error(W_list, b_list, y, x, sda);

        gradient(gd[1 - use], W_list, b_list, y, x, loss, sda);

        /*rep(i, 0, sz(U[1-use]))
            trace(itr, i, gd[1-use][i]);*/

        if(itr >= MAX_BFGS)
            S.erase(S.begin()), Y.erase(Y.begin());
        else
            idx++;
        S.PB(vector<double>(0)), Y.PB(vector<double>(0));   // As this is n+1th 

        vec_sum(S[idx+1], 1, U[1 - use], -1, U[use]);
        vec_sum(Y[idx+1], 1, gd[1 - use], -1, gd[use]);

         
        bfgsMultiply(S, Y, d); 
        use = 1 - use;
    }
}

int main(){
    cin >> m >> n;
    cin >> N;
    vvd W1, W2, x, y, b_list;
    vd b1, b2;
    input(W1, W2, b1, b2, x, y);

    vvvd W_list;
    W_list.PB(W1), W_list.PB(W2);
    b_list.PB(b1), b_list.PB(b2);
    lbfgs(W_list, b_list, y, x, 0);
    
    return 0;
}
