/*
 * C++ replica of the TOPIQ ErrorTensor + execute_graph framework.
 * Benchmark: TOPIQ prediction vs direct computation for all QoI families.
 *
 * Build: g++ -O3 -o topiq_bench topiq_bench.cpp -lm
 * Usage: topiq_bench <meta_bin> <orig_f32> <dec_f32> <alpha> <n_base> <n_queries> <seed>
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <chrono>
#include <variant>
#include <functional>

static const int H = 1800, W = 3600;
static const int BH = 200, BW = 200;
static const int N_PIX = BH * BW;

// ═════════════════════════════════════════════════════════════════
// ErrorTensorVUVC — faithful port of error_tensor_uv.py
// ═════════════════════════════════════════════════════════════════

struct ET {
    double mu, sig2, bias, var_err, vu, vc, cov_xe;

    static ET from_stats(double mu, double var, double bias, double var_err,
                         double cov_xe) {
        return {mu, var, bias, var_err, var_err, 0.0, cov_xe};
    }

    // ── alpha split ──
    static void split(double ve, double alpha, int n, double &vu, double &vc) {
        if (n <= 1) { vu = ve; vc = 0; return; }
        vc = ((alpha - 1.0) * ve) / (n - 1.0);
        vc = std::max(0.0, std::min(vc, ve));
        vu = ve - vc;
    }

    ET with_alpha_base(double alpha_base, int n_base, int n_block) const {
        double r = (n_base <= 1) ? 0.0 : (alpha_base - 1.0) / (n_base - 1.0);
        r = std::max(0.0, std::min(r, 1.0));
        double alpha_n = 1.0 + (n_block - 1.0) * r;
        double vu2, vc2;
        split(var_err, alpha_n, n_block, vu2, vc2);
        return {mu, sig2, bias, var_err, vu2, vc2, cov_xe};
    }

    // ── unary nonlinear ──
    ET square() const {
        double d = 2.0 * mu, h = 2.0;
        double Ex2 = mu * mu + sig2, sc = 4.0 * Ex2;
        return {mu * mu, d * d * sig2,
                d * bias + 0.5 * h * var_err + h * cov_xe,
                sc * vu + sc * vc, sc * vu, sc * vc, cov_xe * d};
    }

    ET sigmoid() const {
        double s = 1.0 / (1.0 + std::exp(-mu));
        double ds = s * (1.0 - s), dds = ds * (1.0 - 2.0 * s);
        double sc = ds * ds;
        return {s + 0.5 * dds * sig2, sc * sig2,
                ds * bias + 0.5 * dds * var_err,
                sc * vu + sc * vc, sc * vu, sc * vc, ds * cov_xe};
    }

    ET reciprocal() const {
        if (std::fabs(mu) < 1e-9) return {0,0,0,0,0,0,0};
        double inv = 1.0 / mu, d = -inv * inv, h = 2.0 * inv * inv * inv;
        double dd = d * d;
        return {inv, dd * sig2,
                d * bias + 0.5 * h * var_err + h * cov_xe,
                dd * vu + dd * vc, dd * vu, dd * vc, cov_xe * d};
    }

    // ── linear ops ──
    ET operator+(const ET &o) const {
        return {mu+o.mu, sig2+o.sig2, bias+o.bias, var_err+o.var_err,
                vu+o.vu, vc+o.vc, cov_xe+o.cov_xe};
    }
    ET operator-(const ET &o) const {
        return {mu-o.mu, sig2+o.sig2, bias-o.bias, var_err+o.var_err,
                vu+o.vu, vc+o.vc, cov_xe-o.cov_xe};
    }
    ET smul(double k) const {
        double kk = k * k;
        return {mu*k, sig2*kk, bias*k, var_err*kk, vu*kk, vc*kk, cov_xe*k};
    }
    ET operator*(const ET &o) const {  // cross-field multiply
        double a = o.mu, b = mu;
        return {a*b, a*a*sig2+b*b*o.sig2+sig2*o.sig2,
                a*bias+b*o.bias,
                a*a*(vu+vc)+b*b*(o.vu+o.vc), a*a*vu+b*b*o.vu, a*a*vc+b*b*o.vc,
                mu*o.cov_xe+o.mu*cov_xe};
    }
    ET operator/(const ET &o) const { return (*this) * o.reciprocal(); }

    // ── aggregation ──
    ET sum(int n) const {
        double nn = (double)n;
        return {mu*nn, sig2*nn, bias*nn,
                vu*nn + vc*nn*nn, vu*nn, vc*nn*nn, cov_xe*nn};
    }
    ET mean(int n) const {
        ET s = this->sum(n);
        double k = 1.0 / n, kk = k * k;
        return {s.mu*k, s.sig2*kk, s.bias*k, s.var_err*kk,
                s.vu*kk, s.vc*kk, s.cov_xe*k};
    }
};

// constant tensor (no error, no variance)
static inline ET ET_const(double val) { return {val, 0, 0, 0, 0, 0, 0}; }

// ═════════════════════════════════════════════════════════════════
// execute_graph — faithful port of executor.py
// ═════════════════════════════════════════════════════════════════

using Val = std::variant<ET, double, int>;

struct GraphNode {
    std::string op, output;
    std::vector<std::string> args;
};

static ET as_et(const Val &v) {
    if (auto *e = std::get_if<ET>(&v)) return *e;
    if (auto *d = std::get_if<double>(&v)) return ET_const(*d);
    if (auto *i = std::get_if<int>(&v)) return ET_const((double)*i);
    return {};
}
static int as_int(const Val &v) {
    if (auto *i = std::get_if<int>(&v)) return *i;
    if (auto *d = std::get_if<double>(&v)) return (int)*d;
    return 0;
}

static std::unordered_map<std::string, Val>
execute_graph(const std::vector<GraphNode> &graph,
              std::unordered_map<std::string, Val> ctx) {
    for (auto &node : graph) {
        std::vector<Val> vals;
        for (auto &a : node.args) {
            auto it = ctx.find(a);
            vals.push_back(it != ctx.end() ? it->second : Val(0.0));
        }
        Val res;
        ET a = as_et(vals[0]);
        if      (node.op == "add")      res = a + as_et(vals[1]);
        else if (node.op == "sub")      res = a - as_et(vals[1]);
        else if (node.op == "mul")      { auto b = as_et(vals[1]); res = a * b; }
        else if (node.op == "div")      { auto b = as_et(vals[1]); res = a / b; }
        else if (node.op == "square")   res = a.square();
        else if (node.op == "sigmoid")  res = a.sigmoid();
        else if (node.op == "reciprocal") res = a.reciprocal();
        else if (node.op == "sum")      res = a.sum(as_int(vals[1]));
        else if (node.op == "mean")     res = a.mean(as_int(vals[1]));
        else if (node.op == "smul")     res = a.smul(std::get<double>(vals[1]));
        else { fprintf(stderr, "Unknown op: %s\n", node.op.c_str()); exit(1); }
        ctx[node.output] = res;
    }
    return ctx;
}

// ═════════════════════════════════════════════════════════════════
// Meta block + interpolation
// ═════════════════════════════════════════════════════════════════

struct MetaBlock { int start[2], stop[2]; double mu, mu_sq, var_err, cov_xe; };

static bool meta_interp(const std::vector<MetaBlock> &meta,
                         int qr0, int qr1, int qc0, int qc1,
                         double &mu, double &var, double &ve, double &cxe) {
    double tw=0,wm=0,wm2=0,wve=0,wcov=0;
    for (const auto &b : meta) {
        int lo0=std::max(qr0,b.start[0]),hi0=std::min(qr1,b.stop[0]); if(hi0<=lo0)continue;
        int lo1=std::max(qc0,b.start[1]),hi1=std::min(qc1,b.stop[1]); if(hi1<=lo1)continue;
        double a=(double)(hi0-lo0)*(hi1-lo1);
        tw+=a;wm+=a*b.mu;wm2+=a*b.mu_sq;wve+=a*b.var_err;wcov+=a*b.cov_xe;
    }
    if(tw==0)return false;
    mu=wm/tw; var=std::max(0.0,wm2/tw-mu*mu); ve=wve/tw; cxe=wcov/tw;
    return true;
}

static ET make_tensor(const std::vector<MetaBlock> &meta,
                       double alpha_raw, int n_base, int qr, int qc) {
    double mu,var,ve,cxe;
    meta_interp(meta,qr,qr+BH,qc,qc+BW,mu,var,ve,cxe);
    ET t = ET::from_stats(mu,var,0.0,ve,cxe);
    return t.with_alpha_base(alpha_raw, n_base, N_PIX);
}

// ═════════════════════════════════════════════════════════════════
// Direct QoI computation (needs orig+dec)
// ═════════════════════════════════════════════════════════════════

static double direct_mean_square(const float *o, const float *d, int r, int c) {
    double s0=0,s1=0;
    for(int i=0;i<BH;i++) for(int j=0;j<BW;j++){
        int idx=(r+i)*W+(c+j); s0+=(double)o[idx]*o[idx]; s1+=(double)d[idx]*d[idx];
    }
    return (s1-s0)/N_PIX;
}

static double direct_weighted_sum(const float *ox, const float *dx,
                                   const float *ow, const float *dw, int r, int c) {
    double s0=0,s1=0;
    for(int i=0;i<BH;i++) for(int j=0;j<BW;j++){
        int idx=(r+i)*W+(c+j); s0+=(double)ox[idx]*ow[idx]; s1+=(double)dx[idx]*dw[idx];
    }
    return s1-s0;
}

static double direct_cloudy(const float *ofc, const float *dfc,
                             const float *ofl, const float *dfl,
                             const float *oct, const float *dct,
                             int r, int c, double k, double tau, double eps) {
    double n0=0,n1=0,d0=0,d1=0;
    for(int i=0;i<BH;i++) for(int j=0;j<BW;j++){
        int idx=(r+i)*W+(c+j);
        double m0=1.0/(1.0+exp(-k*(oct[idx]-tau)));
        double m1=1.0/(1.0+exp(-k*(dct[idx]-tau)));
        n0+=m0*(ofc[idx]-ofl[idx]); n1+=m1*(dfc[idx]-dfl[idx]);
        d0+=m0; d1+=m1;
    }
    return n1/(d1+eps)-n0/(d0+eps);
}

// ═════════════════════════════════════════════════════════════════
// Timer
// ═════════════════════════════════════════════════════════════════

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void start(){ t0=std::chrono::high_resolution_clock::now(); }
    double us(){ return std::chrono::duration<double,std::micro>(
                   std::chrono::high_resolution_clock::now()-t0).count(); }
};

// ═════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════

int main(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr, "Usage: %s <meta_bin> <orig> <dec> <alpha> <n_base> <nq> <seed>\n", argv[0]);
        return 1;
    }
    double alpha_raw=atof(argv[4]); int n_base=atoi(argv[5]);
    int NQ=atoi(argv[6]); int seed=atoi(argv[7]);

    // Load meta
    FILE *f=fopen(argv[1],"rb");
    int nm; fread(&nm,4,1,f);
    std::vector<MetaBlock> meta(nm); fread(meta.data(),sizeof(MetaBlock),nm,f); fclose(f);

    // Load field
    std::vector<float> orig(H*W), dec(H*W);
    f=fopen(argv[2],"rb"); fread(orig.data(),4,H*W,f); fclose(f);
    f=fopen(argv[3],"rb"); fread(dec.data(),4,H*W,f); fclose(f);

    // Define QoI graphs (matching Python's GRAPH definitions)
    // QoI 1: mean(x²) = square -> mean
    std::vector<GraphNode> graph_ms = {
        {"square", "x_sq", {"input"}},
        {"mean", "out", {"x_sq", "N"}},
    };
    // QoI 2: weighted_sum = mul -> sum
    std::vector<GraphNode> graph_ws = {
        {"mul", "xw", {"x", "w"}},
        {"sum", "out", {"xw", "N"}},
    };
    // QoI 4: cloudy ratio
    std::vector<GraphNode> graph_cloudy = {
        {"sub", "X", {"flutc", "flut"}},
        {"sub", "d", {"cldtot", "tau"}},
        {"smul", "kd", {"d", "k"}},
        {"sigmoid", "m", {"kd"}},
        {"mul", "mx", {"m", "X"}},
        {"sum", "num", {"mx", "N"}},
        {"sum", "den", {"m", "N"}},
        {"add", "den_eps", {"den", "eps"}},
        {"div", "out", {"num", "den_eps"}},
    };

    // Random queries
    std::mt19937 qrng(seed);
    std::uniform_int_distribution<int> dr(0,H-BH-1), dc(0,W-BW-1);
    std::vector<int> qr(NQ),qc(NQ);
    for(int i=0;i<NQ;i++){qr[i]=dr(qrng);qc[i]=dc(qrng);}

    Timer tm;
    volatile double sink=0;

    printf("=== C++ Benchmark: %d queries, %dx%d block ===\n\n", NQ, BH, BW);
    printf("%-20s %12s %12s %8s\n", "QoI", "Direct(us)", "TOPIQ(us)", "Speedup");
    printf("%-20s %12s %12s %8s\n", "---", "----------", "---------", "-------");

    // ── QoI 1: mean(x²) ──
    tm.start();
    for(int i=0;i<NQ;i++) sink+=direct_mean_square(orig.data(),dec.data(),qr[i],qc[i]);
    double d1=tm.us()/NQ;
    tm.start();
    for(int i=0;i<NQ;i++){
        ET t=make_tensor(meta,alpha_raw,n_base,qr[i],qc[i]);
        std::unordered_map<std::string,Val> ctx={{"input",t},{"N",(int)N_PIX}};
        auto res=execute_graph(graph_ms,ctx);
        ET out=std::get<ET>(res["out"]); sink+=out.bias+out.var_err;
    }
    double t1=tm.us()/NQ;
    printf("%-20s %12.2f %12.2f %7.0fx\n","mean(x²)",d1,t1,d1/t1);

    // ── QoI 2: weighted_sum ──
    tm.start();
    for(int i=0;i<NQ;i++) sink+=direct_weighted_sum(orig.data(),dec.data(),orig.data(),dec.data(),qr[i],qc[i]);
    double d3=tm.us()/NQ;
    tm.start();
    for(int i=0;i<NQ;i++){
        ET t=make_tensor(meta,alpha_raw,n_base,qr[i],qc[i]);
        std::unordered_map<std::string,Val> ctx={{"x",t},{"w",t},{"N",(int)N_PIX}};
        auto res=execute_graph(graph_ws,ctx);
        ET out=std::get<ET>(res["out"]); sink+=out.bias+out.var_err;
    }
    double t3=tm.us()/NQ;
    printf("%-20s %12.2f %12.2f %7.0fx\n","weighted_sum",d3,t3,d3/t3);

    // ── QoI 4: cloudy_ratio ──
    double kv=20.0, tau_v=0.5, eps_v=1e-6;
    tm.start();
    for(int i=0;i<NQ;i++) sink+=direct_cloudy(orig.data(),dec.data(),orig.data(),dec.data(),
                                               orig.data(),dec.data(),qr[i],qc[i],kv,tau_v,eps_v);
    double d4=tm.us()/NQ;
    tm.start();
    for(int i=0;i<NQ;i++){
        ET t=make_tensor(meta,alpha_raw,n_base,qr[i],qc[i]);
        std::unordered_map<std::string,Val> ctx={
            {"flutc",t},{"flut",t},{"cldtot",t},
            {"tau",ET_const(tau_v)},{"k",tau_v},  // k is double for smul
            {"eps",ET_const(eps_v)},{"N",(int)N_PIX}};
        // smul needs double as second arg
        ctx["k"] = Val(kv);
        auto res=execute_graph(graph_cloudy,ctx);
        ET out=std::get<ET>(res["out"]); sink+=out.bias+out.var_err;
    }
    double t4=tm.us()/NQ;
    printf("%-20s %12.2f %12.2f %7.0fx\n","cloudy_ratio",d4,t4,d4/t4);

    printf("\n(sink=%.6e)\n",sink);
    return 0;
}
