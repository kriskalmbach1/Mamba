#pragma once


struct MambaParams {
    //dimension
    int d_model;
    int n_layers;
    int inputs;
    //1
    int outputs;
    //16
    int d_state;
    //2
    int expand_factor;
    //4
    int d_conv;
    //0.001, 0.1
    float dt_min, dt_max;
    //expand_factor * D
    int d_inner;
    int dt_rank;
};


struct NNLinear {
    int inputs, outputs;
    //size: inputs X outputs
    float* WG, * BG;
    float* W;
    //size: outputs
    float* B;
};
struct NNConv1D {
    int ichan, ochan;
    int kernel_size;
    int padding;
    //ichan X 1 X kernel_size
    float* WG, * BG;
    float* W;

    //inchan
    float* B;
};

struct MambaBlock {
    //D X 2 * d_inner 
    //bias = false
    NNLinear in_proj;

    //d_inner X d_inner
    //kernel size: d_conv
    //weight size: [d_inner X 1, X d_conv]
    //bias size: d_inner
    NNConv1D conv;

    //d_inner X dt_rank + 2 * d_state
    //bias = false
    NNLinear x_proj;

    //dt_rank X d_inner
    //bias = true
    NNLinear dt_proj;

    //d_inner X D
    //bias = true
    NNLinear out_proj;

    //d_inner X d_state
    float* A;
    //d_inner
    float* D;

    float* AG, * DG;

};
struct ResidualBlock {
    MambaBlock m;
    //RmsNorm
};
struct Mamba {
    MambaParams p;

    //inputs X hidden
    NNLinear linput;

    ResidualBlock* blocks;
    //hidden X output
    NNLinear loutput;



    int b, l, d;

};
/*
for multi-threaded operations:

*/
struct MambaBatch {
    Mamba** m;
    MambaParams* p;
    int n;
};


/*
Adam-based gradient update for Mamba
*/

struct AdamItem {
    float* wm, * wv;
    float* bm, * bv;

    int wn, bn;

};


struct MambaAdam {
    float lr, b1, b2, b1_t, b2_t;
    AdamItem* in, * out;

    AdamItem* lin[4];

    AdamItem* conv;
    AdamItem* A;
    AdamItem* D;

    int n, t;
};

int createMambaBatch(MambaParams* p, MambaBatch** mb, int n);
void freeMambaBatch(MambaBatch* mb);

/*
Allocate memory associated with mamba structure
can call freeMamba() or simply free() to deallocate memory
*/

int createMamba(MambaParams* p, Mamba** m);

/*
Train mamba 

*/
int trainMamba(Mamba* m, const float* X, const float* Y, int N, int D, int iterations, float lr, float* buf_rmse, float* buf_acc);
/*
Inference
*/
int forwardMamba(Mamba* m, const float* X, int N, int D, float* out);

/*
Multi-threaded
*/
int trainMambaBatch(MambaBatch* mba, const float* X, const float* Y, int mcnt, int N, int D, int iterations, float lr, int threads);
void setMambaWeights(Mamba* m, const float* IW, const float* IB, const float* IPW, const float* CW, const float* CB, const float* XW, const float* DTW, const float* DTB, const float* OPW, const float* OW, const float* OB);


void initMambaParams(MambaParams* p, int D, int hidden);