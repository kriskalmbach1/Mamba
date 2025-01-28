

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>

/*
OS-specific windows libraries
*/

#include <windows.h>



#include "Mamba.h"


#define apply_adam(w, g, m, v, b1, b2, b1_t, b2_t, lr)  \
        apply_adam2(&(w), g, &(m), &(v), b1, b2, b1_t, b2_t, lr)


void initMambaParams(MambaParams* p, int D, int hidden)
{
    p->d_model = hidden;
    p->n_layers = 2;
    p->inputs = D;
    p->outputs = 1;
    //p->hidden = hidden;
    p->d_state = 16;
    p->expand_factor = 2;
    p->d_conv = 4;
    p->dt_min = 0.001;
    p->dt_max = 0.1;
    p->d_inner = p->expand_factor * p->d_model;
    p->dt_rank = ceil(p->d_model / 16);
}

void apply_adam2(float* w, float g, float* m, float* v, float b1, float b2, float b1_t, float b2_t, float lr)
{
    *m = b1 * *m + (1 - b1) * g;
    *v = b2 * *v + (1 - b2) * g * g;
    float mhat = *m / (1 - b1_t);
    float vhat = *v / (1 - b2_t);
    float wold = *w;
    *w -= (lr * mhat) / (sqrt(vhat) + 1e-6);

    if (!isfinite(*w)) {
        printf("apply_adam2: !isfinite");
        exit(0);
    }
}


size_t getLinearWeightSize(int inputs, int outputs, char bias)
{
    size_t sz = inputs * outputs * sizeof(float) * 2;
    if (bias) sz += outputs * sizeof(float) * 2;
    return sz;
}
char* setLinearMemory(NNLinear* l, int inputs, int outputs, char bias, char* mem)
{
    l->inputs = inputs;
    l->outputs = outputs;

    l->W = (float*)mem; mem += outputs * inputs * sizeof(float);
    l->WG = (float*)mem; mem += outputs * inputs * sizeof(float);
    if (bias) {
        l->B = (float*)mem; mem += outputs * sizeof(float);
        l->BG = (float*)mem; mem += outputs * sizeof(float);
    }
    else {
        l->B = l->BG = 0;
    }

    return mem;

}
void randomLinearWeight(NNLinear* l)
{
    for (int i = 0; i < l->outputs; ++i) {
        if (l->B) l->B[i] = 0;
        for (int j = 0; j < l->inputs; ++j) {
            l->W[i * l->inputs + j] = (float)rand() / (float)RAND_MAX - 0.5;
        }
    }
}

size_t getConv1DWeightSize(int inputs, int outputs, int kernel_size, char bias)
{
    size_t sz = inputs * 1 * kernel_size * sizeof(float) * 2;
    if (bias) sz += outputs * sizeof(float) * 2;
    return sz;
}
char* setConv1DMemory(NNConv1D* c, int ichan, int ochan, int kernel_size, int padding, char bias, char* mem)
{
    c->ichan = ichan;
    c->ochan = ochan;
    c->kernel_size = kernel_size;
    c->padding = padding;


    c->W = (float*)mem; mem += ichan * 1 * kernel_size * sizeof(float);
    c->WG = (float*)mem; mem += ichan * 1 * kernel_size * sizeof(float);
    if (bias) {
        c->B = (float*)mem; mem += ochan * sizeof(float);
        c->BG = (float*)mem; mem += ochan * sizeof(float);
    }
    else {
        c->B = c->BG = 0;
    }


    return mem;
}
void randomConv1DWeight(NNConv1D* c)
{
    for (int i = 0; i < c->ichan; ++i) {
        if (c->B) c->B[i] = 0;
        for (int j = 0; j < c->kernel_size; ++j) {
            c->W[i * c->kernel_size + j] = (float)rand() / (float)RAND_MAX - 0.5;
        }
    }

}
size_t getMambaMatrixSize(MambaParams* p)
{
    return (p->d_inner * p->d_state + p->d_inner) * sizeof(float) * 2;
}
int getMambaBlockSize(MambaParams* p, char include_struct)
{
    size_t sz = (include_struct ? sizeof(MambaBlock) : 0);
    sz += getLinearWeightSize(p->d_model, 2 * p->d_inner, 0);                         //in_proj
    sz += getConv1DWeightSize(p->d_inner, p->d_inner, p->d_conv, 1);        //conv
    sz += getLinearWeightSize(p->d_inner, 2 * p->d_state + p->dt_rank, 0);               //x_proj
    sz += getLinearWeightSize(p->dt_rank, p->d_inner, 1);                   //dt_proj
    sz += getLinearWeightSize(p->d_inner, p->d_model, 0);                         //out_proj
    sz += getMambaMatrixSize(p);
    return sz;
}
size_t getResidualBlockSize(MambaParams* p, char include_struct)
{
    size_t sz = (include_struct ? sizeof(ResidualBlock) : 0);
    sz += getMambaBlockSize(p, 0);
    return sz;
}

size_t calcDimLinear(Mamba* m, NNLinear* l, int N, int D, int* buf_N, int* buf_D)
{
    int outN = 0, outD = 0;
    if (D != l->inputs) {
        return 0;
    }

    if (buf_N) *buf_N = N;
    if (buf_D) *buf_D = l->outputs;
    return N * l->outputs;
}


Mamba* allocMamba(MambaParams* p)
{
    int rbsz = getResidualBlockSize(p, 1) * p->n_layers;
    int linsz = getLinearWeightSize(p->inputs, p->d_model, 1) +                        //linput
        getLinearWeightSize(p->d_model, p->outputs, 1);
    size_t sz = sizeof(Mamba) + rbsz + linsz;


    Mamba* ret = (Mamba*)calloc(1, sz);
    ret->blocks = (ResidualBlock*)(((char*)ret) + sizeof(Mamba));
    char* mem = (char*)(ret->blocks + p->n_layers);


    char* imem = mem;
    mem = setLinearMemory(&ret->loutput, p->d_model, p->outputs, 1, mem);
    mem = setLinearMemory(&ret->linput, p->inputs, p->d_model, 1, mem);
    char* imem2 = mem;

    char* rmem = 0, * rmem2 = 0;
    int rsz = 0;
    for (int i = 0; i < p->n_layers; ++i) {

        rmem = mem;
        rsz = getResidualBlockSize(p, 1);
        ResidualBlock* b = &ret->blocks[i];
        MambaBlock* mb = &b->m;

        mem = setLinearMemory(&mb->in_proj, p->d_model, 2 * p->d_inner, 0, mem);
        mem = setConv1DMemory(&mb->conv, p->d_inner, p->d_inner, p->d_conv, p->d_conv - 1, 1, mem);
        mem = setLinearMemory(&mb->x_proj, p->d_inner, 2 * p->d_state + p->dt_rank, 0, mem);
        mem = setLinearMemory(&mb->dt_proj, p->dt_rank, p->d_inner, 1, mem);
        mem = setLinearMemory(&mb->out_proj, p->d_inner, p->d_model, 0, mem);


        mb->A = (float*)mem; mem += sizeof(float) * p->d_inner * p->d_state;
        mb->AG = (float*)mem; mem += sizeof(float) * p->d_inner * p->d_state;
        mb->D = (float*)mem; mem += sizeof(float) * p->d_inner;
        mb->DG = (float*)mem; mem += sizeof(float) * p->d_inner;


        for (int j = 0; j < p->d_inner; ++j) {
            for (int k = 0; k < p->d_state; ++k) {
                mb->A[j * p->d_state + k] = log(k + 1);
            }
            mb->D[j] = 1;
        }

        randomLinearWeight(&mb->in_proj);
        randomLinearWeight(&mb->x_proj);
        randomLinearWeight(&mb->dt_proj);
        randomLinearWeight(&mb->out_proj);
        randomConv1DWeight(&mb->conv);

        rmem2 = mem;
    }

    randomLinearWeight(&ret->loutput);
    randomLinearWeight(&ret->linput);

    if (mem != ((char*)ret) + sz) {
        printf("sanity check failure");
    }
    return ret;
}
int createMamba(MambaParams* p, Mamba** m)
{
    Mamba* ret = allocMamba(p);
    ret->p = *p;
    *m = ret;
    return 1;
}
void freeMambaBatch(MambaBatch* mb)
{
    for (int i = 0; i < mb->n; ++i) {
        free(mb->m[i]);
    }
    free(mb);
}
int createMambaBatch(MambaParams* p, MambaBatch** mb, int n)
{
    MambaBatch* ret = (MambaBatch*)calloc(1, sizeof(MambaBatch) + sizeof(Mamba*) * n);
    ret->m = (Mamba**)(((char*)ret) + sizeof(MambaBatch));

    for (int i = 0; i < n; ++i) {
        if (!createMamba(p, &ret->m[i])) {
            freeMambaBatch(ret);
            return 0;
        }
        ++ret->n;
    }
    ret->p = &ret->m[0]->p;

    *mb = ret;
    return 1;
}

void setMambaWeights(Mamba* m, const float* IW, const float* IB, const float* IPW, const float* CW, const float* CB, const float* XW, const float* DTW, const float* DTB, const float* OPW, const float* OW, const float* OB)
{
    MambaParams* p = &m->p;
    const int n_layers = p->n_layers;
    const int d_inner = p->d_inner;
    const int d_state = p->d_state;
    const int d_model = p->d_model;
    const int d_conv = p->d_conv;
    const int dt_rank = p->dt_rank;
    const int inputs = p->inputs;


    for (int i = 0; i < d_model; ++i) {
        for (int k = 0; k < inputs; ++k) {
            if (IW) m->linput.W[i * inputs + k] = IW[i * inputs + k];
        }
        if (IB) m->linput.B[i] = IB[i];
    }


    for (int i = 0; i < n_layers; ++i) {
        MambaBlock* mb = &m->blocks[i].m;

        for (int j = 0; j < d_inner * 2; ++j) {
            for (int k = 0; k < d_model; ++k) {
                if (IPW) mb->in_proj.W[j * d_model + k] = IPW[j * d_model + k];
            }
        }

        for (int j = 0; j < d_inner; ++j) {
            for (int k = 0; k < d_conv; ++k) {
                if (CW) mb->conv.W[j * d_conv + k] = CW[j * d_conv + k];
            }
            if (CB) mb->conv.B[j] = CB[j];
        }

        for (int j = 0; j < dt_rank + d_state * 2; ++j) {
            for (int k = 0; k < d_inner; ++k) {
                if (XW) mb->x_proj.W[j * d_inner + k] = XW[j * d_inner + k];
            }
        }

        for (int j = 0; j < d_inner; ++j) {
            for (int k = 0; k < dt_rank; ++k) {
                if (DTW) mb->dt_proj.W[j * dt_rank + k] = DTW[j * dt_rank + k];
            }
            if (DTB) mb->dt_proj.B[j] = DTB[j];
        }

        for (int j = 0; j < d_model; ++j) {
            for (int k = 0; k < d_inner; ++k) {
                if (OPW) mb->out_proj.W[j * d_inner + k] = OPW[j * d_inner + k];
            }
        }
    }

    for (int i = 0; i < 1; ++i) {
        for (int k = 0; k < d_model; ++k) {
            if (OW) m->loutput.W[i * d_model + k] = OW[i * d_model + k];
        }
        if (OB) m->loutput.B[i] = OB[i];
    }

}

int applyRBlockSGD(Mamba* m, ResidualBlock* rb, float lr)
{
    for (int i = 0; i < m->p.d_inner; ++i) {
        for (int j = 0; j < m->p.d_state; ++j) {
            rb->m.A[i * m->p.d_state + j] -= rb->m.AG[i * m->p.d_state + j] * lr;

        }
        rb->m.D[i] -= rb->m.DG[i] * lr;
    }
    return 1;
}

int applyRBlockAdam(Mamba* m, ResidualBlock* rb, MambaAdam* ma, AdamItem* aA, AdamItem* aD)
{
    for (int i = 0; i < m->p.d_inner; ++i) {
        for (int j = 0; j < m->p.d_state; ++j) {
            apply_adam(rb->m.A[i * m->p.d_state + j], rb->m.AG[i * m->p.d_state + j], aA->wm[i * m->p.d_state + j], aA->wv[i * m->p.d_state + j], ma->b1, ma->b2, ma->b1_t, ma->b2_t, ma->lr);
        }

        apply_adam(rb->m.D[i], rb->m.DG[i], aD->wm[i], aD->wv[i], ma->b1, ma->b2, ma->b1_t, ma->b2_t, ma->lr);
    }
    return 1;
}
int runLinearFixed(NNLinear* l, const float* X, int N, int D, float* O)
{

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < l->outputs; ++j) {
            O[i * l->outputs + j] = (l->B ? l->B[j] : 0);
            for (int k = 0; k < l->inputs; ++k) {
                O[i * l->outputs + j] += X[i * D + k] * l->W[j * l->inputs + k];
            }
        }
    }
    return 1;
}
int runLinear(Mamba* m, NNLinear* l, const float* X, int N, int D, float** buffer, int* bcap, int* buf_N, int* buf_D)
{
    size_t cnt = calcDimLinear(m, l, N, D, 0, 0);
    if (!cnt) return 0;

    if (*bcap < cnt) {
        *bcap = cnt;
        *buffer = (float*)realloc(*buffer, *bcap * sizeof(float));
    }

    runLinearFixed(l, X, N, D, *buffer);
    if (buf_N) *buf_N = N;
    if (buf_D) *buf_D = l->outputs;


    return 1;

}
int backLinearFixed(NNLinear* l, const float* X, int N, const float* DO, float* D)
{
    if (D) memset(D, 0, sizeof(float) * l->inputs * N);
    memset(l->WG, 0, sizeof(float) * l->inputs * l->outputs);
    if (l->BG) memset(l->BG, 0, sizeof(float) * l->outputs);


    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < l->outputs; ++j) {

            if (l->BG) l->BG[j] += DO[i * l->outputs + j];

            for (int k = 0; k < l->inputs; ++k) {
                l->WG[j * l->inputs + k] += X[i * l->inputs + k] * DO[i * l->outputs + j];

                if (D) {
                    D[i * l->inputs + k] += DO[i * l->outputs + j] * l->W[j * l->inputs + k];
                }

            }
        }
    }
    return 1;
}
int backLinear(Mamba* m, NNLinear* l, const float* X, int N, const float* DO, float** D, int* dcap)
{

    if (*dcap < N * l->inputs) {
        *dcap = N * l->inputs;
        *D = (float*)realloc(*D, *dcap * sizeof(float));
    }
    backLinearFixed(l, X, N, DO, (D ? *D : 0));

    return 1;
}
int applyLinearSGD(NNLinear* l, float lr)
{
    for (int i = 0; i < l->outputs; ++i) {
        if (l->B) l->B[i] -= l->BG[i] * lr;
        for (int j = 0; j < l->inputs; ++j) {
            l->W[i * l->inputs + j] -= l->WG[i * l->inputs + j] * lr;
        }
    }
    return 1;
}
int applyLinearAdam(NNLinear* l, MambaAdam* ma, AdamItem* ai)
{
    for (int i = 0; i < l->outputs; ++i) {
        if (l->B) apply_adam(l->B[i], l->BG[i], ai->bm[i], ai->bv[i], ma->b1, ma->b2, ma->b1_t, ma->b2_t, ma->lr);

        for (int j = 0; j < l->inputs; ++j) {

            apply_adam(l->W[i * l->inputs + j], l->WG[i * l->inputs + j], ai->wm[i * l->inputs + j], ai->wv[i * l->inputs + j], ma->b1, ma->b2, ma->b1_t, ma->b2_t, ma->lr);
        }
    }
    return 1;
}
int backConv1DFixed(NNConv1D* c, const float* X, int N, int D, const float* DO, float* DI, char row_wise)
{
    if (DI) memset(DI, 0, N * D * sizeof(float));
    memset(c->WG, 0, c->ichan * 1 * c->kernel_size * sizeof(float));
    if (c->BG) memset(c->BG, 0, c->ochan * sizeof(float));

    if (row_wise) {
        if (c->ichan != D) {
            printf("runConv1D: size mismatch: inchan=%d  D=%d\n", c->ichan, D);
            return 0;
        }

        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < N; ++j) {

                float* WG = c->WG + i * c->kernel_size;
                const float* W = c->W + i * c->kernel_size;



                int sidx = 0;
                int xidx = j - c->padding;
                int cnt = c->kernel_size;
                if (j < c->padding) {
                    sidx += c->padding - j;
                    cnt -= c->padding - j;
                    xidx = 0;
                }

                if (c->BG) c->BG[i] += DO[j * D + i];

                for (int k = 0; k < cnt; ++k) {

                    if (DI) {
                        DI[(xidx + k) * D + i] += W[k + sidx] * DO[j * D + i];
                    }

                    WG[k + sidx] += X[(xidx + k) * D + i] * DO[j * D + i];
                }
            }
        }

    }
    else {
        if (c->ichan != N) {
            printf("runConv1D: size mismatch: inchan=%d  D=%d\n", c->ichan, D);
            return 0;
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < D; ++j) {

                float* WG = c->WG + i * c->kernel_size;
                const float* W = c->W + i * c->kernel_size;

                int sidx = 0;
                int xidx = j - c->padding;
                int cnt = c->kernel_size;
                if (j < c->padding) {
                    sidx += c->padding - j;
                    cnt -= c->padding - j;
                    xidx = 0;
                }

                if (c->BG) c->BG[i] += DO[j * D + i];

                for (int k = 0; k < cnt; ++k) {
                    if (DI) DI[i * D + (xidx + k)] += W[k + sidx] * DO[i * D + j];
                    WG[k + sidx] += X[i * D + (xidx + k)] * DO[i * D + j];
                }
            }
        }

    }


}
int backConv1D(NNConv1D* c, const float* X, int N, int D, const float* DO, float** Dbuf, int* dcap, char row_wise)
{
    if (*dcap < N * D) {
        *dcap = N * D;
        *Dbuf = (float*)realloc(*Dbuf, *dcap * sizeof(float));
    }

    return backConv1DFixed(c, X, N, D, DO, (Dbuf ? *Dbuf : 0), row_wise);

}
int applyConv1DSGD(NNConv1D* c, float lr)
{
    for (int i = 0; i < c->ichan; ++i) {
        if (c->B) c->B[i] -= c->BG[i] * lr;

        for (int j = 0; j < c->kernel_size; ++j) {
            c->W[i * c->kernel_size + j] -= c->WG[i * c->kernel_size + j] * lr;
        }
    }
    return 1;
}
int applyConv1DAdam(NNConv1D* c, MambaAdam* ma, AdamItem* ai)
{
    for (int i = 0; i < c->ichan; ++i) {
        if (c->B) apply_adam(c->B[i], c->BG[i], ai->bm[i], ai->bv[i], ma->b1, ma->b2, ma->b1_t, ma->b2_t, ma->lr);

        for (int j = 0; j < c->kernel_size; ++j) {
            apply_adam(c->W[i * c->kernel_size + j], c->WG[i * c->kernel_size + j], ai->wm[i * c->kernel_size + j], ai->wv[i * c->kernel_size + j], ma->b1, ma->b2, ma->b1_t, ma->b2_t, ma->lr);
        }
    }
    return 1;
}
int runConv1DFixed(NNConv1D* c, const float* X, int N, int D, float* O, char row_wise)
{
    if (row_wise) {
        if (c->ichan != D) {
            printf("runConv1D: size mismatch: inchan=%d  D=%d\n", c->ichan, D);
            return 0;
        }

        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < N; ++j) {

                const float* W = c->W + i * c->kernel_size;


                int sidx = 0;
                int xidx = j - c->padding;
                int cnt = c->kernel_size;
                if (j < c->padding) {
                    sidx += c->padding - j;
                    cnt -= c->padding - j;
                    xidx = 0;
                }

                double o = (c->B ? c->B[i] : 0);

                for (int k = 0; k < cnt; ++k) {
                    o += W[k + sidx] * X[(xidx + k) * D + i];
                }

                O[j * D + i] = o;
            }
        }

    }
    else {
        if (c->ichan != N) {
            printf("runConv1D: size mismatch: inchan=%d  D=%d\n", c->ichan, D);
            return 0;
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < D; ++j) {

                const float* W = c->W + i * c->kernel_size;


                int sidx = 0;
                int xidx = j - c->padding;
                int cnt = c->kernel_size;
                if (j < c->padding) {
                    sidx += c->padding - j;
                    cnt -= c->padding - j;
                    xidx = 0;
                }

                double o = (c->B ? c->B[i] : 0);

                for (int k = 0; k < cnt; ++k) {
                    o += W[k + sidx] * X[i * D + (xidx + k)];

                }

                O[i * D + j] = o;

            }
        }

    }
}
int runConv1D(Mamba* m, NNConv1D* c, const float* X, int N, int D, float** buf, int* bcap, int* buf_N, int* buf_D, char row_wise)
{
    if (*bcap < N * D) {
        *bcap = N * D * sizeof(float);
        *buf = (float*)realloc(*buf, *bcap * sizeof(float));
    }

    runConv1DFixed(c, X, N, D, *buf, row_wise);

    if (row_wise) {
        if (buf_N) *buf_N = N;
        if (buf_D) *buf_D = D;
    }
    else {
        if (buf_N) *buf_N = N;
        if (buf_D) *buf_D = D;
    }
    return 1;
}

int rmsNorm(Mamba* m, float* X, int N, int D)
{

    for (int i = 0; i < N; ++i) {
        double rms = 0;
        for (int k = 0; k < D; ++k) {
            rms += X[i * D + k] * X[i * D + k];
        }
        rms = sqrt(rms / (double)D);
        for (int k = 0; k < D; ++k) {
            X[i * D + k] /= rms;
        }
    }

    return 1;
}
int backRmsNorm(Mamba* m, const float* X, float* DO, int N, int D)
{


    float* dtmp = (float*)_malloca(D * sizeof(float));

    for (int i = 0; i < N; ++i) {

        double di = 1.0 / (double)D;
        double dsum = 0;
        double rms = 0, rms2 = 0;
        for (int k = 0; k < D; ++k) {

            const float x = X[i * D + k];
            dtmp[k] = di * x * 2;

            rms += x * x;
        }

        rms = sqrt(rms / (double)D);
        rms2 = rms * rms;

        for (int k = 0; k < D; ++k) {
            const float x = X[i * D + k];

            dtmp[k] *= 1.0 / (2.0 * rms);
            dsum += dtmp[k];
        }

        for (int k = 0; k < D; ++k) {
            const float x = X[i * D + k];

            double d = (rms - x * dtmp[k]) / rms2;
            //tensorflow: (rms - x * dsum)/rms2

            DO[i * D + k] *= d;
        }

    }
    _freea(dtmp);

    return 1;
}

void SiLU(float* X, int N, int D)
{
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < D; ++k) {
            X[i * D + k] = X[i * D + k] * 1.0 / (1.0 + exp(-X[i * D + k]));
        }
    }
}
void SiLUCopy(const float* X, float* O, int N, int D)
{
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < D; ++k) {
            O[i * D + k] = X[i * D + k] * 1.0 / (1.0 + exp(-X[i * D + k]));
        }
    }
}
void SiLUBackward(const float* X, float* dx, int N, int D)
{
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < D; ++k) {
            const float x = X[i * D + k];
            const float sig = 1.0 / (1 + exp(-x));

            dx[i * D + k] *= sig * (1 + x * (1 - sig));
        }
    }
}
void SoftPlus(float* X, int N, int D)
{
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < D; ++k) {
            X[i * D + k] = log(1 + exp(X[i * D + k]));
        }
    }
}
void SoftPlusBackward(const float* O, float* dx, int N, int D)
{
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < D; ++k) {
            //tmp = exp(X[i * D + k])

            const double tmp = exp(O[i * D + k]) - 1;
            const double invexp = 1.0 / tmp;
            dx[i * D + k] *= 1.0 / (1 + invexp);
        }
    }
}
void transpose(const float* A, int N, int D, float** B, int* Nb, int* Db, int* cap)
{
    //allocate a slice of memory to store columns


    if (*cap < N * D * sizeof(float)) {
        *cap = N * D * sizeof(float);
        *B = (float*)realloc(*B, *cap * sizeof(float));
    }


    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < D; ++k) {
            (*B)[k * N + i] = A[i * D + k];
        }
    }

    *Nb = D;
    *Db = N;

}
void splitFixed(float* A, float* B, int N, int D, int colcnt)
{
    for (int i = 0; i < N; i++) {
        for (int k = colcnt; k < D; k++) {
            B[i * colcnt + (k - colcnt)] = A[i * D + k];

            if (i > 0) {
                A[i * colcnt + (k - colcnt)] = A[i * D + (k - colcnt)];
            }

        }
    }
}
void split(int cnt, float** a, float** b, int* n_a, int* d_a, int* n_b, int* d_b, int* cb)
{
    ///split A column-wise into A and B

    int colcnt = *d_a / cnt;
    size_t cnt2 = colcnt * *n_a;

    if (cnt2 > *cb) {
        *cb = cnt2;
        *b = (float*)realloc(*b, *cb * sizeof(float));
    }

    splitFixed(*a, *b, *n_a, *d_a, colcnt);


    if (n_b) *n_b = *n_a;
    if (d_b) *d_b = colcnt;
    if (d_a) *d_a = colcnt;
}
void splitInPlace(float* A, int N, int D)
{
    int colcnt = D / 2;
    int size = 0;
    for (int i = 0; i < N - 1; i++) {
        for (int k = 0; k < colcnt; k++) {

            int right_idx = i * D + (k + colcnt);
            int left_idx = (i + 1) * D + k;

            float tmp = A[left_idx];
            A[left_idx] = A[right_idx];
            A[right_idx] = tmp;
            ++size;
        }

        for (int k = 0; k < colcnt; ++k) {
            int left_idx = (i + 1) * D + k;
            int right_idx = (i + 1) * D + (k + colcnt);

            float tmp = A[left_idx];
            A[left_idx] = A[right_idx];
            A[right_idx] = tmp;

        }
        //switch rows
    }
    int sidx = colcnt;
    while (1) {
        float tmp = A[sidx];

    }

}
void multiply(float* A, float* B, int N, int D)
{
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < D; ++k) {
            A[i * D + k] *= B[i * D + k];
        }
    }
}
void add(float* A, float* B, int N, int D)
{
    for (int i = 0; i < N * D; ++i) {
        A[i] += B[i];
    }
}


int splitDeltaFixed(const float* deltaBC, float* delta, float* B, float* C, int N, int D, int D1, int D2, int D3)
{
    for (int i = 0; i < N; ++i) {
        int colidx = 0;
        for (int k = 0; k < D1; ++k) {
            delta[i * D1 + k] = deltaBC[i * D + (k + colidx)];
        }
        colidx += D1;
        for (int k = 0; k < D2; ++k) {
            B[i * D2 + k] = deltaBC[i * D + (k + colidx)];
        }

        colidx += D2;
        for (int k = 0; k < D3; ++k) {
            C[i * D3 + k] = deltaBC[i * D + (k + colidx)];
        }
    }
    return 1;
}
int splitDelta(const float* deltaBC, float** delta, float** B, float** C, int N, int D, int Dd, int Db, int Dc,
    int* dcap, int* bcap, int* ccap)
{

    if (D != Dd + Db + Dc) {
        printf("size mismatch in split3\n");
        return 0;
    }

    if (*dcap < N * Dd) {
        *dcap = N * Dd;
        *delta = (float*)realloc(*delta, *dcap * sizeof(float));
    }
    if (*bcap < N * Db) {
        *bcap = N * Db;
        *B = (float*)realloc(*B, *bcap * sizeof(float));
    }
    if (*ccap < N * Dc) {
        *ccap = N * Dc;
        *C = (float*)realloc(*C, *ccap * sizeof(float));
    }


    splitDeltaFixed(deltaBC, *delta, *B, *C, N, D, Dd, Db, Dc);
    return 1;
}


void selectiveScanLowMem(float* X, const float* delta, const float* A, const float* B, const float* C, const float* D,
    float* buffer, int N, int d_inner, int d_state)
{
    //ED = d_inner = 32
    //S = d_state = 16
    //N = sample size

    //X: N X d_inner
    //delta: N X d_inner
    //A: d_inner X d_state 
    //B: N X d_state
    //C: N X d_state
    //D: d_inner
    //buffer: d_inner X d_state
    //Y: N X d_inner

    //BX, deltaA, deltaB: N X d_inner X d_state

    //deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, N, d_inner, d_state)
    // 
    //build a new A matrix that's multiplied by the columns of delta
    //
    //deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, N, d_inner, d_state)
    //BX = deltaB * (x.unsqueeze(-1)) # (B, N, d_inner, d_state)
    memset(buffer, 0, d_inner * d_state * sizeof(float));

    for (int i = 0; i < N; ++i) {
        float* b = buffer;


        for (int j = 0; j < d_inner; ++j) {
            const float Xi = X[i * d_inner + j];
            X[i * d_inner + j] = 0;

            for (int k = 0; k < d_state; ++k) {
                //deltaA
                //b1[j * d_state + k] = A[j * d_state + k] * delta[i * d_inner + j];
                //deltaB
                //b1[j * d_state + k] = B[i * d_state + k] * delta[i * d_inner + j];
                //BX
                //b1[j * d_state + k] = B[i * d_state + k] * delta[i * d_inner + j] * X[i * d_inner + j];

                //h = deltaA[:, t] * h + BX[:, t]
                const float dA = exp(A[j * d_state + k] * delta[i * d_inner + j]);
                const float bX = B[i * d_state + k] * delta[i * d_inner + j] * Xi;

                b[j * d_state + k] = dA * b[j * d_state + k] + bX;

                X[i * d_inner + j] += b[j * d_state + k] * C[i * d_state + k];
            }

            X[i * d_inner + j] += D[j] * Xi;
        }
    }
}

void selectiveScan(const float* X, const float* delta, const float* A, const float* B, const float* C, const float* D,
    float* buffer, float* Y_buf, int N, int d_inner, int d_state)
{
    //ED = d_inner = 32
    //S = d_state = 16
    //N = sample size

    //X: N X d_inner
    //delta: N X d_inner
    //A: d_inner X d_state 
    //B: N X d_state
    //C: N X d_state
    //D: d_inner
    //buffer: d_inner X d_state
    //Y: N X d_inner

    //BX, deltaA, deltaB: N X d_inner X d_state

    //deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, N, d_inner, d_state)
    // 
    //build a new A matrix that's multiplied by the columns of delta
    //
    //deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, N, d_inner, d_state)
    //BX = deltaB * (x.unsqueeze(-1)) # (B, N, d_inner, d_state)

    memset(Y_buf, 0, N * d_inner * sizeof(float));

    for (int i = 0; i < N; ++i) {
        float* b = buffer + i * d_inner * d_state;
        float* y = Y_buf;



        for (int j = 0; j < d_inner; ++j) {

            for (int k = 0; k < d_state; ++k) {
                //deltaA
                //b1[j * d_state + k] = exp(A[j * d_state + k] * delta[i * d_inner + j]);
                //deltaB
                //b1[j * d_state + k] = B[i * d_state + k] * delta[i * d_inner + j];
                //BX
                //b1[j * d_state + k] = B[i * d_state + k] * delta[i * d_inner + j] * X[i * d_inner + j];

                //h = deltaA[:, t] * h + BX[:, t]
                const float dA = exp(A[j * d_state + k] * delta[i * d_inner + j]);
                const float bX = B[i * d_state + k] * delta[i * d_inner + j] * X[i * d_inner + j];

                if (i == 0) {
                    b[j * d_state + k] = bX;
                }
                else {
                    const float* b0 = buffer + (i - 1) * d_inner * d_state;
                    b[j * d_state + k] = dA * b0[j * d_state + k] + bX;
                }

                y[i * d_inner + j] += b[j * d_state + k] * C[i * d_state + k];
            }

            y[i * d_inner + j] += D[j] * X[i * d_inner + j];
        }
    }
}
void selectiveScanBackward(const float* X, const float* delta, const float* A, const float* B, const float* C, const float* D, float* dy,
    float* buffer, float* buf_dx, float* buf_ddelta, float* buf_da, float* buf_db, float* buf_dc, float* buf_dd, int N, int d_inner, int d_state)
{

    //X: N X d_inner
    //delta: N X d_inner
    //A: d_inner X d_state 
    //B: N X d_state
    //C: N X d_state
    //D: d_inner
    //buffer: d_inner X d_state
    //Y: N X d_inner

    memset(buf_da, 0, d_inner * d_state * sizeof(float));
    memset(buf_dd, 0, d_inner * sizeof(float));
    //memset(buf_dx, 0, d_inner * N * sizeof(float));
    memset(buf_db, 0, N * d_state * sizeof(float));
    memset(buf_dc, 0, N * d_state * sizeof(float));
    memset(buf_ddelta, 0, N * d_inner * sizeof(float));


    for (int i = N - 1; i >= 0; --i) {
        float* b = buffer + i * d_inner * d_state;

        for (int j = 0; j < d_inner; ++j) {

            //dy/dd = X[0] + X[1] ...
            buf_dd[j] += X[i * d_inner + j] * dy[i * d_inner + j];

            //dy/dx
            buf_dx[i * d_inner + j] = D[j] * dy[i * d_inner + j];

            for (int k = 0; k < d_state; ++k) {


                const float bX = B[i * d_state + k] * delta[i * d_inner + j] * X[i * d_inner + j];


                buf_dc[i * d_state + k] += b[j * d_state + k] * dy[i * d_inner + j];

                //dy/dbuf
                b[j * d_state + k] = C[i * d_state + k] * dy[i * d_inner + j];


                if (i < N - 1) {
                    const float dA_prev = exp(A[j * d_state + k] * delta[(i + 1) * d_inner + j]);
                    const float* bNext = buffer + (i + 1) * d_inner * d_state;
                    b[j * d_state + k] += dA_prev * bNext[j * d_state + k];
                }



                //dy/db = dy/dbuf * dbuf/db
                buf_db[i * d_state + k] += delta[i * d_inner + j] * X[i * d_inner + j] * b[j * d_state + k];

                //dy/ddelta = dy/dbuf * dbuf/ddelta
                buf_ddelta[i * d_inner + j] += X[i * d_inner + j] * B[i * d_state + k] * b[j * d_state + k];
                if (i > 0) {
                    const float* b0 = buffer + (i - 1) * d_inner * d_state;
                    const float dA = exp(A[j * d_state + k] * delta[i * d_inner + j]);
                    buf_ddelta[i * d_inner + j] += dA * A[j * d_state + k] * b0[j * d_state + k] * b[j * d_state + k];

                    buf_da[j * d_state + k] += dA * delta[i * d_inner + j] * b0[j * d_state + k] * b[j * d_state + k];
                }

                buf_dx[i * d_inner + j] += B[i * d_state + k] * delta[i * d_inner + j] * b[j * d_state + k];



            }



        }

    }
}


int createMambaAdam(MambaAdam** adam, Mamba* m)
{
    MambaParams* p = &m->p;

    MambaAdam* ret = 0;
    size_t sz = sizeof(MambaAdam) +
        sizeof(AdamItem) * 2 + //in, out
        sizeof(AdamItem) * (p->n_layers * 3) + //conv, A, D
        sizeof(AdamItem) * p->n_layers * 4;



    sz += m->linput.outputs * m->linput.inputs * sizeof(float) * 2;
    sz += m->loutput.outputs * m->loutput.inputs * sizeof(float) * 2;

    if (m->linput.B) sz += m->linput.outputs * sizeof(float) * 2;
    if (m->loutput.B) sz += m->loutput.outputs * sizeof(float) * 2;

    for (int i = 0; i < p->n_layers; ++i) {
        ResidualBlock* rb = &m->blocks[i];
        MambaBlock* mb = &rb->m;

        NNLinear* l[4] = { &mb->in_proj, &mb->dt_proj, &mb->x_proj, &mb->out_proj };

        for (int k = 0; k < 4; ++k) {
            sz += l[k]->outputs * l[k]->inputs * sizeof(float) * 2;
            if (l[k]->B) sz += l[k]->outputs * sizeof(float) * 2;
        }

        sz += mb->conv.ichan * mb->conv.kernel_size * sizeof(float) * 2;
        if (mb->conv.B) sz += mb->conv.ichan * sizeof(float) * 2;

        sz += p->d_inner * p->d_state * sizeof(float) * 2;
        sz += p->d_inner * sizeof(float) * 2;
    }


    ret = (MambaAdam*)calloc(1, sz);
    ret->in = (AdamItem*)(((char*)ret) + sizeof(MambaAdam));
    ret->out = ret->in + 1;
    ret->lin[0] = ret->out + 1;
    ret->lin[1] = ret->lin[0] + p->n_layers;
    ret->lin[2] = ret->lin[1] + p->n_layers;
    ret->lin[3] = ret->lin[2] + p->n_layers;
    ret->conv = ret->lin[3] + p->n_layers;
    ret->A = ret->conv + p->n_layers;
    ret->D = ret->A + p->n_layers;

    float* fmem = (float*)(ret->D + p->n_layers);

    ret->in->wm = fmem; fmem += m->linput.outputs * m->linput.inputs;
    ret->in->wv = fmem; fmem += m->linput.outputs * m->linput.inputs;
    ret->out->wm = fmem; fmem += m->loutput.outputs * m->loutput.inputs;
    ret->out->wv = fmem; fmem += m->loutput.outputs * m->loutput.inputs;



    if (m->linput.B) {
        ret->in->bm = fmem; fmem += m->linput.outputs;
        ret->in->bv = fmem; fmem += m->linput.outputs;
    }

    if (m->loutput.B) {
        ret->out->bm = fmem; fmem += m->loutput.outputs;
        ret->out->bv = fmem; fmem += m->loutput.outputs;
    }


    for (int i = 0; i < p->n_layers; ++i) {
        ResidualBlock* rb = &m->blocks[i];
        MambaBlock* mb = &rb->m;

        NNLinear* l[4] = { &mb->in_proj, &mb->dt_proj, &mb->x_proj, &mb->out_proj };

        for (int k = 0; k < 4; ++k) {
            ret->lin[k][i].wm = fmem; fmem += l[k]->outputs * l[k]->inputs;
            ret->lin[k][i].wv = fmem; fmem += l[k]->outputs * l[k]->inputs;
            ret->lin[k][i].wn = l[k]->outputs * l[k]->inputs;

            if (l[k]->B) {
                ret->lin[k][i].bm = fmem; fmem += l[k]->outputs;
                ret->lin[k][i].bv = fmem; fmem += l[k]->outputs;
                ret->lin[k][i].bn = l[k]->outputs;
            }
        }

        ret->conv[i].wm = fmem; fmem += mb->conv.ichan * mb->conv.kernel_size;
        ret->conv[i].wv = fmem; fmem += mb->conv.ichan * mb->conv.kernel_size;
        ret->conv[i].wn = mb->conv.ichan * mb->conv.kernel_size;

        if (mb->conv.B) {
            ret->conv[i].bm = fmem; fmem += mb->conv.ichan;
            ret->conv[i].bv = fmem; fmem += mb->conv.ichan;
            ret->conv[i].bn = mb->conv.ichan;
        }

        ret->A[i].wm = fmem; fmem += p->d_inner * p->d_state;
        ret->A[i].wv = fmem; fmem += p->d_inner * p->d_state;
        ret->A[i].wn = p->d_inner * p->d_state;

        ret->D[i].wm = fmem; fmem += p->d_inner;
        ret->D[i].wv = fmem; fmem += p->d_inner;
        ret->D[i].wn = p->d_inner;
    }

    if ((char*)fmem != ((char*)ret) + sz) {
        printf("createMambaAdam: sanity check failure\n");
    }

    ret->lr = 0.001;
    ret->b1 = ret->b1_t = 0.9;
    ret->b2 = ret->b2_t = 0.999;
    *adam = ret;

    return 1;

}

void applyGradientsSGD(Mamba* m, float lr)
{
    applyLinearSGD(&m->linput, lr);
    for (int i = 0; i < m->p.n_layers; ++i) {
        ResidualBlock* rb = &m->blocks[i];

        applyLinearSGD(&rb->m.in_proj, lr);
        applyLinearSGD(&rb->m.dt_proj, lr);
        applyLinearSGD(&rb->m.x_proj, lr);
        applyLinearSGD(&rb->m.out_proj, lr);
        applyConv1DSGD(&rb->m.conv, lr);
        applyRBlockSGD(m, rb, lr);


    }
    applyLinearSGD(&m->loutput, lr);

}


void applyGradientsAdam(Mamba* m, MambaAdam* ma)
{
    applyLinearAdam(&m->linput, ma, ma->in);
    for (int i = 0; i < m->p.n_layers; ++i) {
        ResidualBlock* rb = &m->blocks[i];

        applyLinearAdam(&rb->m.in_proj, ma, &ma->lin[0][i]);
        applyLinearAdam(&rb->m.dt_proj, ma, &ma->lin[1][i]);
        applyLinearAdam(&rb->m.x_proj, ma, &ma->lin[2][i]);
        applyLinearAdam(&rb->m.out_proj, ma, &ma->lin[3][i]);
        applyConv1DAdam(&rb->m.conv, ma, &ma->conv[i]);
        applyRBlockAdam(m, rb, ma, &ma->A[i], &ma->D[i]);

    }
    applyLinearAdam(&m->loutput, ma, ma->out);

    ma->b1_t *= ma->b1;
    ma->b2_t *= ma->b2;

    //printf("C: b1_t=%.4g  b2_t=%.4g\n", ma->b1_t, ma->b2_t);

}

int forwardMamba(Mamba* m, const float* X, int N, int D, float* out)
{
    const int d_inner = m->p.d_inner;
    const int d_state = m->p.d_state;
    const int dt_rank = m->p.dt_rank;

    size_t sz = N * ((d_inner * 2) + (d_state * 2 + dt_rank) + d_inner * 2 + d_state * 3 + dt_rank) * sizeof(float);
    sz += d_inner * d_state * sizeof(float);
    sz += d_inner * d_state * N * sizeof(float);

    float* oi2 = (float*)malloc(sz);
    float* oi_0 = oi2 + N * d_inner * 2;
    float* oi_1 = oi_0 + N * d_inner;
    float* olarge = oi_1 + N * d_inner;
    float* os_0 = olarge + N * (d_state * 2 + dt_rank);
    float* os_1 = os_0 + N * d_state;
    float* os_2 = os_1 + N * d_state;
    float* or = os_2 + N * d_state;
    float* ois = or +N * dt_rank;
    float* buf = ois + d_inner * d_state;


    if ((char*)(buf + N * d_inner * d_state) != ((char*)oi2) + sz) {
        printf("forwardMamba: sanity check failure\n");
    }

    float* x, * o, * delta, * A, * B, * C, * y, * z;

    runLinearFixed(&m->linput, X, N, D, os_0);

    for (int i = 0; i < m->p.n_layers; ++i) {
        ResidualBlock* bl = &m->blocks[i];
        MambaBlock* mb = &bl->m;

        for (int j = 0; j < N * d_state; ++j) {
            os_1[j] = os_0[j];
        }

        rmsNorm(m, os_1, N, d_state);


        runLinearFixed(&mb->in_proj, os_1, N, d_state, oi2);

        splitFixed(oi2, oi_0, N, 2 * d_inner, d_inner);

        //handle x
        runConv1DFixed(&mb->conv, oi2, N, d_inner, oi_1, 1);

        SiLU(oi_1, N, d_inner);

        for (int j = 0; j < m->p.d_inner; ++j) {
            for (int k = 0; k < m->p.d_state; ++k) {
                ois[j * m->p.d_state + k] = -exp(mb->A[j * m->p.d_state + k]);
            }
        }

        runLinearFixed(&mb->x_proj, oi_1, N, d_inner, olarge);

        splitDeltaFixed(olarge, or , os_1, os_2, N, dt_rank + d_state * 2, dt_rank, d_state, d_state);

        runLinearFixed(&mb->dt_proj, or , N, dt_rank, oi2);
        SoftPlus(oi2, N, d_inner);

        selectiveScanLowMem(oi_1, oi2, ois, os_1, os_2, mb->D, buf, N, d_inner, d_state);

        SiLU(oi_0, N, d_inner);

        for (int j = 0; j < N * d_inner; ++j) {
            oi2[j] = oi_0[j] * oi_1[j];
        }

        runLinearFixed(&mb->out_proj, oi2, N, d_inner, os_1);

        for (int j = 0; j < N * d_state; ++j) {
            os_0[j] += os_1[j];
        }

        rmsNorm(m, os_0, N, d_state);

    }

    runLinearFixed(&m->loutput, os_0, N, d_state, out);


    free(oi2);
    return 1;
}


int trainMamba(Mamba* m, const float* X, const float* Y, int N, int D, int iterations, float lr, float* buf_rmse, float* buf_acc)
{

    const MambaParams* p = &m->p;
    MambaAdam* ma = 0;
    const int ocnt = 15;

    createMambaAdam(&ma, m);

    float* b1 = 0, * b2 = 0, * b3 = 0, * b4 = 0;

    int bn1 = 0, bd1 = 0, bn2 = 0, bd2 = 0, bn3 = 0, bd3 = 0, bn4 = 0, bd4 = 0;
    int bc1 = 0, bc2 = 0, bc3 = 0, bc4 = 0;

    int matsz = sizeof(float) * p->d_inner * p->d_state +  //Ai
        sizeof(float) * p->d_inner;                    //Di

    int ysz = N * p->d_inner * sizeof(float);

    float* Ai = (float*)malloc(matsz);
    float* Di = Ai + m->p.d_inner * m->p.d_state;

    float* delta = (float*)malloc(N * sizeof(float));



    size_t ssz = sizeof(float*) * p->n_layers +
        N * p->d_inner * p->d_state * p->n_layers * sizeof(float);

    int merged_size = (p->d_state * 2 + p->dt_rank > p->d_inner * 2 ? p->d_state * 2 + p->dt_rank : p->d_inner * 2);

    size_t dsz = sizeof(float) * (p->d_inner * N * 2 + p->d_state * N * 2 + p->d_inner * p->d_state + p->d_inner + N * merged_size);

    size_t sz = sizeof(float**) * p->n_layers +
        sizeof(float*) * p->n_layers * ocnt +
        sizeof(float) * p->n_layers * N * (p->d_state * 2 + p->d_model * 3 + p->d_inner * 9 + (p->d_state * 2 + p->dt_rank) + p->dt_rank);


    float* out = (float*)malloc(N * p->d_state * sizeof(float));
    float** buf = (float**)malloc(ssz);
    float*** O = (float***)malloc(sz);
    float* dx = (float*)malloc(dsz);
    float* ddelta = dx + p->d_inner * N;
    float* da = ddelta + p->d_inner * N;
    float* db = da + p->d_inner * p->d_state;
    float* dc = db + N * p->d_state;
    float* dd = dc + N * p->d_state;
    float* dmerged = dd + p->d_inner;

    if ((char*)(dmerged + N * merged_size) != ((char*)dx) + dsz) {
        printf("trainMamba: sanity check failure: dx\n");
    }

    char* mem = (char*)(O + p->n_layers);
    float* bmem = (float*)(buf + p->n_layers);

    for (int i = 0; i < p->n_layers; ++i) {
        O[i] = (float**)mem; mem += sizeof(float*) * ocnt;
        O[i][0] = (float*)mem; mem += sizeof(float) * N * p->d_model;
        O[i][1] = (float*)mem; mem += sizeof(float) * N * p->d_model;
        O[i][2] = (float*)mem; mem += sizeof(float) * N * p->d_inner * 2;
        O[i][3] = (float*)mem; mem += sizeof(float) * N * p->d_inner;
        O[i][4] = (float*)mem; mem += sizeof(float) * N * p->d_inner;
        O[i][5] = (float*)mem; mem += sizeof(float) * N * p->d_inner;
        O[i][6] = (float*)mem; mem += sizeof(float) * N * (p->d_state * 2 + p->dt_rank);
        O[i][7] = (float*)mem; mem += sizeof(float) * N * p->dt_rank;
        O[i][8] = (float*)mem; mem += sizeof(float) * N * p->d_state;
        O[i][9] = (float*)mem; mem += sizeof(float) * N * p->d_state;
        O[i][10] = (float*)mem; mem += sizeof(float) * N * p->d_inner;
        O[i][11] = (float*)mem; mem += sizeof(float) * N * p->d_inner;
        O[i][12] = (float*)mem; mem += sizeof(float) * N * p->d_inner;
        O[i][13] = (float*)mem; mem += sizeof(float) * N * p->d_inner;
        O[i][14] = (float*)mem; mem += sizeof(float) * N * p->d_model;

        buf[i] = bmem; bmem += N * p->d_inner * p->d_state;
    }

    if (mem != ((char*)O) + sz) {
        printf("trainMamba sanity check failure\n");
    }
    if ((char*)bmem != ((char*)buf) + ssz) {
        printf("trainMamba sanity check failure: bmem\n");
    }


    for (int it = 0; it < iterations; ++it) {

        runLinear(m, &m->linput, X, N, D, &b1, &bc1, 0, 0);

        for (int i = 0; i < m->p.n_layers; ++i) {
            ResidualBlock* bl = &m->blocks[i];

            //Mamba:
            //INPUT: N X d_state
            //OUTPUT: N X d_state

            //ResidualBlock::Forward

            float** o = O[i];

            //save xr to add to output of MambaBlock::Forward
            for (int k = 0; k < N * p->d_model; ++k) {
                o[0][k] = o[1][k] = b1[k];
            }

            //INPUT 0: N X d_state
            //OUTPUT 1: N X d_state
            rmsNorm(m, o[1], N, p->d_model);

            //MambaForward
            //INPUT: N X d_model
            //OUTPUT: N X d_model


            //INPUT 1: N X d_model
            //OUTPUT 2: N X 2 * p_dinner
            //runLinear(m, &bl->m.in_proj, b1, N, p->d_state, &b2, &bc2, &bn2, &bd2);
            runLinearFixed(&bl->m.in_proj, o[1], N, p->d_model, o[2]);



            //INPUT: N X 2 * d_inner
            //OUTPUT: N X d_inner (2 matricies)
            splitFixed(o[2], o[3], N, 2 * p->d_inner, p->d_inner);
            float* z = o[3];
            float* x = o[2];



            //INPUT: N X d_inner
            //OUTPUT: N X d_inner
            runConv1DFixed(&bl->m.conv, o[2], N, p->d_inner, o[4], 1);

            //SiLU(x, N, p->d_inner);
            SiLUCopy(o[4], o[5], N, p->d_inner);



            //ssm algorithm
            //b1 = delta
            //b2 = delta (temporary)
            //b3 = B
            //b4 = C

            //INPUT: N X d_inner
            //OUTPUT: N X 2 * d_state + dt_rank
            runLinearFixed(&bl->m.x_proj, o[5], N, p->d_inner, o[6]);



            //INPUT: N X 2 * d_state + dt_rank
            //OUTPUT: N X (dt_rank, d_state, d_state)
            //splitDelta(b1, &b2, &b3, &b4, bn1, bd1, p->dt_rank, p->d_state, p->d_state, &bc2, &bc3, &bc4);

            splitDeltaFixed(o[6], o[7], o[8], o[9], N, p->d_state * 2 + p->dt_rank, p->dt_rank, p->d_state, p->d_state);


            //INPUT: N X dt_rank
            //OUTPUT: N X d_inner

            //runLinear(m, &bl->m.dt_proj, b2, N, p->dt_rank, &b1, &bc1, &bn1, &bd1);
            //SoftPlus(b1, bn1, bd1);
            runLinearFixed(&bl->m.dt_proj, o[7], N, p->dt_rank, o[10]);
            SoftPlus(o[10], N, p->d_inner);



            //selective scan
            //INPUT: N X d_inner
            //OUTPUT: N X d_inner

            for (int j = 0; j < m->p.d_inner; ++j) {
                for (int k = 0; k < m->p.d_state; ++k) {
                    Ai[j * m->p.d_state + k] = -exp(m->blocks[0].m.A[j * m->p.d_state + k]);
                }
            }


            //INPUT and SIZES:
            //o[5] (X) - N X d_inner
            //o[10] (delta) - N X dt_rank
            //Ai (A) - d_inner X d_state
            //o[8] (B) - N X d_state
            //o[9] (C) - N X d_state
            //D - d_inner
            //buf - N X d_inner X d_state
            //OUTPUT: N X d_inner

            selectiveScan(o[5], o[10], Ai, o[8], o[9], m->blocks[0].m.D, buf[i], o[11], N, p->d_inner, p->d_state);



            //ending selective scan
            //ending ssm
            
            SiLUCopy(o[3], o[13], N, p->d_inner);



            //multiply(o[11], o[12], N, p->d_inner);
            for (int j = 0; j < N * p->d_inner; ++j) {
                o[12][j] = o[13][j] * o[11][j];
            }

            //INPUT: N X d_inner
            //OUTPUT: N X d_model
            runLinearFixed(&bl->m.out_proj, o[12], N, p->d_inner, o[14]);

            //ending MambaForward

            add(o[14], o[0], N, p->d_model);

            for (int j = 0; j < N * p->d_model; ++j) {
                b1[j] = o[14][j];
            }


            rmsNorm(m, b1, N, p->d_model);



            //ending ResidualBlock::Forward


        }

        runLinearFixed(&m->loutput, b1, N, p->d_model, out);
        float mse = 0, pl = 0, pl_std = 0;
        int cor = 0;


        for (int i = 0; i < N; ++i) {
            delta[i] = (2.0 * (out[i] - Y[i])) / (double)N;
            float dd;
            float pli;

            if ((out[i] < 0 && Y[i] < 0) || (out[i] > 0 && Y[i] > 0)) {
                ++cor;
            }

            mse += (out[i] - Y[i]) * (out[i] - Y[i]);
        }

        mse = sqrt(mse / (double)N);
        if (buf_rmse) *buf_rmse = mse;
        if (buf_acc) *buf_acc = (float)cor / (float)N;

        backLinear(m, &m->loutput, b1, N, delta, &b2, &bc2);

        for (int i = m->p.n_layers - 1; i >= 0; --i) {
            ResidualBlock* bl = &m->blocks[i];

            backRmsNorm(m, O[i][14], b2, N, p->d_model);

            backLinear(m, &bl->m.out_proj, O[i][12], N, b2, &b1, &bc1);

            //b1 = dz
            //b2 = dy

            bc2 = N * p->d_inner;
            b2 = (float*)realloc(b2, bc2 * sizeof(float));


            //backprop multiply()
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < p->d_inner; ++k) {
                    b2[j * p->d_inner + k] = b1[j * p->d_inner + k] * O[i][13][j * p->d_inner + k];
                    b1[j * p->d_inner + k] *= O[i][11][j * p->d_inner + k];
                }
            }

            for (int j = 0; j < m->p.d_inner; ++j) {
                for (int k = 0; k < m->p.d_state; ++k) {
                    Ai[j * m->p.d_state + k] = -exp(m->blocks[0].m.A[j * m->p.d_state + k]);
                }
            }


            selectiveScanBackward(O[i][5], O[i][10], Ai, O[i][8], O[i][9], m->blocks[0].m.D, b2, buf[i], dx, ddelta, bl->m.AG, db, dc, bl->m.DG, N, p->d_inner, p->d_state);
            SoftPlusBackward(O[i][10], ddelta, N, p->d_inner);
            backLinear(m, &bl->m.dt_proj, O[i][7], N, ddelta, &b3, &bc3);

            //merge deltaBC

            for (int j = 0; j < N; ++j) {
                int colidx = 0;
                const int colcnt = p->dt_rank + p->d_state * 2;
                for (int k = 0; k < p->dt_rank; ++k) {
                    dmerged[j * colcnt + k] = b3[j * p->dt_rank + k];
                }
                colidx += p->dt_rank;

                for (int k = 0; k < p->d_state; ++k) {
                    dmerged[j * colcnt + (k + colidx)] = db[j * p->d_state + k];
                }
                colidx += p->d_state;

                for (int k = 0; k < p->d_state; ++k) {
                    dmerged[j * colcnt + (k + colidx)] = dc[j * p->d_state + k];
                }
            }



            backLinear(m, &bl->m.x_proj, O[i][5], N, dmerged, &b2, &bc2);


            //add dx to input delta
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < p->d_inner; ++k) {
                    b2[j * p->d_inner + k] += dx[j * p->d_inner + k];
                }
            }


            //z = F.silu(z)
            SiLUBackward(O[i][3], b1, N, p->d_inner);


            //o = F.silue(o)
            SiLUBackward(O[i][4], b2, N, p->d_inner);


            backConv1D(&bl->m.conv, O[i][2], N, p->d_inner, b2, &b3, &bc3, 1);

            //merge x and z
            for (int j = 0; j < N; ++j) {
                int colidx = 0;
                const int colcnt = p->d_inner * 2;
                for (int k = 0; k < p->d_inner; ++k) {
                    dmerged[j * colcnt + k] = b3[j * p->d_inner + k];
                }
                colidx += p->d_inner;
                for (int k = 0; k < p->d_inner; ++k) {
                    dmerged[j * colcnt + (k + colidx)] = b1[j * p->d_inner + k];
                }
            }


            backLinear(m, &bl->m.in_proj, O[i][1], N, dmerged, &b2, &bc2);


            backRmsNorm(m, O[i][0], b2, N, p->d_model);

        }

        backLinear(m, &m->linput, X, N, b2, &b1, &bc1);

        //applyGradientsSGD(m, lr);
        applyGradientsAdam(m, ma);


    }


    if (out) free(out);
    if (O) free(O);
    if (buf) free(buf);
    if (dx) free(dx);
    if (delta) free(delta);
    if (Ai) free(Ai);
    if (b1) free(b1);
    if (b2) free(b2);
    if (b3) free(b3);
    if (b4) free(b4);
    if (ma) free(ma);

    return 1;

}

/*
Windows API multi-threading

*/

struct MBStruct {
    Mamba* m;
    float rmse, acc;
    const float* X;
    const float* Y;
    int N, D;
    float lr;
    int iterations;
    int idx;
};

DWORD WINAPI MBCallback(LPVOID p)
{
    MBStruct* ms = (MBStruct*)p;
    trainMamba(ms->m, ms->X, ms->Y, ms->N, ms->D, ms->iterations, ms->lr, &ms->rmse, &ms->acc);
    return 1;
}

int trainMambaBatch(MambaBatch* mba, const float* X, const float* Y, int mcnt, int N, int D, int iterations, float lr, int threads)
{
    int ret = 1;
    if (mcnt < threads) threads = mcnt;
    size_t sz = (sizeof(HANDLE) + sizeof(MBStruct) + sizeof(MBStruct*)) * threads;
    HANDLE* h = (HANDLE*)calloc(1, sz);
    MBStruct* mb = (MBStruct*)(h + threads);
    MBStruct** mp = (MBStruct**)(mb + threads);

    for (int i = 0; i < threads; ++i) {
        mb[i].N = N;
        mb[i].D = D;
        mb[i].iterations = iterations;
        mb[i].lr = lr;
        mp[i] = &mb[i];
    }


    int idx = 0;
    int active = 0;
    int rem = mcnt;
    int avail_idx = 0;

    while (rem || active) {
        for (int i = avail_idx; i < threads; ++i) {
            if (!rem) break;

            mp[i]->X = X + idx * N * D;
            mp[i]->Y = Y + idx * N;
            mp[i]->m = mba->m[idx];
            mp[i]->idx = idx;

            h[i] = CreateThread(0, 0, MBCallback, mp[i], 0, 0);

            if (!h[i]) {
                printf("trainMambaBatch: createthread failed\n");
                ret = 0;

                if (active) {
                    WaitForMultipleObjects(active, h, TRUE, INFINITE);
                    for (int k = 0; k < active; ++k) CloseHandle(h[k]);
                }

                rem = 0;
                ret = 0;
                break;
            }

            ++idx;
            --rem;
            ++active;
        }

        DWORD ret = WaitForMultipleObjects(active, h, FALSE, INFINITE);
        int curidx = ret - WAIT_OBJECT_0;
        CloseHandle(h[curidx]);
            
        MBStruct* ptr = mp[curidx];
        for (int i = curidx; i < active - 1; ++i) {
            h[i] = h[i + 1];
            mp[i] = mp[i + 1];
        }

        mp[active - 1] = ptr;
        --active;
        avail_idx = active;

    }


    free(h);
    return ret;
}
