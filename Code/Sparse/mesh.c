#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mesh.h"
#include "linearAlgebra.h"

float* generateTx(int N)
{
    float *tx = mallocVector(N + 1);
    
    float delta = 1.0 / N;
    
    for (int i = 0; i < N+1; i++) {
        tx[1 + i] = 0.5 * (1 - cos(M_PI * i * delta));
    }
    
    return tx;
}

float* generateTh(int N)
{
    float *tx = mallocVector(N + 1);
    float *th = mallocVector(N);
    
    float delta = 1.0 / N;
    
    for (int i = 0; i < N+1; i++) {
        tx[1 + i] = 0.5 * (1 - cos(M_PI * i * delta));
    }
    
    for (int i = 0; i < N; i++) {
        th[1 + i] = tx[1 + i+1] - tx[1 + i];
    }
    
    free(tx);
    
    return th;
}

float* generateX(int N)
{
    float *tx = mallocVector(N + 1);
    float *x = mallocVector(N + 2);
    
    float delta = 1.0 / N;
    
    for (int i = 0; i < N+1; i++) {
        tx[1 + i] = 0.5 * (1 - cos(M_PI * i * delta));
    }
    
    x[1] = 0;
    x[1 + N+1] = 1;
    
    for (int i = 1; i < N+1; i++) {
        x[1 + i] = 0.5 * (tx[1 + i-1] + tx[1 + i]);
    }
    
    free(tx);
    
    return x;
}

float* generateH(int N)
{
    float *tx = mallocVector(N + 1);
    float *x = mallocVector(N + 2);
    float *h = mallocVector(N + 1);
    
    float delta = 1.0 / N;
    
    for (int i = 0; i < N+1; i++) {
        tx[1 + i] = 0.5 * (1 - cos(M_PI * i * delta));
    }
    
    x[1] = 0;
    x[1 + N+1] = 1;
    
    for (int i = 1; i < N+1; i++) {
        x[1 + i] = 0.5 * (tx[1 + i-1] + tx[1 + i]);
    }
    
    for (int i = 0; i < N+1; i++) {
        h[1 + i] = x[1 + i+1] - x[1 + i];
    }
    
    free(tx);
    free(x);
    
    return h;
}

void checkMesh(int N, float *tx, float *th, float *x, float *h)
{
    fprintf(stdout, "Outer oriented grid:\n\ntx =\t");
    
    for (int i = 0; i < N+1; i++) {
        fprintf(stdout, "%f\t", tx[1 + i]);
    }
    
    fprintf(stdout, "\nth =\t");
    
    for (int i = 0; i < N; i++) {
        fprintf(stdout, "\t%f", th[1 + i]);
    }
    
    fprintf(stdout, "\n\nInner oriented grid:\n\nx  =\t");
    
    for (int i = 0; i < N+2; i++) {
        fprintf(stdout, "%f\t", x[1 + i]);
    }
    
    fprintf(stdout, "\nh  =\t");
    
    for (int i = 0; i < N+1; i++) {
        fprintf(stdout, "\t%f", h[1 + i]);
    }
    
    fprintf(stdout, "\n");
}
