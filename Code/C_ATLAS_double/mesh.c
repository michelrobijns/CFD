#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mesh.h"
#include "linearAlgebra.h"

double* generateTx(int N)
{
    double *tx = mallocVector(N + 1);
    
    double delta = 1.0 / N;
    
    for (int i = 0; i < N+1; i++) {
        tx[1 + i] = 0.5 * (1 - cos(M_PI * i * delta));
    }
    
    return tx;
}

double* generateTh(int N)
{
    double *tx = mallocVector(N + 1);
    double *th = mallocVector(N);
    
    double delta = 1.0 / N;
    
    for (int i = 0; i < N+1; i++) {
        tx[1 + i] = 0.5 * (1 - cos(M_PI * i * delta));
    }
    
    for (int i = 0; i < N; i++) {
        th[1 + i] = tx[1 + i+1] - tx[1 + i];
    }
    
    free(tx);
    
    return th;
}

double* generateX(int N)
{
    double *tx = mallocVector(N + 1);
    double *x = mallocVector(N + 2);
    
    double delta = 1.0 / N;
    
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

double* generateH(int N)
{
    double *tx = mallocVector(N + 1);
    double *x = mallocVector(N + 2);
    double *h = mallocVector(N + 1);
    
    double delta = 1.0 / N;
    
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

void checkMesh(int N, double *tx, double *th, double *x, double *h)
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
