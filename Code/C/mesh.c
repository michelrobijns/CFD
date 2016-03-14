#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mesh.h"

double* generateTx(int N)
{
    double *tx = malloc(sizeof(double) * (N + 1));
    
    if (tx == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }
    
    double delta = 1.0 / N;
    
    for (int i = 0; i < N+1; i++) {
        tx[i] = 0.5 * (1 - cos(M_PI * i * delta));
    }
    
    return tx;
}

double* generateTh(int N)
{
    double *tx = malloc(sizeof(double) * (N + 1));
    double *th = malloc(sizeof(double) * N);
    
    if (tx == NULL || th == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }
    
    double delta = 1.0 / N;
    
    for (int i = 0; i < N+1; i++) {
        tx[i] = 0.5 * (1 - cos(M_PI * i * delta));
    }
    
    for (int i = 0; i < N; i++) {
        th[i] = tx[i+1] - tx[i];
    }
    
    free(tx);
    
    return th;
}

double* generateX(int N)
{
    double *tx = malloc(sizeof(double) * (N + 1));
    double *x = malloc(sizeof(double) * (N + 2));
    
    if (tx == NULL || x == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }
    
    double delta = 1.0 / N;
    
    for (int i = 0; i < N+1; i++) {
        tx[i] = 0.5 * (1 - cos(M_PI * i * delta));
    }
    
    x[0] = 0;
    x[N+1] = 1;
    
    for (int i = 1; i < N+1; i++) {
        x[i] = 0.5 * (tx[i-1] + tx[i]);
    }
    
    free(tx);
    
    return x;
}

double* generateH(int N)
{
    double *tx = malloc(sizeof(double) * (N + 1));
    double *x = malloc(sizeof(double) * (N + 2));
    double *h = malloc(sizeof(double) * (N + 1));
    
    if (tx == NULL || x == NULL || h == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }
    
    double delta = 1.0 / N;
    
    for (int i = 0; i < N+1; i++) {
        tx[i] = 0.5 * (1 - cos(M_PI * i * delta));
    }
    
    x[0] = 0;
    x[N+1] = 1;
    
    for (int i = 1; i < N+1; i++) {
        x[i] = 0.5 * (tx[i-1] + tx[i]);
    }
    
    for (int i = 0; i < N+1; i++) {
        h[i] = x[i+1] - x[i];
    }
    
    free(tx);
    free(x);
    
    return h;
}

void checkMesh(int N, double *tx, double *th, double *x, double *h)
{
    fprintf(stdout, "Outer oriented grid:\n\ntx =\t");
    
    for (int i = 0; i < N+1; i++) {
        fprintf(stdout, "%f\t", tx[i]);
    }
    
    fprintf(stdout, "\nth =\t");
    
    for (int i = 0; i < N; i++) {
        fprintf(stdout, "\t%f", th[i]);
    }
    
    fprintf(stdout, "\n\nInner oriented grid:\n\nx  =\t");
    
    for (int i = 0; i < N+2; i++) {
        fprintf(stdout, "%f\t", x[i]);
    }
    
    fprintf(stdout, "\nh  =\t");
    
    for (int i = 0; i < N+1; i++) {
        fprintf(stdout, "\t%f", h[i]);
    }
    
    fprintf(stdout, "\n");
}
