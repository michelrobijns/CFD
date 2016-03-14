#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mesh.h"
#include "matrices.h"
#include "linearAlgebra.h"

#define ARGS 2

int main(int argc, char **argv)
{
    if (argc != ARGS) {
        fprintf(stderr, "Wrong number of arguments\nUsage: %s N\n", argv[0]);
        exit(-1);
    }
    
    int Re, N;
    double dt, tol;
    
    Re = 3200;
    N = atoi(argv[1]);
    dt = 0.01;
    tol = 1e-5;
    
    // Generate mesh
    double *tx = generateTx(N);
    double *th = generateTh(N);
    double *x = generateX(N);
    double *h = generateH(N);
    
    // Generate matrices and vectors
    int rowsTE21, columnsTE21;
    double **tE21 = generateTE21(N, tE21, &rowsTE21, &columnsTE21);
    
    int rowsE10 = columnsTE21, columnsE10 = rowsTE21;
    double **E10 = matTranspose(tE21, rowsTE21, columnsTE21);
    matScalarMult(E10, -1, rowsE10, columnsE10);
    
    int rowsE21K, columnsE21K;
    double **E21K = generateE21K(N, E21K, &rowsE21K, &columnsE21K);

    int rowsE21, columnsE21;
    double **E21 = generateE21(N, E21, &rowsE21, &columnsE21);
    
    int rowsTE10 = columnsE21, columnsTE10 = rowsE21;
    double **tE10 = matTranspose(E21, rowsE21, columnsE21);
    
    int rowsH1t1, columnsH1t1;
    double **H1t1 = generateH1t1(N, H1t1, &rowsH1t1, &columnsH1t1, th);
    
    int rowsHt11 = rowsH1t1, columnsHt11 = columnsH1t1;
    double **Ht11 = matInvert(H1t1, rowsH1t1, columnsH1t1);

    int rowsHt02, columnsHt02;
    double **Ht02 = generateHt02(N, Ht02, &rowsHt02, &columnsHt02, h);

    int rowsA = rowsH1t1, columnsA = columnsE10;
    double **A = matMatMult(Ht11, rowsH1t1, columnsH1t1, E10, rowsE10, columnsE10);
    A = matMatMult(tE21, rowsTE21, columnsTE21, A, rowsA, columnsA);
    rowsA = rowsTE21;
    
    int rowsU;
    double *u = generateU(N, &rowsU);
    double *uOld = generateUOld(N);
    
    int rowsUK;
    double *uK = generateUK(N, &rowsUK, h);
        
    double *uPres = matVecMult(E21K, rowsE21K, columnsE21K, uK, rowsUK);
    int rowsUPres = rowsE21K;
    uPres = matVecMult(Ht02, rowsHt02, columnsHt02, uPres, rowsUPres);
    rowsUPres = rowsHt02;
    uPres = matVecMult(tE10, rowsTE10, columnsTE10, uPres, rowsUPres);
    rowsUPres = rowsTE10;
    uPres = matVecMult(H1t1, rowsH1t1, columnsH1t1, uPres, rowsUPres);
    rowsUPres = rowsH1t1;

    int rowsC0 = rowsHt02, columnsC0 = columnsE21;
    double **C0 = matMatMult(Ht02, rowsHt02, columnsHt02, 
                             E21, rowsE21, columnsE21);

    int rowsC1 = rowsTE21, columnsC1 = columnsHt11;
    double **C1 = matMatMult(tE21, rowsTE21, columnsTE21,
                             Ht11, rowsHt11, columnsHt11);
    
    int rowsC2 = rowsTE10, columnsC2 = columnsC0;
    double **C2 = matMatMult(tE10, rowsTE10, columnsTE10,
                             C0, rowsC0, columnsC0);
    C2 = matMatMult(H1t1, rowsH1t1, columnsH1t1, C2, rowsC2, columnsC2);
    rowsC2 = rowsH1t1;
    
    int rowsC3 = rowsUPres;
    double *C3 = malloc(rowsUPres * sizeof(double));
    memcpy(C3, uPres, rowsUPres * sizeof(double));
    vecScalarMult(C3, rowsUPres, 1 / (double) Re);
    
    LUFactorization(A, rowsA, columnsA);
    
    // Free unnecessary matrices
    freeMatrix(tE21, rowsTE21);
    freeMatrix(E21K, rowsE21K);
    freeMatrix(E21, rowsE21);
    freeMatrix(tE10, rowsTE10);
    freeMatrix(H1t1, rowsH1t1);
    freeMatrix(Ht02, rowsHt02);
    
    // Iterate
    double *C4, *xi, *convective, *rhs, *P;
    double diff = 1;
    int iteration = 0;
    
    while (diff > tol) {
        iteration++;
        
        C4 = generateC4(C2, rowsC2, columnsC2, u, rowsU, C3, rowsC3, Re);
        
        xi = matVecMult(C0, rowsC0, columnsC0, u, rowsU);
        
        convective = generateConvective(N, xi, u, uK, h);
        
        rhs = generateRhs(C1, rowsC1, columnsC1, u, rowsU, convective, C4,
                          dt);
        
        P = LUSubstitution(A, rowsA, columnsA, rhs, rowsA);
        
        memcpy(uOld, u, rowsU * sizeof(double));
        
        updateU(E10, rowsE10, columnsE10, u, rowsU, convective, P, C4, dt);
        
        diff = computeDiff(u, rowsU, uOld, dt);
        
        fprintf(stdout, "Iteration %d\tdiff = %.11e\n", iteration, diff);
    }
    
    fprintf(stdout, "Converged after %d iterations.\n", iteration);
    
    // Free dynamically allocated memory
    freeMatrix(E10, rowsE10);
    freeMatrix(Ht11, rowsHt11);
    freeMatrix(A, rowsA);
    freeMatrix(C0, rowsC0);
    freeMatrix(C1, rowsC1);
    freeMatrix(C2, rowsC2);
    
    free(tx);
    free(th);
    free(x);
    free(h);
    free(u);
    free(uOld);
    free(uK);
    free(uPres);
    free(C3);
    free(C4);
    free(xi);
    free(convective);
    free(rhs);
    free(P);
    
    return 0;
}
