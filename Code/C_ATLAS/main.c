#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
//#include <omp.h>
#include "mesh.h"
#include "matrices.h"
#include "linearAlgebra.h"

#define ARGS 3

int main(int argc, char **argv)
{
    if (argc != ARGS) {
        fprintf(stderr, "Wrong number of arguments\nUsage: %s N dt\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    int Re, N;
    float dt, tol;
    
    tol = 1e-5;
    Re = 3200;
    dt = atof(argv[1]);
    N = atoi(argv[2]);
        
    // Generate mesh
    float *tx = generateTx(N);
    float *th = generateTh(N);
    float *x = generateX(N);
    float *h = generateH(N);
    
    // Generate matrices
    float *tE21 = generateTE21(N);
    float *E21K = generateE21K(N);
    float *E21 = generateE21(N);
    float *H1t1 = generateH1t1(N, th);
    float *Ht11 = diagMatInvert(H1t1);
    float *Ht02 = generateHt02(N, h);
    float *A = generateA(Ht11, tE21);
    float *C0 = matMatMult(Ht02, E21);
    float *C1 = matMatMult(tE21, Ht11);
    float *C2 = generateC2(H1t1, E21, C0);
    
    // Generate vectors
    float *u = callocVector(2 * N * (N - 1));
    float *uOld = mallocVector(2 * N * (N - 1));
    float *uK = generateUK(N, h);
    float *uPres = generateUPres(H1t1, E21, Ht02, E21K, uK);
    float *C3 = vecScalarMult(uPres, (float) 1 / Re);
        
    // Free redundant matrices
    free(E21K);
    free(E21);
    free(H1t1);
    free(Ht11);
    free(Ht02);
    
    // LU decomposition of A
    factorizeA(N, A);
    
    // Allocate memory for loop vectors
    float *C4 = mallocVector(C2[0]);
    float *xi = mallocVector(C0[0]);
    float *convective = mallocVector(2 * N * (N - 1));
    float *rhs = mallocVector(C1[0]);
    float *P = mallocVector(A[0]);
    
    float diff = 1;
    int iteration = 0;
        
    // Simulation loop
    while (diff > tol) {
        iteration++;
        
        updateXi(xi, C0, u);
        updateConvective(convective, N, xi, u, uK, h);
        updateC4(C4, C2, u, C3, convective, Re);
        updateRhs(rhs, C1, u, C4, dt, convective);
        updateP(P, N, A, rhs);
        cblas_scopy(u[0], &u[1], 1, &uOld[1], 1);
        updateU(u, tE21, P, C4, dt);
        
        if (iteration % 1000 == 0) {
            updateDiff(&diff, u, uOld, dt);
            fprintf(stdout, "Iteration %d\t\tdiff = %6.5f\n", iteration, diff);
        }
    }
    
    fprintf(stdout, "Converged after %d iterations.\n", iteration);
    
    // Free dynamically allocated memory
    free(tE21);
    free(A);
    free(C0);
    free(C1);
    free(C2);
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
