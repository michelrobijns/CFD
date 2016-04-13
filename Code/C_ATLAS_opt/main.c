#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
//#include <omp.h>
#include "mesh.h"
#include "matrices.h"
#include "linearAlgebra.h"
#include "fast.h"

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
    Re = 1000;
    dt = atof(argv[1]);
    N = atoi(argv[2]);
        
    // Generate mesh
    printf("Generating mesh... (1/17)\n");
    float *tx = generateTx(N);
    float *th = generateTh(N);
    float *x = generateX(N);
    float *h = generateH(N);
    
    // Generate matrices
    printf("Generating tE21... (2/17)\n");
    float *tE21 = generateTE21(N);
    printf("Generating E21K... (3/17)\n");
    float *E21K = generateE21K(N);
    printf("Generating E21... (4/17)\n");
    float *E21 = generateE21(N);
    printf("Generating H1t1Vec... (5/17)\n");
    float *H1t1Vec = generateH1t1Vec(N, th);
    printf("Generating Ht11Vec... (6/17)\n");
    float *Ht11Vec = vecInvert(H1t1Vec);
    printf("Generating Ht02Vec... (7/17)\n");
    float *Ht02Vec = generateHt02Vec(N, h);
    printf("Generating A... (8/17)\n");
    float *A = generateAFast(N, Ht11Vec);
    printf("Generating C0... (9/17)\n");
    float *C0 = generateC0Fast(N, Ht02Vec);
    printf("Generating C1... (10/17)\n");
    float *C1 = generateC1Fast(N, Ht11Vec);
    printf("Generating C2... (11/17)\n");
    float *C2 = generateC2Fast(N, H1t1Vec, C0);
            
    // Generate vectors
    printf("Generating u... (12/17)\n");
    float *u = callocVector(2 * N * (N - 1));
    printf("Generating uOld... (13/17)\n");
    float *uOld = mallocVector(2 * N * (N - 1));
    printf("Generating uK... (14/17)\n");
    float *uK = generateUK(N, h);
    printf("Generating uPres... (15/17)\n");
    float *uPres = generateUPresFast(H1t1Vec, E21, Ht02Vec, E21K, uK);
    printf("Generating C3... (16/17)\n");
    float *C3 = vecScalarMult(uPres, (float) 1 / Re);
        
    // Free redundant matrices
    free(E21K);
    free(E21);
    free(H1t1Vec);
    free(Ht02Vec);
        
    // LU decomposition of A
    printf("Generating LU factorization... (17/17)\n");
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
                
        if (iteration % 1 == 0) {
            updateDiff(&diff, u, uOld, dt);
            fprintf(stdout, "Iteration %d\tdiff = %6.5f\n", iteration, diff);
        }
    }
    
    fprintf(stdout, "Converged after %d iterations.\n", iteration);
    
    // Store the results in data files
    //storeVector(u, "u32.dat");
    //storeVector(xi, "xi32.dat");
    //storeVector(P, "P32.dat");
            
    storeStreamFunction(N, Ht11Vec, u, tx);
    storeVorticity(N, xi, tx);
    storePressure(N, x, h, u, uK, P);
    
    // Free dynamically allocated memory
    free(tE21);
    free(Ht11Vec);
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
