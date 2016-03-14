#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrices.h"
#include "linearAlgebra.h"

double** generateTE21(int N, double **tE21, int *rows, int *columns)
{
    *rows = N * N;
    *columns = 2 * N * (N-1);
        
    tE21 = callocMatrix(*rows, *columns);
    
    int i = 0, k = 0;

    for (int j = 0; j < *columns / 2; j++) {
        tE21[i][j] = 1;
        tE21[i+1][j] = -1;

        tE21[k][j+*columns/2] = 1;
        tE21[k+N][j+*columns/2] = -1;

        i += ((j + 1) % (N - 1) == 0 ? 2 : 1);
        k += 1;
    }
    
    return tE21;
}

double** generateE21K(int N, double **E21K, int *rows, int *columns)
{
    *rows = (N + 1) * (N + 1);
    *columns = 4 * (N + 1) + 4 * N;
        
    E21K = callocMatrix(*rows, *columns);
    
    int i = 0, k = *rows - 1;

    for (int j = 0; j < N + 1; j++) {
        E21K[i][j] = 1;
        E21K[k][*columns / 2 - 1 - j] = -1;

        E21K[i][j + *columns / 2] = -1;
        E21K[i][j + *columns / 2 + 1] = 1;

        E21K[k][*columns - j - 1] = 1;
        E21K[k][*columns - j - 2] = -1;

        i += 1;
        k -= 1;
    }

    i = 0;

    for (int j = N + 1; j < *columns / 2 - (N + 1); j++) {
        E21K[i][j] = -1;
        E21K[i+N+1][j] = 1;

        i += ((j - N - 1) % 2 == 0 ? N : 1);
    }

    i = N + 1;

    for (int j = *columns / 2 + N + 2; j < *columns - (N + 2); j += 2) {
        E21K[i][j] = -1;
        E21K[i+N][j+1] = 1;

        i += N + 1;
    }

    return E21K;
}

double** generateE21(int N, double **E21, int *rows, int *columns)
{
    *rows = (N + 1) * (N + 1);
    *columns = 2 * N * (N - 1);

    E21 = callocMatrix(*rows, *columns);

    int i = 1, k = N + 1;

    for (int j = 0; j < *columns / 2; j++) {
        E21[i][j] = -1;
        E21[i+N+1][j] = 1;

        E21[k][j+*columns/2] = 1;
        E21[k+1][j+*columns/2] = -1;
        
        i += ((j + 1) % (N - 1) == 0 ? 3 : 1);
        k += ((j + 1) % N == 0 ? 2 : 1);

    }

    return E21;
}

double** generateH1t1(int N, double **H1t1, int *rows, int *columns,
                      double *th)
{
    *rows = 2 * N * (N - 1);
    *columns = 2 * N * (N - 1);

    H1t1 = callocMatrix(*rows, *columns);
    
    double velocity, length;
    int k;

    for (int j = 0; j < N; j++) {
        velocity = 1 / th[j];

        for (int i = 0; i < N - 1; i++) {
            length = 0.5 * (th[i] + th[i+1]);

            k = i + (N - 1) * j;

            H1t1[k][k] = velocity * length;
        }
    }

    for (int j = 0; j < N - 1; j++) {
        length = 0.5 * (th[j] + th[j+1]);

        for (int i = 0; i < N; i++) {
            velocity = 1 / th[i];

            k = *rows / 2 + i + N * j;

            H1t1[k][k] = velocity * length;
        }
    }

    return  H1t1;
}

double** generateHt02(int N, double **Ht02, int *rows, int *columns,
                      double *h)
{
    *rows = (N + 1) * (N + 1);
    *columns = (N + 1) * (N + 1);

    Ht02 = callocMatrix(*rows, *columns);
    
    int k;

    for (int i = 0; i < N + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            k = j + (N + 1) * i;

            Ht02[k][k] = 1 / (h[i] * h[j]);
        }
    }

    return Ht02;
}

double* generateConvective(int N, double *xi, double *u, double *uK,
                           double *h)
{
    double *convective = calloc(2 * N * (N - 1), sizeof(double));
    
    if (convective == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }

    double *uTotal = combineUAndUK(N, u, uK);
    double V1, V2, U1, U2;
    
    int offset = (N + 1) * (N + 2), k;

    for (int i = 1; i < N; i++) {
        for (int j = 0; j < N; j++) {
            k = (i - 1) + (N - 1) * j;
            
            V1 = uTotal[offset+i+j*(N+2)] + uTotal[offset+(i+1)+j*(N+2)];
            V2 = uTotal[offset+i+(j+1)*(N+2)] + 
                 uTotal[offset+(i+1)+(j+1)*(N+2)];

            convective[k] = - h[i] / (4 * h[j]) * V1 * xi[i+j*(N+1)] - 
                            h[i] / (4 * h[j+1]) * V2 * xi[i+(j+1)*(N+1)];
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 1; j < N; j++) {
            k = N * (N - 1) + i + N * (j - 1);

            U1 = uTotal[i+j*(N+1)] + uTotal[i+(j+1)*(N+1)];
            U2 = uTotal[(i+1)+j*(N+1)] + uTotal[(i+1)+(j+1)*(N+1)];

            convective[k] = h[j] / (4 * h[i]) * U1 * xi[i+j*(N+1)] + 
                            h[j] / (4 * h[i+1]) * U2 * xi[(i+1)+j*(N+1)];
        }
    }

    return convective;
}

double* combineUAndUK(int N, double *u, double *uK)
{
    double *uTotal = calloc(2 * (N + 1) * (N + 2), sizeof(double));
    
    if (uTotal == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }

    for (int i = 0; i < N + 2; i++) {
        uTotal[(N+1)*(N+2)-1-i] = uK[2*(N+1)+N*2-1-i];
    }

    int k = N + 2;

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N - 1; i++) {
            uTotal[k] = u[i+(N-1)*j] = u[i+(N-1)*j];
            k += 1;
        }
        k += 2;
    }

    k = (N + 1) * (N + 2) + N + 3;

    for (int j = 0; j < N - 1; j++) {
        for (int i = 0; i < N; i++) {
            uTotal[k] = u[i+N*j+N*(N-1)] = u[i+N*j+N*(N-1)];
            k += 1;
        }
        k += 2;
    }

    return uTotal;
}

double* generateU(int N, int *rowsU)
{
    *rowsU = 2 * N * (N - 1);
    
    double *u = calloc(*rowsU, sizeof(double));
    
    if (u == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }

    return u;
}

double* generateUOld(int N)
{
    int rows = 2 * N * (N - 1);
    
    double *uOld = calloc(rows, sizeof(double));
    
    if (uOld == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }

    return uOld;
}

double* generateUK(int N, int *rowsUK, double *h)
{
    *rowsUK = 4 * (N + 1) + 4 * N;
    
    double *uK = calloc(*rowsUK, sizeof(double));
    
    if (uK == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }

    for (int i = 0; i < N + 1; i++) {
        uK[i+N*2+N+1] = -1 * h[i];
    }

    return uK;
}

double* generateC4(double **C2, int rowsC2, int columnsC2, double *u,
                   int rowsU, double *C3, int rowsC3, double Re)
{
    if (columnsC2 != rowsU || rowsU != rowsC3) {
        fprintf(stderr, "Incompatible matrix and vector dimensions.\n");
        exit(-1);
    }
    
    double *C4 = calloc(rowsC2, sizeof(double));
    
    if (C4 == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }
    
    for (int i = 0; i < rowsC2; i++) {
        for (int j = 0; j < columnsC2; j++) {
            C4[i] += C2[i][j] * (u[j] / (double) Re);
        }
        
        C4[i] += C3[i];
    }
    
    return C4;
}

double* generateRhs(double **C1, int rowsC1, int columnsC1, double *u,
                    int rowsU, double *convective, double *C4, double dt)
{
    double *rhs = calloc(rowsC1, sizeof(double));
    
    if (rhs == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }
    
    for (int i = 0; i < rowsC1; i++) {
        for (int j = 0; j < columnsC1; j++) {
            rhs[i] += C1[i][j] * (u[j] / dt - convective[j] - C4[j]);
        }
    }
    
    return rhs;
}

void updateU(double **E10, int rowsE10, int columnsE10, double *u, int rowsU,
             double *convective, double *P, double *C4, double dt)
{
    double temp;
    
    for (int i = 0; i < rowsE10; i++) {
        temp = 0;
        
        for (int j = 0; j < columnsE10; j++) {
            temp += E10[i][j] * P[j];
        }
        
        u[i] -= dt * (convective[i] + temp + C4[i]);
    }
}

double computeDiff(double *u, int rowsU, double *uOld, double dt)
{
    double diff = 0;
    
    for (int i = 0; i < rowsU; i++) {        
        if (fabs((u[i] - uOld[i]) / dt) > diff) {
            diff = fabs((u[i] - uOld[i]) / dt);
        }
    }
    
    return diff;
}
