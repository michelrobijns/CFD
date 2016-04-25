#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
//#include <omp.h>
#include "matrices.h"
#include "linearAlgebra.h"

float* generateAFast(int N, float *Ht11Vec)
{
    int rows = N * N;
    int columns = 2 * N * (N-1);
        
    float *E10D = callocMatrix(rows, columns);
        
    int i = 0, k = 0, col;

    for (int j = 0; j < columns / 2; j++) {
        col = j;
        
        E10D[2 + i * columns + col] = 1 * sqrt(Ht11Vec[1 + col]);
        E10D[2 + (i+1) * columns + col] = -1 * sqrt(Ht11Vec[1 + col]);
        
        col = j+columns/2;

        E10D[2 + k * columns + col] = 1 * sqrt(Ht11Vec[1 + col]);
        E10D[2 + (k+N) * columns + col] = -1 * sqrt(Ht11Vec[1 + col]);

        i += ((j + 1) % (N - 1) == 0 ? 2 : 1);
        k++;
    }
    
    float *A = mallocMatrix(E10D[0], E10D[0]);
    
    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans, A[0], E10D[1], -1.0, &E10D[2], E10D[1], 0.0, &A[2], A[1]);
    
    free(E10D);
        
    rows = A[0];
    columns = A[1];
    
    for (int i = 0; i < columns - 1; i++) {
        A[2 + (i+1) * columns + i] = A[2 + i * columns + (i+1)];
        
        if (i+N < columns) A[2 + (i+N) * columns + i] = A[2 + i * columns + (i+N)];
    }
    
    return A;
}

float* generateH1t1Vec(int N, float *th)
{
    int rows = 2 * N * (N - 1);

    float *H1t1Vec = mallocVector(rows);
    
    float velocity, length;
    int k;

    for (int j = 0; j < N; j++) {
        velocity = 1 / th[1 + j];

        for (int i = 0; i < N - 1; i++) {
            length = 0.5 * (th[1 + i] + th[1 + i+1]);

            k = i + (N - 1) * j;

            H1t1Vec[1 + k] = velocity * length;
        }
    }

    for (int j = 0; j < N - 1; j++) {
        length = 0.5 * (th[1 + j] + th[1 + j+1]);

        for (int i = 0; i < N; i++) {
            velocity = 1 / th[1 + i];

            k = rows / 2 + i + N * j;

            H1t1Vec[1 + k] = velocity * length;
        }
    }

    return H1t1Vec;
}

float* generateHt02Vec(int N, float *h)
{
    int rows = (N + 1) * (N + 1);

    float *Ht02Vec = mallocVector(rows);
    
    int k;

    for (int i = 0; i < N + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            k = j + (N + 1) * i;

            Ht02Vec[1 + k] = 1 / (h[1 + i] * h[1 + j]);
        }
    }

    return Ht02Vec;
}

float* generateC0Fast(int N, float *Ht02Vec)
{
    int rows = (N + 1) * (N + 1);
    int columns = 2 * N * (N - 1);

    float *C0 = callocMatrix(rows, columns);

    int i = 1, k = N + 1, row;

    for (int j = 0; j < columns / 2; j++) {
        row = i;
        
        C0[2 + row * columns + j] = -1 * Ht02Vec[1 + row];
        
        row = i+N+1;
        
        C0[2 + row * columns + j] = 1 * Ht02Vec[1 + row];
        
        row = k;

        C0[2 + row * columns + (j+columns/2)] = 1 * Ht02Vec[1 + row];
        
        row = k+1;
        
        C0[2 + row * columns + (j+columns/2)] = -1 * Ht02Vec[1 + row];
        
        i += ((j + 1) % (N - 1) == 0 ? 3 : 1);
        k += ((j + 1) % N == 0 ? 2 : 1);
    }

    return C0;
}

float* generateC1Fast(int N, float *Ht11Vec)
{
    int rows = N * N;
    int columns = 2 * N * (N-1);
        
    float *C1 = callocMatrix(rows, columns);
        
    int i = 0, k = 0, col;

    for (int j = 0; j < columns / 2; j++) {
        col = j;
        
        C1[2 + i * columns + col] = 1 * Ht11Vec[1 + col];
        C1[2 + (i+1) * columns + col] = -1 * Ht11Vec[1 + col];
        
        col = j+columns/2;

        C1[2 + k * columns + col] = 1 * Ht11Vec[1 + col];
        C1[2 + (k+N) * columns + col] = -1 * Ht11Vec[1 + col];

        i += ((j + 1) % (N - 1) == 0 ? 2 : 1);
        k++;
    }

    return C1;
}

float* generateC2Fast(int N, float *H1t1Vec, float *C0)
{
    int rows = 2 * N * (N - 1);
    int columns = (N + 1) * (N + 1);

    float *temp = callocMatrix(rows, columns);

    int j = 1, k = N + 1, row;

    for (int i = 0; i < rows / 2; i++) {
        row = i;
        
        temp[2 + row * columns + j] = -1 * H1t1Vec[1 + row];
        temp[2 + row * columns + (j+N+1)] = 1 * H1t1Vec[1 + row];
        
        row = i+rows/2;

        temp[2 + row * columns + k] = 1 * H1t1Vec[1 + row];
        temp[2 + row * columns + (k+1)] = -1 * H1t1Vec[1 + row];
        
        j += ((i + 1) % (N - 1) == 0 ? 3 : 1);
        k += ((i + 1) % N == 0 ? 2 : 1);
    }
    
    float *C2 = mallocMatrix(temp[0], C0[1]);
    
    // C2 <- temp * C0
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, temp[0], C0[1], temp[1], 1.0, &temp[2], temp[1], &C0[2], C0[1], 0.0, &C2[2], C2[1]);

    return C2;
}

float* generateUPresFast(float *H1t1Vec, float *E21, float *Ht02Vec, float *E21K, float *uK)
{
    float *temp1 = mallocVector(E21K[0]);
    float *temp2 = mallocVector(Ht02Vec[0]);
    float *temp3 = mallocVector(E21[1]);
    float *uPres = mallocVector(H1t1Vec[0]);
    
    // temp1 <- E21K * uK
    cblas_sgemv(CblasRowMajor, CblasNoTrans, E21K[0], E21K[1], 1.0, &E21K[2], E21K[1], &uK[1], 1, 0.0, &temp1[1], 1);
    
    // temp2 <- Ht02 * temp1    
    cblas_ssbmv(CblasRowMajor, CblasLower, Ht02Vec[0], 0, 1.0, &Ht02Vec[1], 1, &temp1[1], 1, 0.0, &temp2[1], 1);
    
    // temp3 <- tE10 * temp2 = E21' * temp2
    cblas_sgemv(CblasRowMajor, CblasTrans, E21[0], E21[1], 1.0, &E21[2], E21[1], &temp2[1], 1, 0.0, &temp3[1], 1);
    
    // uPres <- H1t1 * temp3
    cblas_ssbmv(CblasRowMajor, CblasLower, H1t1Vec[0], 0, 1.0, &H1t1Vec[1], 1, &temp3[1], 1, 0.0, &uPres[1], 1);
        
    free(temp1);
    free(temp2);
    free(temp3);

    return uPres;
}
