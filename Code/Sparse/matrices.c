#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
//#include <omp.h>
#include "matrices.h"
#include "linearAlgebra.h"

float* generateTE21(int N)
{
    int rows = N * N;
    int columns = 2 * N * (N-1);
        
    float *tE21 = callocMatrix(rows, columns);
        
    int i = 0, k = 0;

    for (int j = 0; j < columns / 2; j++) {
        tE21[2 + i * columns + j] = 1;
        tE21[2 + (i+1) * columns + j] = -1;

        tE21[2 + k * columns + (j+columns/2)] = 1;
        tE21[2 + (k+N) * columns + (j+columns/2)] = -1;

        i += ((j + 1) % (N - 1) == 0 ? 2 : 1);
        k++;
    }
    
    return tE21;
}

float* generateE21K(int N)
{
    int rows = (N + 1) * (N + 1);
    int columns = 4 * (N + 1) + 4 * N;
        
    float *E21K = callocMatrix(rows, columns);
    
    int i = 0, k = rows - 1;

    for (int j = 0; j < N + 1; j++) {
        E21K[2 + i * columns + j] = 1;
        E21K[2 + k * columns + columns / 2 - 1 - j] = -1;

        E21K[2 + i * columns + (j+columns/2)] = -1;
        E21K[2 + i * columns + (j+columns/2+1)] = 1;

        E21K[2 + k * columns + (columns-j-1)] = 1;
        E21K[2 + k * columns + (columns-j-2)] = -1;

        i++;
        k--;
    }

    i = 0;

    for (int j = N + 1; j < columns / 2 - (N + 1); j++) {
        E21K[2 + i * columns + j] = -1;
        E21K[2 + (i+N+1) * columns + j] = 1;

        i += ((j - N - 1) % 2 == 0 ? N : 1);
    }

    i = N + 1;

    for (int j = columns / 2 + N + 2; j < columns - (N + 2); j += 2) {
        E21K[2 + i * columns + j] = -1;
        E21K[2 + (i+N) * columns + (j+1)] = 1;

        i += N + 1;
    }

    return E21K;
}

float* generateE21(int N)
{
    int rows = (N + 1) * (N + 1);
    int columns = 2 * N * (N - 1);

    float *E21 = callocMatrix(rows, columns);

    int i = 1, k = N + 1;

    for (int j = 0; j < columns / 2; j++) {
        E21[2 + i * columns + j] = -1;
        E21[2 + (i+N+1) * columns + j] = 1;

        E21[2 + k * columns + (j+columns/2)] = 1;
        E21[2 + (k+1) * columns + (j+columns/2)] = -1;
        
        i += ((j + 1) % (N - 1) == 0 ? 3 : 1);
        k += ((j + 1) % N == 0 ? 2 : 1);
    }

    return E21;
}

float* generateH1t1(int N, float *th)
{
    int rows = 2 * N * (N - 1);
    int columns = 2 * N * (N - 1);

    float *H1t1 = callocMatrix(rows, columns);
    
    float velocity, length;
    int k;

    for (int j = 0; j < N; j++) {
        velocity = 1 / th[1 + j];

        for (int i = 0; i < N - 1; i++) {
            length = 0.5 * (th[1 + i] + th[1 + i+1]);

            k = i + (N - 1) * j;

            H1t1[2 + k * columns + k] = velocity * length;
        }
    }

    for (int j = 0; j < N - 1; j++) {
        length = 0.5 * (th[1 + j] + th[1 + j+1]);

        for (int i = 0; i < N; i++) {
            velocity = 1 / th[1 + i];

            k = rows / 2 + i + N * j;

            H1t1[2 + k * columns + k] = velocity * length;
        }
    }

    return H1t1;
}

float* generateHt02(int N, float *h)
{
    int rows = (N + 1) * (N + 1);
    int columns = (N + 1) * (N + 1);

    float *Ht02 = callocMatrix(rows, columns);
    
    int k;

    for (int i = 0; i < N + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            k = j + (N + 1) * i;

            Ht02[2 + k * columns + k] = 1 / (h[1 + i] * h[1 + j]);
        }
    }

    return Ht02;
}

float* generateA(float *Ht11, float *tE21)
{
    float *temp = callocMatrix(Ht11[0], tE21[0]);
    
    temp[0] = Ht11[0];
    temp[1] = tE21[0];
    
    // temp = Ht11 * E10 = Ht11 * -tE21'
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Ht11[0], tE21[0], Ht11[1], -1.0, &Ht11[2], Ht11[1], &tE21[2], tE21[1], 0.0, &temp[2], temp[1]);
    
    float *A = mallocMatrix(tE21[0], temp[1]);
    
    // A = tE21 * temp
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, tE21[0], temp[1], tE21[1], 1.0, &tE21[2], tE21[1], &temp[2], temp[1], 0.0, &A[2], A[1]);
    
    return A;
}

float* generateC2(float *H1t1, float *E21, float *C0)
{
    float *temp = callocMatrix(E21[1], C0[1]);
    
    temp[0] = E21[1];
    temp[1] = C0[1];
    
    // temp = tE10 * C0 = E21' * C0
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, E21[1], C0[1], E21[0], 1.0, &E21[2], E21[1], &C0[2], C0[1], 0.0, &temp[2], temp[1]);
    
    float *C2 = mallocMatrix(H1t1[0], temp[1]);
        
    // C2 = H1t1 * temp
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, H1t1[0], temp[1], H1t1[1], 1.0, &H1t1[2], H1t1[1], &temp[2], temp[1], 0.0, &C2[2], C2[1]);
    
    return C2;
}

float* generateUPres(float *H1t1, float *E21, float *Ht02, float *E21K, float *uK)
{
    float *temp = callocVector(E21K[0]);
    float *uPres = callocVector(E21[1]);
    int rows, columns;
    
    // temp = E21K * uK
    cblas_sgemv(CblasRowMajor, CblasNoTrans, E21K[0], E21K[1], 1.0, &E21K[2], E21K[1], &uK[1], 1, 0.0, &temp[1], 1);
    
    // temp := Ht02 * temp
    rows = temp[0];
    columns = temp[0];
    
    for (int i = 0; i < rows; i++) {
        temp[1 + i] *= Ht02[2 + i * columns + i];
    }
    
    // uPres = tE10 * temp = E21' * temp
    cblas_sgemv(CblasRowMajor, CblasTrans, E21[0], E21[1], 1.0, &E21[2], E21[1], &temp[1], 1, 0.0, &uPres[1], 1);
    
    // uPres := H1t1 * uPres
    rows = uPres[0];
    columns = uPres[0];
    
    for (int i = 0; i < rows; i++) {
        uPres[1 + i] *= H1t1[2 + i * columns + i];
    }
    
    return uPres;
}

float* generateUK(int N, float *h)
{
    int rows = 4 * (N + 1) + 4 * N;
    
    float *uK = callocVector(rows);

    for (int i = 0; i < N + 1; i++) {
        uK[1 + i+N*2+N+1] = -1 * h[1 + i];
    }

    return uK;
}

float* combineUAndUK(int N, float *u, float *uK)
{
    float *uTotal = callocVector(2 * (N + 1) * (N + 2));

    for (int i = 0; i < N + 2; i++) {
        uTotal[1 + (N+1)*(N+2)-1-i] = uK[1 + 2*(N+1)+N*2-1-i];
    }

    int k = N + 2;

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N - 1; i++) {
            uTotal[1 + k] = u[1 + i+(N-1)*j];
            k++;
        }
        
        k += 2;
    }

    k = (N + 1) * (N + 2) + N + 3;

    for (int j = 0; j < N - 1; j++) {
        for (int i = 0; i < N; i++) {
            uTotal[1 + k] = u[1 + i+N*j+N*(N-1)];
            k++;
        }
        
        k += 2;
    }

    return uTotal;
}

void factorizeA(int N, float *A)
{    
    int rows = A[0];
    int columns = A[1];
    
    for (int c = 0; c < columns - 1; c++) {
        for (int r = c + 1; r < (c+1+N < rows ? c+1+N : rows); r++) {
            A[2 + r * columns + c] = A[2 + r * columns + c] / A[2 + c * columns + c];

            for (int cc = c + 1; cc < (c+1+N < columns ? c+1+N : columns); cc++) {
                A[2 + r * columns + cc] = A[2 + r * columns + cc] - A[2 + r * columns + c] * A[2 + c * columns + cc];
            }
        }
    }
}

void updateC4(float *C4, float *C2, float *u, float *C3, float *convective, float Re)
{
    // C4 = C2 * (u / Re) + C3 + convective
    
    // convective <- C3 + convective
    cblas_saxpy(C3[0], 1.0, &C3[1], 1, &convective[1], 1);
    
    // convective <- (1.0 / Re) * C2 * u + convective
    cblas_sgemv(CblasRowMajor, CblasNoTrans, C2[0], C2[1], 1.0 / Re, &C2[2], C2[1], &u[1], 1, 1.0, &convective[1], 1);
    
    // REDUNDANT
    // C4 <- convective
    cblas_scopy(convective[0], &convective[1], 1, &C4[1], 1);
}

void updateXi(float *xi, float *C0, float *u)
{
    // xi <- C0 * u
    cblas_sgemv(CblasRowMajor, CblasNoTrans, C0[0], C0[1], 1.0, &C0[2], C0[1], &u[1], 1, 0.0, &xi[1], 1);
}

void updateConvective(float *convective, int N, float *xi, float *u, float *uK, float *h)
{
    float *uTotal = combineUAndUK(N, u, uK);
    
    float V1, V2, U1, U2;
    
    int offset = (N + 1) * (N + 2), k, l;
    
    //#pragma omp parallel for private(k, l, V1, V2, U1, U2)
    for (int i = 1; i < N; i++) {
        for (int j = 0; j < N; j++) {
            k = (i - 1) + (N - 1) * j;
            l = N * (N - 1) + j + N * (i - 1);
            
            V1 = uTotal[1 + offset+i+j*(N+2)] + uTotal[1 + offset+(i+1)+j*(N+2)];
            V2 = uTotal[1 + offset+i+(j+1)*(N+2)] + uTotal[1 + offset+(i+1)+(j+1)*(N+2)];
            U1 = uTotal[1 + j+i*(N+1)] + uTotal[1 + j+(i+1)*(N+1)];
            U2 = uTotal[1 + (j+1)+i*(N+1)] + uTotal[1 + (j+1)+(i+1)*(N+1)];

            convective[1 + k] = - h[1 + i] / (4 * h[1 + j]) * V1 * xi[1 + i+j*(N+1)] - h[1 + i] / (4 * h[1 + j+1]) * V2 * xi[1 + i+(j+1)*(N+1)];
            convective[1 + l] = h[1 + i] / (4 * h[1 + j]) * U1 * xi[1 + j+i*(N+1)] + h[1 + i] / (4 * h[1 + j+1]) * U2 * xi[1 + (j+1)+i*(N+1)];
        }
    }
}

void updateRhs(float *rhs, float *C1, float *u, float *C4, float dt, float *temp)
{
    // -rhs = C1 * (-u / dt + C4)
        
    // temp <- C4
    cblas_scopy(C4[0], &C4[1], 1, &temp[1], 1);
    
    // temp <- (-1.0 / dt) * u + C4
    cblas_saxpy(u[0], -1.0 / dt, &u[1], 1, &temp[1], 1);
            
    // -rhs <- C1 * temp
    cblas_sgemv(CblasRowMajor, CblasNoTrans, C1[0], C1[1], 1.0, &C1[2], C1[1], &temp[1], 1, 0.0, &rhs[1], 1);
    
    // So this will produce -rhs
}

void updateP(float *P, int N, float *A, float *rhs)
{    
    int rows = A[0];
    int columns = A[1];
    float sum;
        
    // Forward substitution step
    for (int r = 0; r < rows; r++)    {
        sum = 0;

        //#pragma omp parallel for reduction(+:sum)
        for (int c = (r - N < 0 ? 0 : r - N); c < r; c++)
            sum += A[2 + r * columns + c] * P[1 + c];

        P[1 + r] = - rhs[1 + r] - sum;
    }

    // Backward substitution step
    for (int r = rows - 1; r >= 0; r--) {
        sum = 0;

        //#pragma omp parallel for reduction(+:sum)
        for (int c = r + 1; c < (r + N + 1 > columns ? columns : r + N + 1); c++)
            sum += A[2 + r * columns + c] * P[1 + c];

        P[1 + r] = (P[1 + r] - sum) / A[2 + r * columns + r];
    }
}

void updateU(float *u, float *tE21, float *P, float *C4, float dt)
{
    // u = u - dt * (-tE21' * P + C4)
        
    // C4 <- -tE21' * P + C4
    cblas_sgemv(CblasRowMajor, CblasTrans, tE21[0], tE21[1], -1.0, &tE21[2], tE21[1], &P[1], 1, 1.0, &C4[1], 1);
    
    // u = - dt * C4 + u
    cblas_saxpy(C4[0], -1.0 * dt, &C4[1], 1, &u[1], 1);
}

void updateDiff(float *diff, float *u, float *uOld, float dt)
{
    float difference = 0;
    int rows = u[0];
    
    //#pragma omp parallel for reduction(max:difference)
    for (int i = 0; i < rows; i++) {
        float value = fabs((u[1 + i] - uOld[1 + i]) / dt);
        
        if (value > difference) {
            difference = value;
        }
    }
    
    *diff = difference;
}
