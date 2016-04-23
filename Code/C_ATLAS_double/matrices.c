#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
//#include <omp.h>
#include "matrices.h"
#include "linearAlgebra.h"

double* generateTE21(int N)
{
    int rows = N * N;
    int columns = 2 * N * (N-1);
        
    double *tE21 = callocMatrix(rows, columns);
        
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

double* generateE21K(int N)
{
    int rows = (N + 1) * (N + 1);
    int columns = 4 * (N + 1) + 4 * N;
        
    double *E21K = callocMatrix(rows, columns);
    
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

double* generateE21(int N)
{
    int rows = (N + 1) * (N + 1);
    int columns = 2 * N * (N - 1);

    double *E21 = callocMatrix(rows, columns);

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

double* generateH1t1(int N, double *th)
{
    int rows = 2 * N * (N - 1);
    int columns = 2 * N * (N - 1);

    double *H1t1 = callocMatrix(rows, columns);
    
    double velocity, length;
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

double* generateHt02(int N, double *h)
{
    int rows = (N + 1) * (N + 1);
    int columns = (N + 1) * (N + 1);

    double *Ht02 = callocMatrix(rows, columns);
    
    int k;

    for (int i = 0; i < N + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            k = j + (N + 1) * i;

            Ht02[2 + k * columns + k] = 1 / (h[1 + i] * h[1 + j]);
        }
    }

    return Ht02;
}

double* generateA(double *Ht11, double *tE21)
{
    double *temp = callocMatrix(Ht11[0], tE21[0]);
    
    temp[0] = Ht11[0];
    temp[1] = tE21[0];
    
    // temp = Ht11 * E10 = Ht11 * -tE21'
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Ht11[0], tE21[0], Ht11[1], -1.0, &Ht11[2], Ht11[1], &tE21[2], tE21[1], 0.0, &temp[2], temp[1]);
    
    double *A = mallocMatrix(tE21[0], temp[1]);
    
    // A = tE21 * temp
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, tE21[0], temp[1], tE21[1], 1.0, &tE21[2], tE21[1], &temp[2], temp[1], 0.0, &A[2], A[1]);
    
    return A;
}

double* generateC2(double *H1t1, double *E21, double *C0)
{
    double *temp = callocMatrix(E21[1], C0[1]);
    
    temp[0] = E21[1];
    temp[1] = C0[1];
    
    // temp = tE10 * C0 = E21' * C0
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, E21[1], C0[1], E21[0], 1.0, &E21[2], E21[1], &C0[2], C0[1], 0.0, &temp[2], temp[1]);
    
    double *C2 = mallocMatrix(H1t1[0], temp[1]);
        
    // C2 = H1t1 * temp
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, H1t1[0], temp[1], H1t1[1], 1.0, &H1t1[2], H1t1[1], &temp[2], temp[1], 0.0, &C2[2], C2[1]);
    
    return C2;
}

double* generateUPres(double *H1t1, double *E21, double *Ht02, double *E21K, double *uK)
{
    double *temp = callocVector(E21K[0]);
    double *uPres = callocVector(E21[1]);
    int rows, columns;
    
    // temp = E21K * uK
    cblas_dgemv(CblasRowMajor, CblasNoTrans, E21K[0], E21K[1], 1.0, &E21K[2], E21K[1], &uK[1], 1, 0.0, &temp[1], 1);
    
    // temp := Ht02 * temp
    rows = temp[0];
    columns = temp[0];
    
    for (int i = 0; i < rows; i++) {
        temp[1 + i] *= Ht02[2 + i * columns + i];
    }
    
    // uPres = tE10 * temp = E21' * temp
    cblas_dgemv(CblasRowMajor, CblasTrans, E21[0], E21[1], 1.0, &E21[2], E21[1], &temp[1], 1, 0.0, &uPres[1], 1);
    
    // uPres := H1t1 * uPres
    rows = uPres[0];
    columns = uPres[0];
    
    for (int i = 0; i < rows; i++) {
        uPres[1 + i] *= H1t1[2 + i * columns + i];
    }
    
    return uPres;
}

double* generateUK(int N, double *h)
{
    int rows = 4 * (N + 1) + 4 * N;
    
    double *uK = callocVector(rows);

    for (int i = 0; i < N + 1; i++) {
        uK[1 + i+N*2+N+1] = -1 * h[1 + i];
    }

    return uK;
}

double* combineUAndUK(int N, double *u, double *uK)
{
    double *uTotal = callocVector(2 * (N + 1) * (N + 2));

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

void factorizeA(int N, double *A)
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

void updateC4(double *C4, double *C2, double *u, double *C3, double *convective, double Re)
{
    // C4 = C2 * (u / Re) + C3 + convective
    
    // convective <- C3 + convective
    cblas_daxpy(C3[0], 1.0, &C3[1], 1, &convective[1], 1);
    
    // convective <- (1.0 / Re) * C2 * u + convective
    cblas_dgemv(CblasRowMajor, CblasNoTrans, C2[0], C2[1], 1.0 / Re, &C2[2], C2[1], &u[1], 1, 1.0, &convective[1], 1);
    
    // REDUNDANT
    // C4 <- convective
    cblas_dcopy(convective[0], &convective[1], 1, &C4[1], 1);
}

void updateXi(double *xi, double *C0, double *u)
{
    // xi <- C0 * u
    cblas_dgemv(CblasRowMajor, CblasNoTrans, C0[0], C0[1], 1.0, &C0[2], C0[1], &u[1], 1, 0.0, &xi[1], 1);
}

void updateConvective(double *convective, int N, double *xi, double *u, double *uK, double *h)
{
    double *uTotal = combineUAndUK(N, u, uK);
    
    double V1, V2, U1, U2;
    
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

void updateRhs(double *rhs, double *C1, double *u, double *C4, double dt, double *temp)
{
    // -rhs = C1 * (-u / dt + C4)
        
    // temp <- C4
    cblas_dcopy(C4[0], &C4[1], 1, &temp[1], 1);
    
    // temp <- (-1.0 / dt) * u + C4
    cblas_daxpy(u[0], -1.0 / dt, &u[1], 1, &temp[1], 1);
            
    // -rhs <- C1 * temp
    cblas_dgemv(CblasRowMajor, CblasNoTrans, C1[0], C1[1], 1.0, &C1[2], C1[1], &temp[1], 1, 0.0, &rhs[1], 1);
    
    // So this will produce -rhs
}

void updateP(double *P, int N, double *A, double *rhs)
{    
    int rows = A[0];
    int columns = A[1];
    double sum;
        
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

void updateU(double *u, double *tE21, double *P, double *C4, double dt)
{
    // u = u - dt * (-tE21' * P + C4)
        
    // C4 <- -tE21' * P + C4
    cblas_dgemv(CblasRowMajor, CblasTrans, tE21[0], tE21[1], -1.0, &tE21[2], tE21[1], &P[1], 1, 1.0, &C4[1], 1);
    
    // u = - dt * C4 + u
    cblas_daxpy(C4[0], -1.0 * dt, &C4[1], 1, &u[1], 1);
}

// void updateUWithModifiedEuler(double *u, double *tE21, double *P, double *C4, double dt)
// {
//     // u = u - dt * (-tE21' * P + C4)
//         
//     // C4 <- -tE21' * P + C4
//     cblas_sgemv(CblasRowMajor, CblasTrans, tE21[0], tE21[1], -1.0, &tE21[2], tE21[1], &P[1], 1, 1.0, &C4[1], 1);
//     
//     // u = - dt * C4 + u
//     cblas_saxpy(C4[0], -1.0 * dt, &C4[1], 1, &u[1], 1);
// }

void updateDiff(double *diff, double *u, double *uOld, double dt)
{
    double difference = 0;
    int rows = u[0];
    
    //#pragma omp parallel for reduction(max:difference)
    for (int i = 0; i < rows; i++) {
        double value = fabs((u[1 + i] - uOld[1 + i]) / dt);
        
        if (value > difference) {
            difference = value;
        }
    }
    
    *diff = difference;
}

void storeStreamFunction(int N, double *Ht11, double *u, double *tx)
{
    int rows = N + 1;
    int columns = N + 1;
    
    double *flux = callocVector(u[0]);
    
    for (int i = 0; i < u[0]; i++) {
        flux[1 + i] = Ht11[1 + i] * u[1 + i];
    }
        
    double *streamFunction = callocMatrix(rows, columns);
    double *X = callocMatrix(rows, columns);
    double *Y = callocMatrix(rows, columns);
        
    for (int i = 1; i < N; i++) {
        for (int j = 1; j < N; j++) {
            int k = (j - 1) + (i - 1) * (N - 1);
            
            streamFunction[2 + (N - i) * columns + j] = flux[1 + k] + streamFunction[2 + (N - i + 1) * columns + j];
        }
    }
    
    for (int i = 0; i < N + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            X[2 + (N - i) * columns + j] = tx[1 + j];
            Y[2 + (N - i) * columns + j] = tx[1 + i];
        }
    }
    
    storeMatrix(streamFunction, "streamFunction.dat");
    storeMatrix(X, "streamFunctionX.dat");
    storeMatrix(Y, "streamFunctionY.dat");
    
    free(flux);
    free(streamFunction);
    free(X);
    free(Y);
}

void storeVorticity(int N, double *xi, double *tx)
{
    int rows = N + 1;
    int columns = N + 1;
        
    double *vorticity = callocMatrix(rows, columns);
    double *X = callocMatrix(rows, columns);
    double *Y = callocMatrix(rows, columns);
    
    for (int i = 0; i < N + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            int k = j + i * (N + 1);
            
            vorticity[2 + (N - i) * columns + j] = xi[1 + k];
            
            X[2 + (N - i) * columns + j] = tx[1 + j];
            Y[2 + (N - i) * columns + j] = tx[1 + i];
        }
    }
    
    storeMatrix(vorticity, "vorticity.dat");
    storeMatrix(X, "vorticityX.dat");
    storeMatrix(Y, "vorticityY.dat");
    
    free(vorticity);
    free(X);
    free(Y);
}

void storePressure(int N, double *x, double *h, double *u, double *uK, double *P)
{
    int rows = N;
    int columns = N;
        
    double *pressure = callocMatrix(rows, columns);
    double *X = callocMatrix(rows, columns);
    double *Y = callocMatrix(rows, columns);
    
    double *uTotal = combineUAndUK(N, u, uK);
    
    for (int i = 1; i < N + 1; i++) {
        for (int j = 1; j < N + 1; j++) {
            int k = (j - 1) + (i - 1) * N;
            int l = j + i * (N + 1);
            int m = j + i * (N + 2) + (N + 1) * (N + 2);

            double uAverage = 0.5 * (uTotal[1+l] / h[1+j] + uTotal[1+l-1] / h[1+j-1]);
            double vAverage = 0.5 * (uTotal[1+m] / h[1+i] + uTotal[1+m-(N+2)] / h[1+i-1]);
            
            double velocity = pow(uAverage, 2) + pow(vAverage, 2);
            
            pressure[2 + (N - i) * columns + j - 1] = P[1 + k] - 0.5 * pow(sqrt(velocity), 2);
            
            X[2 + (N - i) * columns + j - 1] = x[1 + j];
            Y[2 + (N - i) * columns + j - 1] = x[1 + i];
        }
    }
    
    double constant = pressure[2 + (int) floor(N / 2) * columns + (int) floor(N / 2)];
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            pressure[2 + i * columns + j] -= constant;
        }
    }
    
    storeMatrix(pressure, "pressure.dat");
    storeMatrix(X, "pressureX.dat");
    storeMatrix(Y, "pressureY.dat");
    
    free(pressure);
    free(X);
    free(Y);
}
