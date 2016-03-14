#ifndef MATRICES_H_INCLUDED
#define MATRICES_H_INCLUDED

double** generateTE21(int N, double **tE21, int *rows, int *columns);
double** generateE21K(int N, double **E21K, int *rows, int *columns);
double** generateE21(int N, double **E21, int *rows, int *columns);
double** generateH1t1(int N, double **H1t1, int *rows, int *columns,
                      double *th);
double** generateHt02(int N, double **Ht02, int *rows, int *columns,
                      double *h);
double* generateConvective(int N, double *xi, double *u, double *uK,
                           double *h);
double* combineUAndUK(int N, double *u, double *uK);
double* generateU(int N, int *rowsU);
double* generateUOld(int N);
double* generateUK(int N, int *rowsUK, double *h);
double* generateC4(double **C2, int rowsC2, int columnsC2, double *u,
                   int rowsU, double *C3, int rowsC3, double Re);
double* generateRhs(double **C1, int rowsC1, int columnsC1, double *u,
                    int rowsU, double *convective, double *C4, double dt);
void updateU(double **E10, int rowsE10, int columnsE10, double *u, int rowsU,
             double *convective, double *P, double *C4, double dt);
double computeDiff(double *u, int rowsU, double *uOld, double dt);

#endif
