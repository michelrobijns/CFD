#ifndef MATRICES_H_INCLUDED
#define MATRICES_H_INCLUDED

double* generateTE21(int N);
double* generateE21K(int N);
double* generateE21(int N);
double* generateH1t1(int N, double *th);
double* generateHt02(int N, double *h);
double* generateA(double *Ht11, double *tE21);
double* generateC2(double *H1t1, double *E21, double *C0);
double* generateUPres(double *H1t1, double *E21, double *Ht02, double *E21K, double *uK);
double* generateUK(int N, double *h);
double* combineUAndUK(int N, double *u, double *uK);

void factorizeA(int N, double *A);
void updateC4(double *C4, double *C2, double *u, double *C3, double *convective, double Re);
void updateXi(double *xi, double *C0, double *u);
void updateConvective(double *convective, int N, double *xi, double *u, double *uK, double *h);
void updateRhs(double *rhs, double *C1, double *u, double *C4, double dt, double *temp);
void updateP(double *P, int N, double *A, double *rhs);
void updateU(double *u, double *tE21, double *P, double *C4, double dt);
void updateDiff(double *diff, double *u, double *uOld, double dt);

void storeStreamFunction(int N, double *Ht11, double *u, double *tx);
void storeVorticity(int N, double *xi, double *tx);
void storePressure(int N, double *x, double *h, double *u, double *uK, double *P);

#endif
