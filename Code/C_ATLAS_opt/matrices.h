#ifndef MATRICES_H_INCLUDED
#define MATRICES_H_INCLUDED

float* generateTE21(int N);
float* generateE21K(int N);
float* generateE21(int N);
float* generateH1t1(int N, float *th);
float* generateHt02(int N, float *h);
float* generateA(float *Ht11, float *tE21);
float* generateC2(float *H1t1, float *E21, float *C0);
float* generateUPres(float *H1t1, float *E21, float *Ht02, float *E21K, float *uK);
float* generateUK(int N, float *h);
float* combineUAndUK(int N, float *u, float *uK);

void factorizeA(int N, float *A);
void updateC4(float *C4, float *C2, float *u, float *C3, float *convective, float Re);
void updateXi(float *xi, float *C0, float *u);
void updateConvective(float *convective, int N, float *xi, float *u, float *uK, float *h);
void updateRhs(float *rhs, float *C1, float *u, float *C4, float dt, float *temp);
void updateP(float *P, int N, float *A, float *rhs);
void updateU(float *u, float *tE21, float *P, float *C4, float dt);
void updateDiff(float *diff, float *u, float *uOld, float dt);

void storeStreamFunction(int N, float *Ht11, float *u, float *tx);
void storeVorticity(int N, float *xi, float *tx);
void storePressure(int N, float *x, float *h, float *u, float *uK, float *P);

#endif
