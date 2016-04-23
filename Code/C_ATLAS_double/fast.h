#ifndef FAST_H_INCLUDED
#define FAST_H_INCLUDED

double* generateH1t1Vec(int N, double *th);
double* generateHt02Vec(int N, double *h);

double* generateAFast(int N, double *Ht11Vec);
double* generateC0Fast(int N, double *Ht02Vec);
double* generateC1Fast(int N, double *Ht11Vec);
double* generateC2Fast(int N, double *H1t1Vec, double *C0);
double* generateUPresFast(double *H1t1Vec, double *E21, double *Ht02Vec, double *E21K, double *uK);

#endif
