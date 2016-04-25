#ifndef FAST_H_INCLUDED
#define FAST_H_INCLUDED

float* generateH1t1Vec(int N, float *th);
float* generateHt02Vec(int N, float *h);

float* generateAFast(int N, float *Ht11Vec);
float* generateC0Fast(int N, float *Ht02Vec);
float* generateC1Fast(int N, float *Ht11Vec);
float* generateC2Fast(int N, float *H1t1Vec, float *C0);
float* generateUPresFast(float *H1t1Vec, float *E21, float *Ht02Vec, float *E21K, float *uK);

#endif
