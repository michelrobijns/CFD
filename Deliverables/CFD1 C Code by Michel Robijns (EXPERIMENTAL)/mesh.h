#ifndef MESH_H_INCLUDED
#define MESH_H_INCLUDED

float* generateTx(int N);
float* generateTh(int N);
float* generateX(int N);
float* generateH(int N);
void checkMesh(int N, float *tx, float *th, float *x, float *h);

#endif
