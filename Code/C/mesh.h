#ifndef MESH_H_INCLUDED
#define MESH_H_INCLUDED

double* generateTx(int N);
double* generateTh(int N);
double* generateX(int N);
double* generateH(int N);
void checkMesh(int N, double *tx, double *th, double *x, double *h);

#endif
