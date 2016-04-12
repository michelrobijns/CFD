#ifndef SPARSE_H_INCLUDED
#define SPARSE_H_INCLUDED

typedef struct {
    unsigned int rows;
    unsigned int columns;
    unsigned int NNZ;
    float *val;
    unsigned int *col_idx;
    unsigned int *row_ptr;
} matrix;

typedef struct {
    unsigned int rows;
    float *val;
} vector;

sparseMatrix* mallocMatrix(int rows, int columns, int NNZ);
sparseMatrix* callocMatrix(int rows, int columns, int NNZ);
vector* mallocVector_2(int rows);
vector* callocVector_2(int rows);

#endif
