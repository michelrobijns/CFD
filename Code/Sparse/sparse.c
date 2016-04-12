#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
//#include <omp.h>
#include "matrices.h"
#include "linearAlgebra.h"
#include "sparse.h"

matrix* mallocMatrix(int rows, int columns, int NNZ)
{
    if (rows <= 0 || columns <= 0 || NNZ <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
        
    matrix *A = malloc(sizeof(matrix));
    
    if (A == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    A->val = malloc(NNZ * sizeof(float));
    
    if (A->val == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    A->col_idx = malloc(NNZ * sizeof(int));
    
    if (A->col_idx == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    A->row_ptr = malloc(rows * sizeof(int));
    
    if (A->row_ptr == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    // Set dimensions
    A->rows = rows;
    A->columns = columns;
    A->NNZ = NNZ;
    
    return A;
}

matrix* callocMatrix(int rows, int columns, int NNZ)
{
    if (rows <= 0 || columns <= 0 || NNZ <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
        
    matrix *A = calloc(1, sizeof(matrix));
    
    if (A == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    A->val = calloc(NNZ, sizeof(float));
    
    if (A->val == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    A->col_idx = calloc(NNZ, sizeof(int));
    
    if (A->col_idx == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    A->row_ptr = calloc(rows, sizeof(int));
    
    if (A->row_ptr == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    // Set dimensions
    A->rows = rows;
    A->columns = columns;
    A->NNZ = NNZ;
    
    return A;
}

vector* mallocVector_2(int rows)
{
    if (rows <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
    
    vector *x = malloc(sizeof(vector));
    
    if (x == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    x->val = malloc((1 + rows) * sizeof(float));
    
    if (x->val == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    // Set dimensions
    x->rows = rows;
    
    return x;
}

vector* callocVector_2(int rows)
{
    if (rows <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
    
    vector *x = calloc(1, sizeof(vector));
    
    if (x == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    x->val = calloc(1 + rows, sizeof(float));
    
    if (x->val == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    // Set dimensions
    x->rows = rows;
    
    return x;
}
