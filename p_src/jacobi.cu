#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>
#include <cmath>

// Host version of the Jacobi method
void jacobiOnHost(double *x_next, double *A, double *x_now, double *b, int Ni, int Nj) {
    float sigma;
    for (int i = 0; i < Ni; i++) {
        sigma = 0.0;
        for (int j = 0; j < Nj; j++) {
            if (i != j)
                sigma += A[i * Nj + j] * x_now[j];
        }
        x_next[i] = (b[i] - sigma) / A[i * Nj + i];
    }
}

// device version of the Jacobi method
__global__ void kernel(double *x_next, double *A, double *x_now, double *b, int Ni, int Nj) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Ni) {
        float sigma = 0.0;

        int idx_Ai = idx * Nj;

        for (int j = 0; j < Nj; j++)
            if (idx != j)
                sigma += A[idx_Ai + j] * x_now[j];

        x_next[idx] = (b[idx] - sigma) / A[idx_Ai + idx];
    }
}

void fill_by_null(double *arr, int size) {
    for (int i = 0; i < size; i++)
        arr[i] = 0.;
}

void solve(char *fname) {
    double *x_prev, *x, *A, *b, *x_h, *x_result;
    double *x_prev_d, *x_d, *A_d, *b_d;
    int NX = 512, NY = 512, iter = 10000, blockSize = 4;
    int N = NX * NY;

    printf("Parameters:\nN=%d, NX=%d, NY=%d, iterations=%d\n", N, NX, NY, iter);

    x = (double *) malloc(NX * sizeof(double));
    x_prev = (double *) malloc(NX * sizeof(double));
    x_h = (double *) malloc(NX * sizeof(double));
    x_result = (double *) malloc(NX * sizeof(double));

    A = (double *) malloc(N * sizeof(double));
    b = (double *) malloc(NX * sizeof(double));

    fill_by_null(x_prev, NX);
    fill_by_null(x, NX);

    // Read coefficient matrix from file
    FILE *file = fopen(fname, "r");
    if (file == NULL) {
	exit(EXIT_FAILURE);
    }
    char *line;
    size_t len = 0;
    int i = 0;
    while ((getline(&line, &len, file)) != -1) {
        if (i < N)
            A[i] = atof(line);
        else
            b[i - N] = atof(line);
        i++;
    }

    for (int k = 0; k < iter; k++) {
        if (k % 2)
            jacobiOnHost(x_prev, A, x, b, NX, NY);
        else
            jacobiOnHost(x, A, x_prev, b, NX, NY);
    }

    memcpy(x_h, x, NY * sizeof(double));

    fill_by_null(x_prev, NX);
    fill_by_null(x, NX);

    assert(cudaSuccess == cudaMalloc((void **) &x_d, NX * sizeof(double)));
    assert(cudaSuccess == cudaMalloc((void **) &A_d, N * sizeof(double)));
    assert(cudaSuccess == cudaMalloc((void **) &x_prev_d, NX * sizeof(double)));
    assert(cudaSuccess == cudaMalloc((void **) &b_d, NX * sizeof(double)));

    cudaMemcpy(x_d, x, sizeof(double) * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(x_prev_d, x_prev, sizeof(double) * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(double) * NX, cudaMemcpyHostToDevice);

    int nTiles = NX / blockSize + (NX % blockSize == 0 ? 0 : 1);
    int gridHeight = NY / blockSize + (NY % blockSize == 0 ? 0 : 1);
    int gridWidth = NX / blockSize + (NX % blockSize == 0 ? 0 : 1);
    printf("gridWidth = %d, gridHeight = %d\n", gridWidth, gridHeight);
    dim3 dGrid(gridHeight, gridWidth), dBlock(blockSize, blockSize);

    for (int k = 0; k < iter; k++) {
        if (k % 2)
            kernel << < nTiles, blockSize >> > (x_prev_d, A_d, x_d, b_d, NX, NY);
        else
            kernel << < nTiles, blockSize >> > (x_d, A_d, x_prev_d, b_d, NX, NY);
    }

    // Data <- device
    cudaMemcpy(x_result, x_d, sizeof(double) * NX, cudaMemcpyDeviceToHost);

    printf("\nResult after %d iterations:\n", iter);
    double err = 0.0;
    for (i = 0; i < NX; i++) err += fabs(x_h[i] - x_result[i]);
    printf("Relative error: %f\n", err);

    // Free memory
    free(x);
    free(A);
    free(x_prev);
    free(b);
    free(x_h);
    cudaFree(x_d);
    cudaFree(A_d);
    cudaFree(x_prev_d);
    cudaFree(b_d);
    cudaFree(x_result);
}

int main() {
    solve("input.icu");
    return 1;
}