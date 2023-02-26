#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

__global__ void matrixMul(float* A, float* B, float* C, int rowDimA, int colDimA, int colDimB) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float cValue = 0;

    for (int t = 0; t < (colDimA + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < rowDimA && t * TILE_SIZE + tx < colDimA)
            s_A[ty][tx] = A[row * colDimA + t * TILE_SIZE + tx];
        else
            s_A[ty][tx] = 0;

        if (t * TILE_SIZE + ty < colDimA && col < colDimB)
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * colDimB + col];
        else
            s_B[ty][tx] = 0;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            cValue += s_A[ty][i] * s_B[i][tx];
        }

        __syncthreads();
    }

    if (row < rowDimA && col < colDimB)
        C[row * colDimB + col] = cValue;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("usage: ./TiledMatrixMul <rowDimA> <colDimA> <colDimB>\n");
        return 1;
    }

    int rowDimA = atoi(argv[1]);
    int colDimA = atoi(argv[2]);
    int colDimB = atoi(argv[3]);

//     if (colDimA != rowDimB) {
//         printf("error: incompatible matrix dimensions\n");
//         return 1;
//     }

    // allocate host memory for matrices A, B, and C
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(rowDimA * colDimA * sizeof(float));
    h_B = (float*)malloc(colDimA * colDimB * sizeof(float));
    h_C = (float*)malloc(rowDimA * colDimB * sizeof(float));

    // initialize host matrices A and B with random values
    for (int i = 0; i < rowDimA * colDimA; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < colDimA * colDimB; i++) {
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // allocate device memory for matrices A, B, and C
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rowDimA * colDimA * sizeof(float));
    cudaMalloc((void**)&d_B, colDimA * colDimB * sizeof(float));
    cudaMalloc((void**)&d_C, rowDimA * colDimB * sizeof(float));

    // copy host matrices A and B to device
    cudaMemcpy(d_A, h_A, rowDimA * colDimA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, colDimA * colDimB * sizeof(float), cudaMemcpyHostToDevice);

    // calculate grid and block dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((colDimB + TILE_SIZE - 1) / TILE_SIZE, (rowDimA + TILE_SIZE - 1) / TILE_SIZE);

    // invoke CUDA kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rowDimA, colDimA, colDimB);

    // copy results from device to host
    cudaMemcpy(h_C, d_C, rowDimA * colDimB * sizeof(float), cudaMemcpyDeviceToHost);
    
    // print results
    printf("Matrix A:\n");
    for (int i = 0; i < rowDimA; i++) {
        for (int j = 0; j < colDimA; j++) {
            printf("%.2f ", h_A[i * colDimA + j]);
        }
        printf("\n");
    }

    // print results
    printf("Matrix B:\n");
    for (int i = 0; i < colDimA; i++) {
        for (int j = 0; j < colDimB; j++) {
            printf("%.2f ", h_B[i * colDimB + j]);
        }
        printf("\n");
    }
    

    // print results
    printf("Matrix C:\n");
    for (int i = 0; i < rowDimA; i++) {
        for (int j = 0; j < colDimB; j++) {
            printf("%.2f ", h_C[i * colDimB + j]);
        }
        printf("\n");
    }

    // deallocate device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // deallocate host memory
    free(h_A);
    free(h_B);
    free(h_C);

return 0;
}

