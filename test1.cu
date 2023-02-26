#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

__global__ void matrixMul(float* A, float* B, float* C, int rowDimA, int colDimA, int colDimB) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // Csub is used to store the element of the block sub-matrix that is computed by the thread
    float Csub = 0;
    
//     To handle boundary conditions when dealing with arbitrary matrix sizes,
//     we need to make sure that each thread only processes valid elements within
//     the matrix dimensions. We can achieve this by adding conditional statements within
//     the kernel to check whether the thread index is within the valid range.
    
    // We use conditional statements to check whether each thread is within the valid range 
    // for the matrices A and B. If a thread index is outside the valid range, we set the 
    // corresponding shared memory value to 0.0, which effectively ignores those elements 
    // during the computation.
    
    // Note that we only need to add these conditional statements for the shared memory portion of 
    // the computation, since the thread block size is fixed and we can assume that the total number 
    // of threads is always a multiple of the block size. The boundary conditions for the global memory
    // portion of the computation are already handled by the grid size and block size calculation in the host code

    for (int t = 0; t < (colDimA + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        if (row < rowDimA && t * BLOCK_SIZE + tx < colDimA)
            As[ty][tx] = A[row * colDimA + t * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0;

        if (t * BLOCK_SIZE + ty < colDimA && col < colDimB)
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * colDimB + col];
        else
            Bs[ty][tx] = 0;

        __syncthreads();
        

        for (int k = 0; k < BLOCK_SIZE; k++) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < rowDimA && col < colDimB)
        C[row * colDimB + col] = Csub;
}

int main(int argc, char** argv) {
    
    
//     if (argc < 4) {
//         printf("usage: ./TiledMatrixMul <rowDimA> <colDimA> <colDimB>\n");
//         return 1;
//     }

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

    
    // initialize thread block and kernel grid dimensions 
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((colDimB + BLOCK_SIZE - 1) / BLOCK_SIZE, (rowDimA + BLOCK_SIZE - 1) / BLOCK_SIZE);

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

    printf("Matrix B:\n");
    for (int i = 0; i < colDimA; i++) {
        for (int j = 0; j < colDimB; j++) {
            printf("%.2f ", h_B[i * colDimB + j]);
        }
        printf("\n");
    }
    
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
