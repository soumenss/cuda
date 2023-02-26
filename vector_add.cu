#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x;
    if (tid < 10) {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void) {
    int a[10], b[10], c[10];
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, 10*sizeof(int));
    cudaMalloc((void**)&dev_b, 10*sizeof(int));
    cudaMalloc((void**)&dev_c, 10*sizeof(int));

    for (int i = 0; i < 10; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    cudaMemcpy(dev_a, a, 10*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, 10*sizeof(int), cudaMemcpyHostToDevice);

    add<<<10,1>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, 10*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
