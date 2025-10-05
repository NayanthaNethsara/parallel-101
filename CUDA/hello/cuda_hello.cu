#include <stdio.h>

__global__ void hello_kernel() {
    printf("Hello from CUDA thread %d, block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    // Launch 1 block of 5 threads
    hello_kernel<<<1, 5>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    printf("CUDA program finished successfully!\n");
    return 0;
}
