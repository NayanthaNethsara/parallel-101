# CUDA Hello World

This example demonstrates a simple CUDA program that prints messages from GPU threads.

## Source: `cuda_hello.cu`

```cpp
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
```

## Compile & Run

```sh
# Compile
nvcc cuda_hello.cu -o cuda_hello

# Run
./cuda_hello
```

## Expected Output

```
Hello from CUDA thread 0, block 0!
Hello from CUDA thread 1, block 0!
Hello from CUDA thread 2, block 0!
Hello from CUDA thread 3, block 0!
Hello from CUDA thread 4, block 0!
CUDA program finished successfully!
```
