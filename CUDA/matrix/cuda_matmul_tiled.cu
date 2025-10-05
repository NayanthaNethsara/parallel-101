#include <stdio.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matmul_kernel_tiled(double *A, double *B, double *C, int N){
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;

    double sum=0.0;

    for(int t=0; t<(N+TILE-1)/TILE; t++){
        if(row<N && t*TILE+threadIdx.x<N)
            sA[threadIdx.y][threadIdx.x] = A[row*N + t*TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0;

        if(col<N && t*TILE+threadIdx.y<N)
            sB[threadIdx.y][threadIdx.x] = B[(t*TILE+threadIdx.y)*N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for(int k=0;k<TILE;k++)
            sum += sA[threadIdx.y][k]*sB[k][threadIdx.x];

        __syncthreads();
    }

    if(row<N && col<N)
        C[row*N + col] = sum;
}

int main(int argc,char** argv){
    int N=1024;
    if(argc>1) N=atoi(argv[1]);

    size_t bytes = N*N*sizeof(double);
    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);

    // Matching OpenMP/MPI initialization
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            h_A[i*N + j] = i + j*0.5;
            h_B[i*N + j] = i - j*0.5;
        }
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE,TILE);
    dim3 grid((N+TILE-1)/TILE,(N+TILE-1)/TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_kernel_tiled<<<grid,block>>>(d_A,d_B,d_C,N);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms,start,stop);
    cudaMemcpy(h_C,d_C,bytes,cudaMemcpyDeviceToHost);

    printf("CUDA MatMul (Tiled): N=%d grid=(%d,%d) block=(%d,%d) Time=%f ms Sample C[0][0]=%f\n",
           N, grid.x, grid.y, block.x, block.y, ms, h_C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
