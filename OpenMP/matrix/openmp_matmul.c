#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matmul(int N, double **A, double **B, double **C){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            double sum=0.0;
            for(int k=0;k<N;k++){
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main(int argc, char **argv){
    int N = 1024;
    if(argc>1) N = atoi(argv[1]);

    // Allocate matrices
    double **A = malloc(N*sizeof(double*));
    double **B = malloc(N*sizeof(double*));
    double **C = malloc(N*sizeof(double*));
    for(int i=0;i<N;i++){
        A[i] = malloc(N*sizeof(double));
        B[i] = malloc(N*sizeof(double));
        C[i] = malloc(N*sizeof(double));
        for(int j=0;j<N;j++){
            A[i][j] = i + j*0.5;
            B[i][j] = i - j*0.5;
        }
    }

    double t0 = omp_get_wtime();
    matmul(N,A,B,C);
    double t1 = omp_get_wtime();

    printf("OpenMP MatMul: N=%d threads=%d Time=%f s Sample C[0][0]=%f\n",
           N, omp_get_max_threads(), t1-t0, C[0][0]);

    for(int i=0;i<N;i++){ free(A[i]); free(B[i]); free(C[i]); }
    free(A); free(B); free(C);

    return 0;
}
