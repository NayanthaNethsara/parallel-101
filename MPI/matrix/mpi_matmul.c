#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv){
    MPI_Init(&argc,&argv);
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);

    int N = 1024;
    if(argc>1) N = atoi(argv[1]);

    int rows_per_proc = N / nproc;
    double *A = NULL;
    double *B = malloc(N*N*sizeof(double));
    double *C = NULL;

    if(rank==0){
        A = malloc(N*N*sizeof(double));
        C = malloc(N*N*sizeof(double));
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                A[i*N+j] = i+j*0.5;
                B[i*N+j] = i-j*0.5;
            }
        }
    }

    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *A_local = malloc(rows_per_proc * N * sizeof(double));
    double *C_local = malloc(rows_per_proc * N * sizeof(double));

    MPI_Scatter(A, rows_per_proc*N, MPI_DOUBLE,
                A_local, rows_per_proc*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Start timing here (after scatter)
    double t0 = MPI_Wtime();

    for(int i=0;i<rows_per_proc;i++){
        for(int j=0;j<N;j++){
            double sum=0.0;
            for(int k=0;k<N;k++){
                sum += A_local[i*N+k]*B[k*N+j];
            }
            C_local[i*N+j] = sum;
        }
    }

    MPI_Gather(C_local, rows_per_proc*N, MPI_DOUBLE,
               C, rows_per_proc*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime(); // after gather

    if(rank==0){
        printf("MPI MatMul: N=%d Procs=%d Time=%f s Sample C[0][0]=%f\n",
               N, nproc, t1-t0, C[0]);
    }

    free(B); free(A_local); free(C_local);
    if(rank==0){ free(A); free(C); }

    MPI_Finalize();
    return 0;
}
