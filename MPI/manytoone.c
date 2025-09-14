#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank != 0) {
        // Non-root processes send their rank to process 0
        MPI_Send(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        printf("Process %d sent its rank to process 0\n", rank);
    } else {
        // Root (rank 0) receives from all other processes
        for (int i = 1; i < size; i++) {
            int received;
            MPI_Recv(&received, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process 0 received %d from process %d\n", received, i);
        }
    }

    MPI_Finalize();
    return 0;
}
