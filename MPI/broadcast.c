#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int value;

    if (rank == 0) {
        value = 42; // Rank 0 initializes data
        for (int i = 1; i < size; i++) {
            MPI_Send(&value, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            printf("Process 0 sent %d to process %d\n", value, i);
        }
    } else {
        MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received %d from process 0\n", rank, value);
    }

    MPI_Finalize();
    return 0;
}
