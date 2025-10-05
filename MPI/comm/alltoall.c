#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send_value = rank;       // each process has its rank as data
    int recv_values[size];       // array to store data from all processes

    // All-to-all: every process gathers data from all others
    MPI_Allgather(&send_value, 1, MPI_INT, 
                  recv_values, 1, MPI_INT, 
                  MPI_COMM_WORLD);

    printf("Process %d received: ", rank);
    for (int i = 0; i < size; i++) {
        printf("%d ", recv_values[i]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}
