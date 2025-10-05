#include <mpi.h>    
#include <stdio.h>

int main(int argc, char** argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    int world_size;
    // Get total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    // Get rank (ID) of this process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    printf("Hello from process %d out of %d\n", world_rank, world_size);

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
