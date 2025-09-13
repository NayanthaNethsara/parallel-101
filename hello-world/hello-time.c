#include <stdio.h>
#include <omp.h>

int main() {
    double start = omp_get_wtime();   // start timing

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int total = omp_get_num_threads();
        printf("Hello from thread %d out of %d\n", id, total);
    }

    double end = omp_get_wtime();     // end timing
    printf("Execution time: %f seconds\n", end - start);

    return 0;
}
