#include <stdio.h>
#include <omp.h>

int main() {
    long long N = 1000000; // 1 million
    long long sum = 0;
    double start, end;

    start = omp_get_wtime();

    #pragma omp parallel for
    for (long long i = 0; i < N; i++) {
        #pragma omp atomic
        sum += i;
    }

    end = omp_get_wtime();
    printf("Atomic sum = %lld\n", sum);
    printf("Execution time: %f seconds\n", end - start);

    return 0;
}
