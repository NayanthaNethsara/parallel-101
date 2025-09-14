#include <stdio.h>
#include <omp.h>

int main() {
    int sum = 0;

    #pragma omp parallel for
    for (int i = 0; i < 100; i++) {
        sum += i;  // race condition here
    }

    printf("Sum = %d\n", sum);
    return 0;
}
