# OpenMP Loop

This document demonstrates the effects of **race conditions** in parallel loops using OpenMP and compares different synchronization approaches.

---

### Compile with OpenMP support (using GCC):

```bash
gcc -fopenmp sum-critical.c -o sum-critical.o
gcc -fopenmp sum-atomic.c   -o sum-atomic.o
gcc -fopenmp sum-reduction.c -o sum-reduction.o
gcc -fopenmp simple-for.c   -o simple-for.o
```

### Running with specific number of threads

You can control the number of OpenMP threads with the environment variable `OMP_NUM_THREADS`.

I'm (using **8 threads**):

```bash
export OMP_NUM_THREADS=8
./sum-critical.o
./sum-atomic.o
./sum-reduction.o
./simple-for.o
```

---

## Observed Output (with 8 threads)

### Results Comparison

| Program                | Sum (Expected: `499999500000`) | Execution Time (s) | Notes                                                                                |
| ---------------------- | ------------------------------ | ------------------ | ------------------------------------------------------------------------------------ |
| `sum-critical.o`       | `499999500000`                 | `0.028506`         | Uses **`critical`** → correct but slower due to locking overhead on every iteration. |
| `sum-atomic.o`         | `499999500000`                 | `0.017468`         | Uses **`atomic`** → correct, slightly faster than `critical`.                        |
| `sum-reduction.o`      | `499999500000`                 | `0.000945`         | Uses **`reduction(+:sum)`** → fastest and most efficient.                            |
| `simple-for.o` (run 1) | `142004658717`                 | `0.001554`         | Race condition → incorrect result.                                                   |
| `simple-for.o` (run 2) | `30972488803`                  | `0.001158`         | Wrong, nondeterministic.                                                             |
| `simple-for.o` (run 3) | `158977260929`                 | `0.001101`         | Different wrong result each run.                                                     |

---

## Explanation of Each Approach

### 1. `simple-for.o` (Race Condition)

- Multiple threads update the same shared `sum` variable **simultaneously**.
- Updates interfere with each other (lost updates).
- Result: **nondeterministic incorrect sums**.

---

### 2. `critical`

```c
#pragma omp critical
sum += i;
```

- Each thread locks the section before updating `sum`.
- Prevents interference but forces **only one thread at a time** to enter.
- Correct result but **slow** (lock/unlock overhead at each iteration).

---

### 3. `atomic`

```c
#pragma omp atomic
sum += i;
```

- Lighter-weight than `critical`.
- Only the single update is atomic (no block of code).
- Faster than `critical`, but still serialized per update.

---

### 4. `reduction`

```c
#pragma omp parallel for reduction(+:sum)
for (long long i = 0; i < N; i++) {
    sum += i;
}
```

- Compiler gives **each thread a private copy** of `sum`.
- Threads accumulate their own sums **independently (no lock needed)**.
- At the end, OpenMP **reduces** (merges) all partial sums.
- Result: **correct and fastest**.
