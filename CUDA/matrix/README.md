## Overview

This program performs matrix multiplication (C = A × B) using CUDA with shared memory tiling for improved performance.

- **Matrix initialization (same for OpenMP/MPI/CUDA):**
  - `A[i][j] = i + j*0.5`
  - `B[i][j] = i - j*0.5`
- Measures GPU execution time using `cudaEvent`.
- Prints grid, block dimensions, and a sample element (`C[0][0]`) for verification.

## Running on Colab

```bash
# Compile the CUDA program
!nvcc -O3 -arch=sm_75 cuda_matmul_tiled.cu -o cuda_matmul_tiled

# Run the executable with matrix size 1024
!./cuda_matmul_tiled 1024
```

### Example Output

```
CUDA MatMul (Tiled): N=1024 grid=(64,64) block=(16,16) Time=23.707457 ms Sample C[0][0]=178694912.000000
```

## Running on a Local Linux Machine

1. Install NVIDIA GPU driver and CUDA toolkit matching your GPU.
2. Save the program as `cuda_matmul_tiled.cu`.
3. Open a terminal in the program directory.
4. Compile with:
   ```bash
   nvcc -O3 -arch=<your_sm_version> cuda_matmul_tiled.cu -o cuda_matmul_tiled
   ```
   Replace `<your_sm_version>` with your GPU’s compute capability (e.g., `sm_75` for Tesla T4, `sm_80` for A100).
5. Run:
   ```bash
   ./cuda_matmul_tiled 1024
   ```
6. Verify output: `C[0][0]` should match your OpenMP/MPI versions.
