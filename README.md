# Matrix Multiplier using CUDA

This program performs matrix multiplication using CUDA for GPU acceleration and includes CPU calculations for comparison (up to 2048 x 2048 matrices).

## Compilation

Compile the program using the following command:

```bash
nvcc multmatrix.cu -o multmatrix
```

## Performance
| Size          | GPU Time (ms) | CPU Time (ms)        |
|---------------|---------------|----------------------|
| 2 x 2         | 0.6621        | 0.0001               |
| 4 x 4         | 0.0254        | 0.0003               |
| 8 x 8         | 0.0251        | 0.0014               |
| 16 x 16       | 0.1811        | 0.0108               |
| 32 x 32       | 0.0309        | 0.085                |
| 64 x 64       | 0.0191        | 0.6875               |
| 128 x 128     | 0.023         | 5.5319               |
| 256 x 256     | 0.0845        | 44.1627              |
| 512 x 512     | 0.0825        | 349.512              |
| 1024 x 1024   | 0.1169        | 3425.69              |
| 2048 x 2048   | 0.1389        | 29557.7              |
| 4096 x 4096   | 0.2009        | **236,461.6**        |
| 8192 x 8192   | 1.2558        | **1,891,692.8**      |
| 16384 x 16384 | 1.6164        | **15,133,542.4**     |
| 32768 x 32768 | 164.532       | **120,983,820.8**    |

Notes:
  - GPU Time: Measured execution time for matrix multiplication on the GPU.
  - CPU Time: Measured for sizes ≤ 2048×2048; estimated for larger sizes based on O(n<sup>3</sup>) scaling.
