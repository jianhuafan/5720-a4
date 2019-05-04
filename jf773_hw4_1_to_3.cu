/* compile with /usr/local/cuda-10.0/bin/nvcc -arch=compute_XX -o foo.out foo.cu
*  run with ./foo.out N mode
*  N is the size of matrix, mode is different way to multiply matrix
*  1 represents naive gpu, 2 means tiled gpu, 3 means transposed tiled gpu 
*/

#include <stdio.h>
#include <stdlib.h>

#define MAX_MAT_VALUE 1000
#define BLOCK_SIZE 16

void cpu_matrixMul(float *A, float *B, float *C, int N) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float partial = 0.0;
            for (k = 0; k < N; k++) {
                partial += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = partial;
        }
    }
}

__global__ void naive_matrxiMul(float *dev_A, float *dev_B, float *dev_C, int N) {
    float partial = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int k;
    if (row < N && col < N) {
        for (k = 0; k < N; k++) {
            partial += dev_A[row * N + k] * dev_B[k * N + col];
        }
        dev_C[row * N + col] = partial;
    }
}

__global__ void tiled_matrxiMul(float *dev_A, float *dev_B, float *dev_C, int N) {
    __shared__ float A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_tile[BLOCK_SIZE][BLOCK_SIZE];

    float partial = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x, ty = threadIdx.y;

    int m;
    for (m = 0; m < N / blockDim.x; ++m) {
        A_tile[ty][tx] = dev_A[row * N + m * blockDim.x + tx];
        B_tile[ty][tx] = dev_B[N * (m * blockDim.y + ty) + col];
        __syncthreads();
        int k;
        for (k = 0; k < blockDim.x; ++k) {
            partial += A_tile[ty][k] * B_tile[k][tx];
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        dev_C[row * N + col] = partial;
    }
}

__global__ void transposed_tiled_matrxiMul(float *dev_A, float *dev_B, float *dev_C, int N) {
    __shared__ float A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_tile[BLOCK_SIZE][BLOCK_SIZE];

    float partial = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x, ty = threadIdx.y;

    int m;
    for (m = 0; m < N / blockDim.x; ++m) {
        A_tile[ty][tx] = dev_A[row * N + m * blockDim.x + tx];
        B_tile[tx][ty] = dev_B[row * N + m * blockDim.x + tx];
        __syncthreads();
        int k;
        for (k = 0; k < blockDim.x; ++k) {
            partial += A_tile[ty][k] * B_tile[tx][k];
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        dev_C[row * N + col] = partial;
    }
}

void print_matrix(float *A, int N) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%.3f ", A[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        puts("Please specify the size of matrix and mode (1, 2, 3, 4)");
        return 0;
    }

    int N;
    int mode;
    N = atoi(argv[1]);
    mode = atoi(argv[2]);

    float *A, *B, *C, *CC, *dev_A, *dev_B, *dev_C;

    // allocate memory in host
    cudaMallocHost((void **) &A, N * N * sizeof(float));
    cudaMallocHost((void **) &B, N * N * sizeof(float));
    cudaMallocHost((void **) &C, N * N * sizeof(float));
    cudaMallocHost((void **) &CC, N * N * sizeof(float));

    int i, j;
    // initialize matrix A and B
    srand48(1);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i * N + j] = drand48() * MAX_MAT_VALUE;
            B[i * N + j] = drand48() * MAX_MAT_VALUE;
        }
    }

    float gpu_time, cpu_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // allocate memory on the device
    cudaMalloc((void**) &dev_A, N * N * sizeof(float));
    cudaMalloc((void**) &dev_B, N * N * sizeof(float));
    cudaMalloc((void**) &dev_C, N * N * sizeof(float));

    cudaMemcpy(dev_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 Grid(N / Block.x, N / Block.y);

    if (mode == 1) {
        naive_matrxiMul<<< Grid, Block>>>(dev_A, dev_B, dev_C, N);
    } else if (mode == 2) {
        tiled_matrxiMul<<< Grid, Block>>>(dev_A, dev_B, dev_C, N);
    } else if (mode == 3) {
        transposed_tiled_matrxiMul<<< Grid, Block>>>(dev_A, dev_B, dev_C, N);
    } else {
        return 0;
    }
    
    cudaMemcpy(C, dev_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("total time used on gpu matrix multiply: %f ms.\n", gpu_time);

    // start cpu matrix multiply
    cudaEventRecord(start, 0);
    cpu_matrixMul(A, B, CC, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time, start, stop);
    printf("total time used on cpu matrix multiply: %f ms.\n", cpu_time);

    // validate result
    int all_right = 1;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            if (abs(CC[i * N + j] - C[i * N + j]) > 2) {
                printf("%3.f != %3.f\n", C[i * N + j], CC[i * N + j]);
                all_right = 0;
                break;
            }
        }
    }

    if (all_right) {
        printf("all results are right, speed up by gpu = %f.2\n", cpu_time / gpu_time);
    } else {
        printf("wrong results\n");
    }

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(CC);

    return 0;
}