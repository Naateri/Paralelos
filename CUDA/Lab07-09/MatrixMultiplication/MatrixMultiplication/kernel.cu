
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>

#define TILE_WIDTH 32

void init_mat(int size_i, int size_j, float*& A) {
    A = new float[size_i * size_j];
    srand(time(NULL));
    for (int i = 0; i < size_i; i++) {
        for (int j = 0; j < size_j; j++) {
            A[i * size_j + j] = rand() % 1000;
        }
    }
}


// M = m x n, N = n x p, P = m x p
__global__ void MatrixMulKernel(float* M, float* N, float* P, int m, int n, int p) {
    // Calculate row index of the P element and M
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate column index of P and N
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < m) && (col < p)) {
        float Pvalue = 0;
        // each thread computes one element of the block sub-mat
        for (int k = 0; k < n; k++) {
            Pvalue += M[row * n + k] * N[k * p + col];
        }
        P[row * p + col] = Pvalue;
    }
}

// M = m x n, N = n x p, P = m x p
__global__ void MatrixMulTilingKernel(float* d_M, float* d_N, float* d_P, int M, int N, int P) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    //Identify row and column of the d_P element to work on

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    // Loop over d_M and d_N tiles required to compute d_P element
    for (int ph = 0; ph < (N-1) / TILE_WIDTH + 1; ph++) {

        // Collaborative loading of d_M and d_N tiles into shared mem
        if ((row < M) && (ph * TILE_WIDTH + tx) < N)
            Mds[ty][tx] = d_M[(row * N) + (ph * TILE_WIDTH) + tx];
        else Mds[ty][tx] = 0.0f;
        if ((ph * TILE_WIDTH + ty) < N && col < P)
            Nds[ty][tx] = d_N[(ph * TILE_WIDTH + ty) * P + col];
        else Nds[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    if ( (row < M) && (col < P) ) d_P[row * P + col] = Pvalue;
}

// M = m x n, N = n x p, P = m x p
void hostMatrixMult(float* M, float* N, float* P, int m, int n, int p, int kernel_func = 0) {
    float* d_M, * d_N, * d_P;

    cudaMalloc((void**)&d_M, (m*n) * sizeof(float));
    cudaMalloc((void**)&d_N, (n*p) * sizeof(float));
    cudaMalloc((void**)&d_P, (m*p) * sizeof(float));

    cudaMemcpy(d_M, M, (m*n) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, (n*p) * sizeof(float), cudaMemcpyHostToDevice);

    /*
    kernel call
    */
    if (kernel_func == 0) {
        // regular multiplication
        //dim3 dimGrid(ceil(width / 1024.0), 1, 1);
        //dim3 dimBlock(1024, 1, 1);

        dim3 dimGrid(ceil(m / 32.0), ceil(p / 32.0), 1);
        dim3 dimBlock(32, 32, 1);

        MatrixMulKernel << < dimGrid, dimBlock >> > (d_M, d_N, d_P, m, n, p);
    }
    else if (kernel_func == 1) {
        // tiling multiplication
        std::cout << "Tiling\n";
        dim3 dimGrid(ceil(m / 32.0), ceil(p/32.0), 1);
        dim3 dimBlock(32, 32, 1);
        int width = p;

        MatrixMulTilingKernel << < dimGrid, dimBlock >> > (d_M, d_N, d_P, m, n, p);
    }

    cudaMemcpy(P, d_P, (m*p) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main()
{
    // old portion of code
    // used when algorithms only worked with squared matrices

    /*int sizes[] = { 1024, 2048, 4096, 8192, 16384};

    for (int i = 0; i < 5; i++) {

        std::cout << "Size: " << sizes[i] << std::endl;
        float* A, * B;
        init_mat(sizes[i], A);
        init_mat(sizes[i], B);

        float* output = new float[sizes[i] * sizes[i]];

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        hostMatrixMult(A, B, output, sizes[i],0);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float elapsed_time = 0;
        cudaEventElapsedTime(&elapsed_time, start, stop);

        std::cout << "Operation " << 0 << ", time: " << elapsed_time << std::endl;
        
        delete[] A;
        delete[] B;
        delete[] output;

        init_mat(sizes[i], A);
        init_mat(sizes[i], B);

        output = new float[sizes[i] * sizes[i]];

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        hostMatrixMult(A, B, output, sizes[i], 1);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        elapsed_time = 0;
        cudaEventElapsedTime(&elapsed_time, start, stop);

        std::cout << "Operation " << 1 << ", time: " << elapsed_time << std::endl;

        delete[] A;
        delete[] B;
        delete[] output;
    }
    */
    int sizes_i[] = { 4096, 16384 };
    int sizes_j[] = { 1024, 2048, 8192 };

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << "Size i: " << sizes_i[i] << std::endl;
            std::cout << "Size j: " << sizes_j[j] << std::endl;
            float* A, * B;
            init_mat(sizes_i[i], sizes_j[j], A); // A = i x j
            init_mat(sizes_j[j], sizes_i[i], B); // B = j x i

            float* output = new float[sizes_i[i] * sizes_i[i]]; // C = i x i

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            hostMatrixMult(A, B, output, sizes_i[i], sizes_j[j], sizes_i[i], 0);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            float elapsed_time = 0;
            cudaEventElapsedTime(&elapsed_time, start, stop);

            std::cout << "Operation " << 0 << ", time: " << elapsed_time << std::endl;

            delete[] A;
            delete[] B;
            delete[] output;
            
            init_mat(sizes_i[i], sizes_j[j], A); // A = i x j
            init_mat(sizes_j[j], sizes_i[i], B); // B = j x i

            output = new float[sizes_i[i] * sizes_i[i]]; // C = i x i

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            hostMatrixMult(A, B, output, sizes_i[i], sizes_j[j], sizes_i[i], 1);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            
            cudaEventElapsedTime(&elapsed_time, start, stop);

            std::cout << "Operation " << 1 << ", time: " << elapsed_time << std::endl;
            
            delete[] A;
            delete[] B;
            delete[] output;
        }
    }

    return 0;
}