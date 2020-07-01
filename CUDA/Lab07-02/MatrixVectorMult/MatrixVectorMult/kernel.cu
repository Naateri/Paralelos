
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>

void init_mat(int size, float*& A) {
    A = new float[size * size];
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = rand() % 1000;
        }
    }
}

void init_vec(int size, float*& A) {
    A = new float[size];
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        A[i] = rand() % 1000;
    }
}

__global__ void matrix_vector_mult(float* output, float* in_mat, float* in_vec, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0;
        for (int j = 0; j < n; j++) {
            sum += (in_vec[i] * in_mat[i * n + j]);
        }
        output[i] = sum;
    }
}

void host_func(float* output, float* in_mat, float* in_vec, int n) {
    float* c_output, * c_inmat, * c_invec;

    int n2 = n * n;

    cudaMalloc((void**)&c_output, n * sizeof(float));
    cudaMalloc((void**)&c_inmat, n2 * sizeof(float));
    cudaMalloc((void**)&c_invec, n * sizeof(float));

    cudaMemcpy(c_inmat, in_mat, n2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_invec, in_vec, n * sizeof(float), cudaMemcpyHostToDevice);

    // kernel call

    dim3 dimGrid(ceil(n / 1024.0), 1, 1);
    dim3 dimBlock(1024, 1, 1);

    matrix_vector_mult << < dimGrid, dimBlock >> > (c_output, c_inmat, c_invec, n);

    cudaDeviceSynchronize();
    
    cudaMemcpy(output, c_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(c_output);
    cudaFree(c_inmat);
    cudaFree(c_invec);
}

int main()
{
    
    int sizes[] = { 1000, 2500, 5000, 10000, 20000 };

    for (int i = 0; i < 5; i++) {
        float* A, * B;
        init_mat(sizes[i], A);
        init_vec(sizes[i], B);

        float* output = new float[sizes[i]];

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        host_func(output, A, B, sizes[i]);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float elapsed_time = 0;
        cudaEventElapsedTime(&elapsed_time, start, stop);

        std::cout << "Size: " << sizes[i] << ", time: " << elapsed_time << std::endl;

        delete[] A;
        delete[] B;
        delete[] output;
    }

    return 0;
}
