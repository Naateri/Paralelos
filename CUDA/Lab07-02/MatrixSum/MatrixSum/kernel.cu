// Summatory of two matrices
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
            A[i*size + j] = rand() % 1000;
        }
    }
}

/*
B) Write a kernel that has each thread to produce one output matrix
element. Fill in the execution configuration parameters for this design.
*/

__global__ void element_per_thread(float* c_output, float* in_1, float* in_2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        c_output[i * n + j] = in_1[i * n + j] + in_2[i * n + j];
    }
}

/*
C) Write a kernel that has each thread to produce one output matrix row.
Fill in the execution configuration parameters for the design.
*/

__global__ void thread_per_row(float* c_output, float* in_1, float* in_2, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        for (int j = 0; j < n; j++) {
            c_output[n * row + j] = in_1[n * row + j] + in_2[n * row + j];
        }
    }
}

/*
D) Write a kernel that has each thread to produce one output matrix column.
Fill in the execution configuration parameters for the design.
*/

__global__ void thread_per_col(float* c_output, float* in_1, float* in_2, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        for (int i = 0; i < n; i++) {
            c_output[i * n + col] = in_1[i * n + col] + in_2[i * n + col];
        }
    }
}

/*
A) Write the host stub function by allocating memory for the input and
output matrices, transferring input data to device; launch the kernel,
transferring the output data to host and freeing the device memory for
the input and output data.Leave the execution configuration parameters
open for this step. 
*/

void host_func(float* output, float* in_1, float* in_2, int n, int kernel_func=0) {
    float* c_output, * c_in1, * c_in2;

    int n2 = n * n;

    cudaMalloc((void**)&c_output, n2 * sizeof(float));
    cudaMalloc((void**)&c_in1, n2 * sizeof(float));
    cudaMalloc((void**)&c_in2, n2 * sizeof(float));

    cudaMemcpy(c_in1, in_1, n2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_in2, in_2, n2 * sizeof(float), cudaMemcpyHostToDevice);

    /*
    kernel call
    */

    if (kernel_func == 0) {
        // thread per element
        dim3 dimGrid(ceil(n / 32.0), ceil(n / 32.0), 1);
        dim3 dimBlock(32, 32, 1);

        element_per_thread << < dimGrid, dimBlock >> > (c_output, c_in1, c_in2, n);
    }
    else if (kernel_func == 1) {
        // thread per row
        dim3 dimGrid(ceil(n / 1024.0), 1, 1);
        dim3 dimBlock(1024, 1, 1);

        thread_per_row << < dimGrid, dimBlock >> > (c_output, c_in1, c_in2, n);
    }
    else if (kernel_func == 2) {
        // thread per col
        dim3 dimGrid(ceil(n / 1024.0), 1, 1);
        dim3 dimBlock(1024, 1, 1);

        thread_per_col << < dimGrid, dimBlock >> > (c_output, c_in1, c_in2, n);
    }

    cudaMemcpy(output, c_output, n2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(c_output);
    cudaFree(c_in1);
    cudaFree(c_in2);
}

int main()
{
    int sizes[] = { 1000, 2500, 5000, 10000 };
    
    for (int j = 0; j < 4; j++) {

        for (int i = 0; i < 3; i++) {

            std::cout << "Size: " << sizes[j] << std::endl;
            float* A, * B;
            init_mat(sizes[j], A);
            init_mat(sizes[j], B);

            float* output = new float[sizes[j] * sizes[j]];

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            host_func(output, A, B, sizes[j], i);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            float elapsed_time = 0;
            cudaEventElapsedTime(&elapsed_time, start, stop);

            std::cout << "Operation " << i << ", time: " << elapsed_time << std::endl;

            delete[] A;
            delete[] B;
            delete[] output;
        }
    }
    

    return 0;
}