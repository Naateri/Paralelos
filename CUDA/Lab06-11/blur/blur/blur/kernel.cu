#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bmp.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <stdio.h>

#define BLUR_SIZE 7

__global__ void blurKernel(unsigned char* in, unsigned char* out, int w, int h) {

    int CHANNELS = 3;

    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < w && Row < h) {
        int pixVal = 0;
        int pixels = 0;

        int rVal = 0;
        int gVal = 0;
        int bVal = 0;

        int cur_pixel = Row * w + Col;

        int rgbOffset = cur_pixel * CHANNELS;
        
        // Get the Average of the surrounding BLUR_SIZE * BLUR_SIZE box

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;

                // Verify we have a valid image pixel

                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    int cur_pixel = curRow * w + curCol;
                    //pixVal += (int) in[curRow * w + curCol];

                    cur_pixel *= 3;

                    gVal += in[cur_pixel];
                    rVal += in[cur_pixel + 1];
                    bVal += in[cur_pixel + 2];

                    pixels++;
                }
            }

        }
        //out[Row * w + Col] = (unsigned char)(pixVal / pixels);
        out[rgbOffset] = (unsigned char)(gVal / pixels);
        out[rgbOffset +1] = (unsigned char)(rVal / pixels);
        out[rgbOffset +2] = (unsigned char)(bVal / pixels);
    }
}


// pass vector to char*
void vector_to_char(std::vector<uint8_t> vec, unsigned char*& out) {
    for (int i = 0; i < vec.size(); i++) {
        out[i] = (unsigned char)(vec[i]);
        //std::cout << (int)out[i] << std::endl;
    }
}

// pass char* to vector
void char_to_vector(unsigned char* in, std::vector<uint8_t>& vec, int size) {
    for (int i = 0; i < size; i++) {
        vec.push_back((uint8_t)in[i]);
        //std::cout << "char to vec " << (int)vec[i] << std::endl;
    }
}

void blurImage(const char* filename) {
    char* fileout = "blurpic.bmp";
    BMP image(filename);
    std::vector<uint8_t> image_info = image.data;
    int size = image_info.size();
    std::cout << "data size " << size << std::endl;

    if (image.bmp_info_header.bit_count == 24) {
        std::cout << "in: RGB Colors, 3 channels\n";
    }

    int width = image.bmp_info_header.width;
    int height = image.bmp_info_header.height;

    unsigned char* Pin = new unsigned char[size];
    unsigned char* Pout = new unsigned char[size];

    // Storing image info vector's in Pin array

    vector_to_char(image_info, Pin);

    /*for (int i = 0; i < 100000; i++) {
        std::cout << "Pin[i] value after copying from vector " << (int)Pin[i] << std::endl;
    }*/

    unsigned char* d_Pin;
    unsigned char* d_Pout;

    // Separating space in GPU's memory
    // And sending Pin to memory
    cudaMalloc((void**)&d_Pin, size);
    cudaMemcpy(d_Pin, Pin, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_Pout, size);

    // kernel call

    dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
    dim3 dimBlock(16, 16, 1);
    blurKernel << < dimGrid, dimBlock >> > (d_Pin, d_Pout, width, height);

    // get blur data from gpu
    cudaMemcpy(Pout, d_Pout, size, cudaMemcpyDeviceToHost);

    std::vector<uint8_t> blur_data;
    char_to_vector(Pout, blur_data, size);

    std::cout << "blur data size " << blur_data.size() << std::endl;

    BMP image2(width, height, false);

    if (image2.bmp_info_header.bit_count == 24) {
        std::cout << "out: RGB Colors, 3 channels\n";
    }

    std::cout << "Image 2 size before modifying " << image2.data.size() << std::endl;

    /* for (int i = 0; i < image2.data.size(); i++) {
         image2.data[i] = Pout[i];
     }*/

    image2.data = blur_data;
    image2.write(fileout);


    // free memory on gpu
    cudaFree(d_Pin);
    cudaFree(d_Pout);

    // free memory
    delete[] Pin;
    delete[] Pout;

}

int main()
{
    blurImage("4pics.bmp");

    return 0;
}
