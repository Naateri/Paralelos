
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bmp.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <stdio.h>


// Snippet of code from book
// Programming Massively Parallel Processors A Hands-on Approach
__global__ void colorToGreyscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    int CHANNELS = 3;
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height) {
        // get 1D coordinate for the grayscale image   
        int greyOffset = Row * width + Col;
        // one can think of the RGB image having
        // CHANNEL times columns than the grayscale image
        int rgbOffset = greyOffset * CHANNELS;
        unsigned char b = Pin[rgbOffset]; // blue value for pixel
        unsigned char g = Pin[rgbOffset + 1]; // green value for pixel
        unsigned char r = Pin[rgbOffset + 2]; // red value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        unsigned char conversion = 0.21f * r + 0.71f * g + 0.07f * b;
        Pout[rgbOffset] = conversion;
        Pout[rgbOffset + 1] = conversion;
        Pout[rgbOffset + 2] = conversion;
/*
# if __CUDA_ARCH__>=200
        printf("red %d green %d blue %d\n", r, g, b);
        //printf("conversion value %d \n", conversion);

#endif  */
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
        vec.push_back((uint8_t) in[i]);
        //std::cout << "char to vec " << (int)vec[i] << std::endl;
    }
}

void colorToGreyscale(const char* filename) {
    char* fileout = "greyscale.bmp";
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

    vector_to_char(image_info, Pin);

    /*for (int i = 0; i < 100000; i++) {
        std::cout << "Pin[i] value after copying from vector " << (int)Pin[i] << std::endl;
    }*/

    unsigned char* d_Pin;
    unsigned char* d_Pout;

    cudaMalloc((void**)&d_Pin, size);
    cudaMemcpy(d_Pin, Pin, size, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_Pout, size);

    // kernel call

    dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
    dim3 dimBlock(16, 16, 1);
    colorToGreyscaleConversion <<< dimGrid, dimBlock >>> (d_Pout, d_Pin, width, height);

    // get greyscale data from gpu
    cudaMemcpy(Pout, d_Pout, size, cudaMemcpyDeviceToHost);

    std::vector<uint8_t> grey_data;
    char_to_vector(Pout, grey_data, size);

    std::cout << "grey data size " << grey_data.size() << std::endl;
    
    BMP image2(width, height, false);

    if (image2.bmp_info_header.bit_count == 24) {
        std::cout << "out: RGB Colors, 3 channels\n";
    }

    std::cout << "Image 2 size before modifying " << image2.data.size() << std::endl;

   /* for (int i = 0; i < image2.data.size(); i++) {
        image2.data[i] = Pout[i];
    }*/

    image2.data = grey_data;
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

    colorToGreyscale("4pics.bmp");

    return 0;
}
