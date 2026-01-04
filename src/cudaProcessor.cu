#include <cuda_runtime.h>
#include "cudaProcessor.cuh"
#include <iostream>


__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rgbOffset = (y * width + x) * channels;

        int grayOffset = y * width + x;

        unsigned char r = input[rgbOffset];
        unsigned char g = input[rgbOffset + 1];
        unsigned char b = input[rgbOffset + 2];

        float gray = 0.299f * r + 0.587f * g + 0.114f * b;

        output[grayOffset] = static_cast<unsigned char>(gray);
    }
}

void launchGrayscaleKernel(unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {

    dim3 blockSize(16, 16);

    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    grayscaleKernel <<<gridSize, blockSize >>> (d_input, d_output, width, height, channels);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error to launch it in the kernel: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}
