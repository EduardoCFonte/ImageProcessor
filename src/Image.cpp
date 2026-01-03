#include <iostream>
#include "Image.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include <stdexcept>
#include <cuda_runtime.h>

Image::Image(const std::string& filepath) {
	m_HostData = stbi_load(filepath.c_str(), &m_width, &m_height, &m_channels, 0);
	if (!m_HostData) {
		throw std::runtime_error("Fail to load the image in path:" + filepath + "Due to:" + std::string(stbi_failure_reason()));
	}

	std::cout << "[LOG] Image loaded. Width: " << m_width << "    Height: " << m_height << "    Channels:" << m_channels << std::endl;
}

Image::Image(int width, int height, int channels) : m_width(width), m_height(height), m_channels(channels)
{
	size_t size = static_cast<size_t>(m_width) * m_height * m_channels;
	m_HostData = (unsigned char*)malloc(size); 

	if (!m_HostData) {
		throw std::runtime_error("Falha ao alocar memória para nova imagem.");
	}
}

Image::~Image() {
	if (m_HostData) {
		stbi_image_free(m_HostData);
	}
	if (m_GpuData) {
		cudaFree(m_GpuData); 
	}
}

void Image::save(const std::string& outputPath) {
	if (!m_HostData) 
		return;

	int success = stbi_write_png(outputPath.c_str(), m_width, m_height, m_channels, m_HostData, m_width * m_channels);

	if (success) {
		std::cout << "[LOG] Image saved on " << outputPath << std::endl;
	}
	else {
		std::cerr << "[ERROR] Failed to save the image." << std::endl;
	}
}

int Image::GetWidth() {
	return m_width;
}

int Image::GetHeight() {
	return m_height;
}

int Image::GetChannels() {
	return m_channels;
}

void Image::uploadToGPU() {
	size_t size = static_cast<size_t>(m_width) * m_height * m_channels;

	cudaError_t err = cudaMalloc(&m_GpuData, size);

	if (err != cudaSuccess) {
		throw std::runtime_error("Failed to allocate memory in the GPU: " + std::string(cudaGetErrorString(err)));
	}

	err = cudaMemcpy(m_GpuData, m_HostData, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		throw std::runtime_error("Failed to copy data to the GPU: " + std::string(cudaGetErrorString(err)));
	}

	std::cout << "[LOG] Data transfered to the GPU";
}

void Image::downloadFromGPU() {
	size_t size = static_cast<size_t>(m_width) * m_height * m_channels;

	cudaError_t err = cudaMemcpy(m_HostData, m_GpuData, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		throw std::runtime_error("Failed to load data from GPU: " + std::string(cudaGetErrorString(err)));
	}
}