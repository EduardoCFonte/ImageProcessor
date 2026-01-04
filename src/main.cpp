#include <iostream>
#include "Image.h"
#include "cudaProcessor.cuh"
#include <chrono>
#include <fstream>
#include <iomanip>


void saveBenchmark(double cpuTime, double gpuTime, int width, int height) {
	std::ofstream outFile("../out/benchmark_results.txt", std::ios::app); 
	if (outFile.is_open()) {
		outFile << "--- Benchmark: " << width << "x" << height << " ---" << std::endl;
		outFile << "CPU Time: " << std::fixed << std::setprecision(4) << cpuTime << " ms" << std::endl;
		outFile << "GPU Time: " << std::fixed << std::setprecision(4) << gpuTime << " ms" << std::endl;
		outFile << "Speedup: " << cpuTime / gpuTime << "x faster" << std::endl;
		outFile << "-----------------------------------" << std::endl << std::endl;
		outFile.close();
		std::cout << "[LOG] Benchmark saved in benchmark_results.txt" << std::endl;
	}
}


int main() {

	std::cout << "Using it to attach a debugger" << std::endl;
	std::cin.get();

	try
	{
		Image imageToGrey("../Mocks/Test1.jpg");
		int w = imageToGrey.GetWidth();
		int h = imageToGrey.GetHeight();
		int c = imageToGrey.GetChannels();

		auto startCPU = std::chrono::high_resolution_clock::now();

		imageToGrey.processWithCPU();

		auto endCPU = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> cpuMs = endCPU - startCPU;

		imageToGrey.save_with_cpu("GreyResult.png");

		Image greyImage(w, h, 1);

		imageToGrey.uploadToGPU();

		greyImage.allocateGPU();

		auto startGPU = std::chrono::high_resolution_clock::now();

		launchGrayscaleKernel(imageToGrey.getGPUData(), greyImage.getGPUData(), w, h, c);

		auto endGPU = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> gpuMs = endGPU - startGPU;

		saveBenchmark(cpuMs.count(), gpuMs.count(), w, h);

		greyImage.downloadFromGPU();

		greyImage.save("GreyResult.png");

	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}
	return 0;
}