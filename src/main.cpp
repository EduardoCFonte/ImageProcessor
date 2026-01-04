#include <iostream>
#include "Image.h"
#include "cudaProcessor.cuh"

int main() {

	std::cout << "Using it to attach a debugger" << std::endl;
	std::cin.get();

	try
	{
		Image imageToGrey("../Mocks/Test1.jpg");
		int w = imageToGrey.GetWidth();
		int h = imageToGrey.GetHeight();
		int c = imageToGrey.GetChannels();

		Image greyImage(w, h, 1);

		imageToGrey.uploadToGPU();

		greyImage.allocateGPU();

		launchGrayscaleKernel(imageToGrey.getGPUData(), greyImage.getGPUData(), w, h, c);

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