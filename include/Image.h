#pragma once
#include <string>

class Image
{
public:

	//Constructor that receives the path
	Image(const std::string& filepath);

	//Constructor that creates an empty image
	Image(int width, int height, int channels);

	//Destructor
	~Image();

	void uploadToGPU();
	void downloadFromGPU();

	int GetWidth();

	int GetHeight();

	int GetChannels();

	unsigned char* getHostData();

	unsigned char* getGPUData();

	void save(const std::string& outputFilePath);

	void save_with_cpu(const std::string& outputFilePath);

	void allocateGPU();

	void processWithCPU();

private:

	// Width in pixels
	int m_width;

	//Height in pixels
	int m_height;

	//Channels, being 3 a RGB and 4 an RGBA
	int m_channels;

	//A lot of consecutive bytes that actually represents the image for the RAM and CPU
	unsigned char* m_HostData = nullptr;

	//A lot of consecutive bytes that actually represents the image for the VRAM and GPU
	unsigned char* m_GpuData = nullptr;

	//A lot of consecutive bytes just for a test using cpu
	unsigned char* m_CpuOutputData = nullptr;

	std::string setOutputPath(const std::string& filepath);
};
