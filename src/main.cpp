#include <iostream>
#include "Image.h"

int main() {

	std::cout << "Using it to attach a debugger" << std::endl;
	std::cin.get();

	try
	{
		Image("../Mocks/Test1.jpg");
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}
	return 0;
}