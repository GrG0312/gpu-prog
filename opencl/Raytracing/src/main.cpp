
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

#include "OpenCLUtils.hpp"

using namespace std;

struct Vector3
{
	float x, y, z;
};

struct Ray
{
	Vector3 origin; //honnan jon
	Vector3 direction; //merre megy
	float wavelength; //nanometerben
	float intensity;
};

void saveTGA(const std::string& filename, const std::vector<unsigned char>& rgbaData, int width, int height) {
	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Picture couldn't be made!" << std::endl;
		return;
	}

	//18 bajtos standard TGA fejlec (Uncompressed True-Color Image)
	unsigned char header[18] = { 0 };
	header[2] = 2;
	header[12] = width & 0xFF;
	header[13] = (width >> 8) & 0xFF;
	header[14] = height & 0xFF;
	header[15] = (height >> 8) & 0xFF;
	header[16] = 32;
	header[17] = 0x20;

	file.write(reinterpret_cast<char*>(header), 18);

	//TGA formatum BGRA sorrendet var RGBA helyett -> cserelni kell sorrendet
	std::vector<unsigned char> bgraData = rgbaData;
	for (size_t i = 0; i < bgraData.size(); i += 4) {
		unsigned char r = bgraData[i + 0];
		unsigned char b = bgraData[i + 2];
		bgraData[i + 0] = b;
		bgraData[i + 2] = r;
	}

	file.write(reinterpret_cast<char*>(bgraData.data()), bgraData.size());
	file.close();
	std::cout << "Successfully saved: " << filename << std::endl;
}

int main()
{
#pragma region Generated stuff
	cl_int err;

	cl_platform_id platform;
	cl_device_id deviceId;

	// Get the first available platform
	err = clGetPlatformIDs(1, &platform, nullptr);
	checkError(err, "Failed to get OpenCL platform.");

	// Get the first available GPU device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &deviceId, nullptr);
	checkError(err, "Failed to get OpenCL device.");

	// Create an OpenCL context
	cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &err);
	checkError(err, "Failed to create OpenCL context.");

	cl_command_queue rawQueue = clCreateCommandQueue(context, deviceId, 0, &err);
	checkError(err, "Failed to create command queue.");
	cl::CommandQueue queue(rawQueue);

	char deviceName[256];
	err = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
	checkError(err, "Failed to get device name.");

	std::cout << "Device: " << deviceName << std::endl;
#pragma endregion
	//ADDED
	std::vector<Ray> rays;

	//kepernyo
	int screenWidth = 512;
	int screenHeight = 512;
	int numPixels = screenHeight * screenWidth;

	std::vector<float> screenData(3 * numPixels, 0.0f); //3* mert minden pixelhez 3 csatorna kell (rgb), alfa itt meg nem kell -> csak hullamhossz van

	int raysPerSide = 1024;
	int numRays = raysPerSide * raysPerSide;
	rays.reserve(numRays);

	for (int y = 0; y < raysPerSide; ++y) {
		for (int x = 0; x < raysPerSide; ++x) {
			Ray r;

			//x es y koo-kat szetosztjuk [-0.9, +0.9] kozott, h lefedjuk az 1.0 sugaru gomb elejet
			float offsetX = ((x / (float)raysPerSide) * 1.8f) - 0.9f;
			float offsetY = ((y / (float)raysPerSide) * 1.8f) - 0.9f;

			r.origin = { offsetX, offsetY, -5.0f };
			r.direction = { 0.0f, 0.0f, 1.0f };

			//minden fenysugar kapjon egy egyenletesen eloszl hullamhosszat 380 és 780nm kozott 
			int rayIndex = y * raysPerSide + x;
			r.wavelength = 380.0f + ((rayIndex / (float)numRays) * 400.0f);

			r.intensity = 1.0f;
			rays.push_back(r);
		}
	}
	std::cout << "Successfully generated " << rays.size() << " spectral rays in a 2D Grid on the CPU." << std::endl;

	//buffer letrehozas fenysugarknak
	cl::Buffer gpuRaysBuffer(cl::Context(context), CL_MEM_READ_WRITE, sizeof(Ray) * numRays, nullptr, &err);
	checkError(err, "Failed to create GPU buffer for rays!");
	//buffer kepernyonek
	cl::Buffer gpuScreenBuffer(cl::Context(context), CL_MEM_READ_WRITE, sizeof(float) * screenData.size(), nullptr, &err);
	checkError(err, "Failed to create GPU buffer!");

	//masolas
	err = queue.enqueueWriteBuffer(gpuRaysBuffer, CL_TRUE, 0, sizeof(Ray) * numRays, rays.data());
	checkError(err, "Failed to write data to GPU buffer for screen!");
	err = queue.enqueueWriteBuffer(gpuScreenBuffer, CL_TRUE, 0, sizeof(float) * screenData.size(), screenData.data());
	checkError(err, "Failed to write data to GPU buffer for screen!");

	//kernel beolvasas
	std::ifstream kernelFile("../src/raytrace_kernel.cl");
	if (!kernelFile.is_open())
	{
		std::cerr << "Couldn't open raytrace.cl!" << std::endl;
		return -1;
	}
	std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());

	cl::Program::Sources sources;
	sources.push_back({ kernelSource.c_str(), kernelSource.length() });

	cl::Program program(cl::Context(context), sources);
	err = program.build({ cl::Device(deviceId) });
	if (err != CL_SUCCESS) {
		std::cout << " GPU build error: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device(deviceId)) << std::endl;
		return -1;
	}

	cl::Kernel kernel(program, "test_raytrace", &err);
	checkError(err, "Failed to create kernel!");

	//beallijuk a kernelt
	kernel.setArg(0, gpuRaysBuffer);
	kernel.setArg(1, numRays);
	kernel.setArg(2, gpuScreenBuffer);
	kernel.setArg(3, screenWidth);
	kernel.setArg(4, screenHeight);

	cl::NDRange globalSize(numRays);

	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, cl::NullRange);
	checkError(err, "Failed to launch kernel!");

	//megvarjuk a gpu-t
	queue.finish();

	//visszaolvassuk az eredmenyt
	//fenysugarak
	std::vector<Ray> modifiedRays(numRays);
	err = queue.enqueueReadBuffer(gpuRaysBuffer, CL_TRUE, 0, sizeof(Ray) * numRays, modifiedRays.data());
	checkError(err, "Failed to read rays data from GPU buffer!");
	//kepernyo
	err = queue.enqueueReadBuffer(gpuScreenBuffer, CL_TRUE, 0, sizeof(float) * screenData.size(), screenData.data());
	checkError(err, "Failed to read screen data from GPU.");


	//SZINKONVETZIO
	std::vector<unsigned char> rgbaData(4 * numPixels, 0);//itt mar 4 mert kell alfa csatorna is

	// GPU buffer az RGBA kepnek
	cl::Buffer gpuRgbaBuffer(cl::Context(context), CL_MEM_READ_WRITE, sizeof(unsigned char) * rgbaData.size(), nullptr, &err);
	checkError(err, "Failed to create RGBA buffer.");

	cl::Kernel colorKernel(program, "convert_xyz_to_srgb", &err);
	checkError(err, "Failed to create color kernel.");

	colorKernel.setArg(0, gpuScreenBuffer);
	colorKernel.setArg(1, gpuRgbaBuffer);
	colorKernel.setArg(2, screenWidth);
	colorKernel.setArg(3, screenHeight);

	cl::NDRange globalSize2D(screenWidth, screenHeight);

	err = queue.enqueueNDRangeKernel(colorKernel, cl::NullRange, globalSize2D, cl::NullRange);
	checkError(err, "Failed to launch color kernel.");

	queue.finish();

	err = queue.enqueueReadBuffer(gpuRgbaBuffer, CL_TRUE, 0, sizeof(unsigned char) * rgbaData.size(), rgbaData.data());
	checkError(err, "Failed to read RGBA data from GPU.");




	//TESTING

	std::cout << "\n--- First 4 rays new dir after GPU ---" << std::endl;
	for (int i = 0; i < 4; ++i) {
		std::cout << "Ray #" << i << " [" << modifiedRays[i].wavelength << " nm] -> "
			<< "New direction X: " << modifiedRays[i].direction.x
			<< " | Y: " << modifiedRays[i].direction.y
			<< " | Z: " << modifiedRays[i].direction.z << std::endl;
	}

	float totalEnergy = 0.0f;
	for (float val : screenData) {
		totalEnergy += val;
	}
	std::cout << "Simulation ended. The total energy was " << totalEnergy << " that hit the screen!" << std::endl;

	int activePixels = 0;
	int maxR = 0, maxG = 0, maxB = 0;
	int bestX = 0, bestY = 0;

	for (int y = 0; y < screenHeight; ++y) {
		for (int x = 0; x < screenWidth; ++x) {
			int idx = (y * screenWidth + x) * 4;
			int r = rgbaData[idx + 0];
			int g = rgbaData[idx + 1];
			int b = rgbaData[idx + 2];

			if (r > 0 || g > 0 || b > 0) {
				activePixels++;
				if (r + g + b > maxR + maxG + maxB) {
					maxR = r; maxG = g; maxB = b;
					bestX = x; bestY = y;
				}
			}
		}
	}

	std::cout << "\n--- PIC STATISTICS DEBUG ---" << std::endl;
	std::cout << "Colourful pixels: " << activePixels << " / " << numPixels << std::endl;
	if (activePixels > 0) {
		std::cout << "Lightest pixel at: (" << bestX << ", " << bestY << ")" << std::endl;
		std::cout << "Lightest pixel's colour -> R: " << maxR << " | G: " << maxG << " | B: " << maxB << std::endl;
	}
	else {
		std::cout << "ERORO: The pic is totally black! The  energy did not turn into colours or didnt hit the screen properly" << std::endl;
	}

	//kep mentese
	saveTGA("szivarvany.tga", rgbaData, screenWidth, screenHeight);

	return 0;
}


