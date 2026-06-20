
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
	float wavelenght; //nanometerben
	float intensity;
};

int main()
{
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

	//ADDED
	std::vector<Ray> rays;

	int numRays = 1000; //egyenlore konstans
	for (int i = 0; i < numRays; i++)
	{
		Ray r;
		float offset = (i - numRays / 2) * 0.002f;

		r.origin = { offset, 0.0f, -5.0f };
		r.direction = { 0.0f, 0.0f, 1.0f };

		r.wavelenght = 550.0f; //zöld
		r.intensity = 1.0f;

		rays.push_back(r);
	}

	std::cout << "Successfully generated " << rays.size() << " rays on the CPU." << std::endl;

	//buffer letrehozas
	cl::Buffer gpuRaysBuffer(cl::Context(context), CL_MEM_READ_WRITE, sizeof(Ray) * numRays, nullptr, &err);
	checkError(err, "Failed to create GPU buffer!");

	//masolas
	err = queue.enqueueWriteBuffer(gpuRaysBuffer, CL_TRUE, 0, sizeof(Ray) * numRays, rays.data());
	checkError(err, "Failed to write data to GPU buffer!");

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

	cl::NDRange globalSize(numRays);

	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, cl::NullRange);
	checkError(err, "Failed to launch kernel!");

	//megvarjuk a gpu-t
	queue.finish();

	//visszaolvassuk az eredmenyt
	std::vector<Ray> modifiedRays(numRays);
	err = queue.enqueueReadBuffer(gpuRaysBuffer, CL_TRUE, 0, sizeof(Ray) * numRays, modifiedRays.data());
	checkError(err, "Failed to read data from GPU buffer!");

	std::cout << "Original first ray Z: " << rays[0].origin.z << std::endl;
	std::cout << "GPU changed it to Z: " << modifiedRays[0].origin.z << std::endl;

	return 0;
}
