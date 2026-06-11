
#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <vector>

#include "OpenCLUtils.hpp"

using namespace std;

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

	cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, &err);
	checkError(err, "Failed to create command queue.");

	char deviceName[256];
	err = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
	checkError(err, "Failed to get device name.");

	std::cout << "Device: " << deviceName << std::endl;
	return 0;
}
