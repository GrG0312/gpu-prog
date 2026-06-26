
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> //idomereshez

#include <Windows.h>
#define GLEW_STATIC
#include <glew.v140.1.12.0/build/native/include/GL/glew.h>
#include <glfw.3.4.0/build/native/include/GLFW/glfw3.h>
#pragma comment(lib, "opengl32.lib")

#include "OpenCLUtils.hpp"

using namespace std;

//-----------SKALAR-------------
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

//-----------VEKTOROS-------------
struct alignas(16) Vector4 {
	float x, y, z, w;
};

struct alignas(16) RayVectorized {
	Vector4 origin;
	Vector4 direction;
	float wavelength;
	float intensity;
	float dummy1;
	float dummy2;
};

//interaktiv nezethet
float g_rotationAngle = 0.0f;
double g_lastX = 0.0, g_lastY = 0.0;
bool g_leftMouseButtonPressed = false;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			g_leftMouseButtonPressed = true;
			glfwGetCursorPos(window, &g_lastX, &g_lastY);
		}
		else if (action == GLFW_RELEASE) {
			g_leftMouseButtonPressed = false;
		}
	}
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
	if (g_leftMouseButtonPressed) {
		double deltaX = xpos - g_lastX;
		g_rotationAngle += static_cast<float>(deltaX) * 0.005f;
		g_lastX = xpos;
		g_lastY = ypos;
	}
}

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

void runInteractiveOpenGL(cl_context context, cl_device_id deviceId, cl_command_queue rawQueue,
	cl::Program& programVector, cl::Program& programScalar,
	cl::Buffer& bufRays, int numRays, int width, int height, cl::NDRange globalSize,
	const std::vector<RayVectorized>& vectorRays,
	cl::Buffer& bufScreen, cl::Buffer& bufRgba)
{
	//GLFW ablak letrehozasa
	if (!glfwInit()) return;
	GLFWwindow* window = glfwCreateWindow(width, height, "Iteractive Rainbow Simulation - GPU Memory Sharing", NULL, NULL);
	if (!window) { glfwTerminate(); return; }
	glfwMakeContextCurrent(window);

	//GLEW init
	glewInit();

	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);

	//OpenGL textura letrehozata -> ebbe fog a gpu kozvetlenul beleirni
	GLuint glTexture;
	glGenTextures(1, &glTexture);
	glBindTexture(GL_TEXTURE_2D, glTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	std::vector<float> screenData(3 * width * height, 0.0f);
	std::vector<unsigned char> hostRgba(width * height * 4, 0);

	cl::Kernel kernelVector(programVector, "test_raytrace_vectorized");
	cl::Kernel colorKernel(programScalar, "convert_xyz_to_srgb");
	cl::CommandQueue queue(rawQueue);

	std::cout << "\n>>> Interactive window started! Click and drag the mouse around! <<<" << std::endl;

	while (!glfwWindowShouldClose(window)) {//render
		glfwPollEvents();

		std::fill(screenData.begin(), screenData.end(), 0.0f);
		queue.enqueueWriteBuffer(bufScreen, CL_TRUE, 0, sizeof(float) * screenData.size(), screenData.data());

		queue.enqueueWriteBuffer(bufRays, CL_TRUE, 0, sizeof(RayVectorized) * numRays, vectorRays.data());

		kernelVector.setArg(0, bufRays);
		kernelVector.setArg(1, numRays);
		kernelVector.setArg(2, bufScreen);
		kernelVector.setArg(3, width);
		kernelVector.setArg(4, height);
		kernelVector.setArg(5, g_rotationAngle);

		queue.enqueueNDRangeKernel(kernelVector, cl::NullRange, globalSize, cl::NullRange);

		colorKernel.setArg(0, bufScreen);
		colorKernel.setArg(1, bufRgba);
		colorKernel.setArg(2, width);
		colorKernel.setArg(3, height);
		cl::NDRange globalSize2D(width, height);
		queue.enqueueNDRangeKernel(colorKernel, cl::NullRange, globalSize2D, cl::NullRange);

		queue.finish();

		//frissitjuk a texturat
		//FRISSULJ MAR
		queue.enqueueReadBuffer(bufRgba, CL_TRUE, 0, hostRgba.size(), hostRgba.data());

		glBindTexture(GL_TEXTURE_2D, glTexture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, hostRgba.data());

		glClear(GL_COLOR_BUFFER_BIT);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, glTexture);

		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
		glEnd();

		glfwSwapBuffers(window);
	}

	glDeleteTextures(1, &glTexture);
	glfwDestroyWindow(window);
	glfwTerminate();
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
	std::vector<Ray> scalarRays;
	std::vector<RayVectorized> vectorRays;

	//kepernyo
	int screenWidth = 512;
	int screenHeight = 512;
	int numPixels = screenHeight * screenWidth;

	std::vector<float> screenData(3 * numPixels, 0.0f); //3* mert minden pixelhez 3 csatorna kell (rgb), alfa itt meg nem kell -> csak hullamhossz van

	int raysPerSide = 2048;
	int numRays = raysPerSide * raysPerSide;
	scalarRays.reserve(numRays);
	vectorRays.reserve(numRays);

	cl::NDRange globalSize(numRays);

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
			float wave = 380.0f + ((rayIndex / (float)numRays) * 400.0f);
			r.wavelength = wave;

			r.intensity = 1.0f;
			scalarRays.push_back(r);

			//vektoros fenysugar
			RayVectorized r2;
			r2.origin = { offsetX, offsetY, -5.0f, 1.0f };
			r2.direction = { 0.0f, 0.0f, 1.0f, 0.0f };
			r2.wavelength = wave;
			r2.intensity = 1.0f;
			r2.dummy1 = 0.0f; r2.dummy2 = 0.0f;
			vectorRays.push_back(r2);
		}
	}
	std::cout << "Successfully generated " << scalarRays.size() << " spectral rays in a 2D Grid on the CPU." << std::endl;

	////-----------SKALAR BEOLVASAS ES FORDITAS-------------
	//kernel beolvasas
	std::ifstream f1("../src/raytrace_kernel.cl");
	if (!f1.is_open())
	{
		std::cerr << "Couldn't open raytrace.cl!" << std::endl;
		return -1;
	}
	std::string src1((std::istreambuf_iterator<char>(f1)), std::istreambuf_iterator<char>());
	cl::Program::Sources sources1;
	sources1.push_back({ src1.c_str(), src1.length() });
	cl::Program programScalar(cl::Context(context), sources1);
	err = programScalar.build({ cl::Device(deviceId) });
	if (err != CL_SUCCESS) {
		std::cout << " GPU build error: " << programScalar.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device(deviceId)) << std::endl;
		return -1;
	}

	////-----------VEKTOROS BEOLVASAS ES FORDITAS-------------
	//kernel beolvasas
	std::ifstream f2("../src/raytrace_vectorized_kernel.cl");
	if (!f2.is_open())
	{
		std::cerr << "Couldn't open raytrace_vectorized_kernel.cl!" << std::endl;
		return -1;
	}
	std::string src2((std::istreambuf_iterator<char>(f2)), std::istreambuf_iterator<char>());
	cl::Program::Sources sources2;
	sources2.push_back({ src2.c_str(), src2.length() });
	cl::Program programVector(cl::Context(context), sources2);
	err = programVector.build({ cl::Device(deviceId) });
	if (err != CL_SUCCESS) {
		std::cout << " GPU build error: " << programVector.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device(deviceId)) << std::endl;
		return -1;
	}

	////-----------SKALAR BEOLVASAS ES FORDITAS-------------
	//buffer letrehozas fenysugarknak
	cl::Buffer gpuRaysBuffer1(cl::Context(context), CL_MEM_READ_WRITE, sizeof(Ray) * numRays, nullptr, &err);
	checkError(err, "Failed to create GPU buffer for rays!");
	//buffer kepernyonek
	cl::Buffer gpuScreenBuffer1(cl::Context(context), CL_MEM_READ_WRITE, sizeof(float) * screenData.size(), nullptr, &err);
	checkError(err, "Failed to create GPU buffer!");

	//masolas
	err = queue.enqueueWriteBuffer(gpuRaysBuffer1, CL_TRUE, 0, sizeof(Ray) * numRays, scalarRays.data());
	checkError(err, "Failed to write data to GPU buffer for screen!");
	err = queue.enqueueWriteBuffer(gpuScreenBuffer1, CL_TRUE, 0, sizeof(float) * screenData.size(), screenData.data());
	checkError(err, "Failed to write data to GPU buffer for screen!");

	cl::Kernel kernelScalar(programScalar, "test_raytrace", &err);
	checkError(err, "Failed to create kernel!");

	//beallijuk a kernelt
	kernelScalar.setArg(0, gpuRaysBuffer1);
	kernelScalar.setArg(1, numRays);
	kernelScalar.setArg(2, gpuScreenBuffer1);
	kernelScalar.setArg(3, screenWidth);
	kernelScalar.setArg(4, screenHeight);

	auto t1_start = std::chrono::high_resolution_clock::now();//merjuk az idot
	err = queue.enqueueNDRangeKernel(kernelScalar, cl::NullRange, globalSize, cl::NullRange);
	checkError(err, "Failed to launch kernel!");
	queue.finish();//megvarjuk a gpu-t
	auto t1_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> durationScalar = t1_end - t1_start;

	////-----------VEKTOROS BEOLVASAS ES FORDITAS-------------
	std::fill(screenData.begin(), screenData.end(), 0.0f);
	cl::Buffer gpuRaysBuffer2(cl::Context(context), CL_MEM_READ_WRITE, sizeof(RayVectorized) * numRays, nullptr, &err);
	checkError(err, "Failed to create GPU buffer for rays!");
	cl::Buffer gpuScreenBuffer2(cl::Context(context), CL_MEM_READ_WRITE, sizeof(float) * screenData.size(), nullptr, &err);
	checkError(err, "Failed to create GPU buffer!");
	err = queue.enqueueWriteBuffer(gpuRaysBuffer2, CL_TRUE, 0, sizeof(RayVectorized) * numRays, vectorRays.data());
	checkError(err, "Failed to write data to GPU buffer for screen!");
	err = queue.enqueueWriteBuffer(gpuScreenBuffer2, CL_TRUE, 0, sizeof(float) * screenData.size(), screenData.data());
	checkError(err, "Failed to write data to GPU buffer for screen!");

	cl::Kernel kernelVector(programVector, "test_raytrace_vectorized", &err);
	checkError(err, "Failed to create kernel!");

	kernelVector.setArg(0, gpuRaysBuffer2);
	kernelVector.setArg(1, numRays);
	kernelVector.setArg(2, gpuScreenBuffer2);
	kernelVector.setArg(3, screenWidth);
	kernelVector.setArg(4, screenHeight);

	auto t2_start = std::chrono::high_resolution_clock::now();
	queue.enqueueNDRangeKernel(kernelVector, cl::NullRange, globalSize, cl::NullRange);
	queue.finish();
	auto t2_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> durationVector = t2_end - t2_start;


	//visszaolvassuk az eredmenyt
	//fenysugarak
	std::vector<Ray> modifiedRays(numRays);
	err = queue.enqueueReadBuffer(gpuRaysBuffer2, CL_TRUE, 0, sizeof(Ray) * numRays, modifiedRays.data());
	checkError(err, "Failed to read rays data from GPU buffer!");
	//kepernyo
	err = queue.enqueueReadBuffer(gpuScreenBuffer2, CL_TRUE, 0, sizeof(float) * screenData.size(), screenData.data());
	checkError(err, "Failed to read screen data from GPU.");


	//SZINKONVETZIO
	std::vector<unsigned char> rgbaData(4 * numPixels, 0);//itt mar 4 mert kell alfa csatorna is

	// GPU buffer az RGBA kepnek
	cl::Buffer gpuRgbaBuffer(cl::Context(context), CL_MEM_READ_WRITE, sizeof(unsigned char) * rgbaData.size(), nullptr, &err);
	checkError(err, "Failed to create RGBA buffer.");

	cl::Kernel colorKernel(programScalar, "convert_xyz_to_srgb", &err);
	checkError(err, "Failed to create color kernel.");

	colorKernel.setArg(0, gpuScreenBuffer1);
	colorKernel.setArg(1, gpuRgbaBuffer);
	colorKernel.setArg(2, screenWidth);
	colorKernel.setArg(3, screenHeight);

	cl::NDRange globalSize2D(screenWidth, screenHeight);

	err = queue.enqueueNDRangeKernel(colorKernel, cl::NullRange, globalSize2D, cl::NullRange);
	checkError(err, "Failed to launch color kernel.");

	queue.finish();

	err = queue.enqueueReadBuffer(gpuRgbaBuffer, CL_TRUE, 0, sizeof(unsigned char) * rgbaData.size(), rgbaData.data());
	checkError(err, "Failed to read RGBA data from GPU.");

	//TELJESITMENY OSSZEHASONLITAS---
	std::cout << "\n=========================================" << std::endl;
	std::cout << "       PERFORMANCE COMPARISON            " << std::endl;
	std::cout << "=========================================" << std::endl;
	std::cout << "Scalar Kernel Time   : " << durationScalar.count() << " ms" << std::endl;
	std::cout << "Vectorized Kernel Time: " << durationVector.count() << " ms" << std::endl;
	std::cout << "Speedup              : " << (durationScalar.count() / durationVector.count()) << "x" << std::endl;
	std::cout << "=========================================\n" << std::endl;

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

	//interaktiv nezet inditasa
	runInteractiveOpenGL(context, deviceId, rawQueue, programVector, programScalar, gpuRaysBuffer2, numRays, screenWidth, screenHeight, globalSize, vectorRays, gpuScreenBuffer2 , gpuRgbaBuffer);

	return 0;
}


