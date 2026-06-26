#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

#include "OpenCLUtils.hpp"
#include "rng_kernels.hpp"
#include "ChiSquareTest.hpp"
#include "HistogramExport.hpp"
#include "CPUBaseline.hpp"

using namespace std;

// ============================================================
//  Constants
// ============================================================

// Workgroup size used by every kernel. Must be a power of two and <= CL_DEVICE_MAX_WORK_GROUP_SIZE.
static const size_t LOCAL_SIZE = 64;

// All three generators produce this many values per work-item
// Must be even so the Monte Carlo kernel can consume pairs
static const unsigned int RANDOMS_PER_WORK_ITEM = 256;

static const unsigned int NUM_WORK_ITEMS = 1024;
static const unsigned int N = NUM_WORK_ITEMS * RANDOMS_PER_WORK_ITEM; // 262 144
static const unsigned int HIST_BINS = 50;

// ============================================================
//  OpenCL helpers
// ============================================================
struct CLKernel { cl_program program; cl_kernel kernel; };

CLKernel buildKernel(cl_context ctx, cl_device_id dev, const char* src, const char* funcName, const char* label)
{
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    checkError(err, label);
    buildProgramWithLog(prog, dev, label);
    cl_kernel k = clCreateKernel(prog, funcName, &err);
    checkError(err, label);
    return { prog, k };
}

// Reads START and END profiling timestamps from a cl_event and returns the elapsed time in milliseconds
double profilingMs(cl_event ev)
{
    unsigned long t0, t1;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(unsigned long), &t0, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(unsigned long), &t1, nullptr);
    return static_cast<double>(t1 - t0) * 1e-6;
}

// Sums partial hit counts and computes the Pi estimate
// numPairs = total (x,y) pairs across all work-items
double computePi(const vector<unsigned int>& partialHits, unsigned long numPairs)
{
    unsigned long total = 0;
    for (unsigned int h : partialHits) total += h;
    return 4.0 * static_cast<double>(total) / static_cast<double>(numPairs);
}


// ============================================================
//  main function
// ============================================================
int main()
{
    // --------------------------------------------------------
    //  OpenCL setup
    // --------------------------------------------------------
    #pragma region OpenCL Setup

    cl_int err;
    cl_platform_id platform;
    cl_device_id deviceId;

    // Get the first available platform
    err = clGetPlatformIDs(1, &platform, nullptr);
    checkError(err, "Failed to get OpenCL platform.");

    // Get the first available GPU device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &deviceId, nullptr);
    checkError(err, "Failed to get OpenCL device.");

    // Get the GPU name
    char deviceName[256];
    err = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    checkError(err, "Failed to get device name.");

    // Print data
    cout << "Device: " << deviceName << endl;
    cout << "N (values per generator): " << N << "  (work-items = " << NUM_WORK_ITEMS << "; randoms/item = " << RANDOMS_PER_WORK_ITEM << ")" << endl;

    // Get max workgroup size for GPU device
    size_t maxWGSize;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWGSize), &maxWGSize, nullptr);
    checkError(err, "Failed to get device info");

    if (LOCAL_SIZE > maxWGSize) {
        cerr << "LOCAL_SIZE (" << LOCAL_SIZE << ") exceeds device max (" << maxWGSize << ")." << endl;
        return 1;
    }

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkError(err, "clCreateCommandQueue");

    #pragma endregion


    const size_t globalSize = NUM_WORK_ITEMS;
    const size_t localSize = LOCAL_SIZE;

    // Monte Carlo: each work-item contributes (RANDOMS_PER_WORK_ITEM / 2) pairs
    const unsigned int MC_NUM_GROUPS = NUM_WORK_ITEMS / LOCAL_SIZE;
    const unsigned long MC_TOTAL_PAIRS = static_cast<unsigned long>(NUM_WORK_ITEMS) * (RANDOMS_PER_WORK_ITEM / 2);


    // --------------------------------------------------------
    //  LCG
    // --------------------------------------------------------
    #pragma region LCG

    // Seed for LCG
    unsigned long lcg_seed = 43545UL;
    // Results of LCG
    vector<unsigned int> lcg_hostOutput(N);

    CLKernel lcg = buildKernel(ctx, device, lcg_kernel_code, "lcg_kernel", "LCG");
    cl_mem lcg_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(unsigned int) * N, nullptr, &err);
    checkError(err, "LCG buffer error");

    err = clSetKernelArg(lcg.kernel, 0, sizeof(cl_mem), &lcg_buf);
    err |= clSetKernelArg(lcg.kernel, 1, sizeof(unsigned long), &lcg_seed);
    err |= clSetKernelArg(lcg.kernel, 2, sizeof(unsigned int), &RANDOMS_PER_WORK_ITEM);
    err |= clSetKernelArg(lcg.kernel, 3, sizeof(unsigned int) * LOCAL_SIZE, nullptr);
    checkError(err, "LCG arguments error");

    cl_event lcg_event;

    err = clEnqueueNDRangeKernel(queue, lcg.kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, &lcg_event);
    checkError(err, "LCG enqueue error");

    err = clEnqueueReadBuffer(queue, lcg_buf, CL_TRUE, 0, sizeof(unsigned int) * N, lcg_output.data(), 0, nullptr, nullptr);
    checkError(err, "LCG read error");

    double lcg_ms = profilingMs(lcg_event);
    clReleaseEvent(lcg_event);
    cout << "\n[LCG]  GPU time: " << lcg_ms << " ms  (" << N << " values)" << endl;

    #pragma endregion

    #pragma region LCG Chi-square + histogram

    runChiSquareTest(lcg_output, "GPU LCG");
    exportHistogramCSV(lcg_output, HIST_BINS, "histogram_lcg.csv");

    #pragma endregion

    #pragma region LCG Kernel Cleanup

    // lcg_buf is intentionally NOT released here, it is passed to the Monte Carlo kernel later on
    clReleaseKernel(lcg.kernel);
    clReleaseProgram(lcg.program);

    #pragma endregion

    // --------------------------------------------------------
    //  XORSHIFT
    // --------------------------------------------------------

    #pragma region Inicializing xorshift
    unsigned int xor_numWorkItems = 1024;                  // Number of threads   
    unsigned int xor_randomsPerWorkItem = 200;            //Number of generated numbers per thread
    unsigned int xor_N = xor_numWorkItems * xor_randomsPerWorkItem;
    unsigned int xor_seed = 86432U;

    vector<unsigned int> xor_hostOutput(xor_N); // Store for xorshift values

    //Kernel
    cl_program xor_program = clCreateProgramWithSource(context, 1, &xorshift_kernel_code, nullptr, &err);
    checkError(err, "Failed to create xorshift program.");
    err = clBuildProgram(xor_program, 1, &deviceId, nullptr, nullptr, nullptr);
    checkError(err, "Failed to build xorshift program.");
    cl_kernel xor_kernel = clCreateKernel(xor_program, "xorshift_kernel", &err);
    checkError(err, "Failed to create xorshift kernel.");

    // GPU buffer building
    cl_mem xor_deviceOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * xor_N, nullptr, &err);
    checkError(err, "Failed to create xorshift buffer.");

    //Setting the arguments for 
    err = clSetKernelArg(xor_kernel, 0, sizeof(cl_mem), &xor_deviceOutput);
    err |= clSetKernelArg(xor_kernel, 1, sizeof(unsigned int), &xor_seed);
    err |= clSetKernelArg(xor_kernel, 2, sizeof(unsigned int), &xor_randomsPerWorkItem);
    checkError(err, "Failed to set xorshift kernel args.");

    // Running the kernel
    size_t xor_globalSize = xor_numWorkItems;
    err = clEnqueueNDRangeKernel(queue, xor_kernel, 1, nullptr, &xor_globalSize, nullptr, 0, nullptr, nullptr);
    checkError(err, "Failed to enqueue xorshift kernel.");

    //Reading back the values
    err = clEnqueueReadBuffer(queue, xor_deviceOutput, CL_TRUE, 0, sizeof(unsigned int) * xor_N, xor_hostOutput.data(), 0, nullptr, nullptr);
    checkError(err, "Failed to read xorshift buffer.");
    #pragma endregion

    /*

    //==================\\
    ||      XOR CHI     ||
    \\==================//

    */

    #pragma region xorshift Chi-test
    //Running the Chi-test
    cout << "\n=== RUNNING XORSHIFT CHI-SQUARE TEST ===" << endl;
    runChiSquareTest(xor_hostOutput);
    #pragma endregion

    #pragma region Cleanup for xorshift
    clReleaseMemObject(xor_deviceOutput);
    clReleaseKernel(xor_kernel);
    clReleaseProgram(xor_program);
    #pragma endregion
    //5 percent rule 
    #pragma region Global OpenCL Cleanup
    //Closing the context and the que
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    #pragma endregion

    return 0;
}
