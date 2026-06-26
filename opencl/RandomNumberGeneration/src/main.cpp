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

    #pragma region Inicializing XORShift

    // Seed for XORShift
    unsigned int xor_seed = 86432U;
    // Results for XORShift
    vector<unsigned int> xor_output(N);

    CLKernel xorsh = buildKernel(ctx, device, xorshift_kernel_code, "xorshift_kernel", "XORShift");
    cl_mem xor_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(unsigned int) * N, nullptr, &err);
    checkError(err, "XORShift buffer error");

    err = clSetKernelArg(xorsh.kernel, 0, sizeof(cl_mem), &xor_buf);
    err |= clSetKernelArg(xorsh.kernel, 1, sizeof(unsigned int), &xor_seed);
    err |= clSetKernelArg(xorsh.kernel, 2, sizeof(unsigned int), &RANDOMS_PER_WORK_ITEM);
    err |= clSetKernelArg(xorsh.kernel, 3, sizeof(unsigned int) * LOCAL_SIZE, nullptr);
    checkError(err, "XORShift args error");

    cl_event xor_event;

    err = clEnqueueNDRangeKernel(queue, xorsh.kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, &xor_event);
    checkError(err, "XORShift enqueue error");

    err = clEnqueueReadBuffer(queue, xor_buf, CL_TRUE, 0, sizeof(unsigned int) * N, xor_output.data(), 0, nullptr, nullptr);
    checkError(err, "XORShift read");

    double xor_ms = profilingMs(xor_event);
    clReleaseEvent(xor_event);

    cout << "\n[XORShift]  GPU time: " << xor_ms << " ms  (" << N << " values)" << endl;
    #pragma endregion

    #pragma region XORShift Chi-square + histogram

    runChiSquareTest(xor_output, "GPU XORShift");
    exportHistogramCSV(xor_output, HIST_BINS, "histogram_xorshift.csv");

    #pragma endregion

    #pragma region XORShift Kernel Cleanup

    clReleaseKernel(xorsh.kernel);
    clReleaseProgram(xorsh.program);

    #pragma endregion



    // --------------------------------------------------------
    //  MERSENNE TWISTER
    // --------------------------------------------------------

    #pragma region Initializing Mersenne Twister

    // Seed for MT
    unsigned int mt_seed = 19650218U;
    const unsigned int MT_STATE_WORDS = 624;
    // Results for MT
    vector<unsigned int> mt_output(N);

    CLKernel mt = buildKernel(ctx, device, mt_kernel_code, "mt_kernel", "MT");
    cl_mem   mt_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(unsigned int) * N, nullptr, &err);
    checkError(err, "MT output buffer error");

    cl_mem   mt_state = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(unsigned int) * NUM_WORK_ITEMS * MT_STATE_WORDS, nullptr, &err);
    checkError(err, "MT state buffer error");

    err = clSetKernelArg(mt.kernel, 0, sizeof(cl_mem), &mt_buf);
    err |= clSetKernelArg(mt.kernel, 1, sizeof(cl_mem), &mt_state);
    err |= clSetKernelArg(mt.kernel, 2, sizeof(unsigned int), &mt_seed);
    err |= clSetKernelArg(mt.kernel, 3, sizeof(unsigned int), &RANDOMS_PER_WORK_ITEM);
    err |= clSetKernelArg(mt.kernel, 4, sizeof(unsigned int) * LOCAL_SIZE, nullptr);
    checkError(err, "MT args error");

    cl_event mt_event;
    err = clEnqueueNDRangeKernel(queue, mt.kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, &mt_event);
    checkError(err, "MT enqueue error");

    err = clEnqueueReadBuffer(queue, mt_buf, CL_TRUE, 0, sizeof(unsigned int) * N, mt_output.data(), 0, nullptr, nullptr);
    checkError(err, "MT read error");

    double mt_ms = profilingMs(mt_event);
    clReleaseEvent(mt_event);
    clReleaseMemObject(mt_state); // state buffer no longer needed after generation
    cout << "\n[MT]  GPU time: " << mt_ms << " ms  (" << N << " values)" << endl;

    #pragma endregion

    #pragma region MT Chi-square + histogram

    runChiSquareTest(mt_output, "GPU Mersenne Twister");
    exportHistogramCSV(mt_output, HIST_BINS, "histogram_mt.csv");

    #pragma endregion

    #pragma region MT Kernel Cleanup

    clReleaseKernel(mt.kernel);
    clReleaseProgram(mt.program);

    #pragma endregion



    // --------------------------------------------------------
    //  CPU runs
    // --------------------------------------------------------
    runCPUBaseline(N, HIST_BINS);



    // --------------------------------------------------------
    //  Performance summary
    // --------------------------------------------------------
    cout << "\n=== PERFORMANCE SUMMARY ===" << endl;
    cout << "  Generator  | Time (ms) | Values    | Throughput" << endl;
    cout << "  -----------|-----------|-----------|--------------------" << endl;
    cout << "  LCG        | " << lcg_ms << " | " << N << " | " << (N / lcg_ms / 1000.0) << " M/ms" << endl;
    cout << "  XORShift   | " << xor_ms << " | " << N << " | " << (N / xor_ms / 1000.0) << " M/ms" << endl;
    cout << "  MT         | " << mt_ms  << " | " << N << " | " << (N / mt_ms / 1000.0)  << " M/ms" << endl;

    return 0;
}
