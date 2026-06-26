#define _USE_MATH_DEFINES
#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

#include "OpenCLUtils.hpp"
#include "RngKernels.hpp"
#include "MonteCarloKernels.hpp"
#include "MonteCarloUtils.hpp"
#include "ChiSquareTest.hpp"
#include "HistogramExport.hpp"
#include "CpuBaseline.hpp"

using namespace std;

// ============================================================
//  Constants
// ============================================================

// Workgroup size used by every kernel. Must be a power of two and <= CL_DEVICE_MAX_WORK_GROUP_SIZE.
static const size_t LOCAL_SIZE = 64;

// All three generators produce this many values per work-item.
// Must be even so the Monte Carlo kernel can consume pairs.
static const unsigned int RANDOMS_PER_WORK_ITEM = 2^40;

static const unsigned int NUM_WORK_ITEMS = 1024;
static const unsigned int N = NUM_WORK_ITEMS * RANDOMS_PER_WORK_ITEM; // 262 144
static const unsigned int HIST_BINS = 50;


// ============================================================
//  main
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

    // Get and print the GPU name
    char deviceName[256];
    err = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    checkError(err, "Failed to get device name.");
    cout << "Device: " << deviceName << endl;
    cout << "N (values per generator): " << N << "  (work-items = " << NUM_WORK_ITEMS << "; randoms/item = " << RANDOMS_PER_WORK_ITEM << ")" << endl;

    // Verify the device supports the required workgroup size
    size_t maxWGSize;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWGSize), &maxWGSize, nullptr);
    checkError(err, "Failed to get device info.");
    if (LOCAL_SIZE > maxWGSize) {
        cerr << "LOCAL_SIZE (" << LOCAL_SIZE << ") exceeds device max (" << maxWGSize << ")." << endl;
        return 1;
    }

    cl_context ctx = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");

    // CL_QUEUE_PROFILING_ENABLE is required for profilingMs()
    cl_command_queue queue = clCreateCommandQueue(ctx, deviceId, CL_QUEUE_PROFILING_ENABLE, &err);
    checkError(err, "clCreateCommandQueue");

    #pragma endregion


    // Shared launch geometry used by every kernel
    const size_t globalSize = NUM_WORK_ITEMS;
    const size_t localSize = LOCAL_SIZE;

    // Monte Carlo parameters derived from the shared constants
    const unsigned int MC_NUM_GROUPS = NUM_WORK_ITEMS / LOCAL_SIZE;


    // --------------------------------------------------------
    //  LCG
    // --------------------------------------------------------
    #pragma region LCG

    cl_ulong   lcg_seed = 43545UL;//Kronos:64 bites, WIndows 32 bit, can't use unsigned long
    vector<unsigned int> lcg_output(N);

    CLKernel lcg = buildKernel(ctx, deviceId, lcg_kernel_code, "lcg_kernel", "LCG");
    cl_mem   lcg_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(unsigned int) * N, nullptr, &err);
    checkError(err, "LCG buffer error");

    err = clSetKernelArg(lcg.kernel, 0, sizeof(cl_mem), &lcg_buf);
    err |= clSetKernelArg(lcg.kernel, 1, sizeof(cl_ulong), &lcg_seed);
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

    // lcg_buf is intentionally NOT released here — reused by Monte Carlo below
    clReleaseKernel(lcg.kernel);
    clReleaseProgram(lcg.program);

    #pragma endregion


    // --------------------------------------------------------
    //  XORShift
    // --------------------------------------------------------
    #pragma region XORShift

    unsigned int         xor_seed = 86432U;
    vector<unsigned int> xor_output(N);

    CLKernel xorsh = buildKernel(ctx, deviceId, xorshift_kernel_code, "xorshift_kernel", "XORShift");
    cl_mem   xor_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(unsigned int) * N, nullptr, &err);
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
    checkError(err, "XORShift read error");

    double xor_ms = profilingMs(xor_event);
    clReleaseEvent(xor_event);
    cout << "\n[XORShift]  GPU time: " << xor_ms << " ms  (" << N << " values)" << endl;

    #pragma endregion
    

    #pragma region XORShift Chi-square + histogram

    runChiSquareTest(xor_output, "GPU XORShift");
    exportHistogramCSV(xor_output, HIST_BINS, "histogram_xorshift.csv");

    #pragma endregion

    #pragma region XORShift Kernel Cleanup

    // xor_buf intentionally kept alive for Monte Carlo
    clReleaseKernel(xorsh.kernel);
    clReleaseProgram(xorsh.program);

    #pragma endregion


    // --------------------------------------------------------
    //  Mersenne Twister
    // --------------------------------------------------------
    #pragma region Mersenne Twister

    unsigned int         mt_seed = 19650218U;
    const unsigned int   MT_STATE_WORDS = 624;
    vector<unsigned int> mt_output(N);

    CLKernel mt = buildKernel(ctx, deviceId, mt_kernel_code, "mt_kernel", "MT");
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

    // mt_buf intentionally kept alive for Monte Carlo
    clReleaseKernel(mt.kernel);
    clReleaseProgram(mt.program);

    #pragma endregion


    // --------------------------------------------------------
    //  CPU baseline
    // --------------------------------------------------------
    runCPUBaseline(N, HIST_BINS);


    // --------------------------------------------------------
    //  Performance summary
    // --------------------------------------------------------
    cout << "\n=== PERFORMANCE SUMMARY ===" << endl;
    cout << "  Generator  | Time (ms) | Values | Throughput" << endl;
    cout << "  -----------|-----------|--------|--------------------" << endl;
    cout << "  LCG        | " << lcg_ms << " | " << N << " | " << (N / lcg_ms / 1000.0) << " M/ms" << endl;
    cout << "  XORShift   | " << xor_ms << " | " << N << " | " << (N / xor_ms / 1000.0) << " M/ms" << endl;
    cout << "  MT         | " << mt_ms << " | " << N << " | " << (N / mt_ms / 1000.0) << " M/ms" << endl;


    // --------------------------------------------------------
    //  Monte Carlo Pi estimation
    //
    //  One shared results buffer is reused across all three runs.
    //  runMonteCarlo() handles only the mechanical execution and
    //  returns the raw partial hit counts. Pi is computed here
    //  since that interpretation is specific to this simulation.
    // --------------------------------------------------------
    #pragma region Monte Carlo Pi estimation

    // numPairs = total (x,y) pairs across all work-items
    const unsigned long MC_TOTAL_PAIRS = static_cast<unsigned long>(NUM_WORK_ITEMS) * (RANDOMS_PER_WORK_ITEM / 2);

    CLKernel mc_pi = buildKernel(ctx, deviceId, monte_carlo_pi_kernel_code, "monte_carlo_pi_kernel", "MC-Pi");
    cl_mem   mc_hits = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(unsigned int) * MC_NUM_GROUPS, nullptr, &err);
    checkError(err, "MC hits buffer error");

    // Sums partial hit counts and estimates Pi
    auto computePi = [&](const vector<unsigned int>& partial) -> double {
        unsigned long total = 0;
        for (unsigned int h : partial) total += h;
        return 4.0 * static_cast<double>(total) / static_cast<double>(MC_TOTAL_PAIRS);
        };

    cout << "\n=== MONTE CARLO PI ESTIMATION (pairs per generator: " << MC_TOTAL_PAIRS << ") ===" << endl;
    cout << "  True Pi: " << M_PI << endl;
    cout << endl;
    cout << "  Generator  | Estimated Pi | Abs. error   | MC time" << endl;
    cout << "  -----------|--------------|--------------|--------" << endl;

    double mc_ms;
    vector<unsigned int> mc_partial;

    mc_partial = runMonteCarlo(queue, mc_pi.kernel, lcg_buf, mc_hits, RANDOMS_PER_WORK_ITEM, MC_NUM_GROUPS, globalSize, localSize, mc_ms, "MC-LCG");
    double pi_lcg = computePi(mc_partial);
    cout << "  LCG        | " << pi_lcg << "  | " << abs(pi_lcg - M_PI) << "  | " << mc_ms << " ms" << endl;

    mc_partial = runMonteCarlo(queue, mc_pi.kernel, xor_buf, mc_hits, RANDOMS_PER_WORK_ITEM, MC_NUM_GROUPS, globalSize, localSize, mc_ms, "MC-XORShift");
    double pi_xor = computePi(mc_partial);
    cout << "  XORShift   | " << pi_xor << "  | " << abs(pi_xor - M_PI) << "  | " << mc_ms << " ms" << endl;

    mc_partial = runMonteCarlo(queue, mc_pi.kernel, mt_buf, mc_hits, RANDOMS_PER_WORK_ITEM, MC_NUM_GROUPS, globalSize, localSize, mc_ms, "MC-MT");
    double pi_mt = computePi(mc_partial);
    cout << "  MT         | " << pi_mt << "  | " << abs(pi_mt - M_PI) << "  | " << mc_ms << " ms" << endl;

    // Cleanup for Monte Carlo - Pi Estimation
    clReleaseMemObject(mc_hits);
    clReleaseKernel(mc_pi.kernel);
    clReleaseProgram(mc_pi.program);

    #pragma endregion


    // --------------------------------------------------------
    //  RNG buffer cleanup
    // --------------------------------------------------------
    #pragma region RNG Buffer Cleanup

    clReleaseMemObject(lcg_buf);
    clReleaseMemObject(xor_buf);
    clReleaseMemObject(mt_buf);

    #pragma endregion


    // --------------------------------------------------------
    //  Global OpenCL cleanup
    // --------------------------------------------------------
    #pragma region Global OpenCL Cleanup

    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    #pragma endregion

    return 0;
}