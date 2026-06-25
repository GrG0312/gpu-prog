
#include <CL/cl2.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

#include "OpenCLUtils.hpp"
#include "rng_kernels.hpp"

using namespace std;

void runChiSquareTest(const vector<unsigned int>& hostOutput) {
    size_t N = hostOutput.size();

    // A: Sturges-rule
    // K = 1 + log2(N)
    unsigned int K = static_cast<unsigned int>(1 + std::log2(N));
    cout << "\n--- Statistics ---" << endl;
    cout << "Number of generated numbers(N): " << N << endl;
    cout << "Number of slot by Sturges-rule (K): " << K << endl;

    // B: Counting the slots's frequencies
    unsigned long long max_val = 4294967296ULL;
    unsigned long long bin_width = max_val / K; // Length of a slot

    vector<unsigned int> frequencies(K, 0);
    for (unsigned int num : hostOutput) {
        unsigned int bin_index = num / bin_width;
        if (bin_index >= K) bin_index = K - 1; // Correction for rounding
        frequencies[bin_index]++;
    }

    // C: Filtering (Throwing away those slots which have less than 600 numbers in it)
    const unsigned int MIN_FREQUENCY = 600;
    vector<unsigned int> valid_frequencies;
    for (unsigned int i = 0; i < K; ++i) {
        if (frequencies[i] >= MIN_FREQUENCY) {
            valid_frequencies.push_back(frequencies[i]);
        }
        else {
            cout << "Slot #" << i << " thrown, because it had less than " << MIN_FREQUENCY << " elements (" << frequencies[i] << ")" << endl;
        }
    }

    unsigned int valid_K = valid_frequencies.size();
    if (valid_K == 0) {
        cout << "Error: No valid slots left after filtering!" << endl;
        return;
    }

    // D: Counting the expected frequency 
    unsigned int total_remaining_elements = std::accumulate(valid_frequencies.begin(), valid_frequencies.end(), 0);
    double expected_frequency = static_cast<double>(total_remaining_elements) / valid_K;

    // E: Counting Chi square
    // O_i Observed frequency
    // E_i Expected frequency
    // chi^2 = sum( (O_i - E_i)^2 / E_i )
    double chi_square_stat = 0.0;
    for (unsigned int observed : valid_frequencies) {
        double diff = observed - expected_frequency;
        chi_square_stat += (diff * diff) / expected_frequency;
    }

    // F: Writing out the results
    cout << "Remaining slots: " << valid_K << endl;
    cout << "Necessary frequency for the slots: " << expected_frequency << endl;
    cout << "Statistics: " << chi_square_stat << endl;
    cout << "Degree of freedom(df): " << (valid_K - 1) << endl;
}


int main()
{
    #pragma region Generated
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
    #pragma endregion

    #pragma region Inicializing LCG
    unsigned int lcg_numWorkItems = 1024;                  //Number of threads   
    unsigned int lcg_randomsPerWorkItem = 160;             //Number of generated numbers per thread
    unsigned int lcg_N = lcg_numWorkItems * lcg_randomsPerWorkItem; // 163840
    unsigned long long lcg_seed = 43545ULL;

    vector<unsigned int> lcg_hostOutput(lcg_N); // Store for LCG values

    //Kernel
    cl_program lcg_program = clCreateProgramWithSource(context, 1, &lcg_kernel_code, nullptr, &err);
    checkError(err, "Failed to create LCG program.");
    err = clBuildProgram(lcg_program, 1, &deviceId, nullptr, nullptr, nullptr);
    checkError(err, "Failed to build LCG program.");
    cl_kernel lcg_kernel = clCreateKernel(lcg_program, "lcg_kernel", &err);
    checkError(err, "Failed to create LCG kernel.");

    // GPU buffer building
    cl_mem lcg_deviceOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * lcg_N, nullptr, &err);
    checkError(err, "Failed to create LCG buffer.");

    // Setting arguments for the kernel
    err = clSetKernelArg(lcg_kernel, 0, sizeof(cl_mem), &lcg_deviceOutput);
    err = clSetKernelArg(lcg_kernel, 1, sizeof(unsigned long long), &lcg_seed);
    err |= clSetKernelArg(lcg_kernel, 2, sizeof(unsigned int), &lcg_randomsPerWorkItem);
    checkError(err, "Failed to set LCG kernel args.");

    // Running the kernel
    size_t lcg_globalSize = lcg_numWorkItems;
    err = clEnqueueNDRangeKernel(queue, lcg_kernel, 1, nullptr, &lcg_globalSize, nullptr, 0, nullptr, nullptr);
    checkError(err, "Failed to enqueue LCG kernel.");

    //Reading back the values
    err = clEnqueueReadBuffer(queue, lcg_deviceOutput, CL_TRUE, 0, sizeof(unsigned int) * lcg_N, lcg_hostOutput.data(), 0, nullptr, nullptr);
    checkError(err, "Failed to read LCG buffer.");
    #pragma endregion

    #pragma region LCG Chi-test
    //Running the Chi-test
    cout << "\n=== RUNNING LCG CHI-SQUARE TEST ===" << endl;
    runChiSquareTest(lcg_hostOutput);
    #pragma endregion

    #pragma region Cleanup for LCG
    // Csak a specifikus LCG objektumokat töröljük, a context és queue kell még a xorshiftnek!
    clReleaseMemObject(lcg_deviceOutput);
    clReleaseKernel(lcg_kernel);
    clReleaseProgram(lcg_program);
    #pragma endregion

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
    err = clSetKernelArg(xor_kernel, 1, sizeof(unsigned int), &xor_seed);
    err = clSetKernelArg(xor_kernel, 2, sizeof(unsigned int), &xor_randomsPerWorkItem);
    checkError(err, "Failed to set xorshift kernel args.");

    // Running the kernel
    size_t xor_globalSize = xor_numWorkItems;
    err = clEnqueueNDRangeKernel(queue, xor_kernel, 1, nullptr, &xor_globalSize, nullptr, 0, nullptr, nullptr);
    checkError(err, "Failed to enqueue xorshift kernel.");

    //Reading back the values
    err = clEnqueueReadBuffer(queue, xor_deviceOutput, CL_TRUE, 0, sizeof(unsigned int) * xor_N, xor_hostOutput.data(), 0, nullptr, nullptr);
    checkError(err, "Failed to read xorshift buffer.");
    #pragma endregion

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
