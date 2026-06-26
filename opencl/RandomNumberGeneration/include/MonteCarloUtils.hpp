#pragma once

#include <vector>

#include "OpenCLUtils.hpp"

// Runs a Monte Carlo kernel against a pre-generated RNG buffer.
//
// Handles only the mechanical execution: argument binding, kernel launch,
// result readback, and timing. The caller is responsible for interpreting
// the returned partial results and printing any output.
//
// Parameters:
//   queue              - command queue (must have CL_QUEUE_PROFILING_ENABLE)
//   kernel             - compiled Monte Carlo kernel; expected argument layout:
//                          0: __global uint* partialResults  (one entry per workgroup)
//                          1: __global uint* rngBuffer       (pre-generated values)
//                          2: uint           randomsPerWorkItem
//                          3: __local  uint* localBuf        (LOCAL_SIZE slots)
//   rngBuffer          - GPU buffer produced by any RNG kernel
//   resultsBuffer      - pre-allocated output buffer, size = numGroups * sizeof(uint)
//   randomsPerWorkItem - must match the value used when the RNG kernel was run
//   numGroups          - number of workgroups = numWorkItems / localSize
//   globalSize         - total number of work-items
//   localSize          - work-items per workgroup
//   outMs              - out: GPU kernel execution time in milliseconds
//
// Returns the raw per-workgroup partial results.
// The caller sums and interprets these values according to the simulation type.
inline std::vector<unsigned int> runMonteCarlo(
    cl_command_queue queue,
    cl_kernel        kernel,
    cl_mem           rngBuffer,
    cl_mem           resultsBuffer,
    unsigned int     randomsPerWorkItem,
    unsigned int     numGroups,
    size_t           globalSize,
    size_t           localSize,
    double& outMs,
    const char* label)
{
    cl_int err;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &resultsBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &rngBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &randomsPerWorkItem);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int) * localSize, nullptr);
    checkError(err, label);

    cl_event ev;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, &ev);
    checkError(err, label);

    std::vector<unsigned int> partial(numGroups);
    err = clEnqueueReadBuffer(queue, resultsBuffer, CL_TRUE, 0,
        sizeof(unsigned int) * numGroups, partial.data(), 0, nullptr, nullptr);
    checkError(err, label);

    clFinish(queue);

    outMs = profilingMs(ev);
    clReleaseEvent(ev);

    return partial;
}