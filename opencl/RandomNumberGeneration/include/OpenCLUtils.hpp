#pragma once

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>

// Checks a cl_int error code and exits with a message if it indicates failure.
inline void checkError(cl_int err, const char* operation)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "OpenCL Error during " << operation
            << ": error code " << err << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Builds an OpenCL program and prints the full build log on failure.
// Exits if compilation fails.
inline void buildProgramWithLog(cl_program program, cl_device_id deviceId, const char* label)
{
    cl_int err = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "[Build failure] " << label << ":\n" << log.data() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Bundles a compiled OpenCL program and its entry-point kernel together so they can be passed around and released as a unit.
struct CLKernel { cl_program program; cl_kernel kernel; };

// Compiles a kernel from source and returns the resulting CLKernel.
inline CLKernel buildKernel(cl_context ctx, cl_device_id deviceId, const char* src, const char* funcName, const char* label)
{
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    checkError(err, label);
    buildProgramWithLog(prog, deviceId, label);
    cl_kernel k = clCreateKernel(prog, funcName, &err);
    checkError(err, label);
    return { prog, k };
}

// Reads the START and END profiling timestamps from a cl_event and
// returns the elapsed GPU execution time in milliseconds.
// Requires the command queue to have been created with CL_QUEUE_PROFILING_ENABLE.
inline double profilingMs(cl_event ev)
{
    cl_ulong t0, t1;  // was: unsigned long
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t0, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t1, nullptr);
    return static_cast<double>(t1 - t0) * 1e-6;
}