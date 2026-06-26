
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
inline void buildProgramWithLog(cl_program program, cl_device_id device, const char* label)
{
    cl_int err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "[Build failure] " << label << ":\n" << log.data() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}