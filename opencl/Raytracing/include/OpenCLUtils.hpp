
#pragma once

#include <CL/cl.h>
#include <iostream>

inline void checkError(cl_int err, const char* operation)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "OpenCL Error during " << operation
                  << ": " << err << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
