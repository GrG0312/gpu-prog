
cmake_minimum_required(VERSION 3.16)
project(RandomNumberGeneration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)

add_executable(RandomNumberGeneration
    src/main.cpp
)

target_include_directories(RandomNumberGeneration PRIVATE include)
target_link_libraries(RandomNumberGeneration PRIVATE OpenCL::OpenCL)

# List headers explicitly so they appear in IDE project views
target_sources(RandomNumberGeneration PRIVATE
    include/OpenCLUtils.hpp
    include/rng_kernels.hpp
    include/ChiSquareTest.hpp
    include/HistogramExport.hpp
    include/CPUBaseline.hpp
)