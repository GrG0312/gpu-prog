
cmake_minimum_required(VERSION 3.16)
project(Raytracing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)

add_executable(Raytracing
    src/main.cpp
)

target_include_directories(Raytracing PRIVATE include ${OpenCL_INCLUDE_DIRS})
target_link_libraries(Raytracing PRIVATE OpenCL::OpenCL)
