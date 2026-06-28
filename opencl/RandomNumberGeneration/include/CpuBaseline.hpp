#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>

#include "ChiSquareTest.hpp"
#include "HistogramExport.hpp"

// Generates N values using std::mt19937, times the generation,
// runs the chi-square uniformity test, and exports a histogram CSV.
// Returns the generation time in milliseconds so the caller can
// include it in the shared performance summary.
inline double runCPUBaseline(unsigned int N, unsigned int numBins)
{
    std::mt19937 rng(20240101u);
    std::vector<unsigned int> data(N);
 
    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto& v : data) v = rng();
    auto t1 = std::chrono::high_resolution_clock::now();
 
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "\n[CPU mt19937]  time: " << ms << " ms  (" << N << " values)" << std::endl;
 
    runChiSquareTest(data, "CPU std::mt19937");
    exportHistogramCSV(data, numBins, "histogram_cpu_mt19937.csv");
 
    return ms;
}