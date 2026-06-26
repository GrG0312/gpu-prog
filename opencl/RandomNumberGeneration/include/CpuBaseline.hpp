#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>

#include "ChiSquareTest.hpp"
#include "HistogramExport.hpp"

inline void runCPUBaseline(unsigned int N, unsigned int numBins)
{
    std::cout << "\n=== CPU BASELINE (std::mt19937, N=" << N << ") ===" << std::endl;

    std::mt19937 rng(20240101u);
    std::vector<unsigned int> data(N);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto& v : data) v = rng();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  Generation time: " << ms << " ms" << std::endl;

    runChiSquareTest(data, "CPU std::mt19937");
    exportHistogramCSV(data, numBins, "histogram_cpu_mt19937.csv");
}