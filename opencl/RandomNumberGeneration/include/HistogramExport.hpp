#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

// Writes a CSV histogram of the given data to filename.
// The value range [0, 2^32) is divided into numBins equal-width bins.
// Output format: two columns — bin index and observed count.
inline void exportHistogramCSV(const vector<unsigned int>& data,
    unsigned int numBins,
    const string& filename)
{
    const unsigned long long MAX_VAL = 4294967296ULL;
    const unsigned long long BIN_WIDTH = MAX_VAL / numBins;

    vector<unsigned int> freq(numBins, 0u);

    for (unsigned int v : data) {
        unsigned int idx = static_cast<unsigned int>(v / BIN_WIDTH);
        if (idx >= numBins) idx = numBins - 1;
        freq[idx]++;
    }

    ofstream f(filename);
    f << "bin,count\n";
    for (unsigned int i = 0; i < numBins; ++i)
        f << i << "," << freq[i] << "\n";

    cout << "  Histogram CSV -> " << filename << sendl;
}