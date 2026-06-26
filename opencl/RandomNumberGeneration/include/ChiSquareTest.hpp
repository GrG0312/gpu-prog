#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <string>

using namespace std;

double runChiSquareTest(const vector<unsigned int>& data, const string& label) {

    size_t N = data.size();
    // A: Sturges-rule
    // K = 1 + log2(N)
    unsigned int K = static_cast<unsigned int>(1 + std::log2(N));


    cout << "\n--- Chi-square test: " << label << " ---" << endl;
    cout << "  N (total numbers) : " << sz << endl;
    cout << "  K (bins, Sturges) : " << K << endl;


    // B: Counting the slots's frequencies
    unsigned long long MAX_VAL = 4294967296ULL;
    unsigned long long BIN_WIDTH = MAX_VAL / K; // Length of a slot

    vector<unsigned int> frequencies(K, 0u);

    for (unsigned int num : data) {
        unsigned int bin_index = static_cast<unsigned int>(num / BIN_WIDTH);
        if (bin_index >= K) bin_index = K - 1; // Correction for rounding
        frequencies[bin_index]++;
    }

    // C: Filtering (Throwing away those slots which have less than 600 numbers in it)
    const unsigned int MIN_FREQUENCY = 600;
    vector<unsigned int> valid_frequencies;

    for (unsigned int i = 0; i < K; ++i) {
        if (frequencies[i] >= MIN_FREQUENCY)
        {
            valid_frequencies.push_back(frequencies[i]);
        }
        else
        {
            cout << "Slot #" << i << " thrown, because it had less than " << MIN_FREQUENCY << " elements (" << frequencies[i] << ")" << endl;
        }
    }

    unsigned int valid_K = static_cast<unsigned int>(valid_frequencies.size());
    if (valid_K == 0) {
        cout << "ERROR: No valid slots left after filtering!" << endl;
        return -1.0;
    }

    // D: Counting the expected frequency 
    unsigned int total_remaining_elements = accumulate(valid_frequencies.begin(), valid_frequencies.end(), 0u);
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
    cout << "Remaining valid slots: "               << valid_K              << endl;
    cout << "Necessary frequency for the slots: "   << expected_frequency   << endl;
    cout << "ChiSq Statistics: "                    << chi_square_stat      << endl;
    cout << "Degree of freedom(df): "               << (valid_K - 1)        << endl;

    return chi_square_stat;
}