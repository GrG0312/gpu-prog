#pragma once

// ====================================================================
//  Add additional Monte Carlo kernel strings in this file
//  Each kernel should follow the same workgroup-reduction pattern:
//      accumulate a per-work-item result into __local memory, then
//      have lane 0 reduce and write one partial result per workgroup
// ====================================================================



/*

//===========================\\
||   MONTE CARLO PI KERNEL   ||
\\===========================//

Consumes a pre-generated flat buffer of unsigned ints.
Consecutive pairs of values are interpreted as (x, y) coordinates in [0, 2^32).

Layout of the input buffer:
Thread-major order:
    [
        thread0_val0, thread0_val1, ..., thread0_val(R-1),
        thread1_val0, thread1_val1, ..., thread1_val(R-1),
        ...,
        thread(N-1)_val(R-1)
    ]
where
- R = numberOfRandoms
- N = numWorkItems.

Monte Carlo the kernel reads pairs sequentially within each thread's slice [ (2i), (2i+1) ]
- Thread outputs are even ( % 2 == 0 )

Workgroup reduction:
Work-items accumulate their own hit count and stage it in __local memory
Lane 0 reduces the workgroup and writes a partial sum to the output buffer.

The host sums all partial sums.
    Pi ~ 4 * totalHits / (numWorkItems * (numberOfRandoms / 2))

*/
static const char* monte_carlo_pi_kernel_code = R"(
    __kernel void monte_carlo_pi_kernel(
        __global unsigned int* partialHits,  // one entry per workgroup
        __global unsigned int* rngBuffer,    // pre-generated values from any RNG kernel
        unsigned int            numberOfRandoms,
        __local  unsigned int*  localHits)
    {
        unsigned int globalId  = get_global_id(0);
        unsigned int localId   = get_local_id(0);
        unsigned int localSize = get_local_size(0);
        unsigned int groupId   = get_group_id(0);

        // This thread's slice starts at globalId * numberOfRandoms.
        // Pairs (2i, 2i+1) are consumed, so point count = numberOfRandoms / 2.
        unsigned int base     = globalId * numberOfRandoms;
        unsigned int numPairs = numberOfRandoms / 2;
        unsigned int hits     = 0;

        for (unsigned int i = 0; i < numPairs; ++i) {
            // Normalise each raw uint to [0, 1) by dividing by 2^32
            float x = (float)rngBuffer[base + 2 * i]     / 4294967296.0f;
            float y = (float)rngBuffer[base + 2 * i + 1] / 4294967296.0f;
            if (x * x + y * y <= 1.0f) ++hits;
        }

        localHits[localId] = hits;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (localId == 0) {
            unsigned int sum = 0;
            for (unsigned int lane = 0; lane < localSize; ++lane)
                sum += localHits[lane];
            partialHits[groupId] = sum;
        }
    }
)";


/*

//==============================\\
||   MONTE CARLO STOCK KERNEL   ||
\\==============================//
*PARAMS
 * @param partialHits   [out] Global buffer storing the total successful paths per work-group.
 * @param rngBuffer     [in]  Global buffer containing pre-generated raw 32-bit unsigned random numbers.
 * @param numberOfRandoms [in]  Total number of raw random integers assigned to be processed by each single thread.
 * @param localHits     [out] Local scratchpad memory used for work-group parallel reduction.

 MATHEMATICAL FORMULAS USED:
 * * 1. Box-Muller Transform:
 * Converts two independent Uniformly distributed random numbers u1, u2 in [0, 1)
 * into a Standard Normal distributed random variable Z ~ N(0, 1):
 * Z = sqrt(-2.0 * ln(u1)) * cos(2.0 * pi * u2)
 *
 * 2. Geometric Brownian Motion (Asset Price Path):
 * Models the continuous-time stochastic process of the stock price:
 * S_T = S_0 * exp((mu - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
*/

static const char* monte_carlo_stock_kernel_code = R"(
    __kernel void monte_carlo_stock_kernel(
        __global unsigned int* partialHits, 
        __global unsigned int* rngBuffer,    
        unsigned int numberOfRandoms, 
        __local  unsigned int* localHits)   
    {
        unsigned int globalId = get_global_id(0);
        unsigned int localId = get_local_id(0);
        unsigned int localSize = get_local_size(0);
        unsigned int groupId = get_group_id(0);

        // Finantial Constants for the simulation (i took the values of S&P500 index fund, stack)
        float S0 = 5500.0f; // starting price
        float Strike = 6200.0f; // Expected price
        float mu = 0.10; // Expected growth
        float sigma = 0.18; // volatolity
        float T = 1.0f; // time, 1 year

        unsigned int base = globalId * numberOfRandoms;
        unsigned int numPairs = numberOfRandoms / 2;
        unsigned int hits = 0;

        for (unsigned int i = 0; i < numPairs; ++i) {
            //normalising between 0 and 1
            float u1 = (float)rngBuffer[base + 2 * i];
            float u2 = (float)rngBuffer[base + 2 * i + 1];
            
            // against log(0)
            if (u1 < 1e-7f) u1 = 1e-7f; 

            // Box-Muller transformation (Standard Distribution Z)
            float Z = sqrt(-2.0f * log(u1)) * cos(2.0f * 3.14159265f * u2);

            // Geometrical Brown-Movement
            float S_T = S0 * exp((mu - 0.5f * sigma * sigma) * T + sigma * sqrt(T) * Z);

            // Seeing if the counted result fits the expected
            if (S_T > Strike) {
                ++hits;
            }
        }

        // Local Memory
        localHits[localId] = hits;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Sequential reduction
        if (localId == 0) {
            unsigned int sum = 0;
            for (unsigned int lane = 0; lane < localSize; ++lane) {
                sum += localHits[lane];
            }
            partialHits[groupId] = sum;
        }
    }
)";