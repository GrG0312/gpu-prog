#pragma once

// ====================================================================
//  Add additional Monte Carlo kernel strings below
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
