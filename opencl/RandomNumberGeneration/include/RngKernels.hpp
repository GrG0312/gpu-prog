#pragma once

/*

//======================\\
||      LCG KERNEL      ||
\\======================//

Multiplier:   a = 1664525
Increment:    c = 1013904223
Modulus:      m = 2^32 = 4_294_967_296

Local memory optimisation:
- Each work-item generates its values into a __local staging buffer
- Every thread in the workgroup is done => flush the buffers to __global memory
- This reduces global memory transactions from (numWorkItems * numberOfRandoms) to (numWorkItems * numberOfRandoms / LOCAL_SIZE)

*/
static const char* lcg_kernel_code = R"(
    __kernel void lcg_kernel(
        __global unsigned int* output,
        unsigned long seed,
        unsigned int numberOfRandoms,
        __local  unsigned int* localBuf)
    {
        unsigned int globalId = get_global_id(0);
        unsigned int localId = get_local_id(0);
        unsigned int localSize = get_local_size(0);
 
        // Unique per-thread seed
        unsigned long state = seed + (unsigned long)globalId * 2654435761UL;
 
        for (unsigned int i = 0; i < numberOfRandoms; ++i) {
            state = (state * 1664525UL + 1013904223UL) & 0xFFFFFFFFUL;
 
            localBuf[localId] = (unsigned int)state;
            barrier(CLK_LOCAL_MEM_FENCE);
 
            // Lane 0 flushes the full workgroup slice to global memory
            if (localId == 0) {
                unsigned int flushBase = (globalId - localId) * numberOfRandoms + i;
                for (unsigned int lane = 0; lane < localSize; ++lane)
                    output[flushBase + lane * numberOfRandoms] = localBuf[lane];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
)";


/*

//==========================\\
||    XORSHIFT KERNEL       ||
\\==========================//

XORShift shifts:
- 13 left
- 17 right
- 5 left

*/
static const char* xorshift_kernel_code = R"(
    __kernel void xorshift_kernel(
        __global unsigned int* output,
        unsigned int seed,
        unsigned int numberOfRandoms,
        __local  unsigned int* localBuf)
    {
        unsigned int globalId = get_global_id(0);
        unsigned int localId = get_local_id(0);
        unsigned int localSize = get_local_size(0);

        // Per-thread unique seed
        unsigned int state = seed ^ (globalId * 2654435761U + 1U);
        unsigned int base  = globalId * numberOfRandoms;

        for (unsigned int i = 0; i < numberOfRandoms; ++i) {
            // XORShift step
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;

            localBuf[localId] = state;

            barrier(CLK_LOCAL_MEM_FENCE);

            if (localId == 0) {
                unsigned int flushBase = (globalId - localId) * numberOfRandoms + i;
                for (unsigned int lane = 0; lane < localSize; ++lane)
                    output[flushBase + lane * numberOfRandoms] = localBuf[lane];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
)";


/*

//==============================\\
||    MERSENNE TWISTER KERNEL   ||
\\==============================//

MT19937 parameters:
  Word size (w):              32
  Degree of recurrence (n):  624
  Middle word (m):           397
  Separation point (r):       31
  Twist matrix coefficient:    a = 0x9908B0DF
  Tempering masks:             b = 0x9D2C5680,  c = 0xEFC60000
  Tempering shifts:            s = 7, t = 15, u = 11, l = 18
  Init multiplier:             f = 1812433253

*/
static const char* mt_kernel_code = R"(
    #define MT_N 624
    #define MT_M 397
    #define MT_MATRIX_A 0x9908B0DFU
    #define MT_UPPER_MASK 0x80000000U // Most significant bit?
    #define MT_LOWER_MASK 0x7FFFFFFFU // Least significant bits?

    // Tempering
    #define MT_TEMPER_B 0x9D2C5680U
    #define MT_TEMPER_C 0xEFC60000U
    #define MT_TEMPER_S 7
    #define MT_TEMPER_T 15
    #define MT_TEMPER_U 11
    #define MT_TEMPER_L 18
    #define MT_INIT_F 1812433253U

    __kernel void mt_kernel(
        __global unsigned int* output,
        __global unsigned int* stateBuffer,
        unsigned int seed,
        unsigned int numberOfRandoms,
        __local  unsigned int* localBuf)
    {
        unsigned int globalId = get_global_id(0);
        unsigned int localId = get_local_id(0);
        unsigned int localSize = get_local_size(0);

        // Each work-item owns MT_N consecutive words in stateBuffer
        // Words are non-trivial starting numbers
        __global unsigned int* mt = stateBuffer + globalId * MT_N;

        // Seeder
        // Each word is generated from the previous one using multiply-and-mix operation
        // The index i ensures that no two consecutive values are the same
        mt[0] = seed ^ (globalId * 2654435761U);
        for (unsigned int i = 1; i < MT_N; ++i) {
            unsigned int prev = mt[i - 1];
            mt[i] = MT_INIT_F * (prev ^ (prev >> 30)) + i;
        }

        // Generate numberOfRandoms values with local-memory staging
        unsigned int index = 0;

        for (unsigned int n = 0; n < numberOfRandoms; ++n) {
            // Regenerate full state when exhausted
            if (index >= MT_N) {
                unsigned int kk;
                unsigned int y;
                for (kk = 0; kk < MT_N - MT_M; ++kk) {
                    y = (mt[kk] & MT_UPPER_MASK) | (mt[kk + 1] & MT_LOWER_MASK);
                    mt[kk] = mt[kk + MT_M] ^ (y >> 1) ^ ((y & 1U) ? MT_MATRIX_A : 0U);
                }
                for (; kk < MT_N - 1; ++kk) {
                    y = (mt[kk] & MT_UPPER_MASK) | (mt[kk + 1] & MT_LOWER_MASK);
                    mt[kk] = mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ ((y & 1U) ? MT_MATRIX_A : 0U);
                }
                y = (mt[MT_N - 1] & MT_UPPER_MASK) | (mt[0] & MT_LOWER_MASK);
                mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ ((y & 1U) ? MT_MATRIX_A : 0U);
                index = 0;
            }

            // Temper
            unsigned int y = mt[index++];
            y ^= (y >> MT_TEMPER_U);
            y ^= (y << MT_TEMPER_S) & MT_TEMPER_B;
            y ^= (y << MT_TEMPER_T) & MT_TEMPER_C;
            y ^= (y >> MT_TEMPER_L);

            // Stage into local memory
            localBuf[localId] = y;

            barrier(CLK_LOCAL_MEM_FENCE);

            if (localId == 0) {
                unsigned int flushBase = (globalId - localId) * numberOfRandoms + n;
                for (unsigned int lane = 0; lane < localSize; ++lane)
                    output[flushBase + lane * numberOfRandoms] = localBuf[lane];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
)";
