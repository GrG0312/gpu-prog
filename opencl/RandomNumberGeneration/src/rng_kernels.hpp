#ifndef RNG_KERNELS_HPP
#define RNG_KERNELS_HPP

/*

//======================\\
||		LCG KERNEL		||
\\======================//

Multiplier:
a = 1664525;

Increment:
c = 1013904223;

Modulus:
m = 2^32 = 4_294_967_296;

*/
static const char* lcg_kernel_code = R"(
	__kernel void lcg_kernel(__global unsigned int* output, unsigned long seed, unsigned int numberOfRandoms) {
		unsigned int globalId = get_global_id(0);
		unsigned long state = seed + globalId * 1664525UL;
		unsigned int start = globalId * numberOfRandoms;

		for (unsigned int i = 0; i < numberOfRandoms; ++i) {
			state = (state * 1664525UL + 1013904223UL) & 0xFFFFFFFFUL; // Modulo 2^32
			output[start + i] = (unsigned int)state;
		}
	}
)";


/*

//==========================\\
||		XORSHIFT KERNEL		||
\\==========================//

*/
static const char* xorshift_kernel_code = R"(
	__kernel void xorshift_kernel(__global unsigned int* output, unsigned int seed, unsigned int numberOfRandoms) {
		unsigned int globalId = get_global_id(0);
		unsigned int state = seed + globalId * 123456789U; // Unique seed for each work-item
		unsigned int start = globalId * numberOfRandoms;

		for (unsigned int i = 0; i < numberOfRandoms; ++i) {
			state ^= state << 13;
			state ^= state >> 17;
			state ^= state << 5;
			output[start + i] = state;
		}
	}
)";

#endif