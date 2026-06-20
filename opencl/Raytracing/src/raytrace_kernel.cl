typedef struct{
	float x, y, z;
} Vector3;

typedef struct{
	Vector3 origin;
	Vector3 direction;
	float wavelenght;
	float intensity;
} Ray;

__kernel void test_raytrace(__global Ray* rays, int numRays)
{
	int global_id = get_global_id(0);
	if(global_id >= numRays)
	{
		return;
	}

	rays[global_id].origin.z += 1.0f;

}