typedef struct {
    float4 origin;
    float4 direction;
    float wavelength;
    float intensity;
    float dummy1; //paddingek -> hogy a struktura merete/igazotasa megegyezzen a CPU-oldalival
    float dummy2; //more info a doksiban meg itt: https://community.khronos.org/t/alignment-problem/2303/2
} RayVec;

float4 reflect_vec(float4 i, float4 n) {
    return i - 2.0f * dot(n, i) * n;
}

float4 refract_vec(float4 i, float4 n, float eta) {
    float dot_NI = dot(n, i);
    float k = 1.0f - eta * eta * (1.0f - dot_NI * dot_NI);
    if (k < 0.0f) {
        return (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
    return eta * i - (eta * dot_NI + sqrt(k)) * n;
}

float calculate_fresnel_vec(float4 I, float4 N, float n1, float n2) {
    float cos_i = -dot(N, I);
    if (cos_i < 0.0f) cos_i = -cos_i;

    float eta = n1 / n2;
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t >= 1.0f) return 1.0f; 
    
    float cos_t = sqrt(1.0f - sin2_t);

    float r_s = ((n1 * cos_i) - (n2 * cos_t)) / ((n1 * cos_i) + (n2 * cos_t));
    float r_p = ((n1 * cos_t) - (n2 * cos_i)) / ((n1 * cos_i) + (n2 * cos_t));

    return (r_s * r_s + r_p * r_p) * 0.5f;
}

void atomic_add_float_vec(__global float* addr, float val) {
    __global int* target = (__global int*)addr;
    int old_val, new_val;
    do {
        old_val = *target;
        float old_float = as_float(old_val);
        float new_float = old_float + val;
        new_val = as_int(new_float);
    } while (atomic_xchg(target, new_val) != old_val);
}

float4 get_cie_xyz_vec(float wave) {
    float4 xyz = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    
    float x_t1 = (wave - 442.0f) * ((wave < 442.0f) ? 0.0624f : 0.0374f);
    float x_t2 = (wave - 599.8f) * ((wave < 599.8f) ? 0.0264f : 0.0323f);
    float x_t3 = (wave - 501.1f) * ((wave < 501.1f) ? 0.0490f : 0.0382f);
    xyz.x = 0.362f * exp(-0.5f * x_t1 * x_t1) + 1.056f * exp(-0.5f * x_t2 * x_t2) - 0.065f * exp(-0.5f * x_t3 * x_t3);
    
    float y_t1 = (wave - 568.8f) * ((wave < 568.8f) ? 0.0213f : 0.0247f);
    float y_t2 = (wave - 530.9f) * ((wave < 530.9f) ? 0.0613f : 0.0322f);
    xyz.y = 0.821f * exp(-0.5f * y_t1 * y_t1) + 0.286f * exp(-0.5f * y_t2 * y_t2);
    
    float z_t1 = (wave - 437.0f) * ((wave < 437.0f) ? 0.0845f : 0.0278f);
    float z_t2 = (wave - 459.0f) * ((wave < 459.0f) ? 0.0385f : 0.0725f);
    xyz.z = 1.217f * exp(-0.5f * z_t1 * z_t1) + 0.681f * exp(-0.5f * z_t2 * z_t2);
    
    return xyz;
}

__kernel void test_raytrace_vectorized(__global RayVec* rays, int numRays, __global float* screen, int width, int height, float rotationAngle) 
{
    int global_id = get_global_id(0);
    if (global_id >= numRays) 
    {
        return;
    }

    float sphereRadius = 1.0f;
    RayVec currentRay = rays[global_id];

    float4 rayDir = currentRay.direction; rayDir.w = 0.0f;
    float4 rayOrig = currentRay.origin; rayOrig.w = 0.0f;

    //NATIV VEKTOROS MUVELETEK (float4)
    float a = dot(rayDir, rayDir);
    float b = 2.0f * dot(rayOrig, rayDir);
    float c = dot(rayOrig, rayOrig) - (sphereRadius * sphereRadius);
    float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f)
    if(discriminant<0.0f)
	{
		rays[global_id].intensity=0.0f;
		return;
	}

    float t = (-b - sqrt(discriminant)) / (2.0f * a);
    if (t < 0.0f)
    {
		rays[global_id].intensity=0.0f;
        return;
	}

    float4 hitPoint = rayOrig + rayDir * t;
    hitPoint.w = 0.0f; //a 4. koordinata w ne keruljon a szamitasba
    float4 normal = normalize(hitPoint);

    float wave = currentRay.wavelength;
    float n_water = 1.32f + (6788.0f / (wave * wave));
    float n_air = 1.0f;

    float eta=n_air/n_water;

    float F_entry = calculate_fresnel_vec(rayDir, normal, n_air, n_water);
    currentRay.intensity *= (1.0f - F_entry);

    rayDir = refract_vec(rayDir, normal, eta);
    rayOrig = hitPoint;

    int maxIntersections = 4;
    for (int i = 2; i <= maxIntersections; i++) {
        float t_internal = -2.0f * dot(rayOrig, rayDir) / dot(rayDir, rayDir);
        if (t_internal <= 0.0001f) break;

        hitPoint = rayOrig + rayDir * t_internal;
        hitPoint.w = 0.0f;
        normal = normalize(hitPoint);

        if (i == 2) {
            float F_reflect = calculate_fresnel_vec(rayDir, normal, n_water, n_air);
            currentRay.intensity *= F_reflect;
            rayDir = reflect_vec(rayDir, normal);
            rayOrig = hitPoint;
        } 
        else if (i == 3) {
            float F_exit = calculate_fresnel_vec(rayDir, normal, n_water, n_air);
            float4 exitDirection = refract_vec(rayDir, normal, n_water / n_air);
            exitDirection.w = 0.0f;
            
            if (dot(exitDirection, exitDirection) > 0.0f) {
                currentRay.intensity *= (1.0f - F_exit);
                rayDir = exitDirection;
                rayOrig = hitPoint;
                break;
            } else {
                rayDir = reflect_vec(rayDir, normal);
                rayOrig = hitPoint;
            }
        }
    }

    if (rayDir.z >= 0.0f) return;

    float screenZ = -3.0f;
    float t_screen = (screenZ - rayOrig.z) / rayDir.z;
    if (t_screen < 0.0f) return;

    float4 p_screen = rayOrig + rayDir * t_screen;

    //interaktiv forgatashoz
    float cosA = cos(rotationAngle);
    float sinA = sin(rotationAngle);
    float rotatedX = p_screen.x * cosA - p_screen.z * sinA;

    int pixelX = (int)(((rotatedX + 6.0f) / 12.0f) * width);
    int pixelY = (int)(((p_screen.y + 6.0f) / 12.0f) * height);

    if (pixelX >= 0 && pixelX < width && pixelY >= 0 && pixelY < height) {
        int pixelIndex = (pixelY * width + pixelX) * 3;
        float4 xyz = get_cie_xyz_vec(wave);

        atomic_add_float_vec(&screen[pixelIndex + 0], xyz.x * currentRay.intensity);
        atomic_add_float_vec(&screen[pixelIndex + 1], xyz.y * currentRay.intensity);
        atomic_add_float_vec(&screen[pixelIndex + 2], xyz.z * currentRay.intensity);
    }
}