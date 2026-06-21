typedef struct{
	float x, y, z;
} Vector3;

typedef struct{
	Vector3 origin;
	Vector3 direction;
	float wavelength;
	float intensity;
} Ray;

float dot(Vector3 a, Vector3 b)
{
	return a.x* b.x+ a.y*b.y+a.z*b.z;
}

Vector3 sub(Vector3 a, Vector3 b)
{
	Vector3 res = {a.x-b.x,a.y-b.y,a.z-b.z};
	return res;
}
Vector3 add(Vector3 a, Vector3 b)
{
	Vector3 res = {a.x+b.x,a.y+b.y,a.z+b.z};
	return res;
}
Vector3 scalar_mul(Vector3 a, float s)
{
	Vector3 res = {a.x*s,a.y*s,a.z*s};
	return res;
}
Vector3 normalize(Vector3 a)
{
	float len = sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
	if(len > 0.0f)
	{
		Vector3 res = {a.x/len,a.y/len,a.z/len};
		return res;
	}
	return a;
}
//Snell's law -> A beesési szög szinuszának és a törési szög szinuszának aránya a közegekben mért terjedési sebességek (c1,c2) arányával egyenlõ, ami megegyezik a két közeg relatív törésmutatójával (n2,1)
Vector3 refract(Vector3 i, Vector3 n, float eta)//i=beeso fenysugar, n=feluleti norma, eta=n1/n2
{
	float dot_NI = dot(n, i);
    float k = 1.0f - eta * eta * (1.0f - dot_NI * dot_NI);
    
    if (k < 0.0f) {
        //teljes belso visszaverodes
        Vector3 zero = {0.0f, 0.0f, 0.0f};
        return zero;
    }
    
    //T=eta * I - (eta * dot(N,I) + sqrt(k)) * N
    Vector3 part1 = scalar_mul(i, eta);
    Vector3 part2 = scalar_mul(n, eta * dot_NI + sqrt(k));
    return sub(part1, part2);
}

__kernel void test_raytrace(__global Ray* rays, int numRays)
{
	int global_id = get_global_id(0);
	if(global_id >= numRays)
	{
		return;
	}

	float sphereRadius = 1.0f;

	Ray currentRay = rays[global_id];

	float a=dot(currentRay.direction, currentRay.direction);
	float b=2.0f * dot(currentRay.origin, currentRay.direction);
	float c=dot(currentRay.origin, currentRay.origin)-(sphereRadius*sphereRadius);

	float discriminant = b * b - 4.0f * a * c;
	if(discriminant<0.0f)//NEM talalta el a cseppet
	{
		rays[global_id].intensity=0.0f;
		return;
	}

	float t = (-b - sqrt(discriminant)) / (2.0f * a);
	if(t<0.0f)//a metszespont mogottunk van, az sem kell
	{
		rays[global_id].intensity=0.0f;
	}

	Vector3 hitPoint = add(currentRay.origin, scalar_mul(currentRay.direction, t));//cseppel valo talalkozasi ponz
	Vector3 normal = normalize(hitPoint);

	//toresmutato szamitas->Cauchy egyenlettel kozelitjuk(n=A+B/lambda^2)
	float wave=currentRay.wavelength;
	float n_water = 1.32f + (3000.0f/(wave*wave));
	float n_air=1.0f;

	//ELOSZOR levegeo!! -> levegobol lepunk vizbe
	float eta=n_air/n_water;

	Vector3 refractedDirection = refract(currentRay.direction, normal, eta);

	//TESZT
	rays[global_id].origin = hitPoint;
    rays[global_id].direction = refractedDirection;

}