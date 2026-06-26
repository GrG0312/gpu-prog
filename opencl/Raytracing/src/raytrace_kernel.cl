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
//Snell's law -> A beesesi szog szinuszanak es a toresi szog szinuszanak aranya a kozegekben mert terjedesi sebessegek (c1,c2) aranyaval egyenlo, ami megegyezik a ket kozeg relativ toresmutatojaval (n2,1)
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

Vector3 reflect(Vector3 i, Vector3 n){
	//help: https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
	//R = I - 2 * dot(N, I) * N
	float dot_NI=dot(n, i);
	return sub(i, scalar_mul(n, 2.0f * dot_NI));
}

//Fresnel-egyenlet polarizalatlan fanyre ->https://www.rp-photonics.com/fresnel_equations.html jo megertesre
float calculate_fresnel(Vector3 I, Vector3 N, float n1, float n2) {
    float cos_i = -dot(N, I);
    if (cos_i < 0.0f) cos_i = -cos_i; //boztonsagi iranyell, mert akkor is pozitiv cos kell, ha ki vagy be jon a feny

    float eta = n1 / n2;
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    
    //teljes belso visszaverodes eseten minden feny visszaverodik ->nem csokken az energia
    if (sin2_t >= 1.0f) return 1.0f; 
    
    float cos_t = sqrt(1.0f - sin2_t);

    //s-pol, p-pol komponensek
    float r_s = ((n1 * cos_i) - (n2 * cos_t)) / ((n1 * cos_i) + (n2 * cos_t));
    float r_p = ((n1 * cos_t) - (n2 * cos_i)) / ((n1 * cos_t) + (n2 * cos_i));

    return (r_s * r_s + r_p * r_p) * 0.5f;
}

//atomi osszeadas, https://stackoverflow.com/questions/72044986/atomic-addition-to-floating-point-values-in-opencl-for-nvidia-gpus -> itt van gyorsabb ver, de csak nvidia-ra
void atomic_add_float(__global float* addr, float val){
	__global int* target = (__global int*)addr;
    int old_val, new_val;
    do {
        old_val = *target;
        float old_float = as_float(old_val);
        float new_float = old_float + val;
        new_val = as_int(new_float);
    } while (atomic_xchg(target, new_val) != old_val);
}

//https://jcgt.org/published/0002/02/01/ -> Wyman analitikus megkozelitese
Vector3 get_cie_xyz(float wave) {
    Vector3 xyz = {0.0f, 0.0f, 0.0f};
    
    //xFit_1931
    float x_t1 = (wave - 442.0f) * ((wave < 442.0f) ? 0.0624f : 0.0374f);
    float x_t2 = (wave - 599.8f) * ((wave < 599.8f) ? 0.0264f : 0.0323f);
    float x_t3 = (wave - 501.1f) * ((wave < 501.1f) ? 0.0490f : 0.0382f);
    xyz.x = 0.362f * exp(-0.5f * x_t1 * x_t1) + 1.056f * exp(-0.5f * x_t2 * x_t2) - 0.065f * exp(-0.5f * x_t3 * x_t3);
    
    //yFit_1931
    float y_t1 = (wave - 568.8f) * ((wave < 568.8f) ? 0.0213f : 0.0247f);
    float y_t2 = (wave - 530.9f) * ((wave < 530.9f) ? 0.0613f : 0.0322f);
    xyz.y = 0.821f * exp(-0.5f * y_t1 * y_t1) + 0.286f * exp(-0.5f * y_t2 * y_t2);
    
    //zFit_1931
    float z_t1 = (wave - 437.0f) * ((wave < 437.0f) ? 0.0845f : 0.0278f);
    float z_t2 = (wave - 459.0f) * ((wave < 459.0f) ? 0.0385f : 0.0725f);
    xyz.z = 1.217f * exp(-0.5f * z_t1 * z_t1) + 0.681f * exp(-0.5f * z_t2 * z_t2);
    
    return xyz;
}

__kernel void test_raytrace(__global Ray* rays, int numRays, __global float* screen, int width, int height)
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

	//elso talalkozasi pont
	Vector3 hitPoint = add(currentRay.origin, scalar_mul(currentRay.direction, t));//cseppel valo talalkozasi ponz
	Vector3 normal = normalize(hitPoint);

	//toresmutato szamitas->Cauchy egyenlettel kozelitjuk(n=A+B/lambda^2)
	float wave=currentRay.wavelength;
	float n_water = 1.32f + (6788.0f/(wave*wave)); //redditen azt mondtak B=3000-el jobb szamolni?
	float n_air=1.0f;

	//ELOSZOR levegeo!! -> levegobol lepunk vizbe
	float eta=n_air/n_water;

	float F_entry = calculate_fresnel(currentRay.direction, normal, n_air, n_water);
    currentRay.intensity *= (1.0f - F_entry); //csak a bejuto resze a fenynek halad tovabb

	currentRay.direction = refract(currentRay.direction, normal, eta);
	currentRay.origin = hitPoint;

	//KOVETES -> max 4
	int maxIntersections = 4;
	for (int i=2;i<=maxIntersections;i++){

		//eleg csak 1 gyokot szamolni, a masik minig 0 (onnan inulunk)
		//t= -2 * dot(O, D) / dot(D, D)
		float t_internal=-2.0f * dot(currentRay.origin, currentRay.direction) / dot(currentRay.direction, currentRay.direction);
		if(t_internal<=0.0001f) break; //szamolasi hiba eseten
		
		hitPoint = add(currentRay.origin, scalar_mul(currentRay.direction, t_internal));
		normal = normalize(hitPoint);

		if(i==2){//elsodleges szivarvany, itt a feny visszapattan
			float F_reflect = calculate_fresnel(currentRay.direction, normal, n_water, n_air);
            currentRay.intensity *= F_reflect;

			currentRay.direction = reflect(currentRay.direction, normal);
            currentRay.origin = hitPoint;
		}
		else if(i==3){//itt a feny athalad
			float F_exit = calculate_fresnel(currentRay.direction, normal, n_water, n_air);

			Vector3 exitDir=refract(currentRay.direction, normal, eta);
			if(dot(exitDir, exitDir) > 0.0f) //ha sikeres a tores
			{
				currentRay.intensity *= (1.0f - F_exit); //ssak a kijuto resz megy a kepernyore
				currentRay.direction=exitDir;
				currentRay.origin=hitPoint;
				break;
			}
			else{//itt nem csokken az intensity mert teljes belso visszaverodes
				currentRay.direction=reflect(currentRay.direction, normal);
				currentRay.origin=hitPoint;
			}
		}
	}
	
	//kepernyore vetites
	float screenZ= -3.0f;
	if(currentRay.direction.z >= 0.0f) return; //masik iranyba ment

	//screenZ = O + t*D -> atrendezve t = (screenZ - O) / D
	float t_screen=(screenZ - currentRay.origin.z) / currentRay.direction.z;
	if (t_screen < 0.0f) return;

	//utkozesi pont a kepernyon
	Vector3 p_screen = add(currentRay.origin, scalar_mul(currentRay.direction, t_screen));

	int pixelX = (int)(((p_screen.x + 6.0f) / 12.0f) * width);
    int pixelY = (int)(((p_screen.y + 6.0f) / 12.0f) * height);
	if(pixelX>=0 && pixelX < width && pixelY>=0 && pixelY < height ){
		int pixelIndex= (pixelY*width + pixelX)*3;
		Vector3 xyz=get_cie_xyz(wave);

		atomic_add_float(&screen[pixelIndex + 0], xyz.x * currentRay.intensity);
        atomic_add_float(&screen[pixelIndex + 1], xyz.y * currentRay.intensity);
        atomic_add_float(&screen[pixelIndex + 2], xyz.z * currentRay.intensity);
	}
}

inline float native_clamped_to_u8(float val) {
    if (val > 255.0f) return 255.0f;
    if (val < 0.0f) return 0.0f;
    return val;
}

__kernel void convert_xyz_to_srgb(__global const float* screenXYZ, __global unsigned char* outputRGBA, int width, int height)
{
	int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

	int pixelIndexXYZ = (y * width + x) * 3;
    int pixelIndexRGBA = (y * width + x) * 4;

	//beolvassuk az osszegyujtott energiat
	float X = screenXYZ[pixelIndexXYZ + 0];
    float Y = screenXYZ[pixelIndexXYZ + 1];
    float Z = screenXYZ[pixelIndexXYZ + 2];

	//may delete later, keves sugar van, lehet sotet lesz
	float exposure = 15.0f; 
    X *= exposure;
    Y *= exposure;
    Z *= exposure;

	//atvaltasi matrix innen: https://stackoverflow.com/questions/66360637/which-matrix-is-correct-to-map-xyz-to-linear-rgb-for-srgb
	float r_lin =  3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    float g_lin = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    float b_lin =  0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;

	//gamma korrekcio egy2.2 szorzo, azt irtak, h jobb igy az emberi szemnek :)
	r_lin = (r_lin > 0.0f) ? pow(r_lin, 1.0f / 2.2f) : 0.0f;
    g_lin = (g_lin > 0.0f) ? pow(g_lin, 1.0f / 2.2f) : 0.0f;
    b_lin = (b_lin > 0.0f) ? pow(b_lin, 1.0f / 2.2f) : 0.0f;

	outputRGBA[pixelIndexRGBA + 0] = (unsigned char)(native_clamped_to_u8(r_lin * 255.0f));
    outputRGBA[pixelIndexRGBA + 1] = (unsigned char)(native_clamped_to_u8(g_lin * 255.0f));
    outputRGBA[pixelIndexRGBA + 2] = (unsigned char)(native_clamped_to_u8(b_lin * 255.0f));
    outputRGBA[pixelIndexRGBA + 3] = 255; //alfa
}