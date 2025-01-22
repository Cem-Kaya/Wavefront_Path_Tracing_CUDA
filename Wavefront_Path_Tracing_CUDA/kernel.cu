
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <fstream>     
#include <iostream>    
#include <vector>
#include <string>
#include <chrono> 

#include "vec3.h"
#include "cam.h"
#include "ray.h"
#include "hit_record.h"
#include "triangle.h"
#include "material_gpu.h"



#define RUSSIAN_ROULETTE_START_BOUNCE 5
#define SURVIVAL_PROB 0.5f
// TODO put the backgroud color here as well // did not 

// Simple macro for error checking 
#define CUDA_CHECK(call)                                         \
	do {                                                         \
		cudaError_t error = call;                                \
		if (error != cudaSuccess) {                              \
			printf("CUDA Error: %s:%d, ", __FILE__, __LINE__);   \
			printf("code:%d, reason: %s\n", error,               \
				   cudaGetErrorString(error));                   \
			exit(1);                                             \
		}                                                        \
	} while(0)

//////////////////////////////////////////////////////////////////////////////////
// RNG STUFF
//////////////////////////////////////////////////////////////////////////////////
__global__ void initCurandStatesKernel(curandState * states, int width, int height, unsigned long long seed)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;
	// Initialize each state with a seed and a sequence number
	curand_init(seed, /*sequence=*/ idx, /*offset=*/ 0, &states[idx]);
}

// A helper function on the host side
void initCurandStates(curandState * d_states, int width, int height, unsigned long long seed)
{
	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	initCurandStatesKernel << <grid, block >> > (d_states, width, height, seed);
	cudaDeviceSynchronize();
}

// helper sample a random iluminous triangle from an array of indices
__device__ int pickRandomLightIndex(const int* d_lightIndices, int numLights, curandState* randState)
{
	// uniform pick among the lights
	int which = curand(randState) % numLights; // integer pick
	return d_lightIndices[which];
}

// helper sample a random point on a triangle
__device__ Vec3 samplePointOnTriangle(const Triangle& tri, curandState* randState)
{
	float r1 = curand_uniform(randState);
	float r2 = curand_uniform(randState);


	// If r1 + r2 > 1, flip them to remain in the triangle
	if (r1 + r2 > 1.f) {
		r1 = 1.f - r1;
		r2 = 1.f - r2;
	}
	// point = v0 + r1*(v1-v0) + r2*(v2-v0)
	Vec3 p = tri.v0 + (tri.v1 - tri.v0) * r1 + (tri.v2 - tri.v0) * r2;
	return p;
}


//////////////////////////////////////////////
// kernels 
//////////////////////////////////////////////


// Kernel to generate primary rays
__global__ void generatePrimaryRaysKernel(
	Ray* d_rays,
	hit_record* d_hit_records,
	Camera      camera,
	int         width,
	int         height
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if within the image bounds
	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;

	// Generate the primary ray for this pixel
	Ray primaryRay = camera.generateRay(x, y, width, height);

	// Store it in array
	d_rays[pixelIndex] = primaryRay;

	// Initialize the path state 
	hit_record hit_record;
	hit_record.bounceCount = 0;
	hit_record.accumulatedColor = make_vec3(0.f, 0.f, 0.f);
	hit_record.throughput = make_vec3(1.f, 1.f, 1.f);
	hit_record.active = true;

	d_hit_records[pixelIndex] = hit_record;
}

// a function to call the cuda kernal to generate primary rays 
void generatePrimaryRays(
	Ray* d_rays,
	hit_record* d_hit_records,
	const Camera& camera,
	int width,
	int height
)
{
	
	dim3 blockDim(32, 32);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
		(height + blockDim.y - 1) / blockDim.y);

	generatePrimaryRaysKernel <<< gridDim, blockDim >>> (d_rays, d_hit_records, camera, width, height );
	CUDA_CHECK(cudaDeviceSynchronize());
}



__global__ void extendRaysKernel(
	Ray* d_rays,
	Triangle* d_triangles,
	int       numTriangles,
	hit_record* d_hit_records,
	int       width,
	int       height
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;

	// If this path is no longer active, skip warp divergence !!!!
	if (!d_hit_records[pixelIndex].active)
		return;

	Ray ray = d_rays[pixelIndex];

	float tMin = 1e30f;    // BIG NUMBER far plane 
	bool  hitSomething = false;
	Vec3  bestNormal = make_vec3(0.f, 0.f, 0.f);
	int   bestTriIndex = -1; // track which triangle is closest

	// For each triangle, test intersection // BVH travercel woulh have been here ! 
	for (int i = 0; i < numTriangles; i++)
	{
		float t, u, v;
		bool hit = intersectTriangleMT(ray, d_triangles[i], t, u, v);
		if (hit && t < tMin)
		{
			tMin = t;
			hitSomething = true;
			bestNormal = d_triangles[i].normal; // or compute from barycentrics
			bestTriIndex = i;                  // store the index of the triangle
		}
	}

	if (hitSomething)
	{
		// Update the hit_record
		d_hit_records[pixelIndex].t = tMin;
		d_hit_records[pixelIndex].hitPosition = ray.origin + ray.direction * tMin;
		d_hit_records[pixelIndex].normal = bestNormal.normalized();
		d_hit_records[pixelIndex].didHit = true;
		d_hit_records[pixelIndex].triangleIndex = bestTriIndex; // <--- store it
	}
	else
	{
		// No intersection
		d_hit_records[pixelIndex].didHit = false;
		d_hit_records[pixelIndex].t = -1.f;
		d_hit_records[pixelIndex].triangleIndex = -1;  // no triangle
	}
}



void extendRays(
	Ray* d_rays,
	Triangle* d_triangles,
	int          numTriangles,
	hit_record* d_hit_records,
	int          width,
	int          height
)
{
	dim3 blockDim(32, 32);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
		(height + blockDim.y - 1) / blockDim.y);

	extendRaysKernel << <gridDim, blockDim >> > (
		d_rays,
		d_triangles,
		numTriangles,
		d_hit_records,
		width,
		height
		);
	CUDA_CHECK(cudaDeviceSynchronize());
}



// Example threshold and survival probability
#define RUSSIAN_ROULETTE_START_BOUNCE 5
#define SURVIVAL_PROB 0.5f

__global__ void shadeKernel(
	hit_record* d_hit_records,
	Ray* d_rays,
	Triangle* d_triangles,
	GPU_Material* d_materials,
	curandState* d_randStates,
	int           width,
	int           height,
	int           numTriangles
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;
	hit_record& rec = d_hit_records[idx];
	Ray& ray = d_rays[idx];

	// No need to do any thing if not active
	if (!rec.active) return;

	// If  didn't hit anything, it hit the background  sky skycolor !!!!!
	if (!rec.didHit) {
		//  background color
		rec.accumulatedColor = rec.accumulatedColor +  rec.throughput * make_vec3(0.0f, 0.0f, 0.0f);
		rec.active = false;
		return;
	}

	//  which triangle was hit
	int triIdx = rec.triangleIndex;
	if (triIdx < 0 || triIdx >= numTriangles) {
		rec.active = false;
		return;
	}

	Triangle     tri = d_triangles[triIdx];
	GPU_Material m = d_materials[tri.materialID];

	// 1) Emission
	Vec3 emission = emitted(m, rec);
	rec.accumulatedColor = rec.accumulatedColor +  rec.throughput * emission;

	// 2) Scatter
	Vec3 attenuation;
	Ray scattered;
	bool didScatter = scatter(m, ray, rec, attenuation, scattered, d_randStates, idx);

	if (didScatter)
	{
		// Update throughput
		rec.throughput = rec.throughput *  attenuation;
		ray = scattered;
		rec.bounceCount++;

		// --- RUSSIAN ROULETTE TERMINATION ---
		//  start after RUSSIAN_ROULETTE_START_BOUNCE
		if (rec.bounceCount > RUSSIAN_ROULETTE_START_BOUNCE) {
			float randVal = curand_uniform(&d_randStates[idx]);  // Random float in [0,1)

			// With probability (1 - SURVIVAL_PROB), terminate the ray.
			if (randVal > SURVIVAL_PROB) {
				rec.active = false;
				return;
			}
			else {
				// Survives, so we must compensate throughput
				rec.throughput = rec.throughput /  SURVIVAL_PROB;
			}
		}
	}
	else
	{
		// No valid  terminate
		rec.active = false;
	}
}




void shadeRays(
	hit_record* d_hit_records,
	Ray* d_rays,
	Triangle* d_triangles,
	GPU_Material* d_materials,
	curandState* d_randStates,
	int           width,
	int           height,
	int           numTriangles
)
{
	dim3 blockDim(32, 32);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
		(height + blockDim.y - 1) / blockDim.y);

	shadeKernel << <gridDim, blockDim >> > (
		d_hit_records,
		d_rays,
		d_triangles,
		d_materials,
		d_randStates,
		width,
		height,
		numTriangles
		);
	CUDA_CHECK(cudaDeviceSynchronize());
}





// Kernel that generates shadow rays for next event estimation
__global__ void generateShadowRaysKernel(
	hit_record* d_hit_records,
	Ray* d_primaryRays,
	Triangle* d_triangles,
	GPU_Material* d_materials,
	const int* d_lightIndices,
	int numLights,
	ShadowRay* d_shadowRays,
	int width,
	int height,
	curandState* d_randStates)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;
	hit_record& rec = d_hit_records[idx];

	// If path ended or no hit  skip
	if (!rec.active || !rec.didHit) {
		d_shadowRays[idx].active = false;
		return;
	}

	// Find the material of the triangle we just hit
	int triIdx = rec.triangleIndex;
	Triangle tri = d_triangles[triIdx];
	GPU_Material hitMat = d_materials[tri.materialID];

	// ONLY do next event estimation if it's Lambertian //dielectrics are removed btw 
	if (hitMat.type != MATERIAL_LAMBERTIAN) {
		d_shadowRays[idx].active = false;
		return;
	}

	
	curandState* rs = &d_randStates[idx];

	// 1) Pick a random iluminous triangle
	int lightIndex = pickRandomLightIndex(d_lightIndices, numLights, rs);
	Triangle lightTri = d_triangles[lightIndex];
	GPU_Material lightMat = d_materials[lightTri.materialID];

	// 2) Sample a random point on the light// to be honest it looks same as tthe center but maybe debends ont he scene ???
	Vec3 lightPoint = samplePointOnTriangle(lightTri, rs);

	// 3) Create the shadow ray
	Vec3 origin = rec.hitPosition + rec.normal * 0.001f; // offset
	Vec3 dir = lightPoint - origin;
	float dist = dir.length();
	dir = dir / dist;  // normalize

	ShadowRay& sray = d_shadowRays[idx];
	sray.pixelIdx = idx;
	sray.ray.origin = origin;
	sray.ray.direction = dir;
	sray.active = true;

	// Store distance, light normal, etc.
	sray.distanceToLight = dist;
	sray.lightNormal = lightTri.normal.normalized();
	sray.lightEmission = lightMat.emission;

	
}




// Kernel that intersects shadow rays
__global__ void extendShadowRaysKernel(
	ShadowRay* d_shadowRays,
	Triangle* d_triangles,
	int numTriangles,
	shadow_hit_record* d_shadowHitRecords,
	int width,
	int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;
	ShadowRay& sray = d_shadowRays[idx];

	if (!sray.active) {
		d_shadowHitRecords[idx].didHit = false;
		d_shadowHitRecords[idx].t = -1.f;
		return;
	}

	Ray r = sray.ray;

	float tMin = 1e30f;
	bool hitSomething = false;

	// For each triangle, test intersection  // again BVH would have been here 
	for (int i = 0; i < numTriangles; i++) {
		float t, u, v;
		bool hit = intersectTriangleMT(r, d_triangles[i], t, u, v);
		//  only care if there is  intersection up to the distance to the light	
		if (hit && t < tMin && t > 0.0001f) {
			tMin = t;
			hitSomething = true;
		}
	}

	d_shadowHitRecords[idx].didHit = hitSomething;
	d_shadowHitRecords[idx].t = (hitSomething) ? tMin : -1.f;
}


//  a kernel to accumulate the shadow 
__global__ void finalizeShadowKernel(
	hit_record* d_hit_records,
	ShadowRay* d_shadowRays,
	shadow_hit_record* d_shadowHits,
	int width,
	int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;
	ShadowRay& sray = d_shadowRays[idx];
	if (!sray.active) {
		// No shadow ray 
		return;
	}

	shadow_hit_record& sh = d_shadowHits[idx];

	// 1) Distance check if the shadow ray hits "before" it reaches the light then it causes light leak or no shadows basicly 
	bool inShadow = false;
	if (sh.didHit) {
		float eps = 1e-4f;
		if (sh.t > eps && sh.t < sray.distanceToLight - eps) {
			inShadow = true;
		}
	}

	if (!inShadow) {
		// 2) Not shadowed => accumulate tthe light
		int pixelIdx = sray.pixelIdx;

		// surface normal at the shading point
		Vec3 surfN = d_hit_records[pixelIdx].normal;
		// direction from shading point to light
		Vec3 L = sray.ray.direction;

		// cosines
		float cosSurf = fmaxf(0.f, surfN.dot( L));
		float cosLight = fmaxf(0.f, sray.lightNormal.dot(  L* - 1.0f));

		// distance^2 for 1/r^2 falloff pbr stuff 
		float dist2 = sray.distanceToLight * sray.distanceToLight;

		// a geometry factor 
		float G = (cosSurf * cosLight) / dist2;

		//  direct lighting
		Vec3 direct = sray.brdfFactor * sray.lightEmission * G;

		// accumulate
		d_hit_records[pixelIdx].accumulatedColor = d_hit_records[pixelIdx].accumulatedColor +  direct;
	}
}














//////////////////////////////////////////////////////////////////////////////////
// Debug functions 
//////////////////////////////////////////////////////////////////////////////////
void createMaterials(std::vector<GPU_Material>& mats)
{
	mats.clear();
	mats.reserve(12);

	// 0) Lambertian White
	{
		GPU_Material lam;
		lam.type = MATERIAL_LAMBERTIAN;
		lam.albedo = make_vec3(0.90f, 0.90f, 0.90f); // white
		lam.fuzz = 0.0f;
		lam.ir = 1.0f;
		lam.emission = make_vec3(0.f, 0.f, 0.f);
		mats.push_back(lam);
	}

	// 1) Lambertian Red
	{
		GPU_Material lam;
		lam.type = MATERIAL_LAMBERTIAN;
		lam.albedo = make_vec3(1.0f, 0.0f, 0.0f); // red
		lam.fuzz = 0.0f;
		lam.ir = 1.0f;
		lam.emission = make_vec3(0.f, 0.f, 0.f);
		mats.push_back(lam);
	}

	// 2) Lambertian Green
	{
		GPU_Material lam;
		lam.type = MATERIAL_LAMBERTIAN;
		lam.albedo = make_vec3(0.0f, 1.0f, 0.0f); // green
		lam.fuzz = 0.0f;
		lam.ir = 1.0f;
		lam.emission = make_vec3(0.f, 0.f, 0.f);
		mats.push_back(lam);
	}

	// 3) Lambertian Blue
	{
		GPU_Material lam;
		lam.type = MATERIAL_LAMBERTIAN;
		lam.albedo = make_vec3(0.0f, 0.0f, 1.0f); // blue
		lam.fuzz = 0.0f;
		lam.ir = 1.0f;
		lam.emission = make_vec3(0.f, 0.f, 0.f);
		mats.push_back(lam);
	}

	// 4) Lambertian Yellow
	{
		GPU_Material lam;
		lam.type = MATERIAL_LAMBERTIAN;
		lam.albedo = make_vec3(1.0f, 1.0f, 0.0f); // yellow
		lam.fuzz = 0.0f;
		lam.ir = 1.0f;
		lam.emission = make_vec3(0.f, 0.f, 0.f);
		mats.push_back(lam);
	}

	// 5) Lambertian Magenta
	{
		GPU_Material lam;
		lam.type = MATERIAL_LAMBERTIAN;
		lam.albedo = make_vec3(1.0f, 0.0f, 1.0f); // magenta
		lam.fuzz = 0.0f;
		lam.ir = 1.0f;
		lam.emission = make_vec3(0.f, 0.f, 0.f);
		mats.push_back(lam);
	}

	// 6) Lambertian Cyan
	{
		GPU_Material lam;
		lam.type = MATERIAL_LAMBERTIAN;
		lam.albedo = make_vec3(0.0f, 1.0f, 1.0f); // cyan
		lam.fuzz = 0.0f;
		lam.ir = 1.0f;
		lam.emission = make_vec3(0.f, 0.f, 0.f);
		mats.push_back(lam);
	}

	// 7) Lambertian Orange
	{
		GPU_Material lam;
		lam.type = MATERIAL_LAMBERTIAN;
		lam.albedo = make_vec3(1.0f, 0.5f, 0.0f); // orange
		lam.fuzz = 0.0f;
		lam.ir = 1.0f;
		lam.emission = make_vec3(0.f, 0.f, 0.f);
		mats.push_back(lam);
	}

	// 8) Mirror-like Metal
	{
		GPU_Material metalMat;
		metalMat.type = MATERIAL_METAL;
		metalMat.albedo = make_vec3(0.95f, 0.95f, 0.95f); // near mirror
		metalMat.fuzz = 0.2f;  // no fuzz => perfect reflection
		metalMat.ir = 1.0f;
		metalMat.emission = make_vec3(0.f, 0.f, 0.f);
		mats.push_back(metalMat);
	}

	// 9) White Light
	{
		GPU_Material light;
		light.type = MATERIAL_DIFFUSE_LIGHT;
		light.albedo = make_vec3(1.f, 1.f, 1.f);
		light.fuzz = 0.f;
		light.ir = 1.f;
		light.emission = make_vec3(5.f, 5.f, 5.f); // bright white
		mats.push_back(light);
	}

	// 10) Red Light
	{
		GPU_Material light;
		light.type = MATERIAL_DIFFUSE_LIGHT;
		light.albedo = make_vec3(1.f, 0.f, 0.f);
		light.fuzz = 0.f;
		light.ir = 1.f;
		light.emission = make_vec3(5.f, 0.f, 0.f); // bright red
		mats.push_back(light);
	}

	// 11) Blue Light
	{
		GPU_Material light;
		light.type = MATERIAL_DIFFUSE_LIGHT;
		light.albedo = make_vec3(0.f, 0.f, 1.f);
		light.fuzz = 0.f;
		light.ir = 1.f;
		light.emission = make_vec3(0.f, 0.f, 5.f); // bright blue
		mats.push_back(light);
	}
}


void createSimpleScene(std::vector<Triangle>& triangles) {
	// Triangle with Lambertian material
	Triangle tri;
	tri.v0 = make_vec3(-1.f, 0.f, -1.f);
	tri.v1 = make_vec3(1.f, 0.f, -1.f);
	tri.v2 = make_vec3(0.f, 1.f, -1.f);
	tri.normal = cross(tri.v1 - tri.v0, tri.v2 - tri.v0).normalized();
	tri.materialID = 0; // e.g. Lambertian
	triangles.push_back(tri);

	// Another triangle with metal,
}

void createMoreComplexScene(std::vector<Triangle>& triangles)
{
	triangles.clear();

	//
	// 1) Floor (two big triangles)
	//
	{
		// coordinates for the floor corners
		Vec3 A(-4.f, 0.f, -4.f);
		Vec3 B(4.f, 0.f, -4.f);
		Vec3 C(4.f, 0.f, 4.f);
		Vec3 D(-4.f, 0.f, 4.f);

		// first floor triangle
		triangles.push_back(makeTriangle(A, B, C, /*materialID=*/0));
		// second floor triangle
		triangles.push_back(makeTriangle(A, C, D, /*materialID=*/9));
	}

	//
	// 2) Back wall (two triangles)
	//
	{
		Vec3 A(-4.f, 0.f, -5.f);
		Vec3 B(4.f, 0.f, -5.f);
		Vec3 C(4.f, 4.f, -5.f);
		Vec3 D(-4.f, 4.f, -5.f);

		triangles.push_back(makeTriangle(A, B, C, /*materialID=*/3));
		triangles.push_back(makeTriangle(A, C, D, /*materialID=*/3));
	}

	//
	// 3) A slanted metal triangle
	//
	{
		Vec3 A(-1.f, 0.f, -2.f);
		Vec3 B(1.f, 0.f, -2.f);
		Vec3 C(0.f, 2.f, -2.f);

		triangles.push_back(makeTriangle(A, B, C, /*materialID=*/1));
	}

	//
	// 4) A small glass triangle
	//
	{
		Vec3 A(2.f, 0.f, -1.f);
		Vec3 B(3.f, 0.f, -2.f);
		Vec3 C(2.5f, 2.f, -1.5f);

		triangles.push_back(makeTriangle(A, B, C, /*materialID=*/2));
	}

	std::string myObjFile = "../mesh/mesh_from_nerf.obj";
	int objMaterialID = 1; // or any valid material ID in your system

	std::vector<Triangle> loadedTriangles = parseObj(myObjFile, objMaterialID);

	triangles.insert(triangles.end(), loadedTriangles.begin(), loadedTriangles.end());

}



void createEnclosedCubeScene(std::vector<Triangle>& triangles)
{
	triangles.clear();

	float boxMin = 0.f;
	float boxMax = 5.f;
	float yCeiling = 5.f;

	//
	// 1) Floor: white lambert (ID=0)
	//
	{
		Vec3 A(boxMin, 0.f, boxMin);
		Vec3 B(boxMax, 0.f, boxMin);
		Vec3 C(boxMax, 0.f, boxMax);
		Vec3 D(boxMin, 0.f, boxMax);

		triangles.push_back(makeTriangle(A, C, B, /*materialID=*/0));
		triangles.push_back(makeTriangle(A, D, C, /*materialID=*/0));
	}

	//
	// 2) Left Wall: red (ID=1)
	//
	{
		Vec3 A(boxMin, 0.f, boxMax);
		Vec3 B(boxMin, 0.f, boxMin);
		Vec3 C(boxMin, yCeiling, boxMin);
		Vec3 D(boxMin, yCeiling, boxMax);

		triangles.push_back(makeTriangle(A, B, C, 8));
		triangles.push_back(makeTriangle(A, C, D, 8));
	}

	//
	// 3) Right Wall: green (ID=2)
	//
	{
		Vec3 A(boxMax, 0.f, boxMin);
		Vec3 B(boxMax, 0.f, boxMax);
		Vec3 C(boxMax, yCeiling, boxMax);
		Vec3 D(boxMax, yCeiling, boxMin);

		triangles.push_back(makeTriangle(A, B, C, 3));
		triangles.push_back(makeTriangle(A, C, D, 3));
	}

	//
	// 4) Back Wall: white (ID=0)
	//
	{
		Vec3 A(boxMin, 0.f, boxMax);
		Vec3 B(boxMax, 0.f, boxMax);
		Vec3 C(boxMax, yCeiling, boxMax);
		Vec3 D(boxMin, yCeiling, boxMax);

		triangles.push_back(makeTriangle(A, C, B, 0));
		triangles.push_back(makeTriangle(A, D, C, 0));
	}

	//
	// 5) Ceiling + Emissive Light
	//
	{
		// Big white ceiling
		Vec3 A(boxMin, yCeiling, boxMin);
		Vec3 B(boxMax, yCeiling, boxMin);
		Vec3 C(boxMax, yCeiling, boxMax);
		Vec3 D(boxMin, yCeiling, boxMax);

		triangles.push_back(makeTriangle(A, B, C, 0));
		triangles.push_back(makeTriangle(A, C, D, 0));

		// Smaller rectangle for the light (ID=3)
		float lightSize = 2.f;
		float start = (boxMax - lightSize) * 0.5f;
		float end = start + lightSize;

		Vec3 L1(start, yCeiling - 0.001f, start);
		Vec3 L2(end, yCeiling - 0.001f, start);
		Vec3 L3(end, yCeiling - 0.001f, end);
		Vec3 L4(start, yCeiling - 0.001f, end);

		triangles.push_back(makeTriangle(L1, L2, L3, 9));
		triangles.push_back(makeTriangle(L1, L3, L4, 9));
	}

	//
	// 6) Spheres: loaded from .obj, offset inside the box
	//
	{
		// sphere.obj, offset  to  left
		Vec3 offsetLeft = make_vec3(1.2f, 2.f, 2.0f);
		std::vector<Triangle> sphereLeft = parseObj("../mesh/sphere.obj", /*materialID=*/1);

		// Add the offset can store and do it for ray but this also works 
		for (auto& tri : sphereLeft) {
			tri.v0 = tri.v0 +  offsetLeft;
			tri.v1 = tri.v1 +  offsetLeft;
			tri.v2 = tri.v2 +  offsetLeft;
			tri.normal = computeNormal(tri.v0, tri.v1, tri.v2);
		}
		triangles.insert(triangles.end(), sphereLeft.begin(), sphereLeft.end());

		// sphere_right.obj, offset it to the right
		Vec3 offsetRight = make_vec3(3.2f, 3.0f, 5.0f);
		std::vector<Triangle> sphereRight = parseObj("../mesh/sphere.obj", /*materialID=*/2);
		for (auto& tri : sphereRight) {
			tri.v0 = tri.v0 + offsetRight;
			tri.v1 = tri.v1 + offsetRight;
			tri.v2 = tri.v2 + offsetRight;
			tri.normal = computeNormal(tri.v0, tri.v1, tri.v2);
		}
		triangles.insert(triangles.end(), sphereRight.begin(), sphereRight.end());
	}
}


void createEnclosedCubeScene2(std::vector<Triangle>& triangles)
{
	triangles.clear();

	float boxMin = 0.f;
	float boxMax = 5.f;
	float yCeiling = 5.f;

	//
	// 1) Floor: white lambert (ID=0)
	//
	{
		Vec3 A(boxMin, 0.f, boxMin);
		Vec3 B(boxMax, 0.f, boxMin);
		Vec3 C(boxMax, 0.f, boxMax);
		Vec3 D(boxMin, 0.f, boxMax);

		triangles.push_back(makeTriangle(A, C, B, /*materialID=*/0));
		triangles.push_back(makeTriangle(A, D, C, /*materialID=*/0));
	}

	

	

	//
	// 5) Ceiling + Emissive Light
	//
	{
		// Big white ceiling
		Vec3 A(boxMin, yCeiling, boxMin);
		Vec3 B(boxMax, yCeiling, boxMin);
		Vec3 C(boxMax, yCeiling, boxMax);
		Vec3 D(boxMin, yCeiling, boxMax);

		//triangles.push_back(makeTriangle(A, B, C, 0));
		//triangles.push_back(makeTriangle(A, C, D, 0));

		// Smaller rectangle for the light (ID=3)
		float lightSize = 2.f;
		float start = (boxMax - lightSize) * 0.5f;
		float end = start + lightSize;

		Vec3 L1(start, yCeiling - 0.001f, start);
		Vec3 L2(end, yCeiling - 0.001f, start);
		Vec3 L3(end, yCeiling - 0.001f, end);
		Vec3 L4(start, yCeiling - 0.001f, end);

		triangles.push_back(makeTriangle(L1, L2, L3, 9));
		triangles.push_back(makeTriangle(L1, L3, L4, 9));
	}

	//
	// 6) Spheres: loaded from .obj, offset inside the box
	//
	{
		// sphere.obj, offset  to  left
		Vec3 offsetLeft = make_vec3(1.2f, 2.f, 2.0f);
		std::vector<Triangle> sphereLeft = parseObj("../mesh/mesh_from_nerf.obj", /*materialID=*/1);

		// Add the offset can store and do it for ray but this also works 
		for (auto& tri : sphereLeft) {
			tri.v0 = tri.v0 + offsetLeft;
			tri.v1 = tri.v1 + offsetLeft;
			tri.v2 = tri.v2 + offsetLeft;
			tri.normal = computeNormal(tri.v0, tri.v1, tri.v2);
		}
		triangles.insert(triangles.end(), sphereLeft.begin(), sphereLeft.end());

		// sphere_right.obj, offset it to the right
		Vec3 offsetRight = make_vec3(3.2f, 3.0f, 5.0f);
		std::vector<Triangle> sphereRight = parseObj("../mesh/sphere.obj", /*materialID=*/2);
		for (auto& tri : sphereRight) {
			tri.v0 = tri.v0 + offsetRight;
			tri.v1 = tri.v1 + offsetRight;
			tri.v2 = tri.v2 + offsetRight;
			tri.normal = computeNormal(tri.v0, tri.v1, tri.v2);
		}
		triangles.insert(triangles.end(), sphereRight.begin(), sphereRight.end());
	}
}


// Writes each pixel’s intersection data to CSV:

void writeDebugCSV(const char* filename, const hit_record* records, int width,	int height)
{
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error opening file " << filename << " for writing!\n";
		return;
	}

	// Optional CSV header
	file << "pixelIndex,didHit,t,normalX,normalY,normalZ\n";

	// Loop over all pixels
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int pixelIndex = y * width + x;
			const hit_record& rec = records[pixelIndex];

			file << pixelIndex << ","
				<< (rec.didHit ? 1 : 0) << ","  // store didHit as 0 or 1
				<< rec.t << ","
				<< rec.normal.x << ","
				<< rec.normal.y << ","
				<< rec.normal.z << "\n";
		}
	}

	file.close();
	std::cout << "Wrote debug CSV to " << filename << "\n";
}

void writeImageToCSV(const char* filename,	const hit_record* d_hit_records,int width,int height)
{
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error opening file " << filename << " for writing!\n";
		return;
	}

	//  CSV header
	file << "pixelIndex,colorR,colorG,colorB\n";

	// Loop over all pixels
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int pixelIndex = y * width + x;
			const hit_record& rec = d_hit_records[pixelIndex];

			// Gather final color
			float r = rec.accumulatedColor.x;
			float g = rec.accumulatedColor.y;
			float b = rec.accumulatedColor.z;

			// Write to file
			file << pixelIndex << ","
				<< r << ","
				<< g << ","
				<< b << "\n";
		}
	}

	file.close();
	std::cout << "Wrote final color CSV to " << filename << "\n";
}


void writeImageToPPM(const char* filename, const Vec3* buffer, int width, int height)
{
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error opening file " << filename << " for writing!\n";
		return;
	}

	// Write the PPM header P3, width, height, max color value
	file << "P3\n" << width << " " << height << "\n255\n";

	// Loop over rows top-to-bottom, left-to-right
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * width + x;
			

			float r = buffer[idx].x;
			float g = buffer[idx].y;
			float b = buffer[idx].z;

			// Clamp each color component to [0,1]
			r = (r < 0.f) ? 0.f : ((r > 1.f) ? 1.f : r);
			g = (g < 0.f) ? 0.f : ((g > 1.f) ? 1.f : g);
			b = (b < 0.f) ? 0.f : ((b > 1.f) ? 1.f : b);

			// Convert to [0..255]
			int ir = static_cast<int>(255.99f * r);
			int ig = static_cast<int>(255.99f * g);
			int ib = static_cast<int>(255.99f * b);

			file << ir << " " << ig << " " << ib << "\n";
		}
	}

	file.close();
	std::cout << "Wrote PPM to " << filename << "\n";
}

void convertAndOpenImage(const char* ppmFilename, const char* jpgFilename)
{
	// 1) Convert PPM to JPEG using ImageMagick
	//  in cli  "magick convert input.ppm output.jpg"
	{
		std::string cmd = std::string("magick convert ")
			+ ppmFilename
			+ " "
			+ jpgFilename;
		int retCode = system(cmd.c_str());
		if (retCode != 0) {
			std::cerr << "ImageMagick conversion failed!\n";
			return;
		}
	}

	// 2) Open the resulting JPEG in your system's default viewer
	//    On Windows:
	{
		std::string cmd = std::string("start ") + jpgFilename;
		system(cmd.c_str());
	}

	// If you're on macOS, how do you even run cuda ? 

	// If on Linux, do:
	//    std::string cmd = "xdg-open " + std::string(jpgFilename);

	std::cout << "Converted " << ppmFilename
		<< " to " << jpgFilename
		<< " and opened it.\n";
}







__global__ void accumulateKernel(
	const hit_record* d_hit_records,
	Vec3* d_accumBuffer,
	int width,
	int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;

	// Add the color from this sample into the accumulation buffer
	d_accumBuffer[idx] = d_accumBuffer[idx] + d_hit_records[idx].accumulatedColor;
}

__global__ void averageKernel(
	Vec3* d_accumBuffer,
	int width,
	int height,
	int samplesPerPixel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;
	d_accumBuffer[idx] = d_accumBuffer[idx] * (1.0f / samplesPerPixel);
}


__global__ void setzerosKernel(
	Vec3* d_accumBuffer,
	int width,
	int height,
	int samplesPerPixel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	int idx = y * width + x;
	d_accumBuffer[idx] = make_vec3(0.f, 0.f, 0.f);
}


// A  kernel to set each buffer element to zero
__global__ void initAccumBufferKernel(Vec3* buffer, int totalPixels)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalPixels) return;
	buffer[idx] = make_vec3(0.f, 0.f, 0.f);
}

// invoke the kernel
void clearAccumulationBuffer(Vec3* d_accumBuffer, int width, int height)
{
	int totalPixels = width * height;
	int blockSize = 256;
	int gridSize = (totalPixels + blockSize - 1) / blockSize;
	initAccumBufferKernel << <gridSize, blockSize >> > (d_accumBuffer, totalPixels);
	cudaDeviceSynchronize();
}




__global__ void checkAnyActiveKernel(const hit_record* d_hit_records,int width,	int height,	int* d_anyActive)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;

	// If an active path, set d_anyActive to 1.
	if (d_hit_records[idx].active) {
		//atomicExch(d_anyActive, 1); // i dont think i need atomics in this consistancy model 
		*d_anyActive = 1;  // Non-atomic write

	}
}






int main()
{
	//  image resolution
	int width = 1920;
	int height = 1080;
	int maxBounces = 15;
	int samplesPerPixel = 512; // for demonstration


	
	auto lastTime = std::chrono::high_resolution_clock::now();
	auto now = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastTime).count();
	lastTime = now;


	



	Vec3* d_accumBuffer = nullptr;
	cudaMallocManaged(&d_accumBuffer, width * height * sizeof(Vec3));

	// Initialize the accumulation buffer to zero (on CPU  // turn this to GPU very slow ! )
	clearAccumulationBuffer(d_accumBuffer, width, height);
	//for (int i = 0; i < width * height; i++) {
	//	d_accumBuffer[i] = make_vec3(0.f, 0.f, 0.f);
	//}
	int* d_anyActive = nullptr;
	cudaMalloc(&d_anyActive, sizeof(int));



	// Create a camera on host
	Camera h_camera;
	h_camera.setLookAt(
		make_vec3(2.5f, 2.5f, -10.0f),  // camera position: 
		//   x=2.5, y=2.5, z=-10  (in front of the box along negative Z)
		make_vec3(2.5f, 2.5f, 2.5f),  // look at the center of the box
		make_vec3(0.f, 1.f, 0.f), // 'up' vector
		35.0f,                          // FOV in degrees
		static_cast<float>(width) / static_cast<float>(height) // aspect ratio
	);



	// Allocate arrays on device
	Ray* d_rays = nullptr;
	hit_record* d_hit_records = nullptr;
	cudaMallocManaged(&d_rays, width * height * sizeof(Ray));
	cudaMallocManaged(&d_hit_records, width * height * sizeof(hit_record));
	

	curandState* d_randStates = nullptr;
	// init rng states
	cudaMallocManaged(&d_randStates, width * height * sizeof(curandState));
	initCurandStates(d_randStates, width, height, 1234ULL);
	

	// 1) Create materials on host
	std::vector<GPU_Material> h_mats;
	createMaterials(h_mats);
	int numMaterials = static_cast<int>(h_mats.size());

	// 2) Allocate a GPU/managed array of GPU_Material
	GPU_Material* d_materials = nullptr;
	cudaMallocManaged(&d_materials, numMaterials * sizeof(GPU_Material));

	// 3) Copy from host to device
	cudaMemcpy(d_materials, h_mats.data(),numMaterials * sizeof(GPU_Material), cudaMemcpyHostToDevice);








	

	
	//  Create scene triangles
	std::vector<Triangle> h_triangles;
	createEnclosedCubeScene(h_triangles);
	//createEnclosedCubeScene2(h_triangles);

	int numTriangles = static_cast<int>(h_triangles.size());

	// Allocate triangles on device (managed for simplicity)
	Triangle* d_triangles = nullptr;
	cudaMallocManaged(&d_triangles, numTriangles * sizeof(Triangle));
	// Copy from host to device
	cudaMemcpy(d_triangles, h_triangles.data(),
		numTriangles * sizeof(Triangle),
		cudaMemcpyHostToDevice);














	// ---------------------------
	// Identify Luminous Triangles
	// ---------------------------
	std::vector<int> lightIndices;
	for (int i = 0; i < numTriangles; i++) {
		// If the material is "MATERIAL_DIFFUSE_LIGHT", treat it as a light
		int matID = h_triangles[i].materialID;
		if (matID >= 0 && matID < numMaterials) {
			if (h_mats[matID].type == MATERIAL_DIFFUSE_LIGHT) {
				lightIndices.push_back(i);
			}
		}
	}
	int numLights = (int)lightIndices.size();
	if (numLights == 0) {
		std::cerr << "No emissive triangles found!\n";
	}

	// Copy light indices to device
	int* d_lightIndices = nullptr;
	CUDA_CHECK(cudaMallocManaged(&d_lightIndices, numLights * sizeof(int)));
	cudaMemcpy(d_lightIndices, lightIndices.data(),
		numLights * sizeof(int),
		cudaMemcpyHostToDevice);

	//  buffers for shadow rays
	ShadowRay* d_shadowRays = nullptr;
	CUDA_CHECK(cudaMallocManaged(&d_shadowRays, width * height * sizeof(ShadowRay)));

	shadow_hit_record* d_shadowHitRecords = nullptr;
	CUDA_CHECK(cudaMallocManaged(&d_shadowHitRecords, width * height * sizeof(shadow_hit_record)));




	now = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastTime).count();
	std::cout << "Time taken  for scene set up  " << elapsed << " micro sec\n";
	lastTime = now;


	// -----------------------
	// RENDER LOOP
	// -----------------------
	//  multiple samples anit aliasing per pixel. 
	for (int s = 0; s < samplesPerPixel; s++)
	{
		// 1) Generate primary rays
		generatePrimaryRays(d_rays, d_hit_records, h_camera, width, height);
		now = std::chrono::high_resolution_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastTime).count();
		std::cout << "Time taken  for Primary  " << elapsed << " micro sec\n";
		lastTime = now;
		// 2) For each bounce
		for (int b = 0; b < maxBounces; b++)
		{
			// Extend rays
			extendRays(d_rays, d_triangles, numTriangles, d_hit_records, width, height);
			now = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastTime).count();
			std::cout << "Time taken  for extand  " << elapsed << " micro sec\n";
			lastTime = now;
			// Shade
			{
				dim3 block(32, 32);
				dim3 grid((width + 15) / 16, (height + 15) / 16);
				shadeKernel << <grid, block >> > (
					d_hit_records,
					d_rays,
					d_triangles,
					d_materials,
					d_randStates,
					width,
					height,
					numTriangles					
					);
				CUDA_CHECK(cudaDeviceSynchronize());
			}
			now = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastTime).count();
			std::cout << "Time taken  for shade  " << elapsed << " micro sec\n";
			lastTime = now;
			

			// ---------------------------------------------------------
			// Only do the shadow-ray logic if "useShadowRays" is true
			//
			// ---------------------------------------------------------
			bool useShadowRays = true; 
			if (useShadowRays)
			{
				// Next Event Estimation (Shadow Rays)
				{
					dim3 block(32, 32);
					dim3 grid((width + 31) / 32, (height + 31) / 32);
					generateShadowRaysKernel << <grid, block >> > (
						d_hit_records,
						d_rays,
						d_triangles,
						d_materials,
						d_lightIndices,
						numLights,
						d_shadowRays,
						width,
						height,
						d_randStates
						);
					CUDA_CHECK(cudaDeviceSynchronize());
				}

				// Intersect shadow rays
				{
					dim3 block(32, 32);
					dim3 grid((width + 31) / 32, (height + 31) / 32);
					extendShadowRaysKernel << <grid, block >> > (
						d_shadowRays,
						d_triangles,
						numTriangles,
						d_shadowHitRecords,
						width,
						height
						);
					CUDA_CHECK(cudaDeviceSynchronize());
				}

				// Accumulate shadow
				{
					dim3 block(32, 32);
					dim3 grid((width + 31) / 32, (height + 31) / 32);
					finalizeShadowKernel << <grid, block >> > (
						d_hit_records,
						d_shadowRays,
						d_shadowHitRecords,
						width,
						height
						);
					CUDA_CHECK(cudaDeviceSynchronize());
				}
			} // end if(useShadowRays)
			now = std::chrono::high_resolution_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastTime).count();
			std::cout << "Time taken  for shadowray  " << elapsed << " micro sec\n";
			lastTime = now;
			// If all paths are inactive, we could break early, but  not implemented , yet // did it 

			cudaMemset(d_anyActive, 0, sizeof(int));

			dim3 block(32, 32);
			dim3 grid((width + block.x - 1) / block.x,
				(height + block.y - 1) / block.y);

			checkAnyActiveKernel << <grid, block >> > (	d_hit_records,width,height,	d_anyActive);
			cudaDeviceSynchronize();

			int h_anyActive = 0;
			cudaMemcpy(&h_anyActive, d_anyActive, sizeof(int),cudaMemcpyDeviceToHost);

			if (h_anyActive == 0) {
				// No active paths remain, break the bounce loop.
				break;
			}


		} // end bounce loop


		{
			dim3 block(32, 32);
			dim3 grid((width + 15) / 16, (height + 15) / 16);
			accumulateKernel << <grid, block >> > (d_hit_records, d_accumBuffer, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		now = std::chrono::high_resolution_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastTime).count();
		std::cout << "Time taken  for accumulate  " << elapsed << " micro sec\n";
		lastTime = now;
		
	} // end sample loop
	{
		dim3 block(32, 32);
		dim3 grid((width + 15) / 16, (height + 15) / 16);
		averageKernel << <grid, block >> > (d_accumBuffer, width, height, samplesPerPixel);
		CUDA_CHECK(cudaDeviceSynchronize());

		now = std::chrono::high_resolution_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastTime).count();
		std::cout << "Time taken  for avarage  " << elapsed << " micro sec\n";
		lastTime = now;
	}

	writeImageToCSV("final_color.csv", d_hit_records, width, height);
	writeImageToPPM("output.ppm", d_accumBuffer, width, height);
	convertAndOpenImage("output.ppm", "output.jpg");





	// Cleanup
	CUDA_CHECK(cudaFree(d_triangles));
	CUDA_CHECK(cudaFree(d_rays));
	CUDA_CHECK(cudaFree(d_hit_records));
	CUDA_CHECK(cudaFree(d_materials));
	CUDA_CHECK(cudaFree(d_randStates));
	CUDA_CHECK(cudaFree(d_lightIndices));
	CUDA_CHECK(cudaFree(d_shadowRays));
	CUDA_CHECK(cudaFree(d_shadowHitRecords));



	return 0;
}