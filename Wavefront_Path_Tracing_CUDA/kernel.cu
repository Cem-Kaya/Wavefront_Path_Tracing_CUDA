
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <stdio.h>

#include <fstream>     // for std::ofstream
#include <iostream>    // for std::cerr
#include <vector>
#include <string>


#include "cam.h"
#include "ray.h"
#include "hit_record.h"
#include "triangle.h"
#include "material_gpu.h"


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
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	initCurandStatesKernel << <grid, block >> > (d_states, width, height, seed);
	cudaDeviceSynchronize();
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

	// Store it in our global array
	d_rays[pixelIndex] = primaryRay;

	// Initialize the path state (if wavefront style, each pixel = one path)
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
	// Typically you choose block and grid sizes to match your image dimension
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

	// If this path is no longer active, skip
	if (!d_hit_records[pixelIndex].active)
		return;

	Ray ray = d_rays[pixelIndex];

	float tMin = 1e30f;    // big sentinel value
	bool  hitSomething = false;
	Vec3  bestNormal = make_vec3(0.f, 0.f, 0.f);
	int   bestTriIndex = -1; // track which triangle is closest

	// For each triangle, test intersection
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


__global__ void shadeKernel(
	hit_record* d_hit_records,
	Ray* d_rays,
	Triangle* d_triangles,
	GPU_Material* d_materials,
	curandState* d_randStates,
	int width,
	int height,
	int numTriangles,
	Vec3 lightPos,   // Light source position
	Vec3 lightColor  // Light source color
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int idx = y * width + x;
	hit_record& rec = d_hit_records[idx];
	Ray& ray = d_rays[idx];

	if (!rec.active) return;           // No need to shade if inactive
	if (!rec.didHit) {
		// Sky color for rays that missed
		rec.accumulatedColor = rec.accumulatedColor + rec.throughput * make_vec3(0.5f, 0.7f, 1.0f);
		rec.active = false;
		return;
	}

	// Look up which triangle was hit
	int triIdx = rec.triangleIndex;
	if (triIdx < 0 || triIdx >= numTriangles) {
		rec.active = false;
		return;
	}

	Triangle tri = d_triangles[triIdx];
	GPU_Material mat = d_materials[tri.materialID];

	// Emission
	Vec3 emission = emitted(mat, rec);
	rec.accumulatedColor = rec.accumulatedColor + rec.throughput * emission;

	// Shadow ray check
	bool inShadow = is_in_shadow(rec.hitPosition, lightPos, d_triangles, numTriangles);

	if (!inShadow) {
		// Lambertian diffuse shading
		Vec3 toLight = (lightPos - rec.hitPosition).normalized();
		float diffuse = rec.normal.dot(toLight);
		rec.accumulatedColor = rec.accumulatedColor + rec.throughput * lightColor * mat.albedo * diffuse;
	}

	// Scatter (for indirect illumination)
	Vec3 attenuation;
	Ray scattered;
	bool didScatter = scatter(mat, ray, rec, attenuation, scattered, d_randStates, idx);

	if (didScatter) {
		rec.throughput = rec.throughput * attenuation;
		ray = scattered;
		rec.bounceCount++;
		if (rec.bounceCount > 5) {
			rec.active = false;
		}
	}
	else {
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
		numTriangles,
		lightPos,
		lightColor
		);
	CUDA_CHECK(cudaDeviceSynchronize());
}

















// Debug function 
// Suppose you have some triangles in a host array:
void createMaterials(std::vector<GPU_Material>& mats) {
	{
		// Lambertian material
		GPU_Material lam;
		lam.type = MATERIAL_LAMBERTIAN;
		lam.albedo = make_vec3(0.8f, 0.3f, 0.3f);
		lam.fuzz = 0.f;
		lam.ir = 1.f;
		lam.emission = make_vec3(0.f, 0.f, 0.f);
		mats.push_back(lam);
	}
	{
		// Metal material
		GPU_Material metalMat;
		metalMat.type = MATERIAL_METAL;
		metalMat.albedo = make_vec3(0.8f, 0.8f, 0.8f);
		metalMat.fuzz = 0.05f;
		metalMat.ir = 1.f;
		metalMat.emission = make_vec3(0.f, 0.f, 0.f);
		mats.push_back(metalMat);
	}
	{
		// Dielectric (glass)
		GPU_Material glass;
		glass.type = MATERIAL_DIELECTRIC;
		glass.albedo = make_vec3(1.f, 1.f, 1.f);
		glass.ir = 1.5f; // index of refraction
		glass.fuzz = 0.f;
		glass.emission = make_vec3(0.f, 0.f, 0.f);
		mats.push_back(glass);
	}
	{
		// Diffuse light
		GPU_Material light;
		light.type = MATERIAL_DIFFUSE_LIGHT;
		light.albedo = make_vec3(1.f, 1.f, 1.f); // not used for scattering
		light.fuzz = 0.f;
		light.ir = 1.f;
		// Emission color
		light.emission = make_vec3(5.f, 4.f, 2.f);
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

	// Another triangle with metal, etc.
}

// Example "bigger" scene
void createMoreComplexScene(std::vector<Triangle>& triangles)
{
	triangles.clear();

	//
	// 1) Floor (two big triangles)
	//
	// -z is “into the screen” in many setups, so we can make a floor at y=0.
	// We'll use materialID=0 for a Lambertian floor.
	{
		// coordinates for the floor corners
		Vec3 A(-4.f, 0.f, -4.f);
		Vec3 B(4.f, 0.f, -4.f);
		Vec3 C(4.f, 0.f, 4.f);
		Vec3 D(-4.f, 0.f, 4.f);

		// first floor triangle
		triangles.push_back(makeTriangle(A, B, C, /*materialID=*/0));
		// second floor triangle
		triangles.push_back(makeTriangle(A, C, D, /*materialID=*/0));
	}

	//
	// 2) Back wall (two triangles)
	//
	// Let’s place a wall at z=-5, from y=0 up to y=4, x from -4 to 4.
	// We'll use materialID=3 for diffuse light (if you defined that in your createMaterials()).
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
	// We'll place it near the center, leaning. Use materialID=1 for metal.
	{
		Vec3 A(-1.f, 0.f, -2.f);
		Vec3 B(1.f, 0.f, -2.f);
		Vec3 C(0.f, 2.f, -2.f);

		triangles.push_back(makeTriangle(A, B, C, /*materialID=*/1));
	}

	//
	// 4) A small glass triangle
	//
	// Let’s place it near the right side. Use materialID=2 for dielectric (glass).
	{
		Vec3 A(2.f, 0.f, -1.f);
		Vec3 B(3.f, 0.f, -2.f);
		Vec3 C(2.5f, 2.f, -1.5f);

		triangles.push_back(makeTriangle(A, B, C, /*materialID=*/2));
	}

	std::string myObjFile = "../mesh/mesh_from_nerf.obj";
	int objMaterialID = 1; // or any valid material ID in your system

	std::vector<Triangle> loadedTriangles = parseObj(myObjFile, objMaterialID);

	// Now insert them into your scene’s global triangle list
	triangles.insert(triangles.end(), loadedTriangles.begin(), loadedTriangles.end());

}



// Writes each pixel’s intersection data to CSV:
//   pixelIndex, didHit, t, normalX, normalY, normalZ
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

	// Optional CSV header
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


int main()
{
	// Example image resolution
	int width = 1920;
	int height = 1080;

	// Create a camera on host
	Camera h_camera;
	h_camera.setLookAt(
		make_vec3(0.f, 1.f, 3.f),   // position
		make_vec3(0.f, 1.f, 0.f),   // target
		make_vec3(0.f, 1.f, 0.f),   // up
		45.0f,                      // fov
		static_cast<float>(width) / static_cast<float>(height)
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
	createMoreComplexScene(h_triangles);
	int numTriangles = static_cast<int>(h_triangles.size());

	// Allocate triangles on device (managed for simplicity)
	Triangle* d_triangles = nullptr;
	cudaMallocManaged(&d_triangles, numTriangles * sizeof(Triangle));
	// Copy from host to device
	cudaMemcpy(d_triangles, h_triangles.data(),
		numTriangles * sizeof(Triangle),
		cudaMemcpyHostToDevice);






	// 1) Generate rays
	generatePrimaryRays(d_rays, d_hit_records, h_camera, width, height);


	// 2) Extend stage (intersection)
	//extendRays(d_rays, d_triangles, numTriangles, d_hit_records, width, height);

	// ... At this point, d_hit_records contains intersection info so to double check i writte stuff to a csv file 
	//writeDebugCSV("debug_output.csv", d_hit_records, width, height);

	int samplesPerPixel = 10;  
	int maxBounces = 8;


	//  Vec3* buffer 
	Vec3* d_finalColor = nullptr;
	cudaMallocManaged(&d_finalColor, width * height * sizeof(Vec3));

	//Initialize  on the CPU
	for (int i = 0; i < width * height; i++) {
		d_finalColor[i] = make_vec3(0.f, 0.f, 0.f);
	}


	for (int bounce = 0; bounce < maxBounces; bounce++)
	{
		// 2) Extend stage (intersection)
		extendRays(d_rays, d_triangles, numTriangles, d_hit_records, width, height);

		
		// 3) Shade (one bounce)
		shadeRays(d_hit_records, d_rays, d_triangles, d_materials, d_randStates, width, height, numTriangles);


		// (TODO) check if all rays are inactive:
	}







	writeImageToCSV("final_color.csv", d_hit_records, width, height);





	// Cleanup
	CUDA_CHECK(cudaFree(d_triangles));
	CUDA_CHECK(cudaFree(d_rays));
	CUDA_CHECK(cudaFree(d_hit_records));
	CUDA_CHECK(cudaFree(d_materials));
	CUDA_CHECK(cudaFree(d_randStates));
	CUDA_CHECK(cudaFree(d_finalColor));



	return 0;
}