#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


#include "vec3.h"

struct Ray
{
	Vec3 origin;
	Vec3 direction;

	__host__ __device__ Ray() {}
	__host__ __device__ Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d) {}
};


struct ShadowRay {
	Ray ray;            // The actual ray
	int pixelIdx;       // which pixel/path does this shadow ray belong to?
	Vec3 lightNormal;  // The normal of the light source
	Vec3 lightEmission; // store the emission color of the sampled light
	Vec3 brdfFactor;    //BSDF to be more spesific how much we multiply if unoccluded pbrt book 3rd edition 
	bool active; 	// is this ray still active
	float cosSurf; // cos of the angle between the surface normal and the ray direction
	float distanceToLight; // somehow light leakes to the other side
};