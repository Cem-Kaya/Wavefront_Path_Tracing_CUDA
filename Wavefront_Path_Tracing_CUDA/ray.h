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
	Vec3 lightNormal;  
	Vec3 lightEmission; // store the emission color of the sampled light
	Vec3 brdfFactor;    // how much we multiply if unoccluded
	bool active;
	float cosSurf; 
	float distanceToLight; // somehow light leakes to the other side
};