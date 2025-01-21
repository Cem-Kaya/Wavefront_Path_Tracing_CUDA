#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>

#include "vec3.h"
#include "ray.h"

struct hit_record
{
	Vec3 accumulatedColor; // Accumulated color for the path
	Vec3 throughput; 	 // Attenuation  Throughput for the path like the full energy of the photon
	int  bounceCount; // the number of bounces for this path
	bool active; // is live or dead

	// Intersection specific data
	float t;               // Distance to intersection
	Vec3  hitPosition;     // World space hit position
	Vec3  normal;          // Surface normal at intersection
	bool  didHit;          // Did we intersect anything?

	int   triangleIndex;   // Index of the triangle we hit better then pointer 

	__host__ __device__
		hit_record()
		: accumulatedColor(0, 0, 0)
		, throughput(1, 1, 1)
		, bounceCount(0)
		, active(true)
		, t(-1.f)
		, hitPosition(0, 0, 0)
		, normal(0, 0, 0)
		, didHit(false)
		, triangleIndex(-1) // default to -1 = "no triangle"
	{
	}
};




// For shadow rays
struct shadow_hit_record {
	bool didHit; // did we hit anything
	float t; // distance to intersection
	// delete unused stuff 
};