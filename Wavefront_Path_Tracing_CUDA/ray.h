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
