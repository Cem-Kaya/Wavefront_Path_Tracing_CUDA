#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>

#include "vec3.h"
#include "ray.h"


struct Camera
{
	Vec3 position;  // Camera position in world space in 3d 
	Vec3 forward; // Forward direction 
	Vec3 right; // Right direction
	Vec3 up;	// Up direction
	float fov;     // field of view in arc degrees
    float aspect;  // width / height

    __host__ __device__
        Camera() : position(0, 0, 0), forward(0, 0, -1), right(1, 0, 0), up(0, 1, 0),
        fov(45.0f), aspect(1.0f)
    {
    }

    // A  function to configure camera for look-at settings
    __host__ __device__
        void setLookAt(const Vec3& pos, const Vec3& target, const Vec3& upVec, float fovDeg, float aspectRatio)
    {
        position = pos;
        forward = (target - pos).normalized();
        right = cross(forward, upVec).normalized();
        up = cross(right, forward).normalized();
        fov = fovDeg;
        aspect = aspectRatio;
    }

    // Generate a primary ray for a pixel (x, y) in [0, width), [0, height)
    __device__ Ray generateRay(int x, int y, int width, int height) const
    {
        // Convert pixel coordinates to [0,1] range
        float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
        float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);

        // Map u,v to [-1,1] range in camera space
        float tanHalfFov = tanf((fov * 0.5f) * (3.14159 / 180.0f));
        float px = (2.0f * u - 1.0f) * aspect * tanHalfFov;
        float py = (1.0f - 2.0f * v) * tanHalfFov;

        // Create the ray direction in world space
        Vec3 rayDir = (forward + right * px + up * py).normalized();

        return Ray(position, rayDir);
    }
};
