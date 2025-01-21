#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


// 3D vector

struct Vec3
{
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    __host__ __device__ inline Vec3 operator+(const Vec3& v) const
    {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ inline Vec3 operator-(const Vec3& v) const
    {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ inline Vec3 operator*(float s) const
    {
        return Vec3(x * s, y * s, z * s);
    }

    __host__ __device__ inline Vec3 operator/(float s) const
    {
        return Vec3(x / s, y / s, z / s);
    }

    __host__ __device__ inline Vec3 operator*(const Vec3& v) const
    {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }

    __host__ __device__ inline float dot(const Vec3& v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    __host__ __device__ inline float length() const
    {
        return sqrtf(dot(*this));
    }

    __host__ __device__ inline Vec3 normalized() const
    {
        float len = length();
        return (len > 0) ? (*this) * (1.0f / len) : Vec3(0, 0, 0);
    }
    


};

// Utility function
__host__ __device__ inline Vec3 make_vec3(float x, float y, float z)
{
    return Vec3(x, y, z);
}

__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b)
{
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
