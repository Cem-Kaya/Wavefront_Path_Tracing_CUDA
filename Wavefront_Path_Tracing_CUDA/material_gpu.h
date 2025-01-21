#pragma once
#ifndef MATERIAL_GPU_H
#define MATERIAL_GPU_H

#include <curand_kernel.h>  // for curandState, curand_uniform
#include "vec3.h"
#include "ray.h"
#include "hit_record.h"

//-----------------------------------------------------------------------------
// Reflection & Refraction Helpers
//-----------------------------------------------------------------------------
__device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - n * (2.0f * v.dot(n));
}

__device__ inline bool refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted) {
    float dt = v.dot(n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.f - dt * dt);
    if (discriminant > 0.f) {
        refracted = (v - n * dt) * ni_over_nt - n * sqrtf(discriminant);
        return true;
    }
    return false;
}

//-----------------------------------------------------------------------------
// Material Enum  Data for GPU layout
//-----------------------------------------------------------------------------
enum MaterialType {
    MATERIAL_LAMBERTIAN = 0,
    MATERIAL_METAL,
    MATERIAL_DIELECTRIC,
    MATERIAL_DIFFUSE_LIGHT,
    MATERIAL_ISOTROPIC
};

struct GPU_Material {
    MaterialType type;

    // Base color or albedo (used by lambertian, metal, colored dielectrics )
    Vec3 albedo;

    // For metal: fuzz parameter
    float fuzz;

    // For dielectric: index of refraction
    float ir;

    // For diffuse_light: emission color & intensity
    Vec3 emission;
};

//-----------------------------------------------------------------------------
// Random Helpers using cuRAND
//-----------------------------------------------------------------------------
__device__ inline float rand01(curandState* state) {
    return curand_uniform(state); // range (0,1]
}

// Return a random point uniformly distributed inside a unit sphere
__device__ Vec3 random_in_unit_sphere(curandState* state) {
    while (true) {
        float x = rand01(state) * 2.f - 1.f;
        float y = rand01(state) * 2.f - 1.f;
        float z = rand01(state) * 2.f - 1.f;
        Vec3 p = make_vec3(x, y, z);
        if (p.dot(p) < 1.f) {
            return p;
        }
    }
}

//-----------------------------------------------------------------------------
// Emitted Only diffuse_light returns non-zero emission
//-----------------------------------------------------------------------------
__device__ Vec3 emitted(const GPU_Material& mat, const hit_record& rec)
{
    if (mat.type == MATERIAL_DIFFUSE_LIGHT) {
        return mat.emission;
    }
    return make_vec3(0.f, 0.f, 0.f);
}

//-----------------------------------------------------------------------------
// Scatter Function
//-----------------------------------------------------------------------------
__device__ bool scatter(
    const GPU_Material& mat,
    const Ray& r_in,
    const hit_record& rec,
    Vec3& attenuation,   // out: how the color is attenuated
    Ray& scattered,      // out: new scattered ray
    curandState* d_randStates,
    int idx
)
{
    curandState& localState = d_randStates[idx];  // reference to this thread's RNG state

    switch (mat.type) {
    case MATERIAL_LAMBERTIAN: {
        // Lambertian reflection
        Vec3 target = rec.hitPosition + rec.normal + random_in_unit_sphere(&localState);
        scattered.origin = rec.hitPosition;
        scattered.direction = (target - rec.hitPosition).normalized();
        attenuation = mat.albedo;
        return true;
    }
    case MATERIAL_METAL: {
        // Reflect + fuzz
        Vec3 reflected = reflect(r_in.direction.normalized(), rec.normal);
        Vec3 fuzzVec = (mat.fuzz > 0.f) ? random_in_unit_sphere(&localState) * mat.fuzz  : make_vec3(0.f, 0.f, 0.f);
        scattered.origin = rec.hitPosition;
        scattered.direction = (reflected + fuzzVec).normalized();
        attenuation = mat.albedo;
        float check = scattered.direction.dot(rec.normal);
        return (check > 0.f);
    }
    case MATERIAL_DIELECTRIC: {
        // Refract or reflect (Schlick)
        attenuation = make_vec3(1.f, 1.f, 1.f);
        Vec3 outwardNormal;
        Vec3 reflected = reflect(r_in.direction, rec.normal);
        float ni_over_nt;
        float cosine;
        if (r_in.direction.dot(rec.normal) > 0.f) {
            outwardNormal = rec.normal* -1.0f ;
            ni_over_nt = mat.ir;  // going out
            cosine = mat.ir * r_in.direction.dot(rec.normal) / r_in.direction.length();
        }
        else {
            outwardNormal = rec.normal;
            ni_over_nt = 1.f / mat.ir;
            cosine = -r_in.direction.dot(rec.normal) / r_in.direction.length();
        }

        Vec3 refracted;
        if (refract(r_in.direction, outwardNormal, ni_over_nt, refracted)) {
            // Schlick approximation
            float r0 = (1.f - mat.ir) / (1.f + mat.ir);
            r0 = r0 * r0;
            float schlickFactor = r0 + (1.f - r0) * powf((1.f - cosine), 5.f);
            if (rand01(&localState) < schlickFactor) {
                scattered.direction = reflected.normalized();
            }
            else {
                scattered.direction = refracted.normalized();
            }
        }
        else {
            // Total internal reflection
            scattered.direction = reflected.normalized();
        }
        scattered.origin = rec.hitPosition;
        return true;
    }
    case MATERIAL_DIFFUSE_LIGHT: {
        // Typically, lights do not scatter further
        return false;
    }
    case MATERIAL_ISOTROPIC: {
        scattered.origin = rec.hitPosition;
        scattered.direction = random_in_unit_sphere(&localState).normalized();
        attenuation = mat.albedo;
        return true;
    }
    } // switch

    // Fallback if material type is unrecognized
    return false;
}

#endif // MATERIAL_GPU_H
