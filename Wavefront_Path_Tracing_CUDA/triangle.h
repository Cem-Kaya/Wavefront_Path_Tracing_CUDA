
#pragma once

#include "vec3.h"
#include "ray.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>


struct Triangle
{
    // Triangle vertex positions
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    int  materialID;  // <--- the index in the GPU_Material array

    // Optional: per-triangle normal (e.g., flat shading)
    // You might compute this once in your setup:
    //   normal = cross(v1 - v0, v2 - v0).normalized();
    Vec3 normal;
};

// constructer on the CPU ! 
Triangle makeTriangle(const Vec3& v0, const Vec3& v1, const Vec3& v2, int materialID)
{
    Triangle tri;
    tri.v0 = v0;
    tri.v1 = v1;
    tri.v2 = v2;
    tri.normal = cross((v1 - v0), (v2 - v0)).normalized();
    tri.materialID = materialID;
    return tri;
}


__device__ bool intersectTriangleMT(const Ray& ray,
    const Triangle& tri,
    float& tOut,
    float& uOut,
    float& vOut)
{
    const float EPSILON = 1e-6f;
    Vec3 edge1 = tri.v1 - tri.v0;
    Vec3 edge2 = tri.v2 - tri.v0;

    Vec3 h = cross(ray.direction, edge2);
    float a = edge1.dot(h);
    if (fabs(a) < EPSILON)
        return false; // This means the ray is parallel to the triangle.

    float f = 1.0f / a;
    Vec3 s = ray.origin - tri.v0;
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f)
        return false;

    Vec3 q = cross(s, edge1);
    float v = f * ray.direction.dot(q);
    if (v < 0.0f || (u + v) > 1.0f)
        return false;

    // At this stage, we can compute t to find out where the intersection point is on the ray.
    float t = f * edge2.dot(q);
    if (t > EPSILON)
    {
        // ray intersection
        tOut = t;
        uOut = u;
        vOut = v;
        return true;
    }

    // This means that there is a line intersection, but not a ray intersection.
    return false;
}



// Helper function: compute 
inline Vec3 computeNormal(const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 n = cross((v1 - v0), (v2 - v0));
    float len = n.dot(n);
    if (len > 1e-16f) {
        n = n * (1.0f / sqrtf(len));
    }
    return n;
}

// A simple .obj parser that reads vertex positions & face indices
inline std::vector<Triangle> parseObj(const std::string& filename, int materialID)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open .obj file: " + filename);
    }

    std::vector<Vec3> vertices;        // store all 'v' lines
    std::vector<Triangle> triangles;   // final result

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream stream(line);
        std::string prefix;
        stream >> prefix;

        if (prefix == "v") {
            // Vertex position
            float x, y, z;
            stream >> x >> y >> z;
            vertices.push_back(Vec3(x, y, z));
        }
        else if (prefix == "f") {
            // Face (triangle)
            // Typically something like: "f 1 2 3" or "f 1/1 2/2 3/3"
            // We'll parse up to 3 vertex indices. 
            std::string v1Str, v2Str, v3Str;
            stream >> v1Str >> v2Str >> v3Str;
            if (v1Str.empty() || v2Str.empty() || v3Str.empty()) {
                std::cerr << "Warning: malformed face line: " << line << std::endl;
                continue;
            }

            // function to parse something like "12/4/7" -> 12 (the position index)
            auto parseIndex = [&](const std::string& token) -> int {
                // find first slash or end
                size_t slashPos = token.find('/');
                std::string idxStr = (slashPos == std::string::npos)
                    ? token
                    : token.substr(0, slashPos);
                // convert to int, then subtract 1 because .obj is 1-based
                int idx = std::stoi(idxStr) - 1;
                return idx;
                };

            int i0 = parseIndex(v1Str);
            int i1 = parseIndex(v2Str);
            int i2 = parseIndex(v3Str);

            // boundary check
            if (i0 < 0 || i0 >= (int)vertices.size() ||
                i1 < 0 || i1 >= (int)vertices.size() ||
                i2 < 0 || i2 >= (int)vertices.size())
            {
                std::cerr << "Warning: face index out of range in " << filename << std::endl;
                continue;
            }

            // retrieve actual positions
            Vec3 v0 = vertices[i0];
            Vec3 v1 = vertices[i1];
            Vec3 v2 = vertices[i2];

            // Build a Triangle
            Triangle tri;
            tri.v0 = v0;
            tri.v1 = v1;
            tri.v2 = v2;
            tri.normal = computeNormal(v0, v1, v2); // if you want a flat normal
            tri.materialID = materialID;

            triangles.push_back(tri);
        }
        // we ignore "vt", "vn", "usemtl", etc. lines
    }

    file.close();
    return triangles;
}
