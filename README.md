# Wavefront Path Tracing with CUDA

## Overview
This repository contains a Wavefront Path Tracing implementation using NVIDIA CUDA. The program is designed to perform photorealistic rendering using advanced GPU-based algorithms for efficiency and accuracy.

The implementation leverages CUDA's parallel computing capabilities to achieve high-performance ray tracing. The project includes support for:

- Path tracing with Russian Roulette termination.
- Lambertian, Metal, Dielectric, and Emissive materials.
- Shadow rays for next event estimation.
- Support for complex scenes loaded from .obj files.
- Multi-sampling for anti-aliasing.

---

## System Requirements

### Minimum Requirements
- **NVIDIA GPU**: A CUDA-capable GPU with at least **4 GB of VRAM**.
- **CUDA Toolkit**: Version **12.5** or later.
- **Operating System**: A 64-bit OS (e.g., Windows 10, Linux).
- **Compiler**: Visual Studio 2022 or a compatible compiler.
- **ImageMagick**: For converting and viewing output images (must be added to the system PATH).

### Optional Requirements
- **Mesh Files**: Some scenes require `.obj` files for 3D geometry. Ensure these files are available if you plan to render complex scenes.

---

## Features

### Key Components
1. **GPU-Based Path Tracing**:
   - Fully parallelized path tracing with CUDA.
   - Russian Roulette sampling for efficient termination of paths.

2. **Material Support**:
   - Lambertian, Metal, Dielectric, and Diffuse Light materials.

3. **Scene Management**:
   - Predefined scenes (e.g., enclosed cube, complex geometry).
   - `.obj` file parsing for custom 3D models.

4. **Output Formats**:
   - CSV for debug information.
   - PPM for rendered images (converted to JPG using ImageMagick).

---

## Installation

### Prerequisites
1. Install **Visual Studio 2022**. ( if on linux NVCC works)
2. Install the **CUDA Toolkit** (12.5 or newer).
3. Install **ImageMagick** and ensure it is added to the system PATH.

### Cloning the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/Cem-Kaya/Wavefront_Path_Tracing_CUDA
cd Wavefront_Path_Tracing_CUDA
