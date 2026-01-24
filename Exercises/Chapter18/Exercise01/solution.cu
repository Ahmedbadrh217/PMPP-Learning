#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#include "solution.h"
#include "../../../Common/utils.cuh"

// ============================================================================
// 常量内存声明
// ============================================================================

// 原子数据：每个原子 4 个 float (x, y, z, charge)
__constant__ float c_atoms[CHUNK_SIZE * 4];

// ============================================================================
// CUDA Kernel 实现
// ============================================================================

/**
 * Scatter Kernel (Fig. 18.5)
 * 每个线程处理一个原子，将其贡献散射到所有网格点
 * 需要 atomicAdd 因为多个原子可能更新同一个网格点
 */
__global__ void cenergyScatterKernel(float* energygrid, dim3 grid, float gridspacing,
                                     float z, int atoms_in_chunk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = idx * 4;  // 每个原子占 4 个 float

    if (idx < atoms_in_chunk) {
        float atom_x = c_atoms[n];
        float atom_y = c_atoms[n + 1];
        float atom_z = c_atoms[n + 2];
        float charge = c_atoms[n + 3];

        // 预计算 dz（同一 z 切片所有点共享）
        float dz = z - atom_z;
        float dz2 = dz * dz;

        // z 切片在能量网格中的起始位置
        int k = (int)(z / gridspacing);
        int grid_slice_offset = grid.x * grid.y * k;

        // 遍历所有网格点
        for (int j = 0; j < (int)grid.y; j++) {
            float y = gridspacing * (float)j;
            float dy = y - atom_y;
            float dy2 = dy * dy;

            int grid_row_offset = grid_slice_offset + grid.x * j;

            for (int i = 0; i < (int)grid.x; i++) {
                float x = gridspacing * (float)i;
                float dx = x - atom_x;

                float r = sqrtf(dx * dx + dy2 + dz2);
                if (r > 0.001f) {
                    atomicAdd(&energygrid[grid_row_offset + i], charge / r);
                }
            }
        }
    }
}

/**
 * Gather Kernel (Fig. 18.6)
 * 每个线程处理一个网格点，收集所有原子的贡献
 * 无需原子操作
 */
__global__ void cenergyGatherKernel(float* energygrid, dim3 grid_dim, float gridspacing,
                                    float z, int atoms_in_chunk, int chunk_start) {
    // 2D 网格坐标
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < (int)grid_dim.x && j < (int)grid_dim.y) {
        // 计算网格点的物理坐标
        float x = gridspacing * (float)i;
        float y = gridspacing * (float)j;

        // z 索引
        int k = (int)(z / gridspacing);
        int grid_point_idx = grid_dim.x * grid_dim.y * k + grid_dim.x * j + i;

        // 第一个 chunk 时初始化
        if (chunk_start == 0) {
            energygrid[grid_point_idx] = 0.0f;
        }

        // 累加当前 chunk 中所有原子的贡献
        float energy = 0.0f;
        for (int n = 0; n < atoms_in_chunk; n++) {
            int atom_idx = n * 4;
            float dx = x - c_atoms[atom_idx];
            float dy = y - c_atoms[atom_idx + 1];
            float dz = z - c_atoms[atom_idx + 2];
            float charge = c_atoms[atom_idx + 3];

            float r = sqrtf(dx * dx + dy * dy + dz * dz);
            if (r > 0.001f) {
                energy += charge / r;
            }
        }

        energygrid[grid_point_idx] += energy;
    }
}

/**
 * Thread Coarsening Kernel (Fig. 18.8)
 * 每个线程处理 COARSEN_FACTOR 个连续的网格点
 */
__global__ void cenergyCoarsenKernel(float* energygrid, dim3 grid, float gridspacing,
                                     float z, int atoms_in_chunk) {
    // 每个线程的起始 x 索引
    int base_i = (blockIdx.x * blockDim.x + threadIdx.x) * COARSEN_FACTOR;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= (int)grid.y) {
        return;
    }

    int k = (int)(z / gridspacing);
    float y = gridspacing * (float)j;

    // 存储 COARSEN_FACTOR 个点的能量值
    float energies[COARSEN_FACTOR];
    for (int c = 0; c < COARSEN_FACTOR; c++) {
        energies[c] = 0.0f;
    }

    // 遍历所有原子
    for (int n = 0; n < atoms_in_chunk * 4; n += 4) {
        float dy = y - c_atoms[n + 1];
        float dz = z - c_atoms[n + 2];
        float dysqdzq = dy * dy + dz * dz;  // 预计算 dy² + dz²
        float charge = c_atoms[n + 3];

        // 计算 COARSEN_FACTOR 个点的能量
        for (int c = 0; c < COARSEN_FACTOR; c++) {
            int i = base_i + c;
            if (i < (int)grid.x) {
                float x = gridspacing * (float)i;
                float dx = x - c_atoms[n];
                float r = sqrtf(dx * dx + dysqdzq);
                if (r > 0.001f) {
                    energies[c] += charge / r;
                }
            }
        }
    }

    // 写回结果
    for (int c = 0; c < COARSEN_FACTOR; c++) {
        int i = base_i + c;
        if (i < (int)grid.x) {
            energygrid[grid.x * grid.y * k + grid.x * j + i] += energies[c];
        }
    }
}

/**
 * Memory Coalescing Kernel (Fig. 18.10)
 * 在 Thread Coarsening 基础上优化内存访问模式
 * 相邻线程访问相邻内存地址
 */
__global__ void cenergyCoalescingKernel(float* energygrid, dim3 grid, float gridspacing,
                                        float z, int atoms_in_chunk) {
    // 相邻线程处理相邻的起始 x 索引
    int base_i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= (int)grid.y) {
        return;
    }

    int k = (int)(z / gridspacing);
    float y = gridspacing * (float)j;

    // 存储 COARSEN_FACTOR 个点的能量值
    float energies[COARSEN_FACTOR];
    for (int c = 0; c < COARSEN_FACTOR; c++) {
        energies[c] = 0.0f;
    }

    // 遍历所有原子
    for (int n = 0; n < atoms_in_chunk * 4; n += 4) {
        float dx_base = gridspacing * (float)base_i - c_atoms[n];
        float dy = y - c_atoms[n + 1];
        float dz = z - c_atoms[n + 2];
        float dysqdzq = dy * dy + dz * dz;
        float charge = c_atoms[n + 3];

        // 计算 COARSEN_FACTOR 个点的能量（间隔 blockDim.x）
        for (int c = 0; c < COARSEN_FACTOR; c++) {
            float dx = dx_base + c * blockDim.x * gridspacing;
            float r = sqrtf(dx * dx + dysqdzq);
            if (r > 0.001f) {
                energies[c] += charge / r;
            }
        }
    }

    // 写回结果（保持内存合并）
    for (int c = 0; c < COARSEN_FACTOR; c++) {
        int idx = base_i + c * blockDim.x;
        if (idx < (int)grid.x) {
            energygrid[grid.x * grid.y * k + grid.x * j + idx] += energies[c];
        }
    }
}

// ============================================================================
// CPU 实现
// ============================================================================

void cenergySequential(float* energygrid, dim3 grid, float gridspacing,
                       float z, const float* atoms, int numatoms) {
    int k = (int)(z / gridspacing);
    int atomarrdim = numatoms * 4;

    for (int j = 0; j < (int)grid.y; j++) {
        float y = gridspacing * (float)j;
        for (int i = 0; i < (int)grid.x; i++) {
            float x = gridspacing * (float)i;
            float energy = 0.0f;

            for (int n = 0; n < atomarrdim; n += 4) {
                float dx = x - atoms[n];
                float dy = y - atoms[n + 1];
                float dz = z - atoms[n + 2];
                float r = sqrtf(dx * dx + dy * dy + dz * dz);
                if (r > 0.001f) {
                    energy += atoms[n + 3] / r;
                }
            }

            energygrid[grid.x * grid.y * k + grid.x * j + i] = energy;
        }
    }
}

void cenergySequentialOptimized(float* energygrid, dim3 grid, float gridspacing,
                                float z, const float* atoms, int numatoms) {
    int k = (int)(z / gridspacing);
    int atomarrdim = numatoms * 4;
    int grid_slice_offset = grid.x * grid.y * k;

    // 初始化为 0
    for (int j = 0; j < (int)grid.y; j++) {
        for (int i = 0; i < (int)grid.x; i++) {
            energygrid[grid_slice_offset + grid.x * j + i] = 0.0f;
        }
    }

    // 先遍历原子（更好的缓存利用）
    for (int n = 0; n < atomarrdim; n += 4) {
        float atom_x = atoms[n];
        float atom_y = atoms[n + 1];
        float atom_z = atoms[n + 2];
        float charge = atoms[n + 3];

        float dz = z - atom_z;
        float dz2 = dz * dz;

        for (int j = 0; j < (int)grid.y; j++) {
            float y = gridspacing * (float)j;
            float dy = y - atom_y;
            float dy2 = dy * dy;
            int grid_row_offset = grid_slice_offset + grid.x * j;

            for (int i = 0; i < (int)grid.x; i++) {
                float x = gridspacing * (float)i;
                float dx = x - atom_x;
                float r = sqrtf(dx * dx + dy2 + dz2);
                if (r > 0.001f) {
                    energygrid[grid_row_offset + i] += charge / r;
                }
            }
        }
    }
}

// ============================================================================
// GPU 主机函数实现
// ============================================================================

void cenergyParallelScatter(float* host_energygrid, dim3 grid, float gridspacing,
                            float z, const float* host_atoms, int numatoms) {
    // 分配设备内存
    float* d_energygrid = NULL;
    size_t grid_size = grid.x * grid.y * grid.z * sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&d_energygrid, grid_size));
    CHECK_CUDA(cudaMemset(d_energygrid, 0, grid_size));

    // 分块处理原子
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = (start_atom + CHUNK_SIZE <= numatoms)
                                 ? CHUNK_SIZE
                                 : (numatoms - start_atom);

        // 复制到常量内存
        size_t chunk_bytes = atoms_in_chunk * 4 * sizeof(float);
        CHECK_CUDA(cudaMemcpyToSymbol(c_atoms, &host_atoms[start_atom * 4], chunk_bytes));

        // 启动 kernel
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (atoms_in_chunk + threadsPerBlock - 1) / threadsPerBlock;

        cenergyScatterKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_energygrid, grid, gridspacing, z, atoms_in_chunk);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // 复制结果回主机
    CHECK_CUDA(cudaMemcpy(host_energygrid, d_energygrid, grid_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_energygrid));
}

void cenergyParallelGather(float* host_energygrid, dim3 grid_dim, float gridspacing,
                           float z, const float* host_atoms, int numatoms) {
    // 分配设备内存
    float* d_energygrid = NULL;
    size_t grid_size = grid_dim.x * grid_dim.y * grid_dim.z * sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&d_energygrid, grid_size));
    CHECK_CUDA(cudaMemset(d_energygrid, 0, grid_size));

    // 分块处理原子
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // 2D 线程块和网格
    dim3 threadsPerBlock(16, 16);  // 256 线程每块
    dim3 blocksPerGrid((grid_dim.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (grid_dim.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = (start_atom + CHUNK_SIZE <= numatoms)
                                 ? CHUNK_SIZE
                                 : (numatoms - start_atom);

        // 复制到常量内存
        size_t chunk_bytes = atoms_in_chunk * 4 * sizeof(float);
        CHECK_CUDA(cudaMemcpyToSymbol(c_atoms, &host_atoms[start_atom * 4], chunk_bytes));

        // 启动 kernel
        cenergyGatherKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_energygrid, grid_dim, gridspacing, z, atoms_in_chunk, start_atom);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // 复制结果回主机
    CHECK_CUDA(cudaMemcpy(host_energygrid, d_energygrid, grid_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_energygrid));
}

void cenergyParallelCoarsen(float* host_energygrid, dim3 grid_dim, float gridspacing,
                            float z, const float* host_atoms, int numatoms) {
    // 分配设备内存
    float* d_energygrid = NULL;
    size_t grid_size = grid_dim.x * grid_dim.y * grid_dim.z * sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&d_energygrid, grid_size));
    CHECK_CUDA(cudaMemset(d_energygrid, 0, grid_size));

    // 分块处理原子
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // 考虑 coarsening factor 的网格尺寸
    dim3 blockDim(16, 16);
    dim3 gridDim(((grid_dim.x + COARSEN_FACTOR - 1) / COARSEN_FACTOR + blockDim.x - 1) / blockDim.x,
                 (grid_dim.y + blockDim.y - 1) / blockDim.y);

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = (start_atom + CHUNK_SIZE <= numatoms)
                                 ? CHUNK_SIZE
                                 : (numatoms - start_atom);

        // 复制到常量内存
        size_t chunk_bytes = atoms_in_chunk * 4 * sizeof(float);
        CHECK_CUDA(cudaMemcpyToSymbol(c_atoms, &host_atoms[start_atom * 4], chunk_bytes));

        // 启动 kernel
        cenergyCoarsenKernel<<<gridDim, blockDim>>>(
            d_energygrid, grid_dim, gridspacing, z, atoms_in_chunk);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // 复制结果回主机
    CHECK_CUDA(cudaMemcpy(host_energygrid, d_energygrid, grid_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_energygrid));
}

void cenergyParallelCoalescing(float* host_energygrid, dim3 grid_dim, float gridspacing,
                               float z, const float* host_atoms, int numatoms) {
    // 分配设备内存
    float* d_energygrid = NULL;
    size_t grid_size = grid_dim.x * grid_dim.y * grid_dim.z * sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&d_energygrid, grid_size));
    CHECK_CUDA(cudaMemset(d_energygrid, 0, grid_size));

    // 分块处理原子
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // 考虑 coarsening 和 coalescing 的网格尺寸
    dim3 blockDim(16, 16);
    dim3 gridDim((grid_dim.x + blockDim.x * COARSEN_FACTOR - 1) / (blockDim.x * COARSEN_FACTOR),
                 (grid_dim.y + blockDim.y - 1) / blockDim.y);

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = (start_atom + CHUNK_SIZE <= numatoms)
                                 ? CHUNK_SIZE
                                 : (numatoms - start_atom);

        // 复制到常量内存
        size_t chunk_bytes = atoms_in_chunk * 4 * sizeof(float);
        CHECK_CUDA(cudaMemcpyToSymbol(c_atoms, &host_atoms[start_atom * 4], chunk_bytes));

        // 启动 kernel
        cenergyCoalescingKernel<<<gridDim, blockDim>>>(
            d_energygrid, grid_dim, gridspacing, z, atoms_in_chunk);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // 复制结果回主机
    CHECK_CUDA(cudaMemcpy(host_energygrid, d_energygrid, grid_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_energygrid));
}
