#ifndef CHAPTER_18_SOLUTION_H
#define CHAPTER_18_SOLUTION_H

#include <cuda_runtime.h>

// ============================================================================
// 常量定义
// ============================================================================

#define BLOCK_SIZE 256          // 每个 Block 的线程数
#define CHUNK_SIZE 1024         // 每次传输到常量内存的原子数
#define COARSEN_FACTOR 8        // Thread Coarsening 因子
#define RANDOM_SEED 12345       // 随机数种子（用于可重复测试）

// ============================================================================
// CPU 实现
// ============================================================================

/**
 * CPU 串行实现 - 基础版本
 * 遍历每个网格点，对所有原子求和
 */
void cenergySequential(float* energygrid, dim3 grid, float gridspacing,
                       float z, const float* atoms, int numatoms);

/**
 * CPU 串行实现 - 优化版本
 * 先遍历原子，再遍历网格点（更好的缓存利用）
 */
void cenergySequentialOptimized(float* energygrid, dim3 grid, float gridspacing,
                                float z, const float* atoms, int numatoms);

// ============================================================================
// GPU 实现
// ============================================================================

/**
 * GPU Scatter 实现 (Fig. 18.5)
 * 每个线程处理一个原子，散射贡献到所有网格点
 * 需要 atomicAdd 避免写冲突
 */
void cenergyParallelScatter(float* host_energygrid, dim3 grid, float gridspacing,
                            float z, const float* host_atoms, int numatoms);

/**
 * GPU Gather 实现 (Fig. 18.6)
 * 每个线程处理一个网格点，收集所有原子的贡献
 * 无需原子操作，更高效
 */
void cenergyParallelGather(float* host_energygrid, dim3 grid, float gridspacing,
                           float z, const float* host_atoms, int numatoms);

/**
 * GPU Thread Coarsening 实现 (Fig. 18.8)
 * 每个线程处理 COARSEN_FACTOR 个连续网格点
 * 减少线程数，提高每线程工作量
 */
void cenergyParallelCoarsen(float* host_energygrid, dim3 grid, float gridspacing,
                            float z, const float* host_atoms, int numatoms);

/**
 * GPU Memory Coalescing 实现 (Fig. 18.10)
 * 在 Thread Coarsening 基础上优化内存访问模式
 * 确保相邻线程访问相邻内存地址
 */
void cenergyParallelCoalescing(float* host_energygrid, dim3 grid, float gridspacing,
                               float z, const float* host_atoms, int numatoms);

#endif // CHAPTER_18_SOLUTION_H
