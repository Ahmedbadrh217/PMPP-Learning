/**
 * 第七章：卷积 - Tiled 2D卷积实现
 * 
 * 参考：chapter-07/code/conv2d_kernels.cu
 * 
 * 本实现包含：
 * 1. Tiled 2D卷积 kernel（图7.12）
 * 2. Tiled 2D卷积 + L2缓存利用 kernel（图7.15）
 */

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>

// 常量内存定义（用于存储滤波器）
__constant__ float d_F[(2 * 9 + 1) * (2 * 9 + 1)];

/**
 * 打印矩阵
 */
void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) 
                      << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

/**
 * CPU 参考实现
 */
void conv2d_cpu(float* N, float* F, float* P, int r, int height, int width) {
    int filterSize = 2 * r + 1;
    
    for (int outRow = 0; outRow < height; ++outRow) {
        for (int outCol = 0; outCol < width; ++outCol) {
            float Pvalue = 0.0f;
            
            for (int fRow = 0; fRow < filterSize; ++fRow) {
                for (int fCol = 0; fCol < filterSize; ++fCol) {
                    int inRow = outRow - r + fRow;
                    int inCol = outCol - r + fCol;
                    
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        Pvalue += N[inRow * width + inCol] * F[fRow * filterSize + fCol];
                    }
                }
            }
            P[outRow * width + outCol] = Pvalue;
        }
    }
}

/**
 * Tiled 2D卷积 Kernel（图7.12）
 * 
 * 使用共享内存存储输入 tile（包含 halo cells）
 * 每个 block 处理一个 OUT_TILE_SIZE × OUT_TILE_SIZE 的输出区域
 * 但需要加载 IN_TILE_SIZE × IN_TILE_SIZE 的输入区域
 */
__global__ void conv2d_tiled_kernel(float* N, float* P, int r, int height, int width) {
    // 计算全局位置（考虑 halo）
    int row = blockIdx.y * OUT_TILE_SIZE + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.x * OUT_TILE_SIZE + threadIdx.x - FILTER_RADIUS;
    
    // 共享内存存储输入 tile
    __shared__ float N_s[IN_TILE_SIZE][IN_TILE_SIZE];
    
    // 协作加载输入 tile 到共享内存
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;  // Ghost cells
    }
    __syncthreads();
    
    // 计算输出位置在 tile 中的相对坐标
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileCol = threadIdx.x - FILTER_RADIUS;
    
    // 只有输出 tile 内部的线程计算结果
    if (tileRow >= 0 && tileRow < OUT_TILE_SIZE &&
        tileCol >= 0 && tileCol < OUT_TILE_SIZE) {
        // 检查全局边界
        if (row >= 0 && row < height && col >= 0 && col < width) {
            float Pvalue = 0.0f;
            int filterSize = 2 * FILTER_RADIUS + 1;
            
            for (int fRow = 0; fRow < filterSize; ++fRow) {
                for (int fCol = 0; fCol < filterSize; ++fCol) {
                    int filterIndex = fRow * filterSize + fCol;
                    // 从共享内存读取输入
                    Pvalue += N_s[tileRow + fRow][tileCol + fCol] * d_F[filterIndex];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

/**
 * Tiled 2D卷积 + L2缓存利用 Kernel（图7.15）
 * 
 * 只将 output tile 对应的输入存入共享内存
 * halo 区域依赖 L2 缓存命中
 */
__global__ void conv2d_tiled_cached_kernel(float* N, float* P, int r, int height, int width) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // 共享内存只存储 tile 中心区域
    __shared__ float N_s[TILE_SIZE][TILE_SIZE];
    
    // 加载当前位置到共享内存
    if (row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    int filterSize = 2 * FILTER_RADIUS + 1;
    
    if (row < height && col < width) {
        float Pvalue = 0.0f;
        
        for (int fRow = 0; fRow < filterSize; ++fRow) {
            for (int fCol = 0; fCol < filterSize; ++fCol) {
                int filterIndex = fRow * filterSize + fCol;
                
                // 计算共享内存索引
                int sy = threadIdx.y - FILTER_RADIUS + fRow;
                int sx = threadIdx.x - FILTER_RADIUS + fCol;
                
                // 如果在共享内存 tile 范围内，从共享内存读取
                if (sy >= 0 && sy < TILE_SIZE && sx >= 0 && sx < TILE_SIZE) {
                    Pvalue += N_s[sy][sx] * d_F[filterIndex];
                } else {
                    // 否则从全局内存读取（期望 L2 缓存命中）
                    int gy = row - FILTER_RADIUS + fRow;
                    int gx = col - FILTER_RADIUS + fCol;
                    
                    if (gy >= 0 && gy < height && gx >= 0 && gx < width) {
                        Pvalue += N[gy * width + gx] * d_F[filterIndex];
                    }
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}

// 辅助函数
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

/**
 * Tiled 2D卷积（主机接口）
 */
void conv2d_tiled(float* h_N, float* h_F, float* h_P, int r, int height, int width) {
    float *d_N, *d_P;
    size_t matrixSize = height * width * sizeof(float);
    int filterSize = 2 * r + 1;
    size_t filterBytes = filterSize * filterSize * sizeof(float);
    
    // 拷贝滤波器到常量内存
    CHECK_CUDA(cudaMemcpyToSymbol(d_F, h_F, filterBytes));
    
    // 分配设备内存
    CHECK_CUDA(cudaMalloc(&d_N, matrixSize));
    CHECK_CUDA(cudaMalloc(&d_P, matrixSize));
    
    CHECK_CUDA(cudaMemcpy(d_N, h_N, matrixSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_P, 0, matrixSize));
    
    // 配置 kernel
    dim3 dimBlock(IN_TILE_SIZE, IN_TILE_SIZE);
    dim3 dimGrid(cdiv(width, OUT_TILE_SIZE), cdiv(height, OUT_TILE_SIZE));
    
    conv2d_tiled_kernel<<<dimGrid, dimBlock>>>(d_N, d_P, r, height, width);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_P, d_P, matrixSize, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}

/**
 * Tiled 2D卷积 + L2缓存利用（主机接口）
 */
void conv2d_tiled_cached(float* h_N, float* h_F, float* h_P, int r, int height, int width) {
    float *d_N, *d_P;
    size_t matrixSize = height * width * sizeof(float);
    int filterSize = 2 * r + 1;
    size_t filterBytes = filterSize * filterSize * sizeof(float);
    
    // 拷贝滤波器到常量内存
    CHECK_CUDA(cudaMemcpyToSymbol(d_F, h_F, filterBytes));
    
    // 分配设备内存
    CHECK_CUDA(cudaMalloc(&d_N, matrixSize));
    CHECK_CUDA(cudaMalloc(&d_P, matrixSize));
    
    CHECK_CUDA(cudaMemcpy(d_N, h_N, matrixSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_P, 0, matrixSize));
    
    // 配置 kernel
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid(cdiv(width, TILE_SIZE), cdiv(height, TILE_SIZE));
    
    conv2d_tiled_cached_kernel<<<dimGrid, dimBlock>>>(d_N, d_P, r, height, width);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_P, d_P, matrixSize, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}
