/**
 * 第七章：卷积 - 3D卷积实现
 * 
 * 参考：chapter-07 README.md 练习 8-10
 * 
 * 本实现包含：
 * 1. 练习 8：基础3D卷积 kernel
 * 2. 练习 9：常量内存3D卷积 kernel
 * 3. 练习 10：Tiled 3D卷积 kernel
 */

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>

// 常量内存定义（支持最大 5x5x5 滤波器）
__constant__ float d_F_3d[(2 * 2 + 1) * (2 * 2 + 1) * (2 * 2 + 1)];

/**
 * 打印3D矩阵的一个切片
 */
void print3DSlice(float* matrix, int width, int height, int depth, int slice) {
    std::cout << "Slice z=" << slice << ":" << std::endl;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = slice * width * height + y * width + x;
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) 
                      << matrix[idx] << " ";
        }
        std::cout << std::endl;
    }
}

/**
 * CPU 参考实现 - 3D卷积
 */
void conv3d_cpu(float* N, float* F, float* P, int r,
                int width, int height, int depth) {
    int filterSize = 2 * r + 1;
    
    for (int outZ = 0; outZ < depth; ++outZ) {
        for (int outY = 0; outY < height; ++outY) {
            for (int outX = 0; outX < width; ++outX) {
                float Pvalue = 0.0f;
                
                for (int fZ = 0; fZ < filterSize; ++fZ) {
                    for (int fY = 0; fY < filterSize; ++fY) {
                        for (int fX = 0; fX < filterSize; ++fX) {
                            int inZ = outZ - r + fZ;
                            int inY = outY - r + fY;
                            int inX = outX - r + fX;
                            
                            if (inZ >= 0 && inZ < depth &&
                                inY >= 0 && inY < height &&
                                inX >= 0 && inX < width) {
                                int inIdx = inZ * width * height + inY * width + inX;
                                int fIdx = fZ * filterSize * filterSize + fY * filterSize + fX;
                                Pvalue += N[inIdx] * F[fIdx];
                            }
                        }
                    }
                }
                int outIdx = outZ * width * height + outY * width + outX;
                P[outIdx] = Pvalue;
            }
        }
    }
}

/**
 * 练习 8：基础3D卷积 Kernel
 * 
 * 扩展图7.7的2D版本到3D
 * 每个线程计算一个输出元素
 */
__global__ void conv3d_basic_kernel(float* N, float* F, float* P, int r,
                                    int width, int height, int depth) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outZ = blockIdx.z * blockDim.z + threadIdx.z;
    
    int filterSize = 2 * r + 1;
    float Pvalue = 0.0f;
    
    if (outX < width && outY < height && outZ < depth) {
        for (int fZ = 0; fZ < filterSize; ++fZ) {
            for (int fY = 0; fY < filterSize; ++fY) {
                for (int fX = 0; fX < filterSize; ++fX) {
                    int inZ = outZ - r + fZ;
                    int inY = outY - r + fY;
                    int inX = outX - r + fX;
                    
                    if (inZ >= 0 && inZ < depth &&
                        inY >= 0 && inY < height &&
                        inX >= 0 && inX < width) {
                        int inIdx = inZ * width * height + inY * width + inX;
                        int fIdx = fZ * filterSize * filterSize + fY * filterSize + fX;
                        Pvalue += N[inIdx] * F[fIdx];
                    }
                }
            }
        }
        int outIdx = outZ * width * height + outY * width + outX;
        P[outIdx] = Pvalue;
    }
}

/**
 * 练习 9：常量内存3D卷积 Kernel
 * 
 * 扩展图7.9的2D版本到3D
 * 滤波器存储在常量内存中
 */
__global__ void conv3d_const_memory_kernel(float* N, float* P, int r,
                                           int width, int height, int depth) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outZ = blockIdx.z * blockDim.z + threadIdx.z;
    
    int filterSize = 2 * r + 1;
    
    if (outX < width && outY < height && outZ < depth) {
        float Pvalue = 0.0f;
        
        for (int fZ = 0; fZ < filterSize; ++fZ) {
            for (int fY = 0; fY < filterSize; ++fY) {
                for (int fX = 0; fX < filterSize; ++fX) {
                    int inZ = outZ - r + fZ;
                    int inY = outY - r + fY;
                    int inX = outX - r + fX;
                    
                    if (inZ >= 0 && inZ < depth &&
                        inY >= 0 && inY < height &&
                        inX >= 0 && inX < width) {
                        int inIdx = inZ * width * height + inY * width + inX;
                        int fIdx = fZ * filterSize * filterSize + fY * filterSize + fX;
                        Pvalue += N[inIdx] * d_F_3d[fIdx];
                    }
                }
            }
        }
        int outIdx = outZ * width * height + outY * width + outX;
        P[outIdx] = Pvalue;
    }
}

/**
 * 练习 10：Tiled 3D卷积 Kernel
 * 
 * 扩展图7.12的2D版本到3D
 * 使用3D共享内存存储输入 tile
 */
__global__ void conv3d_tiled_kernel(float* N, float* P, int r,
                                    int width, int height, int depth) {
    // 计算全局位置
    int x = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int y = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int z = blockIdx.z * OUT_TILE_DIM + threadIdx.z - FILTER_RADIUS;
    
    // 3D 共享内存
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    
    // 协作加载输入 tile
    if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = N[z * width * height + y * width + x];
    } else {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // 计算 tile 内的相对坐标
    int tileX = threadIdx.x - FILTER_RADIUS;
    int tileY = threadIdx.y - FILTER_RADIUS;
    int tileZ = threadIdx.z - FILTER_RADIUS;
    
    // 只有输出 tile 内部的线程计算
    if (tileX >= 0 && tileX < OUT_TILE_DIM &&
        tileY >= 0 && tileY < OUT_TILE_DIM &&
        tileZ >= 0 && tileZ < OUT_TILE_DIM) {
        
        if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
            float Pvalue = 0.0f;
            int filterSize = 2 * FILTER_RADIUS + 1;
            
            for (int fZ = 0; fZ < filterSize; ++fZ) {
                for (int fY = 0; fY < filterSize; ++fY) {
                    for (int fX = 0; fX < filterSize; ++fX) {
                        int fIdx = fZ * filterSize * filterSize + fY * filterSize + fX;
                        Pvalue += N_s[tileZ + fZ][tileY + fY][tileX + fX] * d_F_3d[fIdx];
                    }
                }
            }
            P[z * width * height + y * width + x] = Pvalue;
        }
    }
}

// 辅助函数
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

/**
 * 练习 8：基础3D卷积（主机接口）
 */
void conv3d_basic(float* h_N, float* h_F, float* h_P, int r, 
                  int width, int height, int depth) {
    float *d_N, *d_F, *d_P;
    size_t volumeSize = width * height * depth * sizeof(float);
    int filterSize = 2 * r + 1;
    size_t filterBytes = filterSize * filterSize * filterSize * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_N, volumeSize));
    CHECK_CUDA(cudaMalloc(&d_F, filterBytes));
    CHECK_CUDA(cudaMalloc(&d_P, volumeSize));
    
    CHECK_CUDA(cudaMemcpy(d_N, h_N, volumeSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_F, h_F, filterBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_P, 0, volumeSize));
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y), cdiv(depth, dimBlock.z));
    
    conv3d_basic_kernel<<<dimGrid, dimBlock>>>(d_N, d_F, d_P, r, width, height, depth);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_P, d_P, volumeSize, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_F));
    CHECK_CUDA(cudaFree(d_P));
}

/**
 * 练习 9：常量内存3D卷积（主机接口）
 */
void conv3d_const_memory(float* h_N, float* h_F, float* h_P, int r,
                         int width, int height, int depth) {
    float *d_N, *d_P;
    size_t volumeSize = width * height * depth * sizeof(float);
    int filterSize = 2 * r + 1;
    size_t filterBytes = filterSize * filterSize * filterSize * sizeof(float);
    
    CHECK_CUDA(cudaMemcpyToSymbol(d_F_3d, h_F, filterBytes));
    
    CHECK_CUDA(cudaMalloc(&d_N, volumeSize));
    CHECK_CUDA(cudaMalloc(&d_P, volumeSize));
    
    CHECK_CUDA(cudaMemcpy(d_N, h_N, volumeSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_P, 0, volumeSize));
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y), cdiv(depth, dimBlock.z));
    
    conv3d_const_memory_kernel<<<dimGrid, dimBlock>>>(d_N, d_P, r, width, height, depth);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_P, d_P, volumeSize, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}

/**
 * 练习 10：Tiled 3D卷积（主机接口）
 */
void conv3d_tiled(float* h_N, float* h_F, float* h_P, int r,
                  int width, int height, int depth) {
    float *d_N, *d_P;
    size_t volumeSize = width * height * depth * sizeof(float);
    int filterSize = 2 * r + 1;
    size_t filterBytes = filterSize * filterSize * filterSize * sizeof(float);
    
    CHECK_CUDA(cudaMemcpyToSymbol(d_F_3d, h_F, filterBytes));
    
    CHECK_CUDA(cudaMalloc(&d_N, volumeSize));
    CHECK_CUDA(cudaMalloc(&d_P, volumeSize));
    
    CHECK_CUDA(cudaMemcpy(d_N, h_N, volumeSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_P, 0, volumeSize));
    
    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGrid(cdiv(width, OUT_TILE_DIM), cdiv(height, OUT_TILE_DIM), cdiv(depth, OUT_TILE_DIM));
    
    conv3d_tiled_kernel<<<dimGrid, dimBlock>>>(d_N, d_P, r, width, height, depth);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_P, d_P, volumeSize, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}
