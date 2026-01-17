/**
 * 第七章：卷积 - 2D卷积实现
 * 
 * 参考：chapter-07/code/conv2d_kernels.cu
 * 
 * 本实现包含：
 * 1. 朴素2D卷积 kernel（图7.7）
 * 2. 使用常量内存的2D卷积 kernel（图7.9）
 */

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>

// 常量内存定义（用于存储滤波器）
// 支持最大滤波器半径为 9（19x19 滤波器）
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
 * 用于验证 GPU 结果的正确性
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
                    
                    // 边界检查（ghost cells 处理为 0）
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
 * 朴素2D卷积 Kernel（图7.7）
 * 
 * 每个线程计算一个输出元素
 * 滤波器从全局内存读取
 */
__global__ void conv2d_basic_kernel(float* N, float* F, float* P, int r, int height, int width) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    
    int filterSize = 2 * r + 1;
    float Pvalue = 0.0f;
    
    for (int fRow = 0; fRow < filterSize; ++fRow) {
        for (int fCol = 0; fCol < filterSize; ++fCol) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            
            // 边界检查（ghost cells 处理为 0）
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += N[inRow * width + inCol] * F[fRow * filterSize + fCol];
            }
        }
    }
    
    if (outRow < height && outCol < width) {
        P[outRow * width + outCol] = Pvalue;
    }
}

/**
 * 使用常量内存的2D卷积 Kernel（图7.9）
 * 
 * 滤波器存储在常量内存中
 * 优势：
 * - 常量内存有专用缓存
 * - 所有线程读取相同位置时，仅需一次内存访问（广播）
 */
__global__ void conv2d_const_memory_kernel(float* N, float* P, int r, int height, int width) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    
    int filterSize = 2 * r + 1;
    
    if (outRow < height && outCol < width) {
        float Pvalue = 0.0f;
        
        for (int fRow = 0; fRow < filterSize; ++fRow) {
            for (int fCol = 0; fCol < filterSize; ++fCol) {
                int inRow = outRow - r + fRow;
                int inCol = outCol - r + fCol;
                
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    int filterIndex = fRow * filterSize + fCol;
                    Pvalue += N[inRow * width + inCol] * d_F[filterIndex];
                }
            }
        }
        P[outRow * width + outCol] = Pvalue;
    }
}

// 辅助函数：向上取整除法
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

/**
 * 朴素2D卷积（主机接口）
 */
void conv2d_basic(float* h_N, float* h_F, float* h_P, int r, int height, int width) {
    float *d_N, *d_F_global, *d_P;
    size_t matrixSize = height * width * sizeof(float);
    int filterSize = 2 * r + 1;
    size_t filterBytes = filterSize * filterSize * sizeof(float);
    
    // 分配设备内存
    CHECK_CUDA(cudaMalloc(&d_N, matrixSize));
    CHECK_CUDA(cudaMalloc(&d_F_global, filterBytes));
    CHECK_CUDA(cudaMalloc(&d_P, matrixSize));
    
    // 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_N, h_N, matrixSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_F_global, h_F, filterBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_P, 0, matrixSize));
    
    // 配置 kernel 参数
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));
    
    // 启动 kernel
    conv2d_basic_kernel<<<dimGrid, dimBlock>>>(d_N, d_F_global, d_P, r, height, width);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_P, d_P, matrixSize, cudaMemcpyDeviceToHost));
    
    // 释放内存
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_F_global));
    CHECK_CUDA(cudaFree(d_P));
}

/**
 * 使用常量内存的2D卷积（主机接口）
 */
void conv2d_const_memory(float* h_N, float* h_F, float* h_P, int r, int height, int width) {
    float *d_N, *d_P;
    size_t matrixSize = height * width * sizeof(float);
    int filterSize = 2 * r + 1;
    size_t filterBytes = filterSize * filterSize * sizeof(float);
    
    // 拷贝滤波器到常量内存
    CHECK_CUDA(cudaMemcpyToSymbol(d_F, h_F, filterBytes));
    
    // 分配设备内存
    CHECK_CUDA(cudaMalloc(&d_N, matrixSize));
    CHECK_CUDA(cudaMalloc(&d_P, matrixSize));
    
    // 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_N, h_N, matrixSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_P, 0, matrixSize));
    
    // 配置 kernel 参数
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));
    
    // 启动 kernel
    conv2d_const_memory_kernel<<<dimGrid, dimBlock>>>(d_N, d_P, r, height, width);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_P, d_P, matrixSize, cudaMemcpyDeviceToHost));
    
    // 释放内存
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}
