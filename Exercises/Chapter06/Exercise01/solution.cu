/**
 * 第六章：性能方面的考虑 - 列主序矩阵乘法（Corner Turning）
 * 
 * 参考：chapter-06/code/excercise1.cu
 * 
 * 本实现演示如何处理列主序存储的矩阵，使用角转换技术
 * 保持全局内存的合并访问。
 */

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#define TILE_WIDTH 16

/**
 * 打印矩阵
 */
void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(6) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

/**
 * 矩阵转置 Kernel
 */
__global__ void TransposeMatrixKernel(float* M, int m, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        // 先读取到寄存器
        float value = M[row * n + col];
        __syncthreads();  // 确保所有线程都读取完毕
        // 写入转置位置
        M[col * m + row] = value;
    }
}

/**
 * 行主序 Tiled 矩阵乘法 Kernel
 * 标准实现
 */
__global__ void TiledMatrixMulKernel(float* M, float* N, float* P, int m, int n, int o) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < numTiles; ++ph) {
        if (row < m && (ph * TILE_WIDTH + tx) < n) {
            Mds[ty][tx] = M[row * n + ph * TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0.0f;
        }

        if ((ph * TILE_WIDTH + ty) < n && col < o) {
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * o + col];
        } else {
            Nds[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    if (row < m && col < o) {
        P[row * o + col] = Pvalue;
    }
}

/**
 * 列主序 Tiled 矩阵乘法 Kernel（Corner Turning）
 * 
 * 关键优化：
 * - N 矩阵以列主序存储（转置后）
 * - 按列访问 N 以保持合并访问
 * - 共享内存用于重排数据
 * 
 * N 的访问模式：N[col * n + row] 而不是 N[row * o + col]
 */
__global__ void TiledMatrixMulKernelColMajor(float* M, float* N, float* P, int m, int n, int o) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < numTiles; ++ph) {
        // 加载 M（行主序，正常访问）
        if (row < m && (ph * TILE_WIDTH + tx) < n) {
            Mds[ty][tx] = M[row * n + ph * TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0.0f;
        }

        // 加载 N（列主序：按列访问以保持合并）
        // N 存储为列主序：N[col][row] = N[col * n + row]
        // 相邻线程访问相邻内存地址，实现合并访问
        if ((ph * TILE_WIDTH + ty) < n && col < o) {
            // 注意：这里 N 是转置存储的
            // N[col * n + (ph * TILE_WIDTH + ty)]
            Nds[ty][tx] = N[col * n + (ph * TILE_WIDTH + ty)];
        } else {
            Nds[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    if (row < m && col < o) {
        P[row * o + col] = Pvalue;
    }
}

// 辅助函数
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

/**
 * 原地矩阵转置（主机接口）
 * 注意：这是简化版本，仅用于演示
 */
void inPlaceMatrixTranspose(float* h_M, int m, int n) {
    float* d_M;
    CHECK_CUDA(cudaMalloc(&d_M, m * n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(n, m);
    dim3 dimGrid(cdiv(n, dimBlock.x), cdiv(m, dimBlock.y));

    TransposeMatrixKernel<<<dimGrid, dimBlock>>>(d_M, m, n);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_M, d_M, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_M));
}

/**
 * 行主序 Tiled 矩阵乘法（主机接口）
 */
void matrixMulTiledRowMajor(float* h_P, const float* h_M, const float* h_N, int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    CHECK_CUDA(cudaMalloc(&d_M, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_N, n * o * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, m * o * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, n * o * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(cdiv(o, dimBlock.x), cdiv(m, dimBlock.y));

    TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}

/**
 * 列主序 Tiled 矩阵乘法（主机接口）
 * N 矩阵需要预先转置（以列主序存储）
 */
void matrixMulTiledColMajor(float* h_P, const float* h_M, const float* h_N_transposed, 
                             int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    CHECK_CUDA(cudaMalloc(&d_M, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_N, n * o * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, m * o * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N_transposed, n * o * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(cdiv(o, dimBlock.x), cdiv(m, dimBlock.y));

    TiledMatrixMulKernelColMajor<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}
