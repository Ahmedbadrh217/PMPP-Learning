#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

/**
 * 计算最优 Tile 宽度
 * 基于硬件规格动态计算，而不是硬编码固定值
 * 
 * 考虑因素：
 * 1. 每块最大线程数
 * 2. 块的各维度最大值
 * 3. 共享内存大小（需要 2 个 Tile）
 * 4. 矩阵维度
 * 5. 2 的幂次（更好的内存对齐）
 */
int calculateOptimalTileWidth(int m, int n, int o) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    // 获取硬件限制
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlockDimX = prop.maxThreadsDim[0];
    int maxBlockDimY = prop.maxThreadsDim[1];
    int sharedMemPerBlock = prop.sharedMemPerBlock;

    // 1. 基于每块最大线程数（方形 Tile）
    int tileWidth = static_cast<int>(sqrt(maxThreadsPerBlock));

    // 2. 基于块的各维度最大值
    tileWidth = std::min(tileWidth, std::min(maxBlockDimX, maxBlockDimY));

    // 3. 基于共享内存（需要 2 个 Tile）
    int maxTileWidthBySharedMem = static_cast<int>(sqrt(sharedMemPerBlock / (2 * sizeof(float))));
    tileWidth = std::min(tileWidth, maxTileWidthBySharedMem);

    // 4. 基于矩阵维度（Tile 不应大于矩阵）
    tileWidth = std::min(tileWidth, std::min(m, std::min(n, o)));

    // 5. 向下取整到 2 的幂次（更好的内存对齐）
    tileWidth = 1 << static_cast<int>(log2(tileWidth));

    // 6. 确保最小实用大小
    tileWidth = std::max(16, tileWidth);

    return tileWidth;
}

/**
 * 朴素矩阵乘法 Kernel
 */
__global__ void MatrixMulKernel(float* M, float* N, float* P, int m, int n, int o) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < o) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += M[row * n + i] * N[i * o + col];
        }
        P[row * o + col] = sum;
    }
}

/**
 * 动态 Tile 大小的 Tiled 矩阵乘法 Kernel
 * 使用动态共享内存（extern __shared__）
 * Tile 大小作为参数传入，而不是编译时常量
 */
__global__ void TiledMatrixMulKernelDynamic(float* M, float* N, float* P, 
                                             int m, int n, int o, int tileWidth) {
    // 动态共享内存
    extern __shared__ float sharedMem[];
    // 分割共享内存为两部分
    float* Mds = sharedMem;
    float* Nds = &sharedMem[tileWidth * tileWidth];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * tileWidth + ty;
    int col = bx * tileWidth + tx;

    float Pvalue = 0.0f;
    int numTiles = (n + tileWidth - 1) / tileWidth;
    
    for (int ph = 0; ph < numTiles; ++ph) {
        // 加载 M 的 Tile
        if (row < m && (ph * tileWidth + tx) < n) {
            Mds[ty * tileWidth + tx] = M[row * n + ph * tileWidth + tx];
        } else {
            Mds[ty * tileWidth + tx] = 0.0f;
        }

        // 加载 N 的 Tile
        if ((ph * tileWidth + ty) < n && col < o) {
            Nds[ty * tileWidth + tx] = N[(ph * tileWidth + ty) * o + col];
        } else {
            Nds[ty * tileWidth + tx] = 0.0f;
        }

        __syncthreads();

        // 计算部分点积
        for (int k = 0; k < tileWidth; ++k) {
            Pvalue += Mds[ty * tileWidth + k] * Nds[k * tileWidth + tx];
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
 * 朴素矩阵乘法（主机接口）
 */
void matrixMul(float* h_P, const float* h_M, const float* h_N, int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    CHECK_CUDA(cudaMalloc(&d_M, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_N, n * o * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, m * o * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, n * o * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(16, 16);
    dim3 dimGrid(cdiv(o, dimBlock.x), cdiv(m, dimBlock.y));

    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}

/**
 * 动态 Tile 大小的 Tiled 矩阵乘法（主机接口）
 */
void matrixMulTilingDynamic(float* h_P, const float* h_M, const float* h_N, int m, int n, int o) {
    float *d_M, *d_N, *d_P;
    
    // 动态计算最优 Tile 宽度
    int tileWidth = calculateOptimalTileWidth(m, n, o);
    printf("使用动态计算的 Tile 宽度: %d\n", tileWidth);

    CHECK_CUDA(cudaMalloc(&d_M, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_N, n * o * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, m * o * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, n * o * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(tileWidth, tileWidth);
    dim3 dimGrid(cdiv(o, dimBlock.x), cdiv(m, dimBlock.y));

    // 动态共享内存大小
    int sharedMemSize = 2 * tileWidth * tileWidth * sizeof(float);
    
    TiledMatrixMulKernelDynamic<<<dimGrid, dimBlock, sharedMemSize>>>(
        d_M, d_N, d_P, m, n, o, tileWidth);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}
