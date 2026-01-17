/**
 * 第八章：模板 - 3D七点模板实现
 * 
 * 参考：chapter-08/code/stencil.cu
 * 
 * 本实现包含5种 kernel：
 * 1. stencil_3d_sequential - CPU顺序实现
 * 2. stencil_kernel - 基础并行（图8.6）
 * 3. stencil_kernel_shared_memory - 共享内存（图8.8）
 * 4. stencil_kernel_thread_coarsening - 线程粗化（图8.10）
 * 5. stencil_kernel_register_tiling - 寄存器优化（图8.12）
 */

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>

// 全局系数定义
int c0 = 0;
int c1 = 1;
int c2 = 1;
int c3 = 1;
int c4 = 1;
int c5 = 1;
int c6 = 1;

/**
 * 打印3D数组的一个切片（调试用）
 */
void print3DSlice(float* arr, int N, int z) {
    std::cout << "Slice z=" << z << ":" << std::endl;
    for (int j = 0; j < N && j < 8; ++j) {
        for (int k = 0; k < N && k < 8; ++k) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(1)
                      << arr[z * N * N + j * N + k] << " ";
        }
        std::cout << std::endl;
    }
}

/**
 * CPU 顺序实现
 * 七点模板：中心 + 上下前后左右
 */
void stencil_3d_sequential(float* in, float* out, unsigned int N,
                           int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    for (unsigned int i = 1; i < N - 1; i++) {
        for (unsigned int j = 1; j < N - 1; j++) {
            for (unsigned int k = 1; k < N - 1; k++) {
                out[i * N * N + j * N + k] =
                    c0 * in[i * N * N + j * N + k] +
                    c1 * in[i * N * N + j * N + (k - 1)] +
                    c2 * in[i * N * N + j * N + (k + 1)] +
                    c3 * in[i * N * N + (j - 1) * N + k] +
                    c4 * in[i * N * N + (j + 1) * N + k] +
                    c5 * in[(i - 1) * N * N + j * N + k] +
                    c6 * in[(i + 1) * N * N + j * N + k];
            }
        }
    }
}

/**
 * 基础并行 Kernel（图8.6）
 * 每个线程计算一个输出点，直接从全局内存读取
 */
__global__ void stencil_kernel(float* in, float* out, unsigned int N,
                               int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i * N * N + j * N + k] =
            c0 * in[i * N * N + j * N + k] +
            c1 * in[i * N * N + j * N + (k - 1)] +
            c2 * in[i * N * N + j * N + (k + 1)] +
            c3 * in[i * N * N + (j - 1) * N + k] +
            c4 * in[i * N * N + (j + 1) * N + k] +
            c5 * in[(i - 1) * N * N + j * N + k] +
            c6 * in[(i + 1) * N * N + j * N + k];
    }
}

/**
 * 基础并行（主机接口）
 */
void stencil_3d_parallel_basic(float* in, float* out, unsigned int N,
                               int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    float *d_in, *d_out;
    size_t size = N * N * N * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));
    CHECK_CUDA(cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, size));
    
    dim3 dimBlock(OUT_TILE_DIM_SMALL, OUT_TILE_DIM_SMALL, OUT_TILE_DIM_SMALL);
    dim3 dimGrid(cdiv(N, dimBlock.x), cdiv(N, dimBlock.y), cdiv(N, dimBlock.z));
    
    stencil_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

/**
 * 共享内存 Kernel（图8.8）
 * 使用 3D 共享内存存储输入 Tile
 */
__global__ void stencil_kernel_shared_memory(float* in, float* out, unsigned int N,
                                             int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    int i = blockIdx.z * OUT_TILE_DIM_SMALL + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM_SMALL + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM_SMALL + threadIdx.x - 1;
    
    __shared__ float in_s[IN_TILE_DIM_SMALL][IN_TILE_DIM_SMALL][IN_TILE_DIM_SMALL];
    
    // 协作加载
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    } else {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // 只有内部线程计算
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM_SMALL - 1 &&
            threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM_SMALL - 1 &&
            threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_SMALL - 1) {
            out[i * N * N + j * N + k] =
                c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
                c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
                c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
                c3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
                c4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
                c5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
                c6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }
}

/**
 * 共享内存（主机接口）
 */
void stencil_3d_parallel_shared_memory(float* in, float* out, unsigned int N,
                                       int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    float *d_in, *d_out;
    size_t size = N * N * N * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));
    CHECK_CUDA(cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, size));
    
    dim3 dimBlock(IN_TILE_DIM_SMALL, IN_TILE_DIM_SMALL, IN_TILE_DIM_SMALL);
    dim3 dimGrid(cdiv(N, OUT_TILE_DIM_SMALL), cdiv(N, OUT_TILE_DIM_SMALL), cdiv(N, OUT_TILE_DIM_SMALL));
    
    stencil_kernel_shared_memory<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

/**
 * 线程粗化 Kernel（图8.10）
 * Z方向粗化：每个线程块处理 OUT_TILE_DIM_BIG 层
 * 使用三层 2D 共享内存作为滑动窗口
 */
__global__ void stencil_kernel_thread_coarsening(float* in, float* out, unsigned int N,
                                                 int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    int iStart = blockIdx.z * OUT_TILE_DIM_BIG;
    int j = blockIdx.y * OUT_TILE_DIM_BIG + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM_BIG + threadIdx.x - 1;
    
    __shared__ float inPrev_s[IN_TILE_DIM_BIG][IN_TILE_DIM_BIG];
    __shared__ float inCurr_s[IN_TILE_DIM_BIG][IN_TILE_DIM_BIG];
    __shared__ float inNext_s[IN_TILE_DIM_BIG][IN_TILE_DIM_BIG];
    
    // 初始化
    inPrev_s[threadIdx.y][threadIdx.x] = 0.0f;
    inCurr_s[threadIdx.y][threadIdx.x] = 0.0f;
    inNext_s[threadIdx.y][threadIdx.x] = 0.0f;
    
    // 加载前一层
    if (iStart - 1 >= 0 && iStart - 1 < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
    }
    // 加载当前层
    if (iStart >= 0 && iStart < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
    }
    
    // Z方向遍历
    for (int i = iStart; i < iStart + OUT_TILE_DIM_BIG; ++i) {
        // 加载下一层
        inNext_s[threadIdx.y][threadIdx.x] = 0.0f;
        if (i + 1 >= 0 && i + 1 < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();
        
        // 计算
        if (i >= 1 && i < (int)N - 1 && j >= 1 && j < (int)N - 1 && k >= 1 && k < (int)N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM_BIG - 1 &&
                threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_BIG - 1) {
                out[i * N * N + j * N + k] =
                    c0 * inCurr_s[threadIdx.y][threadIdx.x] +
                    c1 * inCurr_s[threadIdx.y][threadIdx.x - 1] +
                    c2 * inCurr_s[threadIdx.y][threadIdx.x + 1] +
                    c3 * inCurr_s[threadIdx.y - 1][threadIdx.x] +
                    c4 * inCurr_s[threadIdx.y + 1][threadIdx.x] +
                    c5 * inPrev_s[threadIdx.y][threadIdx.x] +
                    c6 * inNext_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        
        // 滑动窗口
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}

/**
 * 线程粗化（主机接口）
 */
void stencil_3d_parallel_thread_coarsening(float* in, float* out, unsigned int N,
                                           int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    float *d_in, *d_out;
    size_t size = N * N * N * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));
    CHECK_CUDA(cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, size));
    
    dim3 dimBlock(IN_TILE_DIM_BIG, IN_TILE_DIM_BIG, 1);
    dim3 dimGrid(cdiv(N, OUT_TILE_DIM_BIG), cdiv(N, OUT_TILE_DIM_BIG), cdiv(N, OUT_TILE_DIM_BIG));
    
    stencil_kernel_thread_coarsening<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

/**
 * 寄存器优化 Kernel（图8.12）
 * Z方向数据存寄存器（inPrev, inCurr, inNext）
 * XY平面存共享内存
 */
__global__ void stencil_kernel_register_tiling(float* in, float* out, unsigned int N,
                                               int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    int iStart = blockIdx.z * OUT_TILE_DIM_SMALL;
    int j = blockIdx.y * OUT_TILE_DIM_SMALL + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM_SMALL + threadIdx.x - 1;
    
    // Z方向使用寄存器
    float inPrev = 0.0f;
    float inCurr = 0.0f;
    float inNext = 0.0f;
    
    // XY平面使用共享内存
    __shared__ float inCurr_s[IN_TILE_DIM_SMALL][IN_TILE_DIM_SMALL];
    
    // 加载前一层到寄存器
    if (iStart - 1 >= 0 && iStart - 1 < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
        inPrev = in[(iStart - 1) * N * N + j * N + k];
    }
    
    // 加载当前层
    if (iStart >= 0 && iStart < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
        inCurr = in[iStart * N * N + j * N + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }
    
    // Z方向遍历
    for (int i = iStart; i < iStart + OUT_TILE_DIM_SMALL; ++i) {
        // 加载下一层到寄存器
        if (i + 1 >= 0 && i + 1 < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
            inNext = in[(i + 1) * N * N + j * N + k];
        }
        
        __syncthreads();
        
        // 计算
        if (i >= 1 && i < (int)N - 1 && j >= 1 && j < (int)N - 1 && k >= 1 && k < (int)N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM_SMALL - 1 &&
                threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_SMALL - 1) {
                out[i * N * N + j * N + k] =
                    c0 * inCurr +
                    c1 * inCurr_s[threadIdx.y][threadIdx.x - 1] +
                    c2 * inCurr_s[threadIdx.y][threadIdx.x + 1] +
                    c3 * inCurr_s[threadIdx.y - 1][threadIdx.x] +
                    c4 * inCurr_s[threadIdx.y + 1][threadIdx.x] +
                    c5 * inPrev +
                    c6 * inNext;
            }
        }
        __syncthreads();
        
        // 滑动窗口
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
    }
}

/**
 * 寄存器优化（主机接口）
 */
void stencil_3d_parallel_register_tiling(float* in, float* out, unsigned int N,
                                         int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    float *d_in, *d_out;
    size_t size = N * N * N * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));
    CHECK_CUDA(cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, size));
    
    dim3 dimBlock(IN_TILE_DIM_SMALL, IN_TILE_DIM_SMALL, 1);
    dim3 dimGrid(cdiv(N, OUT_TILE_DIM_SMALL), cdiv(N, OUT_TILE_DIM_SMALL), cdiv(N, OUT_TILE_DIM_SMALL));
    
    stencil_kernel_register_tiling<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}
