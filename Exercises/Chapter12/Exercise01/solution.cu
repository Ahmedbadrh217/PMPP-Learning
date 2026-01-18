/**
 * 第十二章：归并 - CUDA 实现
 * 
 * 参考：chapter-12/code/merge.cu
 * 
 * 本实现包含：
 * 1. co_rank - 协同排名函数（二分搜索）
 * 2. merge_sequential - 顺序归并
 * 3. merge_basic_kernel - 基础并行归并（图12.9）
 * 4. merge_tiled_kernel - 分块并行归并（图12.11-12.13）
 */

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// 辅助函数
__host__ __device__ inline int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

// ====================== 核心函数 ======================

/**
 * 协同排名函数（co-rank）
 * 使用二分搜索找到：合并结果中第 k 个元素时，A 中需要取多少个元素
 * 
 * 参数：
 *   k - 合并结果中的位置
 *   A, m - 第一个有序数组及其长度
 *   B, n - 第二个有序数组及其长度
 * 
 * 返回：
 *   i - 使得 A[0:i] 和 B[0:k-i] 合并后正好是 C[0:k]
 */
__host__ __device__ int co_rank(int k, float* A, int m, float* B, int n) {
    // 初始猜测：取尽可能多的 A 元素
    int i = min(k, m);
    int j = k - i;

    // 二分搜索边界
    int i_low = max(0, k - n);
    int j_low = max(0, k - m);

    bool active = true;
    while (active) {
        // 如果 i 太大（A[i-1] > B[j]），需要减少 i
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            int delta = cdiv(i - i_low, 2);
            j_low = j;
            i -= delta;
            j += delta;
        }
        // 如果 i 太小（B[j-1] >= A[i]），需要增加 i
        else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            int delta = cdiv(j - j_low, 2);
            i_low = i;
            i += delta;
            j -= delta;
        }
        // 找到正确位置
        else {
            active = false;
        }
    }
    return i;
}

/**
 * 顺序归并
 * 经典的两指针归并算法
 */
__host__ __device__ void merge_sequential_impl(float* A, int m, float* B, int n, float* C) {
    int i = 0, j = 0, k = 0;
    
    // 同时遍历 A 和 B
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    
    // 复制 A 剩余元素
    while (i < m) {
        C[k++] = A[i++];
    }
    
    // 复制 B 剩余元素
    while (j < n) {
        C[k++] = B[j++];
    }
}

// ====================== Kernels ======================

/**
 * 基础并行归并 Kernel（图12.9）
 * 每个线程独立计算自己负责的输出范围，然后顺序归并
 */
__global__ void merge_basic_kernel(float* A, int m, float* B, int n, float* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m + n;
    
    // 每个线程处理的元素数量
    int elementsPerThread = cdiv(total, blockDim.x * gridDim.x);
    
    // 计算本线程负责的输出范围 [k_curr, k_next)
    int k_curr = tid * elementsPerThread;
    int k_next = min((tid + 1) * elementsPerThread, total);
    
    if (k_curr >= total) return;
    
    // 使用 co-rank 找到对应的 A 和 B 子数组范围
    int i_curr = co_rank(k_curr, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int i_next = co_rank(k_next, A, m, B, n);
    int j_next = k_next - i_next;
    
    // 顺序归并子数组
    merge_sequential_impl(&A[i_curr], i_next - i_curr, 
                          &B[j_curr], j_next - j_curr, 
                          &C[k_curr]);
}

/**
 * 分块并行归并 Kernel（图12.11-12.13）
 * 使用共享内存减少全局内存 co-rank 调用
 */
__global__ void merge_tiled_kernel(float* A, int m, float* B, int n, float* C) {
    // 共享内存：A 和 B 的分块
    extern __shared__ float shareAB[];
    float* A_S = shareAB;
    float* B_S = shareAB + TILE_SIZE;
    
    int total = m + n;
    int chunk = cdiv(total, gridDim.x);
    
    // Block 负责的输出范围
    int C_curr = blockIdx.x * chunk;
    int C_next = min((blockIdx.x + 1) * chunk, total);
    
    // 只有线程 0 计算 Block 级别的 co-rank（减少全局内存访问）
    if (threadIdx.x == 0) {
        A_S[0] = (float)co_rank(C_curr, A, m, B, n);
        A_S[1] = (float)co_rank(C_next, A, m, B, n);
    }
    __syncthreads();
    
    // 所有线程读取 Block 级别的范围
    int A_curr = (int)A_S[0];
    int A_next = (int)A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    
    int total_iteration = cdiv(C_length, TILE_SIZE);
    
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;
    
    // 分块迭代
    for (int counter = 0; counter < total_iteration; counter++) {
        int A_remaining = A_length - A_consumed;
        int B_remaining = B_length - B_consumed;
        int A_tile = min(TILE_SIZE, A_remaining);
        int B_tile = min(TILE_SIZE, B_remaining);
        int tile_merged = min(TILE_SIZE, C_length - C_completed);
        
        // 协作加载 A 和 B 分块到共享内存
        for (int i = threadIdx.x; i < A_tile; i += blockDim.x) {
            A_S[i] = A[A_curr + A_consumed + i];
        }
        for (int i = threadIdx.x; i < B_tile; i += blockDim.x) {
            B_S[i] = B[B_curr + B_consumed + i];
        }
        __syncthreads();
        
        // 每个线程处理分块内的一部分
        int thread_chunk = cdiv(tile_merged, blockDim.x);
        int c_tile_start = threadIdx.x * thread_chunk;
        int c_tile_end = min((threadIdx.x + 1) * thread_chunk, tile_merged);
        
        // 在共享内存上计算 co-rank（比全局内存快得多）
        int a_tile_start = co_rank(c_tile_start, A_S, A_tile, B_S, B_tile);
        int b_tile_start = c_tile_start - a_tile_start;
        int a_tile_end = co_rank(c_tile_end, A_S, A_tile, B_S, B_tile);
        int b_tile_end = c_tile_end - a_tile_end;
        
        // 顺序归并到全局内存
        merge_sequential_impl(A_S + a_tile_start, a_tile_end - a_tile_start,
                              B_S + b_tile_start, b_tile_end - b_tile_start,
                              C + C_curr + C_completed + c_tile_start);
        __syncthreads();
        
        // 更新已消耗的元素数量
        int consumed_from_A = co_rank(tile_merged, A_S, A_tile, B_S, B_tile);
        A_consumed += consumed_from_A;
        B_consumed += (tile_merged - consumed_from_A);
        C_completed += tile_merged;
        
        __syncthreads();
    }
}

// ====================== 主机接口 ======================

/**
 * CPU 顺序归并（主机接口）
 */
void merge_sequential(float* A, int m, float* B, int n, float* C) {
    merge_sequential_impl(A, m, B, n, C);
}

/**
 * 基础并行归并（主机接口）
 */
void merge_basic_gpu(float* A, int m, float* B, int n, float* C) {
    float *d_A, *d_B, *d_C;
    int total = m + n;
    
    CHECK_CUDA(cudaMalloc(&d_A, m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, total * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_A, A, m * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, n * sizeof(float), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int numBlocks = 1;  // 单 Block 简化版
    
    merge_basic_kernel<<<numBlocks, blockSize>>>(d_A, m, d_B, n, d_C);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

/**
 * 分块并行归并（主机接口）
 */
void merge_tiled_gpu(float* A, int m, float* B, int n, float* C) {
    float *d_A, *d_B, *d_C;
    int total = m + n;
    
    CHECK_CUDA(cudaMalloc(&d_A, m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, total * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_A, A, m * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, n * sizeof(float), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int numBlocks = cdiv(total, threadsPerBlock);
    numBlocks = min(numBlocks, 65535);
    
    int sharedMemBytes = 2 * TILE_SIZE * sizeof(float);
    
    merge_tiled_kernel<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(d_A, m, d_B, n, d_C);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(C, d_C, total * sizeof(float), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}
