/**
 * 第十三章：排序 - CUDA 实现
 * 
 * 参考：chapter-13/code/gpu_radix_sort.cu, gpu_merge_sort.cu
 * 
 * 本实现包含：
 * 1. 朴素三核基数排序
 * 2. 内存合并基数排序（练习1）
 * 3. 多位基数排序（练习2）
 * 4. 线程粗化基数排序（练习3）
 * 5. 并行归并排序（练习4）
 */

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

// ====================== 辅助函数 ======================

// 向上取整除法
__host__ __device__ inline int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

// ====================== 1. 朴素三核基数排序 ======================

/**
 * Kernel: 提取每个元素的指定位
 * @param input 输入数组
 * @param bits 输出位数组（0或1）
 * @param N 数组长度
 * @param iter 当前位索引
 */
__global__ void extractBitsKernel(unsigned int* input, unsigned int* bits, int N, int iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int key = input[tid];
        bits[tid] = (key >> iter) & 1;
    }
}

/**
 * Kernel: 分散（scatter）- 根据扫描结果重排元素
 * @param input 输入数组
 * @param output 输出数组
 * @param scannedBits 前缀和结果（1的个数）
 * @param N 数组长度
 * @param iter 当前位索引
 * @param totalOnes 总共的1的个数
 */
__global__ void scatterKernel(unsigned int* input, unsigned int* output, 
                              unsigned int* scannedBits, int N, int iter, unsigned int totalOnes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int key = input[tid];
        unsigned int bit = (key >> iter) & 1;
        unsigned int numOnesBefore = scannedBits[tid];
        
        // 0放在前面，1放在后面
        // 对于bit=0: 目标位置 = 当前位置 - 之前1的个数
        // 对于bit=1: 目标位置 = 总0的个数 + 之前1的个数
        unsigned int dst = (bit == 0) ? (tid - numOnesBefore) : (N - totalOnes + numOnesBefore);
        output[dst] = key;
    }
}

/**
 * 1. 朴素并行基数排序（三核实现）
 * 
 * 算法流程：
 *   对于每一位（从最低位到最高位）：
 *     1. extractBitsKernel: 提取每个元素的当前位
 *     2. thrust::exclusive_scan: 对位数组做前缀和
 *     3. scatterKernel: 根据前缀和结果重排元素
 */
void gpuRadixSortNaive(unsigned int* d_input, int N) {
    unsigned int *d_output, *d_bits;
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_bits, N * sizeof(unsigned int)));
    
    int numBlocks = cdiv(N, BLOCK_SIZE);
    
    // 对32位整数的每一位进行排序
    for (int iter = 0; iter < 32; iter++) {
        // 步骤1：提取当前位
        extractBitsKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_bits, N, iter);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 保存最后一个元素的位值（用于计算总1的个数）
        unsigned int lastBit;
        CHECK_CUDA(cudaMemcpy(&lastBit, d_bits + (N - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
        // 步骤2：前缀和（使用 Thrust）
        thrust::device_ptr<unsigned int> d_bits_ptr(d_bits);
        thrust::exclusive_scan(d_bits_ptr, d_bits_ptr + N, d_bits_ptr);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 获取扫描后最后一个元素的值，加上原始位值得到总1的个数
        unsigned int scannedLast;
        CHECK_CUDA(cudaMemcpy(&scannedLast, d_bits + (N - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));
        unsigned int totalOnes = scannedLast + lastBit;
        
        // 步骤3：分散重排
        scatterKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_bits, N, iter, totalOnes);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 交换输入输出缓冲区
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_bits));
}

// ====================== 2. 内存合并基数排序（练习1） ======================

/**
 * Kernel: 本地扫描 - 每个 Block 内进行前缀和
 * 使用 Blelloch 算法在共享内存中实现
 * 
 * @param d_input 输入数组
 * @param d_localScan 输出：每个元素之前有多少个1
 * @param d_blockOneCount 输出：每个 Block 的1的总数
 * @param N 数组长度
 * @param iter 当前位索引
 */
__global__ void localScanKernel(unsigned int* d_input, unsigned int* d_localScan, 
                                unsigned int* d_blockOneCount, int N, int iter) {
    extern __shared__ unsigned int s_bits[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // 加载当前位到共享内存
    unsigned int bit_val = 0;
    if (gid < N) {
        unsigned int key = d_input[gid];
        bit_val = (key >> iter) & 1;
    }
    s_bits[tid] = bit_val;
    __syncthreads();
    
    // Blelloch 扫描 - 上扫阶段（归约）
    for (unsigned int offset = 1; offset < blockDim.x; offset *= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < blockDim.x) {
            s_bits[index] += s_bits[index - offset];
        }
        __syncthreads();
    }
    
    // 保存 Block 总和并清零最后一个元素
    if (tid == 0) {
        d_blockOneCount[blockIdx.x] = s_bits[blockDim.x - 1];
        s_bits[blockDim.x - 1] = 0;
    }
    __syncthreads();
    
    // Blelloch 扫描 - 下扫阶段
    for (unsigned int offset = blockDim.x / 2; offset >= 1; offset /= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < blockDim.x) {
            unsigned int t = s_bits[index - offset];
            s_bits[index - offset] = s_bits[index];
            s_bits[index] += t;
        }
        __syncthreads();
        if (offset == 1) break;  // 防止无符号整数下溢
    }
    
    // 写回全局内存
    if (gid < N) {
        d_localScan[gid] = s_bits[tid];
    }
}

/**
 * Kernel: 合并写入的分散
 * 根据本地偏移和全局偏移，实现内存合并写入
 */
__global__ void scatterKernelCoalesced(unsigned int* d_input, unsigned int* d_output,
                                       unsigned int* d_localScan, 
                                       unsigned int* d_blockZeroOffsets,
                                       unsigned int* d_blockOneOffsets,
                                       unsigned int totalZeros, int N, int iter) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    if (gid < N) {
        unsigned int key = d_input[gid];
        unsigned int bit = (key >> iter) & 1;
        unsigned int localPrefix = d_localScan[gid];  // 之前有多少个1
        
        unsigned int dest;
        if (bit == 0) {
            // 0的目标位置 = Block内0的起始位置 + (线程ID - 之前1的个数)
            dest = d_blockZeroOffsets[blockIdx.x] + tid - localPrefix;
        } else {
            // 1的目标位置 = 所有0之后 + Block内1的起始位置 + 之前1的个数
            dest = totalZeros + d_blockOneOffsets[blockIdx.x] + localPrefix;
        }
        d_output[dest] = key;
    }
}

/**
 * 2. 内存合并基数排序
 * 练习1：使用共享内存优化内存访问
 */
void gpuRadixSortCoalesced(unsigned int* d_input, int N) {
    unsigned int *d_output, *d_localScan, *d_blockOneCount;
    unsigned int *d_blockZeroOffsets, *d_blockOneOffsets;
    
    int numBlocks = cdiv(N, BLOCK_SIZE);
    
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_localScan, N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_blockOneCount, numBlocks * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_blockZeroOffsets, numBlocks * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_blockOneOffsets, numBlocks * sizeof(unsigned int)));
    
    // 主机端数组用于计算偏移
    unsigned int* h_blockOneCount = new unsigned int[numBlocks];
    unsigned int* h_blockZeroCount = new unsigned int[numBlocks];
    unsigned int* h_blockZeroOffsets = new unsigned int[numBlocks];
    unsigned int* h_blockOneOffsets = new unsigned int[numBlocks];
    
    for (int iter = 0; iter < 32; iter++) {
        // 步骤1：本地扫描
        localScanKernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(unsigned int)>>>
            (d_input, d_localScan, d_blockOneCount, N, iter);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 步骤2：在 CPU 上计算全局偏移
        CHECK_CUDA(cudaMemcpy(h_blockOneCount, d_blockOneCount, 
                              numBlocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
        // 计算每个 Block 的0的个数
        for (int i = 0; i < numBlocks; i++) {
            int blockSize = ((i == numBlocks - 1) && (N % BLOCK_SIZE != 0)) 
                          ? (N % BLOCK_SIZE) : BLOCK_SIZE;
            h_blockZeroCount[i] = blockSize - h_blockOneCount[i];
        }
        
        // 计算前缀和（偏移）
        h_blockZeroOffsets[0] = 0;
        h_blockOneOffsets[0] = 0;
        for (int i = 1; i < numBlocks; i++) {
            h_blockZeroOffsets[i] = h_blockZeroOffsets[i-1] + h_blockZeroCount[i-1];
            h_blockOneOffsets[i] = h_blockOneOffsets[i-1] + h_blockOneCount[i-1];
        }
        
        unsigned int totalZeros = h_blockZeroOffsets[numBlocks-1] + h_blockZeroCount[numBlocks-1];
        
        CHECK_CUDA(cudaMemcpy(d_blockZeroOffsets, h_blockZeroOffsets, 
                              numBlocks * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_blockOneOffsets, h_blockOneOffsets,
                              numBlocks * sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        // 步骤3：分散
        scatterKernelCoalesced<<<numBlocks, BLOCK_SIZE>>>
            (d_input, d_output, d_localScan, d_blockZeroOffsets, d_blockOneOffsets, totalZeros, N, iter);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 交换缓冲区
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    
    delete[] h_blockOneCount;
    delete[] h_blockZeroCount;
    delete[] h_blockZeroOffsets;
    delete[] h_blockOneOffsets;
    
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_localScan));
    CHECK_CUDA(cudaFree(d_blockOneCount));
    CHECK_CUDA(cudaFree(d_blockZeroOffsets));
    CHECK_CUDA(cudaFree(d_blockOneOffsets));
}

// ====================== 3. 多位基数排序（练习2） ======================

/**
 * Kernel: 多位本地扫描
 * 每个 Block 构建直方图并计算本地偏移
 */
__global__ void localScanKernelMultibit(const unsigned int* d_input, unsigned int* d_localOffsets,
                                        unsigned int* d_blockHist, int N, int iter, int r) {
    const unsigned int numBuckets = 1 << r;  // 2^r 个桶
    extern __shared__ unsigned int shared[];
    
    unsigned int* s_hist = shared;                    // 直方图
    unsigned int* s_digits = &s_hist[numBuckets];     // 每个线程的数字
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // 提取当前位组的数字
    unsigned int digit = 0;
    if (gid < N) {
        unsigned int key = d_input[gid];
        digit = (key >> (iter * r)) & (numBuckets - 1);
    }
    s_digits[tid] = digit;
    
    // 初始化直方图
    for (unsigned int i = tid; i < numBuckets; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // 原子操作更新直方图
    if (gid < N) {
        atomicAdd(&s_hist[digit], 1);
    }
    __syncthreads();
    
    // 写入 Block 直方图到全局内存
    for (unsigned int i = tid; i < numBuckets; i += blockDim.x) {
        d_blockHist[blockIdx.x * numBuckets + i] = s_hist[i];
    }
    __syncthreads();
    
    // 计算本地偏移（之前相同数字的个数）
    unsigned int local_offset = 0;
    for (int j = 0; j < tid; j++) {
        if (s_digits[j] == digit) {
            local_offset++;
        }
    }
    if (gid < N) {
        d_localOffsets[gid] = local_offset;
    }
}

/**
 * Kernel: 多位分散
 */
__global__ void scatterKernelMultibit(const unsigned int* d_input, unsigned int* d_output,
                                      const unsigned int* d_localOffsets,
                                      const unsigned int* d_globalOffsets,
                                      int N, int iter, int r) {
    const unsigned int numBuckets = 1 << r;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    if (gid < N) {
        unsigned int key = d_input[gid];
        unsigned int digit = (key >> (iter * r)) & (numBuckets - 1);
        
        unsigned int blockOffset = d_globalOffsets[blockIdx.x * numBuckets + digit];
        unsigned int localOffset = d_localOffsets[gid];
        unsigned int dest = blockOffset + localOffset;
        
        d_output[dest] = key;
    }
}

/**
 * 3. 多位基数排序
 * 练习2：同时处理多位，减少迭代次数
 */
void gpuRadixSortMultibit(unsigned int* d_input, int N, unsigned int r) {
    const unsigned int numBuckets = 1 << r;
    unsigned int numPasses = cdiv(32, r);  // 32位整数需要的轮数
    
    int numBlocks = cdiv(N, BLOCK_SIZE);
    
    unsigned int* d_output;
    unsigned int* d_localOffsets;
    unsigned int* d_blockHist;
    unsigned int* d_globalOffsets;
    
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_localOffsets, N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_blockHist, numBlocks * numBuckets * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_globalOffsets, numBlocks * numBuckets * sizeof(unsigned int)));
    
    unsigned int* h_blockHist = new unsigned int[numBlocks * numBuckets];
    unsigned int* h_globalOffsets = new unsigned int[numBlocks * numBuckets];
    unsigned int* h_totalBucket = new unsigned int[numBuckets];
    unsigned int* h_prefixBucket = new unsigned int[numBuckets];
    
    size_t sharedMemSize = numBuckets * sizeof(unsigned int) + BLOCK_SIZE * sizeof(unsigned int);
    
    for (unsigned int pass = 0; pass < numPasses; pass++) {
        // 步骤1：本地扫描和直方图
        localScanKernelMultibit<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>
            (d_input, d_localOffsets, d_blockHist, N, pass, r);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 步骤2：CPU 计算全局偏移
        CHECK_CUDA(cudaMemcpy(h_blockHist, d_blockHist, 
                              numBlocks * numBuckets * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
        // 计算每个桶的总数
        for (unsigned int bucket = 0; bucket < numBuckets; bucket++) {
            unsigned int sum = 0;
            for (int block = 0; block < numBlocks; block++) {
                sum += h_blockHist[block * numBuckets + bucket];
            }
            h_totalBucket[bucket] = sum;
        }
        
        // 计算桶的前缀和
        h_prefixBucket[0] = 0;
        for (unsigned int bucket = 1; bucket < numBuckets; bucket++) {
            h_prefixBucket[bucket] = h_prefixBucket[bucket - 1] + h_totalBucket[bucket - 1];
        }
        
        // 计算全局偏移
        for (unsigned int bucket = 0; bucket < numBuckets; bucket++) {
            unsigned int sum = 0;
            for (int block = 0; block < numBlocks; block++) {
                h_globalOffsets[block * numBuckets + bucket] = h_prefixBucket[bucket] + sum;
                sum += h_blockHist[block * numBuckets + bucket];
            }
        }
        
        CHECK_CUDA(cudaMemcpy(d_globalOffsets, h_globalOffsets,
                              numBlocks * numBuckets * sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        // 步骤3：分散
        scatterKernelMultibit<<<numBlocks, BLOCK_SIZE>>>
            (d_input, d_output, d_localOffsets, d_globalOffsets, N, pass, r);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 交换缓冲区
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    
    delete[] h_blockHist;
    delete[] h_globalOffsets;
    delete[] h_totalBucket;
    delete[] h_prefixBucket;
    
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_localOffsets));
    CHECK_CUDA(cudaFree(d_blockHist));
    CHECK_CUDA(cudaFree(d_globalOffsets));
}

// ====================== 4. 线程粗化基数排序（练习3） ======================

/**
 * Kernel: 线程粗化的本地扫描
 * 每个线程处理 COARSE_FACTOR 个元素
 */
__global__ void localScanKernelCoarsened(const unsigned int* d_input, unsigned int* d_localOffsets,
                                         unsigned int* d_blockHist, int N, int iter, int r) {
    const unsigned int numBuckets = 1 << r;
    extern __shared__ unsigned int shared[];
    
    unsigned int* s_hist = shared;
    unsigned int* s_digits = &s_hist[numBuckets];
    
    int tid = threadIdx.x;
    int baseIdx = blockIdx.x * blockDim.x * COARSE_FACTOR + tid * COARSE_FACTOR;
    
    // 每个线程处理 COARSE_FACTOR 个元素
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int idx = baseIdx + i;
        unsigned int digit = 0;
        if (idx < N) {
            unsigned int key = d_input[idx];
            digit = (key >> (iter * r)) & (numBuckets - 1);
        }
        s_digits[tid * COARSE_FACTOR + i] = digit;
    }
    __syncthreads();
    
    // 初始化直方图
    for (unsigned int i = tid; i < numBuckets; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // 原子更新直方图
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int idx = baseIdx + i;
        if (idx < N) {
            unsigned int digit = s_digits[tid * COARSE_FACTOR + i];
            atomicAdd(&s_hist[digit], 1);
        }
    }
    __syncthreads();
    
    // 写入直方图
    for (unsigned int i = tid; i < numBuckets; i += blockDim.x) {
        d_blockHist[blockIdx.x * numBuckets + i] = s_hist[i];
    }
    __syncthreads();
    
    // 计算本地偏移
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int globalPos = tid * COARSE_FACTOR + i;
        unsigned int myDigit = s_digits[globalPos];
        unsigned int local_offset = 0;
        
        for (int j = 0; j < globalPos; j++) {
            if (s_digits[j] == myDigit) {
                local_offset++;
            }
        }
        
        int outIdx = baseIdx + i;
        if (outIdx < N) {
            d_localOffsets[outIdx] = local_offset;
        }
    }
}

/**
 * Kernel: 线程粗化的分散
 */
__global__ void scatterKernelCoarsened(const unsigned int* d_input, unsigned int* d_output,
                                       const unsigned int* d_localOffsets,
                                       const unsigned int* d_globalOffsets,
                                       int N, int iter, int r) {
    const unsigned int numBuckets = 1 << r;
    int tid = threadIdx.x;
    int baseIdx = blockIdx.x * blockDim.x * COARSE_FACTOR + tid * COARSE_FACTOR;
    
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int idx = baseIdx + i;
        if (idx < N) {
            unsigned int key = d_input[idx];
            unsigned int digit = (key >> (iter * r)) & (numBuckets - 1);
            
            unsigned int blockOffset = d_globalOffsets[blockIdx.x * numBuckets + digit];
            unsigned int localOffset = d_localOffsets[idx];
            unsigned int dest = blockOffset + localOffset;
            
            d_output[dest] = key;
        }
    }
}

/**
 * 4. 线程粗化基数排序
 * 练习3：每个线程处理多个元素
 */
void gpuRadixSortCoarsened(unsigned int* d_input, int N, unsigned int r) {
    const unsigned int numBuckets = 1 << r;
    unsigned int numPasses = cdiv(32, r);
    
    int numBlocks = cdiv(N, BLOCK_SIZE * COARSE_FACTOR);
    
    unsigned int* d_output;
    unsigned int* d_localOffsets;
    unsigned int* d_blockHist;
    unsigned int* d_globalOffsets;
    
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_localOffsets, N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_blockHist, numBlocks * numBuckets * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_globalOffsets, numBlocks * numBuckets * sizeof(unsigned int)));
    
    unsigned int* h_blockHist = new unsigned int[numBlocks * numBuckets];
    unsigned int* h_globalOffsets = new unsigned int[numBlocks * numBuckets];
    unsigned int* h_totalBucket = new unsigned int[numBuckets];
    unsigned int* h_prefixBucket = new unsigned int[numBuckets];
    
    size_t sharedMemSize = numBuckets * sizeof(unsigned int) + (BLOCK_SIZE * COARSE_FACTOR) * sizeof(unsigned int);
    
    for (unsigned int pass = 0; pass < numPasses; pass++) {
        localScanKernelCoarsened<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>
            (d_input, d_localOffsets, d_blockHist, N, pass, r);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaMemcpy(h_blockHist, d_blockHist,
                              numBlocks * numBuckets * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
        for (unsigned int bucket = 0; bucket < numBuckets; bucket++) {
            unsigned int sum = 0;
            for (int block = 0; block < numBlocks; block++) {
                sum += h_blockHist[block * numBuckets + bucket];
            }
            h_totalBucket[bucket] = sum;
        }
        
        h_prefixBucket[0] = 0;
        for (unsigned int bucket = 1; bucket < numBuckets; bucket++) {
            h_prefixBucket[bucket] = h_prefixBucket[bucket - 1] + h_totalBucket[bucket - 1];
        }
        
        for (unsigned int bucket = 0; bucket < numBuckets; bucket++) {
            unsigned int sum = 0;
            for (int block = 0; block < numBlocks; block++) {
                h_globalOffsets[block * numBuckets + bucket] = h_prefixBucket[bucket] + sum;
                sum += h_blockHist[block * numBuckets + bucket];
            }
        }
        
        CHECK_CUDA(cudaMemcpy(d_globalOffsets, h_globalOffsets,
                              numBlocks * numBuckets * sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        scatterKernelCoarsened<<<numBlocks, BLOCK_SIZE>>>
            (d_input, d_output, d_localOffsets, d_globalOffsets, N, pass, r);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    
    delete[] h_blockHist;
    delete[] h_globalOffsets;
    delete[] h_totalBucket;
    delete[] h_prefixBucket;
    
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_localOffsets));
    CHECK_CUDA(cudaFree(d_blockHist));
    CHECK_CUDA(cudaFree(d_globalOffsets));
}

// ====================== 5. 并行归并排序（练习4） ======================

/**
 * 协同排名函数（co-rank）
 * 使用二分搜索找到归并位置
 */
__host__ __device__ int co_rank(int k, unsigned int* A, int m, unsigned int* B, int n) {
    int i = min(k, m);
    int j = k - i;
    
    int i_low = max(0, k - n);
    int j_low = max(0, k - m);
    
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            int delta = cdiv(i - i_low, 2);
            j_low = j;
            i -= delta;
            j += delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            int delta = cdiv(j - j_low, 2);
            i_low = i;
            i += delta;
            j -= delta;
        } else {
            active = false;
        }
    }
    return i;
}

/**
 * 顺序归并
 */
__host__ __device__ void merge_sequential(unsigned int* A, int m, unsigned int* B, int n, unsigned int* C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

/**
 * Kernel: 归并一轮
 * 每个 Block 归并一对相邻的有序段
 */
__global__ void mergePassKernel(unsigned int* d_in, unsigned int* d_out, int N, int width) {
    int pair = blockIdx.x;
    int start = pair * (2 * width);
    if (start >= N) return;
    
    int mid = min(start + width, N);
    int end = min(start + 2 * width, N);
    int lenA = mid - start;
    int lenB = end - mid;
    
    unsigned int* A = d_in + start;
    unsigned int* B = d_in + mid;
    unsigned int* C = d_out + start;
    
    int total = lenA + lenB;
    
    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    int elementsPerThread = cdiv(total, numThreads);
    
    int k_start = tid * elementsPerThread;
    int k_end = min((tid + 1) * elementsPerThread, total);
    
    int i_start = co_rank(k_start, A, lenA, B, lenB);
    int j_start = k_start - i_start;
    int i_end = co_rank(k_end, A, lenA, B, lenB);
    int j_end = k_end - i_end;
    
    merge_sequential(A + i_start, i_end - i_start, B + j_start, j_end - j_start, C + k_start);
}

/**
 * 5. 并行归并排序
 * 练习4：使用第12章的并行归并
 */
void gpuMergeSort(unsigned int* d_input, int N) {
    unsigned int* d_output;
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(unsigned int)));
    
    int width = 1;
    int numPasses = 0;
    
    while (width < N) {
        int numMerges = cdiv(N, 2 * width);
        
        mergePassKernel<<<numMerges, BLOCK_SIZE>>>(d_input, d_output, N, width);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 交换指针
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
        
        width *= 2;
        numPasses++;
    }
    
    // 如果奇数轮，结果在 d_output 中，需要拷贝回去
    if (numPasses % 2 == 1) {
        CHECK_CUDA(cudaMemcpy(d_input, d_output, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    }
    
    CHECK_CUDA(cudaFree(d_output));
}

// ====================== CPU 参考实现 ======================

static int compare_uint(const void* a, const void* b) {
    unsigned int ua = *(const unsigned int*)a;
    unsigned int ub = *(const unsigned int*)b;
    return (ua > ub) - (ua < ub);
}

void cpuSort(unsigned int* arr, int N) {
    qsort(arr, N, sizeof(unsigned int), compare_uint);
}
