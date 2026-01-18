/**
 * 第十章：归约 - CUDA 实现
 * 
 * 参考：chapter-10/code/reduction_sum_2048.cu, reduction_sum.cu, reduction_max.cu
 * 
 * 本实现包含多种 kernel：
 * 1. simple_sum_reduction_kernel - 简单归约（图10.6）
 * 2. convergent_sum_reduction_kernel - 收敛归约（图10.9）
 * 3. convergent_sum_reduction_kernel_reversed - 反向收敛（练习3）
 * 4. shared_memory_sum_reduction_kernel - 共享内存归约（图10.11）
 * 5. segmented_sum_reduction_kernel - 分段归约
 * 6. coarsened_sum_reduction_kernel - 线程粗化归约（图10.15）
 * 7. coarsened_max_reduction_kernel - 最大值归约（练习4）
 */

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ====================== Kernels ======================

/**
 * 简单归约 Kernel（图10.6）
 * 问题：严重的控制分歧，因为 threadIdx.x % stride == 0 条件
 * 第5次迭代时，所有16个warp都有分歧
 */
__global__ void simple_sum_reduction_kernel(float* input, float* output) {
    unsigned int i = 2 * threadIdx.x;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

/**
 * 收敛归约 Kernel（图10.9）
 * 优化：只有前 stride 个线程活跃，消除分歧
 * 第5次迭代时，只有1个warp活跃，无分歧
 */
__global__ void convergent_sum_reduction_kernel(float* input, float* output) {
    unsigned int i = threadIdx.x;

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

/**
 * 反向收敛归约 Kernel（练习3）
 * 从右向左收敛，结果在最后一个元素
 */
__global__ void convergent_sum_reduction_kernel_reversed(float* input, float* output) {
    unsigned int i = threadIdx.x + blockDim.x;
    
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        // stride 迭代不变，但索引从右侧开始
        if (blockDim.x - threadIdx.x <= stride) {
            input[i] += input[i - stride];
        }
        __syncthreads();
    }
    
    // 结果在最后一个元素
    if (threadIdx.x == blockDim.x - 1) {
        *output = input[i];
    }
}

/**
 * 共享内存归约 Kernel（图10.11）
 * 使用共享内存，减少全局内存访问
 */
__global__ void shared_memory_sum_reduction_kernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;

    // 第一次迭代：从全局内存加载并相加
    input_s[t] = input[t] + input[t + BLOCK_DIM];

    // 后续迭代在共享内存中进行
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (t == 0) {
        *output = input_s[0];
    }
}

/**
 * 分段归约 Kernel
 * 支持任意长度，每个 Block 处理一段，最后用 atomicAdd 合并
 */
__global__ void segmented_sum_reduction_kernel(float* input, float* output, int length) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    // 边界检查
    float val1 = (i < length) ? input[i] : 0.0f;
    float val2 = (i + BLOCK_DIM < length) ? input[i + BLOCK_DIM] : 0.0f;
    input_s[t] = val1 + val2;

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

/**
 * 线程粗化归约 Kernel（图10.15）
 * 每个线程处理 COARSE_FACTOR*2 个元素
 * 支持任意长度
 */
__global__ void coarsened_sum_reduction_kernel(float* input, float* output, int length) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    // 每个线程先累加多个元素
    float sum = 0.0f;
    if (i < length) {
        sum = input[i];
        for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
            if (i + tile * BLOCK_DIM < length) {
                sum += input[i + tile * BLOCK_DIM];
            }
        }
    }
    input_s[t] = sum;

    // Block 内归约
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

/**
 * 线程粗化最大值归约 Kernel（练习4）
 */
__global__ void coarsened_max_reduction_kernel(float* input, float* output, int length) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    // 每个线程先求多个元素的最大值
    float max_val = -INFINITY;
    if (i < length) {
        max_val = input[i];
        for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
            if (i + tile * BLOCK_DIM < length) {
                max_val = fmaxf(max_val, input[i + tile * BLOCK_DIM]);
            }
        }
    }
    input_s[t] = max_val;

    // Block 内归约
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] = fmaxf(input_s[t], input_s[t + stride]);
        }
    }

    if (t == 0) {
        // 使用 atomicMax 需要整型，这里用 atomicExch + 比较
        // 注意：这不是完全原子的，但对于演示足够
        float old_val;
        do {
            old_val = *output;
            if (input_s[0] <= old_val) break;
        } while (atomicCAS((unsigned int*)output, 
                           __float_as_uint(old_val), 
                           __float_as_uint(input_s[0])) != __float_as_uint(old_val));
    }
}

// ====================== 主机接口 ======================

/**
 * CPU 顺序归约
 */
float reduction_sequential(float* data, int length) {
    double total = 0.0;
    for (int i = 0; i < length; i++) {
        total += data[i];
    }
    return (float)total;
}

/**
 * CPU 顺序最大值归约
 */
float max_reduction_sequential(float* data, int length) {
    float max_val = data[0];
    for (int i = 1; i < length; i++) {
        max_val = fmaxf(max_val, data[i]);
    }
    return max_val;
}

// 辅助函数：向上取整除法
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

/**
 * 简单归约（主机接口）
 * 限制：仅支持 2048 元素
 */
float reduction_simple(float* data, int length) {
    if (length != 2 * BLOCK_DIM) {
        printf("错误：simple 归约仅支持 %d 元素\n", 2 * BLOCK_DIM);
        return 0.0f;
    }

    float result;
    float *d_data, *d_result;

    CHECK_CUDA(cudaMalloc(&d_data, length * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    simple_sum_reduction_kernel<<<1, BLOCK_DIM>>>(d_data, d_result);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_result));

    return result;
}

/**
 * 收敛归约（主机接口）
 */
float reduction_convergent(float* data, int length) {
    if (length != 2 * BLOCK_DIM) {
        printf("错误：convergent 归约仅支持 %d 元素\n", 2 * BLOCK_DIM);
        return 0.0f;
    }

    float result;
    float *d_data, *d_result;

    CHECK_CUDA(cudaMalloc(&d_data, length * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    convergent_sum_reduction_kernel<<<1, BLOCK_DIM>>>(d_data, d_result);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_result));

    return result;
}

/**
 * 反向收敛归约（主机接口）
 */
float reduction_convergent_reversed(float* data, int length) {
    if (length != 2 * BLOCK_DIM) {
        printf("错误：reversed 归约仅支持 %d 元素\n", 2 * BLOCK_DIM);
        return 0.0f;
    }

    float result;
    float *d_data, *d_result;

    CHECK_CUDA(cudaMalloc(&d_data, length * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    convergent_sum_reduction_kernel_reversed<<<1, BLOCK_DIM>>>(d_data, d_result);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_result));

    return result;
}

/**
 * 共享内存归约（主机接口）
 */
float reduction_shared_memory(float* data, int length) {
    if (length != 2 * BLOCK_DIM) {
        printf("错误：shared_memory 归约仅支持 %d 元素\n", 2 * BLOCK_DIM);
        return 0.0f;
    }

    float result;
    float *d_data, *d_result;

    CHECK_CUDA(cudaMalloc(&d_data, length * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    shared_memory_sum_reduction_kernel<<<1, BLOCK_DIM>>>(d_data, d_result);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_result));

    return result;
}

/**
 * 分段归约（主机接口）
 * 支持任意长度
 */
float reduction_segmented(float* data, int length) {
    float result = 0.0f;
    float *d_data, *d_result;

    unsigned int numBlocks = cdiv(length, 2 * BLOCK_DIM);

    CHECK_CUDA(cudaMalloc(&d_data, length * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));

    segmented_sum_reduction_kernel<<<numBlocks, BLOCK_DIM>>>(d_data, d_result, length);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_result));

    return result;
}

/**
 * 线程粗化归约（主机接口）
 * 支持任意长度
 */
float reduction_coarsened(float* data, int length) {
    float result = 0.0f;
    float *d_data, *d_result;

    int elementsPerBlock = 2 * COARSE_FACTOR * BLOCK_DIM;
    unsigned int numBlocks = cdiv(length, elementsPerBlock);

    CHECK_CUDA(cudaMalloc(&d_data, length * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));

    coarsened_sum_reduction_kernel<<<numBlocks, BLOCK_DIM>>>(d_data, d_result, length);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_result));

    return result;
}

/**
 * 最大值归约（主机接口）
 */
float max_reduction_coarsened(float* data, int length) {
    float result = -INFINITY;
    float *d_data, *d_result;

    int elementsPerBlock = 2 * COARSE_FACTOR * BLOCK_DIM;
    unsigned int numBlocks = cdiv(length, elementsPerBlock);

    CHECK_CUDA(cudaMalloc(&d_data, length * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_result, &result, sizeof(float), cudaMemcpyHostToDevice));

    coarsened_max_reduction_kernel<<<numBlocks, BLOCK_DIM>>>(d_data, d_result, length);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_result));

    return result;
}
