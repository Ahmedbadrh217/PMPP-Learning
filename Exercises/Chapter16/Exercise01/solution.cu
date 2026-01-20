// ============================================================================
// solution.cu - 第十六章练习1: Pooling层前向传播实现
// ============================================================================
// 实现 Max Pooling 和 Average Pooling 的 CPU 串行和 GPU 并行版本
// 对应书中 Section 16.2 的 Pooling 层描述
// ============================================================================

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cfloat>
#include <cstring>

// ============================================================================
// GPU Kernel 定义
// ============================================================================

// 线程块大小
#define BLOCK_SIZE 256

/**
 * @brief Max Pooling CUDA kernel
 * 
 * 每个线程负责计算一个输出元素
 * 线程索引映射: idx -> (n, c, h_out, w_out)
 */
__global__ void pooling_max_kernel(const float* input, float* output,
                                   int N, int C, int H, int W, int K,
                                   int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * C * H_out * W_out;
    
    if (idx >= total_outputs) return;
    
    // 计算输出索引对应的 (n, c, h_out, w_out)
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (C * H_out * W_out);
    
    // 在 K×K 窗口中找最大值
    float max_val = -FLT_MAX;
    for (int kh = 0; kh < K; kh++) {
        for (int kw = 0; kw < K; kw++) {
            int h_in = h_out * K + kh;
            int w_in = w_out * K + kw;
            int input_idx = ((n * C + c) * H + h_in) * W + w_in;
            float val = input[input_idx];
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    
    output[idx] = max_val;
}

/**
 * @brief Average Pooling CUDA kernel
 */
__global__ void pooling_avg_kernel(const float* input, float* output,
                                   int N, int C, int H, int W, int K,
                                   int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * C * H_out * W_out;
    
    if (idx >= total_outputs) return;
    
    // 计算输出索引对应的 (n, c, h_out, w_out)
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (C * H_out * W_out);
    
    // 计算 K×K 窗口的平均值
    float sum = 0.0f;
    for (int kh = 0; kh < K; kh++) {
        for (int kw = 0; kw < K; kw++) {
            int h_in = h_out * K + kh;
            int w_in = w_out * K + kw;
            int input_idx = ((n * C + c) * H + h_in) * W + w_in;
            sum += input[input_idx];
        }
    }
    
    output[idx] = sum / (K * K);
}

// ============================================================================
// CPU 串行实现 - 直接对应书中算法
// ============================================================================

void pooling_max_forward_cpu(const float* input, float* output,
                             int N, int C, int H, int W, int K) {
    // 参数验证：确保输入尺寸能被池化窗口大小整除
    if (H % K != 0 || W % K != 0) {
        printf("警告: pooling_max_forward_cpu - H=%d 或 W=%d 不能被 K=%d 整除，可能导致边界问题\n", H, W, K);
    }
    int H_out = H / K;
    int W_out = W / K;
    
    // 遍历 batch 中的每个样本
    for (int n = 0; n < N; n++) {
        // 遍历每个通道/特征图
        for (int c = 0; c < C; c++) {
            // 遍历输出的每个位置
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
                    // 在 K×K 窗口中找最大值
                    float max_val = -FLT_MAX;
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int h_in = h_out * K + kh;
                            int w_in = w_out * K + kw;
                            int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                            float val = input[input_idx];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                    int output_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                    output[output_idx] = max_val;
                }
            }
        }
    }
}

void pooling_avg_forward_cpu(const float* input, float* output,
                             int N, int C, int H, int W, int K) {
    // 参数验证：确保输入尺寸能被池化窗口大小整除
    if (H % K != 0 || W % K != 0) {
        printf("警告: pooling_avg_forward_cpu - H=%d 或 W=%d 不能被 K=%d 整除，可能导致边界问题\n", H, W, K);
    }
    int H_out = H / K;
    int W_out = W / K;
    
    // 遍历 batch 中的每个样本
    for (int n = 0; n < N; n++) {
        // 遍历每个通道/特征图
        for (int c = 0; c < C; c++) {
            // 遍历输出的每个位置
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
                    // 计算 K×K 窗口的平均值
                    float sum = 0.0f;
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int h_in = h_out * K + kh;
                            int w_in = w_out * K + kw;
                            int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                            sum += input[input_idx];
                        }
                    }
                    int output_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                    output[output_idx] = sum / (K * K);
                }
            }
        }
    }
}

// ============================================================================
// GPU 并行实现
// ============================================================================

void pooling_max_forward_gpu(const float* d_input, float* d_output,
                             int N, int C, int H, int W, int K) {
    int H_out = H / K;
    int W_out = W / K;
    int total_outputs = N * C * H_out * W_out;
    
    // 计算 grid 和 block 维度
    int num_blocks = (total_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    pooling_max_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_input, d_output, N, C, H, W, K, H_out, W_out);
    
    CHECK_LAST_CUDA_ERROR();
}

void pooling_avg_forward_gpu(const float* d_input, float* d_output,
                             int N, int C, int H, int W, int K) {
    int H_out = H / K;
    int W_out = W / K;
    int total_outputs = N * C * H_out * W_out;
    
    // 计算 grid 和 block 维度
    int num_blocks = (total_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    pooling_avg_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_input, d_output, N, C, H, W, K, H_out, W_out);
    
    CHECK_LAST_CUDA_ERROR();
}
