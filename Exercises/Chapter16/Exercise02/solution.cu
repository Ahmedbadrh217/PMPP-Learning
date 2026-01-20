// ============================================================================
// solution.cu - 第十六章练习2: Conv2D 反向传播实现（简版）
// ============================================================================
// 实现卷积层反向传播中的输入梯度计算（无 padding/stride）
// 对应书中 Section 16.2 的 CNN 反向传播描述
// ============================================================================

#include "solution.h"
#include "../../Common/utils.cuh"
#include <cstring>

// ============================================================================
// GPU Kernel 定义
// ============================================================================

#define BLOCK_SIZE 256

/**
 * @brief Conv2D 反向传播 kernel - 计算输入梯度
 * 
 * 每个线程负责计算一个输入梯度元素 dL/dX[c, h_in, w_in]
 * 
 * 原理：
 *   dL/dX[c, h_in, w_in] = Σ_m Σ_(h_out, w_out) dL/dY[m, h_out, w_out] * W[m, c, kh, kw]
 *   其中 h_in = h_out + kh, w_in = w_out + kw
 */
__global__ void conv2d_backward_input_kernel(const float* grad_output, const float* weights,
                                              float* grad_input,
                                              int M, int C, int H_in, int W_in, int K,
                                              int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = C * H_in * W_in;
    
    if (idx >= total_inputs) return;
    
    // 计算输入索引对应的 (c, h_in, w_in)
    int w_in = idx % W_in;
    int h_in = (idx / W_in) % H_in;
    int c = idx / (H_in * W_in);
    
    float grad_val = 0.0f;
    
    // 遍历所有对此输入位置有贡献的输出位置
    for (int m = 0; m < M; m++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                // 计算对应的输出位置
                int h_out = h_in - kh;
                int w_out = w_in - kw;
                
                // 检查输出位置是否有效
                if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                    int grad_output_idx = m * H_out * W_out + h_out * W_out + w_out;
                    int weight_idx = ((m * C + c) * K + kh) * K + kw;
                    
                    grad_val += grad_output[grad_output_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    grad_input[idx] = grad_val;
}

// ============================================================================
// CPU 串行实现 - 对应书中算法
// ============================================================================

void conv2d_backward_input_cpu(const float* grad_output, const float* weights,
                               float* grad_input,
                               int M, int C, int H_in, int W_in, int K) {
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;
    
    // 初始化梯度为0
    memset(grad_input, 0, C * H_in * W_in * sizeof(float));
    
    // 遍历所有输出位置，将梯度反向传播到输入
    for (int m = 0; m < M; m++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                // 获取输出梯度
                float grad = grad_output[m * H_out * W_out + h_out * W_out + w_out];
                
                // 传播到对应的输入位置
                for (int c = 0; c < C; c++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int h_in = h_out + kh;
                            int w_in = w_out + kw;
                            grad_input[c * H_in * W_in + h_in * W_in + w_in] +=
                                grad * weights[((m * C + c) * K + kh) * K + kw];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// GPU 并行实现
// ============================================================================

void conv2d_backward_input_gpu(const float* d_grad_output, const float* d_weights,
                               float* d_grad_input,
                               int M, int C, int H_in, int W_in, int K) {
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;
    int total_inputs = C * H_in * W_in;
    
    // 计算 grid 和 block 维度
    int num_blocks = (total_inputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    conv2d_backward_input_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_grad_output, d_weights, d_grad_input,
        M, C, H_in, W_in, K, H_out, W_out);
    
    CHECK_LAST_CUDA_ERROR();
}
