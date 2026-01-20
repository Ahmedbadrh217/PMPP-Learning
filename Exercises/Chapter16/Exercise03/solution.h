// ============================================================================
// solution.h - 第十六章: CNN 完整层实现
// ============================================================================
// 包含 Conv2D 和 MaxPool2D 的前向和反向传播完整实现
// 对应参考仓库 conv2d.cu 的完整功能
// ============================================================================

#ifndef SOLUTION_H
#define SOLUTION_H

#include <cuda_runtime.h>

// ============================================================================
// 数据布局: [N, C, H, W] (NCHW 格式)
// ============================================================================

// ============================================================================
// Conv2D 前向传播
// ============================================================================

/**
 * @brief Conv2D 前向传播 - CPU 串行版本
 * 
 * @param input     输入张量 [N, C_in, H, W]
 * @param weights   权重 [C_out, C_in, K_h, K_w]
 * @param bias      偏置 [C_out]
 * @param output    输出张量 [N, C_out, H_out, W_out]
 * @param N         Batch size
 * @param C_in      输入通道数
 * @param H, W      输入高度和宽度
 * @param C_out     输出通道数
 * @param K_h, K_w  卷积核大小
 * @param pad_h, pad_w      填充
 * @param stride_h, stride_w 步长
 */
void conv2d_forward_cpu(const float* input, const float* weights, const float* bias,
                        float* output,
                        int N, int C_in, int H, int W, int C_out,
                        int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w);

void conv2d_forward_gpu(const float* d_input, const float* d_weights, const float* d_bias,
                        float* d_output,
                        int N, int C_in, int H, int W, int C_out,
                        int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w);

// ============================================================================
// Conv2D 反向传播
// ============================================================================

/**
 * @brief Conv2D 反向传播 - 计算输入梯度 dL/dX
 */
void conv2d_backward_input_cpu(const float* weights, const float* grad_output,
                               float* grad_input,
                               int N, int C_in, int H, int W, int C_out,
                               int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w);

void conv2d_backward_input_gpu(const float* d_weights, const float* d_grad_output,
                               float* d_grad_input,
                               int N, int C_in, int H, int W, int C_out,
                               int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w);

/**
 * @brief Conv2D 反向传播 - 计算权重梯度 dL/dW
 */
void conv2d_backward_weights_cpu(const float* input, const float* grad_output,
                                 float* grad_weights,
                                 int N, int C_in, int H, int W, int C_out,
                                 int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w);

void conv2d_backward_weights_gpu(const float* d_input, const float* d_grad_output,
                                 float* d_grad_weights,
                                 int N, int C_in, int H, int W, int C_out,
                                 int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w);

/**
 * @brief Conv2D 反向传播 - 计算偏置梯度 dL/db
 */
void conv2d_backward_bias_cpu(const float* grad_output, float* grad_bias,
                              int N, int C_out, int H_out, int W_out);

void conv2d_backward_bias_gpu(const float* d_grad_output, float* d_grad_bias,
                              int N, int C_out, int H_out, int W_out);

// ============================================================================
// MaxPool2D 前向传播（带索引记录，用于反向传播）
// ============================================================================

/**
 * @brief MaxPool2D 前向传播 - 记录最大值索引
 * 
 * @param input     输入张量 [N, C, H, W]
 * @param output    输出张量 [N, C, H_out, W_out]
 * @param indices   最大值索引 [N, C, H_out, W_out]
 * @param N, C, H, W       输入维度
 * @param K_h, K_w         池化窗口大小
 * @param stride_h, stride_w 步长
 */
void maxpool2d_forward_cpu(const float* input, float* output, int* indices,
                           int N, int C, int H, int W,
                           int K_h, int K_w, int stride_h, int stride_w);

void maxpool2d_forward_gpu(const float* d_input, float* d_output, int* d_indices,
                           int N, int C, int H, int W,
                           int K_h, int K_w, int stride_h, int stride_w);

// ============================================================================
// MaxPool2D 反向传播
// ============================================================================

/**
 * @brief MaxPool2D 反向传播 - 使用前向传播时记录的索引
 * 
 * @param grad_output 输出梯度 [N, C, H_out, W_out]
 * @param indices     前向传播时记录的索引 [N, C, H_out, W_out]
 * @param grad_input  输入梯度 [N, C, H, W]
 */
void maxpool2d_backward_cpu(const float* grad_output, const int* indices,
                            float* grad_input,
                            int N, int C, int H, int W,
                            int K_h, int K_w, int stride_h, int stride_w);

void maxpool2d_backward_gpu(const float* d_grad_output, const int* d_indices,
                            float* d_grad_input,
                            int N, int C, int H, int W,
                            int K_h, int K_w, int stride_h, int stride_w);

// ============================================================================
// 辅助函数
// ============================================================================

inline int compute_output_size(int input_size, int kernel_size, int padding, int stride) {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

#endif // SOLUTION_H
