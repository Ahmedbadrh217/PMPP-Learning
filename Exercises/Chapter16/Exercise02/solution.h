// ============================================================================
// solution.h - 第十六章练习2: Conv2D 反向传播（简版）
// ============================================================================
// 实现卷积层反向传播中的输入梯度计算（无 padding/stride）
// 对应书中 Section 16.2 的 CNN 反向传播描述
// ============================================================================

#ifndef SOLUTION_H
#define SOLUTION_H

#include <cuda_runtime.h>

// ============================================================================
// 数据布局说明
// ============================================================================
// 输入张量 X: [C, H_in, W_in] - 单个样本的输入特征图
// 权重张量 W: [M, C, K, K] - M个输出通道，C个输入通道，K×K卷积核
// 输出梯度 dL/dY: [M, H_out, W_out]
// 输入梯度 dL/dX: [C, H_in, W_in]
// 
// 卷积参数: stride=1, padding=0 (valid convolution)
// H_out = H_in - K + 1
// W_out = W_in - K + 1
// ============================================================================

// ============================================================================
// CPU 串行实现
// ============================================================================

/**
 * @brief Conv2D 反向传播 - 计算输入梯度 dL/dX (CPU 串行版本)
 * 
 * 公式: dL/dX = conv(dL/dY, flip(W))
 * 实现时直接累加: dL/dX[c,h,w] += dL/dY[m,h_out,w_out] * W[m,c,kh,kw]
 * 
 * @param grad_output 输出梯度 dL/dY [M, H_out, W_out]
 * @param weights     权重 W [M, C, K, K]
 * @param grad_input  输入梯度 dL/dX [C, H_in, W_in] (输出参数)
 * @param M           输出通道数
 * @param C           输入通道数
 * @param H_in        输入高度
 * @param W_in        输入宽度
 * @param K           卷积核大小
 */
void conv2d_backward_input_cpu(const float* grad_output, const float* weights,
                               float* grad_input,
                               int M, int C, int H_in, int W_in, int K);

// ============================================================================
// GPU 并行实现
// ============================================================================

/**
 * @brief Conv2D 反向传播 - 计算输入梯度 dL/dX (GPU 并行版本)
 * 
 * 每个输出元素由一个线程计算
 * 
 * @param d_grad_output 设备端输出梯度
 * @param d_weights     设备端权重
 * @param d_grad_input  设备端输入梯度 (输出参数)
 * @param M, C, H_in, W_in, K 维度参数
 */
void conv2d_backward_input_gpu(const float* d_grad_output, const float* d_weights,
                               float* d_grad_input,
                               int M, int C, int H_in, int W_in, int K);

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 计算卷积输出维度 (valid convolution, stride=1)
 */
inline int compute_conv_output_size(int input_size, int kernel_size) {
    return input_size - kernel_size + 1;
}

#endif // SOLUTION_H
