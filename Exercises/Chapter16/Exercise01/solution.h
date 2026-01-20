// ============================================================================
// solution.h - 第十六章练习1: Pooling层前向传播
// ============================================================================
// 实现 Max Pooling 和 Average Pooling 的前向传播
// 对应书中 Section 16.2 的 Pooling 层描述
// ============================================================================

#ifndef SOLUTION_H
#define SOLUTION_H

#include <cuda_runtime.h>

// ============================================================================
// 数据布局说明
// ============================================================================
// 输入张量: [N, C, H, W] - NCHW 格式
//   N: Batch size
//   C: Channels (通道数/特征图数量)
//   H: Height (高度)
//   W: Width (宽度)
// 
// Pooling 窗口: K × K (假设正方形窗口，stride = K，非重叠)
// 输出张量: [N, C, H/K, W/K]
// ============================================================================

// ============================================================================
// CPU 串行实现 - 直接对应书中算法
// ============================================================================

/**
 * @brief Max Pooling 前向传播 - CPU 串行版本
 * 
 * @param input   输入张量 [N, C, H, W]
 * @param output  输出张量 [N, C, H/K, W/K]
 * @param N       Batch size
 * @param C       通道数
 * @param H       输入高度
 * @param W       输入宽度
 * @param K       Pooling 窗口大小
 */
void pooling_max_forward_cpu(const float* input, float* output,
                             int N, int C, int H, int W, int K);

/**
 * @brief Average Pooling 前向传播 - CPU 串行版本
 * 
 * @param input   输入张量 [N, C, H, W]
 * @param output  输出张量 [N, C, H/K, W/K]
 * @param N       Batch size
 * @param C       通道数
 * @param H       输入高度
 * @param W       输入宽度
 * @param K       Pooling 窗口大小
 */
void pooling_avg_forward_cpu(const float* input, float* output,
                             int N, int C, int H, int W, int K);

// ============================================================================
// GPU 并行实现
// ============================================================================

/**
 * @brief Max Pooling 前向传播 - GPU 并行版本
 * 
 * 每个输出元素由一个线程计算，遍历对应的 K×K 输入窗口
 * 
 * @param d_input   设备端输入张量
 * @param d_output  设备端输出张量
 * @param N, C, H, W, K  维度参数
 */
void pooling_max_forward_gpu(const float* d_input, float* d_output,
                             int N, int C, int H, int W, int K);

/**
 * @brief Average Pooling 前向传播 - GPU 并行版本
 * 
 * @param d_input   设备端输入张量
 * @param d_output  设备端输出张量
 * @param N, C, H, W, K  维度参数
 */
void pooling_avg_forward_gpu(const float* d_input, float* d_output,
                             int N, int C, int H, int W, int K);

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 计算输出维度
 */
inline int compute_pool_output_size(int input_size, int kernel_size) {
    return input_size / kernel_size;
}

/**
 * @brief 计算张量总元素数
 */
inline size_t compute_tensor_size(int N, int C, int H, int W) {
    return static_cast<size_t>(N) * C * H * W;
}

#endif // SOLUTION_H
