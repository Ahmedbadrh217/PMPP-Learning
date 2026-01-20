// ============================================================================
// solution.h - 第十六章练习5: cuDNN 封装实现
// ============================================================================
// 使用 NVIDIA cuDNN 库实现 Conv2D 和 MaxPool2D 操作
// 对应参考仓库的 legacy_cudnn_wrapper.cu 功能
// ============================================================================

#ifndef SOLUTION_H
#define SOLUTION_H

#include <cuda_runtime.h>
#include <cudnn.h>

// ============================================================================
// cuDNN 句柄管理
// ============================================================================

/**
 * @brief 初始化 cuDNN 库
 * @return 0 成功，-1 失败
 */
int init_cudnn();

/**
 * @brief 清理 cuDNN 资源
 * @return 0 成功，-1 失败
 */
int cleanup_cudnn();

// ============================================================================
// Conv2D 操作 (使用 cuDNN)
// ============================================================================

/**
 * @brief Conv2D 前向传播 - cuDNN 实现
 * 
 * @param input     输入张量 [N, C_in, H, W]
 * @param weights   权重 [C_out, C_in, K_h, K_w]
 * @param bias      偏置 [C_out]
 * @param output    输出张量 [N, C_out, H_out, W_out]
 * @param batch_size, in_channels, height, width 输入维度
 * @param out_channels  输出通道数
 * @param kernel_h, kernel_w 卷积核大小
 * @param pad_h, pad_w, stride_h, stride_w 填充和步长
 * @return 0 成功，-1 失败
 */
int conv2d_forward_cudnn(float* input, float* weights, float* bias, float* output,
                         int batch_size, int in_channels, int height, int width,
                         int out_channels, int kernel_h, int kernel_w,
                         int pad_h, int pad_w, int stride_h, int stride_w);

/**
 * @brief Conv2D 反向传播 - cuDNN 实现
 * 
 * 计算输入梯度、权重梯度和偏置梯度
 * 
 * @param input       原始输入
 * @param weights     原始权重
 * @param grad_output 输出梯度
 * @param grad_input  输入梯度 (输出)
 * @param grad_weights 权重梯度 (输出)
 * @param grad_bias   偏置梯度 (输出)
 * @return 0 成功，-1 失败
 */
int conv2d_backward_cudnn(float* input, float* weights, float* grad_output,
                          float* grad_input, float* grad_weights, float* grad_bias,
                          int batch_size, int in_channels, int height, int width,
                          int out_channels, int kernel_h, int kernel_w,
                          int pad_h, int pad_w, int stride_h, int stride_w);

// ============================================================================
// MaxPool2D 操作 (使用 cuDNN)
// ============================================================================

/**
 * @brief MaxPool2D 前向传播 - cuDNN 实现
 * 
 * 注意: cuDNN 不直接返回最大值索引，如需索引请使用手写 kernel 版本
 * 
 * @param input   输入张量 [N, C, H, W]
 * @param output  输出张量 [N, C, H_out, W_out]
 * @param batch_size, channels, height, width 输入维度
 * @param kernel_h, kernel_w 池化窗口大小
 * @param stride_h, stride_w 步长
 * @return 0 成功，-1 失败
 */
int maxpool2d_forward_cudnn(float* input, float* output,
                            int batch_size, int channels, int height, int width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w);

/**
 * @brief MaxPool2D 反向传播 - cuDNN 实现
 * 
 * @param input       原始输入
 * @param output      前向输出
 * @param grad_output 输出梯度
 * @param grad_input  输入梯度 (输出)
 * @return 0 成功，-1 失败
 */
int maxpool2d_backward_cudnn(float* input, float* output, float* grad_output,
                             float* grad_input,
                             int batch_size, int channels, int height, int width,
                             int kernel_h, int kernel_w, int stride_h, int stride_w);

// ============================================================================
// CPU 参考实现 (用于验证)
// ============================================================================

void conv2d_forward_cpu(const float* input, const float* weights, const float* bias,
                        float* output,
                        int batch_size, int in_channels, int height, int width,
                        int out_channels, int kernel_h, int kernel_w,
                        int pad_h, int pad_w, int stride_h, int stride_w);

void maxpool2d_forward_cpu(const float* input, float* output,
                           int batch_size, int channels, int height, int width,
                           int kernel_h, int kernel_w, int stride_h, int stride_w);

// ============================================================================
// 辅助函数
// ============================================================================

inline int compute_conv_output_size(int input_size, int kernel_size, int padding, int stride) {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

inline int compute_pool_output_size(int input_size, int kernel_size, int stride) {
    return (input_size - kernel_size) / stride + 1;
}

#endif // SOLUTION_H
