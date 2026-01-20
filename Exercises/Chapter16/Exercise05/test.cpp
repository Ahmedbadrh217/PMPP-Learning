// ============================================================================
// test.cpp - 第十六章练习5: cuDNN 封装测试
// ============================================================================
// 测试 cuDNN 实现与 CPU 参考实现的一致性
// ============================================================================

#include "solution.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// 测试配置
// ============================================================================

const float EPSILON = 1e-3f;  // cuDNN 可能有略微不同的精度

void generate_random_data(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
}

bool compare_arrays(const float* a, const float* b, size_t size, float epsilon = EPSILON) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    bool passed = max_diff <= epsilon;
    printf("  最大差异: %.2e %s\n", max_diff, passed ? "✅" : "❌");
    return passed;
}

// ============================================================================
// Conv2D Forward 测试
// ============================================================================

bool test_conv2d_forward() {
    printf("\n=== cuDNN Conv2D Forward 测试 ===\n");
    
    int N = 2, C_in = 3, H = 8, W = 8, C_out = 4;
    int K = 3, pad = 1, stride = 1;
    int H_out = (H + 2*pad - K) / stride + 1;
    int W_out = (W + 2*pad - K) / stride + 1;
    
    printf("配置: N=%d, C_in=%d, H=%d, W=%d, C_out=%d, K=%d, pad=%d, stride=%d\n",
           N, C_in, H, W, C_out, K, pad, stride);
    printf("输出: H_out=%d, W_out=%d\n", H_out, W_out);
    
    size_t input_size = N * C_in * H * W;
    size_t weight_size = C_out * C_in * K * K;
    size_t bias_size = C_out;
    size_t output_size = N * C_out * H_out * W_out;
    
    float* h_input = new float[input_size];
    float* h_weights = new float[weight_size];
    float* h_bias = new float[bias_size];
    float* h_output_cpu = new float[output_size];
    float* h_output_cudnn = new float[output_size];
    
    generate_random_data(h_input, input_size);
    generate_random_data(h_weights, weight_size);
    generate_random_data(h_bias, bias_size);
    
    // CPU 参考实现
    conv2d_forward_cpu(h_input, h_weights, h_bias, h_output_cpu,
                       N, C_in, H, W, C_out, K, K, pad, pad, stride, stride);
    
    // cuDNN 实现
    int ret = conv2d_forward_cudnn(h_input, h_weights, h_bias, h_output_cudnn,
                                   N, C_in, H, W, C_out, K, K, pad, pad, stride, stride);
    
    bool passed = false;
    if (ret == 0) {
        passed = compare_arrays(h_output_cpu, h_output_cudnn, output_size);
    } else {
        printf("  cuDNN 调用失败 ❌\n");
    }
    
    delete[] h_input; delete[] h_weights; delete[] h_bias;
    delete[] h_output_cpu; delete[] h_output_cudnn;
    
    return passed;
}

// ============================================================================
// Conv2D Backward 测试
// ============================================================================

bool test_conv2d_backward() {
    printf("\n=== cuDNN Conv2D Backward 测试 ===\n");
    
    int N = 2, C_in = 3, H = 8, W = 8, C_out = 4;
    int K = 3, pad = 1, stride = 1;
    int H_out = (H + 2*pad - K) / stride + 1;
    int W_out = (W + 2*pad - K) / stride + 1;
    
    size_t input_size = N * C_in * H * W;
    size_t weight_size = C_out * C_in * K * K;
    size_t bias_size = C_out;
    size_t output_size = N * C_out * H_out * W_out;
    
    float* h_input = new float[input_size];
    float* h_weights = new float[weight_size];
    float* h_grad_output = new float[output_size];
    float* h_grad_input = new float[input_size];
    float* h_grad_weights = new float[weight_size];
    float* h_grad_bias = new float[bias_size];
    
    generate_random_data(h_input, input_size);
    generate_random_data(h_weights, weight_size);
    generate_random_data(h_grad_output, output_size);
    
    // cuDNN 反向传播
    int ret = conv2d_backward_cudnn(h_input, h_weights, h_grad_output,
                                    h_grad_input, h_grad_weights, h_grad_bias,
                                    N, C_in, H, W, C_out, K, K, pad, pad, stride, stride);
    
    bool passed = ret == 0;
    if (passed) {
        // 简单验证：检查梯度不为全0
        float sum_input = 0, sum_weights = 0, sum_bias = 0;
        for (size_t i = 0; i < input_size; i++) sum_input += std::fabs(h_grad_input[i]);
        for (size_t i = 0; i < weight_size; i++) sum_weights += std::fabs(h_grad_weights[i]);
        for (size_t i = 0; i < bias_size; i++) sum_bias += std::fabs(h_grad_bias[i]);
        
        bool non_zero = (sum_input > 0) && (sum_weights > 0) && (sum_bias > 0);
        printf("  梯度非零检查: %s\n", non_zero ? "✅" : "❌");
        printf("    |grad_input| = %.4f, |grad_weights| = %.4f, |grad_bias| = %.4f\n",
               sum_input, sum_weights, sum_bias);
        passed = non_zero;
    } else {
        printf("  cuDNN 调用失败 ❌\n");
    }
    
    delete[] h_input; delete[] h_weights; delete[] h_grad_output;
    delete[] h_grad_input; delete[] h_grad_weights; delete[] h_grad_bias;
    
    return passed;
}

// ============================================================================
// MaxPool2D Forward 测试
// ============================================================================

bool test_maxpool2d_forward() {
    printf("\n=== cuDNN MaxPool2D Forward 测试 ===\n");
    
    int N = 2, C = 3, H = 8, W = 8;
    int K = 2, stride = 2;
    int H_out = (H - K) / stride + 1;
    int W_out = (W - K) / stride + 1;
    
    printf("配置: N=%d, C=%d, H=%d, W=%d, K=%d, stride=%d\n", N, C, H, W, K, stride);
    printf("输出: H_out=%d, W_out=%d\n", H_out, W_out);
    
    size_t input_size = N * C * H * W;
    size_t output_size = N * C * H_out * W_out;
    
    float* h_input = new float[input_size];
    float* h_output_cpu = new float[output_size];
    float* h_output_cudnn = new float[output_size];
    
    generate_random_data(h_input, input_size);
    
    // CPU 参考实现
    maxpool2d_forward_cpu(h_input, h_output_cpu, N, C, H, W, K, K, stride, stride);
    
    // cuDNN 实现
    int ret = maxpool2d_forward_cudnn(h_input, h_output_cudnn, N, C, H, W, K, K, stride, stride);
    
    bool passed = false;
    if (ret == 0) {
        passed = compare_arrays(h_output_cpu, h_output_cudnn, output_size);
    } else {
        printf("  cuDNN 调用失败 ❌\n");
    }
    
    delete[] h_input; delete[] h_output_cpu; delete[] h_output_cudnn;
    
    return passed;
}

// ============================================================================
// MaxPool2D Backward 测试
// ============================================================================

bool test_maxpool2d_backward() {
    printf("\n=== cuDNN MaxPool2D Backward 测试 ===\n");
    
    int N = 2, C = 3, H = 8, W = 8;
    int K = 2, stride = 2;
    int H_out = (H - K) / stride + 1;
    int W_out = (W - K) / stride + 1;
    
    size_t input_size = N * C * H * W;
    size_t output_size = N * C * H_out * W_out;
    
    float* h_input = new float[input_size];
    float* h_output = new float[output_size];
    float* h_grad_output = new float[output_size];
    float* h_grad_input = new float[input_size];
    
    generate_random_data(h_input, input_size);
    generate_random_data(h_grad_output, output_size);
    
    // 先做 forward 获取 output
    maxpool2d_forward_cpu(h_input, h_output, N, C, H, W, K, K, stride, stride);
    
    // cuDNN backward
    int ret = maxpool2d_backward_cudnn(h_input, h_output, h_grad_output, h_grad_input,
                                       N, C, H, W, K, K, stride, stride);
    
    bool passed = ret == 0;
    if (passed) {
        // 验证梯度不为全0
        float sum = 0;
        for (size_t i = 0; i < input_size; i++) sum += std::fabs(h_grad_input[i]);
        bool non_zero = sum > 0;
        printf("  梯度非零检查: %s (sum = %.4f)\n", non_zero ? "✅" : "❌", sum);
        passed = non_zero;
    } else {
        printf("  cuDNN 调用失败 ❌\n");
    }
    
    delete[] h_input; delete[] h_output; delete[] h_grad_output; delete[] h_grad_input;
    
    return passed;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("================================================================\n");
    printf("  第十六章：cuDNN 封装实现测试\n");
    printf("  Conv2D & MaxPool2D - Forward & Backward\n");
    printf("================================================================\n");
    
    srand(42);
    
    // 初始化 cuDNN
    if (init_cudnn() != 0) {
        printf("cuDNN 初始化失败，退出测试\n");
        return 1;
    }
    
    bool all_passed = true;
    
    all_passed &= test_conv2d_forward();
    all_passed &= test_conv2d_backward();
    all_passed &= test_maxpool2d_forward();
    all_passed &= test_maxpool2d_backward();
    
    // 清理 cuDNN
    cleanup_cudnn();
    
    printf("\n================================================================\n");
    if (all_passed) {
        printf("  ✅ 所有测试通过！\n");
    } else {
        printf("  ❌ 部分测试失败\n");
    }
    printf("================================================================\n");
    
    return all_passed ? 0 : 1;
}
