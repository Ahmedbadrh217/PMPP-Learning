// ============================================================================
// test.cpp - 第十六章: CNN 完整层测试
// ============================================================================
// 测试 Conv2D 和 MaxPool2D 的前向/反向传播正确性
// ============================================================================

#include "solution.h"
#include "../../../Common/timer.h"
#include "../../../Common/utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// 测试配置
// ============================================================================

const float EPSILON = 1e-4f;

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
    printf("\n=== Conv2D Forward 测试 ===\n");
    
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
    float* h_output_gpu = new float[output_size];
    
    generate_random_data(h_input, input_size);
    generate_random_data(h_weights, weight_size);
    generate_random_data(h_bias, bias_size);
    
    conv2d_forward_cpu(h_input, h_weights, h_bias, h_output_cpu,
                       N, C_in, H, W, C_out, K, K, pad, pad, stride, stride);
    
    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_weights, weight_size * sizeof(float));
    cudaMalloc(&d_bias, bias_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bias_size * sizeof(float), cudaMemcpyHostToDevice);
    
    conv2d_forward_gpu(d_input, d_weights, d_bias, d_output,
                       N, C_in, H, W, C_out, K, K, pad, pad, stride, stride);
    
    cudaMemcpy(h_output_gpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool passed = compare_arrays(h_output_cpu, h_output_gpu, output_size);
    
    delete[] h_input; delete[] h_weights; delete[] h_bias;
    delete[] h_output_cpu; delete[] h_output_gpu;
    cudaFree(d_input); cudaFree(d_weights); cudaFree(d_bias); cudaFree(d_output);
    
    return passed;
}

// ============================================================================
// Conv2D Backward Input 测试
// ============================================================================

bool test_conv2d_backward_input() {
    printf("\n=== Conv2D Backward Input 测试 ===\n");
    
    int N = 2, C_in = 3, H = 8, W = 8, C_out = 4;
    int K = 3, pad = 1, stride = 1;
    int H_out = (H + 2*pad - K) / stride + 1;
    int W_out = (W + 2*pad - K) / stride + 1;
    
    size_t weight_size = C_out * C_in * K * K;
    size_t grad_output_size = N * C_out * H_out * W_out;
    size_t grad_input_size = N * C_in * H * W;
    
    float* h_weights = new float[weight_size];
    float* h_grad_output = new float[grad_output_size];
    float* h_grad_input_cpu = new float[grad_input_size];
    float* h_grad_input_gpu = new float[grad_input_size];
    
    generate_random_data(h_weights, weight_size);
    generate_random_data(h_grad_output, grad_output_size);
    
    conv2d_backward_input_cpu(h_weights, h_grad_output, h_grad_input_cpu,
                              N, C_in, H, W, C_out, K, K, pad, pad, stride, stride);
    
    float *d_weights, *d_grad_output, *d_grad_input;
    cudaMalloc(&d_weights, weight_size * sizeof(float));
    cudaMalloc(&d_grad_output, grad_output_size * sizeof(float));
    cudaMalloc(&d_grad_input, grad_input_size * sizeof(float));
    
    cudaMemcpy(d_weights, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, h_grad_output, grad_output_size * sizeof(float), cudaMemcpyHostToDevice);
    
    conv2d_backward_input_gpu(d_weights, d_grad_output, d_grad_input,
                              N, C_in, H, W, C_out, K, K, pad, pad, stride, stride);
    
    cudaMemcpy(h_grad_input_gpu, d_grad_input, grad_input_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool passed = compare_arrays(h_grad_input_cpu, h_grad_input_gpu, grad_input_size);
    
    delete[] h_weights; delete[] h_grad_output;
    delete[] h_grad_input_cpu; delete[] h_grad_input_gpu;
    cudaFree(d_weights); cudaFree(d_grad_output); cudaFree(d_grad_input);
    
    return passed;
}

// ============================================================================
// Conv2D Backward Weights 测试
// ============================================================================

bool test_conv2d_backward_weights() {
    printf("\n=== Conv2D Backward Weights 测试 ===\n");
    
    int N = 2, C_in = 3, H = 8, W = 8, C_out = 4;
    int K = 3, pad = 1, stride = 1;
    int H_out = (H + 2*pad - K) / stride + 1;
    int W_out = (W + 2*pad - K) / stride + 1;
    
    size_t input_size = N * C_in * H * W;
    size_t grad_output_size = N * C_out * H_out * W_out;
    size_t grad_weight_size = C_out * C_in * K * K;
    
    float* h_input = new float[input_size];
    float* h_grad_output = new float[grad_output_size];
    float* h_grad_weights_cpu = new float[grad_weight_size];
    float* h_grad_weights_gpu = new float[grad_weight_size];
    
    generate_random_data(h_input, input_size);
    generate_random_data(h_grad_output, grad_output_size);
    
    conv2d_backward_weights_cpu(h_input, h_grad_output, h_grad_weights_cpu,
                                N, C_in, H, W, C_out, K, K, pad, pad, stride, stride);
    
    float *d_input, *d_grad_output, *d_grad_weights;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_grad_output, grad_output_size * sizeof(float));
    cudaMalloc(&d_grad_weights, grad_weight_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, h_grad_output, grad_output_size * sizeof(float), cudaMemcpyHostToDevice);
    
    conv2d_backward_weights_gpu(d_input, d_grad_output, d_grad_weights,
                                N, C_in, H, W, C_out, K, K, pad, pad, stride, stride);
    
    cudaMemcpy(h_grad_weights_gpu, d_grad_weights, grad_weight_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool passed = compare_arrays(h_grad_weights_cpu, h_grad_weights_gpu, grad_weight_size);
    
    delete[] h_input; delete[] h_grad_output;
    delete[] h_grad_weights_cpu; delete[] h_grad_weights_gpu;
    cudaFree(d_input); cudaFree(d_grad_output); cudaFree(d_grad_weights);
    
    return passed;
}

// ============================================================================
// Conv2D Backward Bias 测试
// ============================================================================

bool test_conv2d_backward_bias() {
    printf("\n=== Conv2D Backward Bias 测试 ===\n");
    
    int N = 2, C_out = 4, H_out = 8, W_out = 8;
    
    size_t grad_output_size = N * C_out * H_out * W_out;
    size_t grad_bias_size = C_out;
    
    float* h_grad_output = new float[grad_output_size];
    float* h_grad_bias_cpu = new float[grad_bias_size];
    float* h_grad_bias_gpu = new float[grad_bias_size];
    
    generate_random_data(h_grad_output, grad_output_size);
    
    conv2d_backward_bias_cpu(h_grad_output, h_grad_bias_cpu, N, C_out, H_out, W_out);
    
    float *d_grad_output, *d_grad_bias;
    cudaMalloc(&d_grad_output, grad_output_size * sizeof(float));
    cudaMalloc(&d_grad_bias, grad_bias_size * sizeof(float));
    
    cudaMemcpy(d_grad_output, h_grad_output, grad_output_size * sizeof(float), cudaMemcpyHostToDevice);
    
    conv2d_backward_bias_gpu(d_grad_output, d_grad_bias, N, C_out, H_out, W_out);
    
    cudaMemcpy(h_grad_bias_gpu, d_grad_bias, grad_bias_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool passed = compare_arrays(h_grad_bias_cpu, h_grad_bias_gpu, grad_bias_size);
    
    delete[] h_grad_output; delete[] h_grad_bias_cpu; delete[] h_grad_bias_gpu;
    cudaFree(d_grad_output); cudaFree(d_grad_bias);
    
    return passed;
}

// ============================================================================
// MaxPool2D Forward 测试
// ============================================================================

bool test_maxpool2d_forward() {
    printf("\n=== MaxPool2D Forward 测试 ===\n");
    
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
    float* h_output_gpu = new float[output_size];
    int* h_indices_cpu = new int[output_size];
    int* h_indices_gpu = new int[output_size];
    
    generate_random_data(h_input, input_size);
    
    maxpool2d_forward_cpu(h_input, h_output_cpu, h_indices_cpu,
                          N, C, H, W, K, K, stride, stride);
    
    float *d_input, *d_output;
    int *d_indices;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_indices, output_size * sizeof(int));
    
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    
    maxpool2d_forward_gpu(d_input, d_output, d_indices,
                          N, C, H, W, K, K, stride, stride);
    
    cudaMemcpy(h_output_gpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices_gpu, d_indices, output_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    bool passed = compare_arrays(h_output_cpu, h_output_gpu, output_size);
    
    // 验证索引
    int idx_match = 0;
    for (size_t i = 0; i < output_size; i++) {
        if (h_indices_cpu[i] == h_indices_gpu[i]) idx_match++;
    }
    printf("  索引匹配: %d/%zu\n", idx_match, output_size);
    
    delete[] h_input; delete[] h_output_cpu; delete[] h_output_gpu;
    delete[] h_indices_cpu; delete[] h_indices_gpu;
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_indices);
    
    return passed && (idx_match == (int)output_size);
}

// ============================================================================
// MaxPool2D Backward 测试
// ============================================================================

bool test_maxpool2d_backward() {
    printf("\n=== MaxPool2D Backward 测试 ===\n");
    
    int N = 2, C = 3, H = 8, W = 8;
    int K = 2, stride = 2;
    int H_out = (H - K) / stride + 1;
    int W_out = (W - K) / stride + 1;
    
    size_t input_size = N * C * H * W;
    size_t output_size = N * C * H_out * W_out;
    
    float* h_input = new float[input_size];
    float* h_output = new float[output_size];
    int* h_indices = new int[output_size];
    float* h_grad_output = new float[output_size];
    float* h_grad_input_cpu = new float[input_size];
    float* h_grad_input_gpu = new float[input_size];
    
    generate_random_data(h_input, input_size);
    generate_random_data(h_grad_output, output_size);
    
    // 先做 forward 获取 indices
    maxpool2d_forward_cpu(h_input, h_output, h_indices, N, C, H, W, K, K, stride, stride);
    
    // Backward
    maxpool2d_backward_cpu(h_grad_output, h_indices, h_grad_input_cpu,
                           N, C, H, W, K, K, stride, stride);
    
    float *d_grad_output, *d_grad_input;
    int *d_indices;
    cudaMalloc(&d_grad_output, output_size * sizeof(float));
    cudaMalloc(&d_indices, output_size * sizeof(int));
    cudaMalloc(&d_grad_input, input_size * sizeof(float));
    
    cudaMemcpy(d_grad_output, h_grad_output, output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, output_size * sizeof(int), cudaMemcpyHostToDevice);
    
    maxpool2d_backward_gpu(d_grad_output, d_indices, d_grad_input,
                           N, C, H, W, K, K, stride, stride);
    
    cudaMemcpy(h_grad_input_gpu, d_grad_input, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool passed = compare_arrays(h_grad_input_cpu, h_grad_input_gpu, input_size);
    
    delete[] h_input; delete[] h_output; delete[] h_indices;
    delete[] h_grad_output; delete[] h_grad_input_cpu; delete[] h_grad_input_gpu;
    cudaFree(d_grad_output); cudaFree(d_indices); cudaFree(d_grad_input);
    
    return passed;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("================================================================\n");
    printf("  第十六章：CNN 完整层实现\n");
    printf("  Conv2D & MaxPool2D - Forward & Backward\n");
    printf("================================================================\n");
    
    srand(42);
    
    bool all_passed = true;
    
    all_passed &= test_conv2d_forward();
    all_passed &= test_conv2d_backward_input();
    all_passed &= test_conv2d_backward_weights();
    all_passed &= test_conv2d_backward_bias();
    all_passed &= test_maxpool2d_forward();
    all_passed &= test_maxpool2d_backward();
    
    printf("\n================================================================\n");
    if (all_passed) {
        printf("  ✅ 所有测试通过！\n");
    } else {
        printf("  ❌ 部分测试失败\n");
    }
    printf("================================================================\n");
    
    return all_passed ? 0 : 1;
}
