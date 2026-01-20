// ============================================================================
// test.cpp - 第十六章练习3: Conv2D 反向传播测试程序
// ============================================================================
// 测试卷积层反向传播的正确性和性能
// ============================================================================

#include "solution.h"
#include "../../Common/timer.h"
#include "../../Common/utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// 测试配置
// ============================================================================

// 正确性测试参数
const int TEST_M = 4;      // 输出通道数
const int TEST_C = 3;      // 输入通道数
const int TEST_H = 28;     // 输入高度
const int TEST_W = 28;     // 输入宽度
const int TEST_K = 3;      // 卷积核大小

// 性能测试参数
const int PERF_M = 64;
const int PERF_C = 32;
const int PERF_H = 56;
const int PERF_W = 56;
const int PERF_K = 3;

// 精度容差
const float EPSILON = 1e-4f;

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 生成随机浮点数组
 */
void generate_random_data(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
}

/**
 * @brief 比较两个数组是否相等（允许误差）
 */
bool compare_arrays(const float* a, const float* b, size_t size, float epsilon = EPSILON) {
    float max_diff = 0.0f;
    size_t max_diff_idx = 0;
    
    for (size_t i = 0; i < size; i++) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    
    if (max_diff > epsilon) {
        printf("  最大差异位置 %zu: CPU=%.6f, GPU=%.6f, diff=%.6e\n",
               max_diff_idx, a[max_diff_idx], b[max_diff_idx], max_diff);
        return false;
    }
    
    printf("  最大差异: %.6e (阈值: %.6e)\n", max_diff, epsilon);
    return true;
}

// ============================================================================
// 正确性测试
// ============================================================================

bool test_conv2d_backward_correctness() {
    printf("1. Conv2D 反向传播 (输入梯度) 正确性测试...\n");
    printf("   配置: M=%d, C=%d, H=%d, W=%d, K=%d\n", 
           TEST_M, TEST_C, TEST_H, TEST_W, TEST_K);
    
    int H_out = TEST_H - TEST_K + 1;
    int W_out = TEST_W - TEST_K + 1;
    
    size_t grad_output_size = TEST_M * H_out * W_out;
    size_t weights_size = TEST_M * TEST_C * TEST_K * TEST_K;
    size_t grad_input_size = TEST_C * TEST_H * TEST_W;
    
    // 分配主机内存
    float* h_grad_output = new float[grad_output_size];
    float* h_weights = new float[weights_size];
    float* h_grad_input_cpu = new float[grad_input_size];
    float* h_grad_input_gpu = new float[grad_input_size];
    
    // 生成随机输入
    generate_random_data(h_grad_output, grad_output_size);
    generate_random_data(h_weights, weights_size);
    
    // CPU 计算
    conv2d_backward_input_cpu(h_grad_output, h_weights, h_grad_input_cpu,
                              TEST_M, TEST_C, TEST_H, TEST_W, TEST_K);
    
    // GPU 计算
    float *d_grad_output, *d_weights, *d_grad_input;
    CHECK_CUDA(cudaMalloc(&d_grad_output, grad_output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights, weights_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_input, grad_input_size * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_grad_output, h_grad_output, grad_output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights, weights_size * sizeof(float), cudaMemcpyHostToDevice));
    
    conv2d_backward_input_gpu(d_grad_output, d_weights, d_grad_input,
                              TEST_M, TEST_C, TEST_H, TEST_W, TEST_K);
    
    CHECK_CUDA(cudaMemcpy(h_grad_input_gpu, d_grad_input, grad_input_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 比较结果
    bool passed = compare_arrays(h_grad_input_cpu, h_grad_input_gpu, grad_input_size);
    
    // 清理
    delete[] h_grad_output;
    delete[] h_weights;
    delete[] h_grad_input_cpu;
    delete[] h_grad_input_gpu;
    cudaFree(d_grad_output);
    cudaFree(d_weights);
    cudaFree(d_grad_input);
    
    printf("   %s\n", passed ? "✅ 通过" : "❌ 失败");
    return passed;
}

// ============================================================================
// 多配置测试
// ============================================================================

void test_multiple_configurations() {
    printf("\n=== 多配置测试 ===\n\n");
    
    struct TestConfig {
        int M, C, H, W, K;
        const char* name;
    };
    
    TestConfig configs[] = {
        {2, 3, 5, 5, 3, "小型正方形"},
        {4, 3, 10, 10, 3, "中型正方形"},
        {2, 1, 7, 5, 3, "非正方形"},
        {3, 2, 8, 8, 5, "大卷积核"},
        {1, 3, 6, 6, 2, "小卷积核"}
    };
    
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    int passed = 0;
    
    for (int i = 0; i < num_configs; i++) {
        TestConfig& cfg = configs[i];
        printf("%d. %s (M=%d, C=%d, H=%d, W=%d, K=%d)... ",
               i + 1, cfg.name, cfg.M, cfg.C, cfg.H, cfg.W, cfg.K);
        
        int H_out = cfg.H - cfg.K + 1;
        int W_out = cfg.W - cfg.K + 1;
        
        size_t grad_output_size = cfg.M * H_out * W_out;
        size_t weights_size = cfg.M * cfg.C * cfg.K * cfg.K;
        size_t grad_input_size = cfg.C * cfg.H * cfg.W;
        
        float* h_grad_output = new float[grad_output_size];
        float* h_weights = new float[weights_size];
        float* h_grad_input_cpu = new float[grad_input_size];
        float* h_grad_input_gpu = new float[grad_input_size];
        
        generate_random_data(h_grad_output, grad_output_size);
        generate_random_data(h_weights, weights_size);
        
        conv2d_backward_input_cpu(h_grad_output, h_weights, h_grad_input_cpu,
                                  cfg.M, cfg.C, cfg.H, cfg.W, cfg.K);
        
        float *d_grad_output, *d_weights, *d_grad_input;
        cudaMalloc(&d_grad_output, grad_output_size * sizeof(float));
        cudaMalloc(&d_weights, weights_size * sizeof(float));
        cudaMalloc(&d_grad_input, grad_input_size * sizeof(float));
        
        cudaMemcpy(d_grad_output, h_grad_output, grad_output_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, h_weights, weights_size * sizeof(float), cudaMemcpyHostToDevice);
        
        conv2d_backward_input_gpu(d_grad_output, d_weights, d_grad_input,
                                  cfg.M, cfg.C, cfg.H, cfg.W, cfg.K);
        
        cudaMemcpy(h_grad_input_gpu, d_grad_input, grad_input_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        float max_diff = 0.0f;
        for (size_t j = 0; j < grad_input_size; j++) {
            float diff = std::fabs(h_grad_input_cpu[j] - h_grad_input_gpu[j]);
            if (diff > max_diff) max_diff = diff;
        }
        
        bool test_passed = max_diff < EPSILON;
        printf("%s (max diff: %.2e)\n", test_passed ? "✅" : "❌", max_diff);
        if (test_passed) passed++;
        
        delete[] h_grad_output;
        delete[] h_weights;
        delete[] h_grad_input_cpu;
        delete[] h_grad_input_gpu;
        cudaFree(d_grad_output);
        cudaFree(d_weights);
        cudaFree(d_grad_input);
    }
    
    printf("\n通过 %d/%d 个配置测试\n", passed, num_configs);
}

// ============================================================================
// 性能测试
// ============================================================================

void test_performance() {
    printf("\n=== 性能基准测试 ===\n\n");
    printf("配置: M=%d, C=%d, H=%d, W=%d, K=%d\n",
           PERF_M, PERF_C, PERF_H, PERF_W, PERF_K);
    
    int H_out = PERF_H - PERF_K + 1;
    int W_out = PERF_W - PERF_K + 1;
    
    size_t grad_output_size = PERF_M * H_out * W_out;
    size_t weights_size = PERF_M * PERF_C * PERF_K * PERF_K;
    size_t grad_input_size = PERF_C * PERF_H * PERF_W;
    
    // 分配内存
    float* h_grad_output = new float[grad_output_size];
    float* h_weights = new float[weights_size];
    float* h_grad_input = new float[grad_input_size];
    
    generate_random_data(h_grad_output, grad_output_size);
    generate_random_data(h_weights, weights_size);
    
    float *d_grad_output, *d_weights, *d_grad_input;
    CHECK_CUDA(cudaMalloc(&d_grad_output, grad_output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights, weights_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_input, grad_input_size * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_grad_output, h_grad_output, grad_output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights, weights_size * sizeof(float), cudaMemcpyHostToDevice));
    
    const int NUM_RUNS = 100;
    Timer cpu_timer;
    CudaTimer gpu_timer;
    
    // CPU 性能
    cpu_timer.start();
    for (int i = 0; i < NUM_RUNS; i++) {
        conv2d_backward_input_cpu(h_grad_output, h_weights, h_grad_input,
                                  PERF_M, PERF_C, PERF_H, PERF_W, PERF_K);
    }
    cpu_timer.stop();
    float cpu_time = cpu_timer.elapsed_ms() / NUM_RUNS;
    
    // GPU 性能
    gpu_timer.start();
    for (int i = 0; i < NUM_RUNS; i++) {
        conv2d_backward_input_gpu(d_grad_output, d_weights, d_grad_input,
                                  PERF_M, PERF_C, PERF_H, PERF_W, PERF_K);
    }
    gpu_timer.stop();
    float gpu_time = gpu_timer.elapsed_ms() / NUM_RUNS;
    
    printf("\nConv2D Backward (Input Gradient):\n");
    printf("  CPU: %.3f ms\n", cpu_time);
    printf("  GPU: %.3f ms (%.1fx 加速)\n", gpu_time, cpu_time / gpu_time);
    
    // 清理
    delete[] h_grad_output;
    delete[] h_weights;
    delete[] h_grad_input;
    cudaFree(d_grad_output);
    cudaFree(d_weights);
    cudaFree(d_grad_input);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("================================================================\n");
    printf("  第十六章练习3：Conv2D 反向传播\n");
    printf("  Convolutional Layer Backward Pass - Input Gradient\n");
    printf("================================================================\n\n");
    
    srand(42);  // 固定随机种子以便复现
    
    printf("=== 正确性验证 ===\n\n");
    
    bool passed = test_conv2d_backward_correctness();
    
    test_multiple_configurations();
    
    if (passed) {
        test_performance();
    }
    
    printf("\n================================================================\n");
    printf("  测试完成\n");
    printf("================================================================\n");
    
    return passed ? 0 : 1;
}
