// ============================================================================
// test.cpp - 第十六章练习1: Pooling层测试程序
// ============================================================================
// 测试 Max Pooling 和 Average Pooling 的正确性和性能
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
const int TEST_N = 2;      // Batch size
const int TEST_C = 16;     // Channels
const int TEST_H = 32;     // Height
const int TEST_W = 32;     // Width
const int TEST_K = 2;      // Pooling kernel size

// 性能测试参数
const int PERF_N = 8;
const int PERF_C = 64;
const int PERF_H = 128;
const int PERF_W = 128;
const int PERF_K = 2;

// 精度容差
const float EPSILON = 1e-5f;

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
    for (size_t i = 0; i < size; i++) {
        if (std::fabs(a[i] - b[i]) > epsilon) {
            printf("  差异位置 %zu: CPU=%.6f, GPU=%.6f, diff=%.6e\n",
                   i, a[i], b[i], std::fabs(a[i] - b[i]));
            return false;
        }
    }
    return true;
}

// ============================================================================
// 正确性测试
// ============================================================================

bool test_max_pooling_correctness() {
    printf("1. Max Pooling 正确性测试...\n");
    
    int H_out = TEST_H / TEST_K;
    int W_out = TEST_W / TEST_K;
    size_t input_size = TEST_N * TEST_C * TEST_H * TEST_W;
    size_t output_size = TEST_N * TEST_C * H_out * W_out;
    
    // 分配主机内存
    float* h_input = new float[input_size];
    float* h_output_cpu = new float[output_size];
    float* h_output_gpu = new float[output_size];
    
    // 生成随机输入
    generate_random_data(h_input, input_size);
    
    // CPU 计算
    pooling_max_forward_cpu(h_input, h_output_cpu, TEST_N, TEST_C, TEST_H, TEST_W, TEST_K);
    
    // GPU 计算
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    pooling_max_forward_gpu(d_input, d_output, TEST_N, TEST_C, TEST_H, TEST_W, TEST_K);
    
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 比较结果
    bool passed = compare_arrays(h_output_cpu, h_output_gpu, output_size);
    
    // 清理
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("   %s\n", passed ? "✅ 通过" : "❌ 失败");
    return passed;
}

bool test_avg_pooling_correctness() {
    printf("2. Average Pooling 正确性测试...\n");
    
    int H_out = TEST_H / TEST_K;
    int W_out = TEST_W / TEST_K;
    size_t input_size = TEST_N * TEST_C * TEST_H * TEST_W;
    size_t output_size = TEST_N * TEST_C * H_out * W_out;
    
    // 分配主机内存
    float* h_input = new float[input_size];
    float* h_output_cpu = new float[output_size];
    float* h_output_gpu = new float[output_size];
    
    // 生成随机输入
    generate_random_data(h_input, input_size);
    
    // CPU 计算
    pooling_avg_forward_cpu(h_input, h_output_cpu, TEST_N, TEST_C, TEST_H, TEST_W, TEST_K);
    
    // GPU 计算
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    pooling_avg_forward_gpu(d_input, d_output, TEST_N, TEST_C, TEST_H, TEST_W, TEST_K);
    
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 比较结果
    bool passed = compare_arrays(h_output_cpu, h_output_gpu, output_size);
    
    // 清理
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("   %s\n", passed ? "✅ 通过" : "❌ 失败");
    return passed;
}

// ============================================================================
// 性能测试
// ============================================================================

void test_performance() {
    printf("\n=== 性能基准测试 ===\n\n");
    printf("输入大小: [%d, %d, %d, %d], Pooling K=%d\n",
           PERF_N, PERF_C, PERF_H, PERF_W, PERF_K);
    
    int H_out = PERF_H / PERF_K;
    int W_out = PERF_W / PERF_K;
    size_t input_size = PERF_N * PERF_C * PERF_H * PERF_W;
    size_t output_size = PERF_N * PERF_C * H_out * W_out;
    
    // 分配内存
    float* h_input = new float[input_size];
    float* h_output = new float[output_size];
    generate_random_data(h_input, input_size);
    
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    const int NUM_RUNS = 100;
    Timer cpu_timer;
    CudaTimer gpu_timer;
    
    // Max Pooling CPU
    cpu_timer.start();
    for (int i = 0; i < NUM_RUNS; i++) {
        pooling_max_forward_cpu(h_input, h_output, PERF_N, PERF_C, PERF_H, PERF_W, PERF_K);
    }
    cpu_timer.stop();
    float max_cpu_time = cpu_timer.elapsed_ms() / NUM_RUNS;
    
    // Max Pooling GPU
    gpu_timer.start();
    for (int i = 0; i < NUM_RUNS; i++) {
        pooling_max_forward_gpu(d_input, d_output, PERF_N, PERF_C, PERF_H, PERF_W, PERF_K);
    }
    gpu_timer.stop();
    float max_gpu_time = gpu_timer.elapsed_ms() / NUM_RUNS;
    
    printf("\nMax Pooling:\n");
    printf("  CPU: %.3f ms\n", max_cpu_time);
    printf("  GPU: %.3f ms (%.1fx 加速)\n", max_gpu_time, max_cpu_time / max_gpu_time);
    
    // Average Pooling CPU
    cpu_timer.start();
    for (int i = 0; i < NUM_RUNS; i++) {
        pooling_avg_forward_cpu(h_input, h_output, PERF_N, PERF_C, PERF_H, PERF_W, PERF_K);
    }
    cpu_timer.stop();
    float avg_cpu_time = cpu_timer.elapsed_ms() / NUM_RUNS;
    
    // Average Pooling GPU
    gpu_timer.start();
    for (int i = 0; i < NUM_RUNS; i++) {
        pooling_avg_forward_gpu(d_input, d_output, PERF_N, PERF_C, PERF_H, PERF_W, PERF_K);
    }
    gpu_timer.stop();
    float avg_gpu_time = gpu_timer.elapsed_ms() / NUM_RUNS;
    
    printf("\nAverage Pooling:\n");
    printf("  CPU: %.3f ms\n", avg_cpu_time);
    printf("  GPU: %.3f ms (%.1fx 加速)\n", avg_gpu_time, avg_cpu_time / avg_gpu_time);
    
    // 清理
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("================================================================\n");
    printf("  第十六章练习1：Pooling 层前向传播\n");
    printf("  Max Pooling & Average Pooling Implementation\n");
    printf("================================================================\n\n");
    
    srand(42);  // 固定随机种子以便复现
    
    printf("=== 正确性验证 ===\n\n");
    
    bool all_passed = true;
    all_passed &= test_max_pooling_correctness();
    all_passed &= test_avg_pooling_correctness();
    
    if (all_passed) {
        printf("\n✅ 所有正确性测试通过！\n");
    } else {
        printf("\n❌ 部分测试失败！\n");
        return 1;
    }
    
    test_performance();
    
    printf("\n================================================================\n");
    printf("  测试完成\n");
    printf("================================================================\n");
    
    return 0;
}
