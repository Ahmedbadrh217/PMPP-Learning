// ============================================================================
// test.cpp - 第十六章: cuBLAS SGEMM 测试
// ============================================================================

#include "solution.h"
#include "../../Common/timer.h"
#include "../../Common/utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>

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
// SGEMM 测试
// ============================================================================

bool test_sgemm_basic() {
    printf("\n=== SGEMM 基础测试 ===\n");
    
    int m = 64, n = 128, k = 256;
    printf("矩阵大小: A[%d×%d] × B[%d×%d] = C[%d×%d]\n", m, k, k, n, m, n);
    
    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C_cpu = new float[m * n];
    float* C_gpu = new float[m * n];
    
    generate_random_data(A, m * k);
    generate_random_data(B, k * n);
    
    // CPU 参考计算
    sgemm_cpu(A, B, C_cpu, m, n, k);
    
    // cuBLAS 计算
    int ret = sgemm_wrapper(A, B, C_gpu, m, n, k, 0, 0);
    if (ret != 0) {
        printf("cuBLAS SGEMM 失败\n");
        delete[] A; delete[] B; delete[] C_cpu; delete[] C_gpu;
        return false;
    }
    
    bool passed = compare_arrays(C_cpu, C_gpu, m * n);
    
    delete[] A; delete[] B; delete[] C_cpu; delete[] C_gpu;
    return passed;
}

bool test_sgemm_small() {
    printf("\n=== SGEMM 小矩阵测试 ===\n");
    
    int m = 4, n = 4, k = 4;
    
    // 简单的已知矩阵
    float A[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    float B[] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };  // 单位矩阵
    
    float C_cpu[16];
    float C_gpu[16];
    
    sgemm_cpu(A, B, C_cpu, m, n, k);
    sgemm_wrapper(A, B, C_gpu, m, n, k, 0, 0);
    
    printf("A × I = A 测试\n");
    bool passed = compare_arrays(C_cpu, C_gpu, m * n);
    
    // 也验证结果应该等于 A
    bool correct = compare_arrays(A, C_gpu, m * n);
    printf("  结果 = A: %s\n", correct ? "✅" : "❌");
    
    return passed && correct;
}

bool test_sgemm_performance() {
    printf("\n=== SGEMM 性能测试 ===\n");
    
    int m = 1024, n = 1024, k = 1024;
    printf("矩阵大小: %d × %d × %d\n", m, n, k);
    
    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];
    
    generate_random_data(A, m * k);
    generate_random_data(B, k * n);
    
    const int NUM_RUNS = 10;
    Timer cpu_timer;
    CudaTimer gpu_timer;
    
    // CPU 性能
    cpu_timer.start();
    for (int i = 0; i < NUM_RUNS; i++) {
        sgemm_cpu(A, B, C, m, n, k);
    }
    cpu_timer.stop();
    float cpu_time = cpu_timer.elapsed_ms() / NUM_RUNS;
    
    // GPU 性能
    gpu_timer.start();
    for (int i = 0; i < NUM_RUNS; i++) {
        sgemm_wrapper(A, B, C, m, n, k, 0, 0);
    }
    gpu_timer.stop();
    float gpu_time = gpu_timer.elapsed_ms() / NUM_RUNS;
    
    // 计算 GFLOPS
    double flops = 2.0 * m * n * k;
    double cpu_gflops = flops / (cpu_time * 1e6);
    double gpu_gflops = flops / (gpu_time * 1e6);
    
    printf("CPU: %.2f ms (%.2f GFLOPS)\n", cpu_time, cpu_gflops);
    printf("GPU: %.2f ms (%.2f GFLOPS) - %.1fx 加速\n", 
           gpu_time, gpu_gflops, cpu_time / gpu_time);
    
    delete[] A; delete[] B; delete[] C;
    return true;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("================================================================\n");
    printf("  第十六章：cuBLAS SGEMM 矩阵乘法\n");
    printf("  全连接层核心操作\n");
    printf("================================================================\n");
    
    srand(42);
    
    // 初始化 cuBLAS
    if (init_cublas() != 0) {
        printf("无法初始化 cuBLAS\n");
        return 1;
    }
    
    bool all_passed = true;
    
    all_passed &= test_sgemm_small();
    all_passed &= test_sgemm_basic();
    all_passed &= test_sgemm_performance();
    
    // 清理
    cleanup_cublas();
    
    printf("\n================================================================\n");
    if (all_passed) {
        printf("  ✅ 所有测试通过！\n");
    } else {
        printf("  ❌ 部分测试失败\n");
    }
    printf("================================================================\n");
    
    return all_passed ? 0 : 1;
}
