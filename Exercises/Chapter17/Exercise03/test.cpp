// ============================================================================
// test.cpp - 第十七章练习3: NUFFT Gridding 测试
// ============================================================================

#include "solution.h"
#include "../../../Common/timer.h"
#include "../../../Common/utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>

const float EPSILON = 1e-3f;

void generate_spiral_trajectory(float* kx, float* ky, int num_samples, int grid_size) {
    // 生成螺旋采样轨迹
    for (int i = 0; i < num_samples; i++) {
        float t = (float)i / num_samples * 6.0f * M_PI;  // 3 圈
        float r = (float)i / num_samples * (grid_size / 2.0f - 2.0f);
        kx[i] = grid_size / 2.0f + r * cosf(t);
        ky[i] = grid_size / 2.0f + r * sinf(t);
    }
}

void generate_random_data(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

bool compare_arrays(const float* a, const float* b, int n, float epsilon = EPSILON) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    bool passed = max_diff <= epsilon;
    printf("  最大差异: %.2e %s\n", max_diff, passed ? "✅" : "❌");
    return passed;
}

// ============================================================================
// 正确性测试
// ============================================================================

bool test_gridding_correctness() {
    printf("\n=== Gridding 正确性测试 ===\n");
    
    int num_samples = 512;
    int grid_size = 64;
    float kernel_width = 4.0f;
    float beta = 5.0f;
    
    printf("采样点: %d, 网格: %dx%d\n", num_samples, grid_size, grid_size);
    
    // 分配内存
    float* kx = new float[num_samples];
    float* ky = new float[num_samples];
    float* data_real = new float[num_samples];
    float* data_imag = new float[num_samples];
    float* grid_real_cpu = new float[grid_size * grid_size];
    float* grid_imag_cpu = new float[grid_size * grid_size];
    float* grid_real_gpu = new float[grid_size * grid_size];
    float* grid_imag_gpu = new float[grid_size * grid_size];
    
    generate_spiral_trajectory(kx, ky, num_samples, grid_size);
    generate_random_data(data_real, num_samples);
    generate_random_data(data_imag, num_samples);
    
    // CPU gridding
    gridding_cpu(kx, ky, data_real, data_imag, grid_real_cpu, grid_imag_cpu,
                 num_samples, grid_size, kernel_width, beta);
    
    // GPU gridding
    float *d_kx, *d_ky, *d_data_real, *d_data_imag;
    float *d_grid_real, *d_grid_imag;
    
    cudaMalloc(&d_kx, num_samples * sizeof(float));
    cudaMalloc(&d_ky, num_samples * sizeof(float));
    cudaMalloc(&d_data_real, num_samples * sizeof(float));
    cudaMalloc(&d_data_imag, num_samples * sizeof(float));
    cudaMalloc(&d_grid_real, grid_size * grid_size * sizeof(float));
    cudaMalloc(&d_grid_imag, grid_size * grid_size * sizeof(float));
    
    cudaMemcpy(d_kx, kx, num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ky, ky, num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_real, data_real, num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_imag, data_imag, num_samples * sizeof(float), cudaMemcpyHostToDevice);
    
    gridding_gpu(d_kx, d_ky, d_data_real, d_data_imag, d_grid_real, d_grid_imag,
                 num_samples, grid_size, kernel_width, beta);
    
    cudaMemcpy(grid_real_gpu, d_grid_real, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grid_imag_gpu, d_grid_imag, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("网格实部比较:\n");
    bool passed_real = compare_arrays(grid_real_cpu, grid_real_gpu, grid_size * grid_size);
    printf("网格虚部比较:\n");
    bool passed_imag = compare_arrays(grid_imag_cpu, grid_imag_gpu, grid_size * grid_size);
    
    // 清理
    cudaFree(d_kx); cudaFree(d_ky);
    cudaFree(d_data_real); cudaFree(d_data_imag);
    cudaFree(d_grid_real); cudaFree(d_grid_imag);
    
    delete[] kx; delete[] ky;
    delete[] data_real; delete[] data_imag;
    delete[] grid_real_cpu; delete[] grid_imag_cpu;
    delete[] grid_real_gpu; delete[] grid_imag_gpu;
    
    return passed_real && passed_imag;
}

// ============================================================================
// 性能测试
// ============================================================================

bool test_gridding_performance() {
    printf("\n=== Gridding 性能测试 ===\n");
    
    int num_samples = 16384;
    int grid_size = 256;
    float kernel_width = 4.0f;
    float beta = 5.0f;
    
    printf("采样点: %d, 网格: %dx%d\n", num_samples, grid_size, grid_size);
    
    // 分配内存
    float* kx = new float[num_samples];
    float* ky = new float[num_samples];
    float* data_real = new float[num_samples];
    float* data_imag = new float[num_samples];
    float* grid_real = new float[grid_size * grid_size];
    float* grid_imag = new float[grid_size * grid_size];
    
    generate_spiral_trajectory(kx, ky, num_samples, grid_size);
    generate_random_data(data_real, num_samples);
    generate_random_data(data_imag, num_samples);
    
    // CPU 性能
    Timer cpu_timer;
    cpu_timer.start();
    gridding_cpu(kx, ky, data_real, data_imag, grid_real, grid_imag,
                 num_samples, grid_size, kernel_width, beta);
    cpu_timer.stop();
    printf("CPU: %.2f ms\n", cpu_timer.elapsed_ms());
    
    // GPU 性能
    float *d_kx, *d_ky, *d_data_real, *d_data_imag;
    float *d_grid_real, *d_grid_imag;
    
    cudaMalloc(&d_kx, num_samples * sizeof(float));
    cudaMalloc(&d_ky, num_samples * sizeof(float));
    cudaMalloc(&d_data_real, num_samples * sizeof(float));
    cudaMalloc(&d_data_imag, num_samples * sizeof(float));
    cudaMalloc(&d_grid_real, grid_size * grid_size * sizeof(float));
    cudaMalloc(&d_grid_imag, grid_size * grid_size * sizeof(float));
    
    cudaMemcpy(d_kx, kx, num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ky, ky, num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_real, data_real, num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_imag, data_imag, num_samples * sizeof(float), cudaMemcpyHostToDevice);
    
    CudaTimer gpu_timer;
    gpu_timer.start();
    gridding_gpu(d_kx, d_ky, d_data_real, d_data_imag, d_grid_real, d_grid_imag,
                 num_samples, grid_size, kernel_width, beta);
    gpu_timer.stop();
    printf("GPU: %.2f ms\n", gpu_timer.elapsed_ms());
    
    printf("加速比: %.2fx\n", cpu_timer.elapsed_ms() / gpu_timer.elapsed_ms());
    
    // 清理
    cudaFree(d_kx); cudaFree(d_ky);
    cudaFree(d_data_real); cudaFree(d_data_imag);
    cudaFree(d_grid_real); cudaFree(d_grid_imag);
    
    delete[] kx; delete[] ky;
    delete[] data_real; delete[] data_imag;
    delete[] grid_real; delete[] grid_imag;
    
    return true;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("================================================================\n");
    printf("  第十七章练习3：NUFFT Gridding\n");
    printf("  非均匀快速傅里叶变换的网格化操作\n");
    printf("================================================================\n");
    
    srand(42);
    
    bool all_passed = true;
    all_passed &= test_gridding_correctness();
    all_passed &= test_gridding_performance();
    
    printf("\n================================================================\n");
    if (all_passed) {
        printf("  ✅ 所有测试通过！\n");
    } else {
        printf("  ❌ 部分测试失败\n");
    }
    printf("================================================================\n");
    
    return all_passed ? 0 : 1;
}
