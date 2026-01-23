// ============================================================================
// test.cpp - 第十七章练习2: F^H D 核心计算测试
// ============================================================================

#include "solution.h"
#include "../../../Common/timer.h"
#include "../../../Common/utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>

const float EPSILON = 1e-3f;

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

bool test_correctness() {
    printf("\n=== F^H D 正确性测试 ===\n");
    
    int M = 256;   // k空间采样点
    int N = 64;    // 图像像素
    
    // 分配主机内存
    float* rPhi = new float[M];
    float* iPhi = new float[M];
    float* rD = new float[M];
    float* iD = new float[M];
    float* kx = new float[M];
    float* ky = new float[M];
    float* kz = new float[M];
    float* x = new float[N];
    float* y = new float[N];
    float* z = new float[N];
    float* rFhD_cpu = new float[N];
    float* iFhD_cpu = new float[N];
    float* rFhD_gpu = new float[N];
    float* iFhD_gpu = new float[N];
    
    // 生成测试数据
    generate_random_data(rPhi, M);
    generate_random_data(iPhi, M);
    generate_random_data(rD, M);
    generate_random_data(iD, M);
    generate_random_data(kx, M);
    generate_random_data(ky, M);
    generate_random_data(kz, M);
    generate_random_data(x, N);
    generate_random_data(y, N);
    generate_random_data(z, N);
    
    // CPU 计算
    fhd_compute_cpu(rPhi, iPhi, rD, iD, kx, ky, kz, x, y, z, rFhD_cpu, iFhD_cpu, M, N);
    
    // GPU 计算
    float *d_rPhi, *d_iPhi, *d_rD, *d_iD;
    float *d_kx, *d_ky, *d_kz, *d_x, *d_y, *d_z;
    float *d_rFhD, *d_iFhD;
    
    cudaMalloc(&d_rPhi, M * sizeof(float));
    cudaMalloc(&d_iPhi, M * sizeof(float));
    cudaMalloc(&d_rD, M * sizeof(float));
    cudaMalloc(&d_iD, M * sizeof(float));
    cudaMalloc(&d_kx, M * sizeof(float));
    cudaMalloc(&d_ky, M * sizeof(float));
    cudaMalloc(&d_kz, M * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMalloc(&d_rFhD, N * sizeof(float));
    cudaMalloc(&d_iFhD, N * sizeof(float));
    
    cudaMemcpy(d_rPhi, rPhi, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iPhi, iPhi, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rD, rD, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iD, iD, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kx, kx, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ky, ky, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kz, kz, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, N * sizeof(float), cudaMemcpyHostToDevice);
    
    fhd_compute_gpu(d_rPhi, d_iPhi, d_rD, d_iD, d_kx, d_ky, d_kz, d_x, d_y, d_z,
                    d_rFhD, d_iFhD, M, N);
    
    cudaMemcpy(rFhD_gpu, d_rFhD, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(iFhD_gpu, d_iFhD, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("实部比较:\n");
    bool passed_real = compare_arrays(rFhD_cpu, rFhD_gpu, N);
    printf("虚部比较:\n");
    bool passed_imag = compare_arrays(iFhD_cpu, iFhD_gpu, N);
    
    // 清理
    cudaFree(d_rPhi); cudaFree(d_iPhi); cudaFree(d_rD); cudaFree(d_iD);
    cudaFree(d_kx); cudaFree(d_ky); cudaFree(d_kz);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_rFhD); cudaFree(d_iFhD);
    
    delete[] rPhi; delete[] iPhi; delete[] rD; delete[] iD;
    delete[] kx; delete[] ky; delete[] kz;
    delete[] x; delete[] y; delete[] z;
    delete[] rFhD_cpu; delete[] iFhD_cpu; delete[] rFhD_gpu; delete[] iFhD_gpu;
    
    return passed_real && passed_imag;
}

// ============================================================================
// 性能测试
// ============================================================================

bool test_performance() {
    printf("\n=== F^H D 性能测试 ===\n");
    
    int M = 4096;   // k空间采样点
    int N = 1024;   // 图像像素
    printf("M=%d (k-space), N=%d (image)\n", M, N);
    
    // 分配内存（只需要 GPU 测试性能）
    float *d_rPhi, *d_iPhi, *d_rD, *d_iD;
    float *d_kx, *d_ky, *d_kz, *d_x, *d_y, *d_z;
    float *d_rFhD, *d_iFhD;
    
    cudaMalloc(&d_rPhi, M * sizeof(float));
    cudaMalloc(&d_iPhi, M * sizeof(float));
    cudaMalloc(&d_rD, M * sizeof(float));
    cudaMalloc(&d_iD, M * sizeof(float));
    cudaMalloc(&d_kx, M * sizeof(float));
    cudaMalloc(&d_ky, M * sizeof(float));
    cudaMalloc(&d_kz, M * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMalloc(&d_rFhD, N * sizeof(float));
    cudaMalloc(&d_iFhD, N * sizeof(float));
    
    // 基础版性能
    CudaTimer timer1;
    timer1.start();
    fhd_compute_gpu(d_rPhi, d_iPhi, d_rD, d_iD, d_kx, d_ky, d_kz, d_x, d_y, d_z,
                    d_rFhD, d_iFhD, M, N);
    timer1.stop();
    printf("GPU 基础版: %.2f ms\n", timer1.elapsed_ms());
    
    // 优化版性能
    CudaTimer timer2;
    timer2.start();
    fhd_compute_gpu_optimized(d_rPhi, d_iPhi, d_rD, d_iD, d_kx, d_ky, d_kz, d_x, d_y, d_z,
                              d_rFhD, d_iFhD, M, N);
    timer2.stop();
    printf("GPU 优化版: %.2f ms\n", timer2.elapsed_ms());
    
    printf("优化提升: %.2fx\n", timer1.elapsed_ms() / timer2.elapsed_ms());
    
    // 清理
    cudaFree(d_rPhi); cudaFree(d_iPhi); cudaFree(d_rD); cudaFree(d_iD);
    cudaFree(d_kx); cudaFree(d_ky); cudaFree(d_kz);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_rFhD); cudaFree(d_iFhD);
    
    return true;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("================================================================\n");
    printf("  第十七章练习2：F^H D 核心计算\n");
    printf("  MRI 重建的傅里叶变换共轭转置\n");
    printf("================================================================\n");
    
    srand(42);
    
    bool all_passed = true;
    all_passed &= test_correctness();
    all_passed &= test_performance();
    
    printf("\n================================================================\n");
    if (all_passed) {
        printf("  ✅ 所有测试通过！\n");
    } else {
        printf("  ❌ 部分测试失败\n");
    }
    printf("================================================================\n");
    
    return all_passed ? 0 : 1;
}
