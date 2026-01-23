// ============================================================================
// solution.cu - 第十七章练习2: F^H D 核心计算
// ============================================================================
// 实现 MRI 重建中的 F^H D（傅里叶变换的共轭转置乘以数据）计算
// 对应书中 Figure 17.4, 17.9, 17.11 的各种实现
// ============================================================================

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cmath>
#include <cstring>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================================================
// CUDA Kernel 定义
// ============================================================================

#define FHD_THREADS_PER_BLOCK 256

/**
 * @brief 计算 Mu = Phi * D（复数乘法）
 */
__global__ void compute_mu_kernel(const float* rPhi, const float* iPhi,
                                   const float* rD, const float* iD,
                                   float* rMu, float* iMu, int M) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M) {
        // 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        rMu[m] = rPhi[m] * rD[m] + iPhi[m] * iD[m];  // 注意符号（共轭）
        iMu[m] = rPhi[m] * iD[m] - iPhi[m] * rD[m];
    }
}

/**
 * @brief 基础版 F^H D kernel
 * 
 * 每个线程计算一个输出像素 FhD[n]
 */
__global__ void fhd_kernel_basic(const float* rMu, const float* iMu,
                                  const float* kx, const float* ky, const float* kz,
                                  const float* x, const float* y, const float* z,
                                  float* rFhD, float* iFhD,
                                  int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    
    float xn = x[n];
    float yn = y[n];
    float zn = z[n];
    
    float rSum = 0.0f;
    float iSum = 0.0f;
    
    for (int m = 0; m < M; m++) {
        float expFhD = 2.0f * M_PI * (kx[m] * xn + ky[m] * yn + kz[m] * zn);
        float cArg = cosf(expFhD);
        float sArg = sinf(expFhD);
        
        rSum += rMu[m] * cArg - iMu[m] * sArg;
        iSum += iMu[m] * cArg + rMu[m] * sArg;
    }
    
    rFhD[n] = rSum;
    iFhD[n] = iSum;
}

/**
 * @brief 优化版 F^H D kernel（使用寄存器）
 * 
 * 对应书中 Figure 17.11
 * 将频繁访问的坐标和输出加载到寄存器
 */
__global__ void fhd_kernel_optimized(const float* rMu, const float* iMu,
                                      const float* kx, const float* ky, const float* kz,
                                      const float* x, const float* y, const float* z,
                                      float* rFhD, float* iFhD,
                                      int M, int N) {
    int n = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;
    if (n >= N) return;
    
    // 将频繁访问的坐标加载到寄存器
    float xn_r = x[n];
    float yn_r = y[n];
    float zn_r = z[n];
    
    // 累加器也在寄存器中
    float rFhDn_r = 0.0f;
    float iFhDn_r = 0.0f;
    
    for (int m = 0; m < M; m++) {
        float expFhD = 2.0f * M_PI * (kx[m] * xn_r + ky[m] * yn_r + kz[m] * zn_r);
        float cArg = cosf(expFhD);
        float sArg = sinf(expFhD);
        
        rFhDn_r += rMu[m] * cArg - iMu[m] * sArg;
        iFhDn_r += iMu[m] * cArg + rMu[m] * sArg;
    }
    
    // 写回全局内存
    rFhD[n] = rFhDn_r;
    iFhD[n] = iFhDn_r;
}

// ============================================================================
// CPU 实现
// ============================================================================

void compute_mu_cpu(const float* rPhi, const float* iPhi,
                    const float* rD, const float* iD,
                    float* rMu, float* iMu, int M) {
    for (int m = 0; m < M; m++) {
        rMu[m] = rPhi[m] * rD[m] + iPhi[m] * iD[m];
        iMu[m] = rPhi[m] * iD[m] - iPhi[m] * rD[m];
    }
}

void fhd_with_mu_cpu(const float* rMu, const float* iMu,
                     const float* kx, const float* ky, const float* kz,
                     const float* x, const float* y, const float* z,
                     float* rFhD, float* iFhD,
                     int M, int N) {
    // 初始化输出
    memset(rFhD, 0, N * sizeof(float));
    memset(iFhD, 0, N * sizeof(float));
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float expFhD = 2.0f * M_PI * (kx[m] * x[n] + ky[m] * y[n] + kz[m] * z[n]);
            float cArg = cosf(expFhD);
            float sArg = sinf(expFhD);
            
            rFhD[n] += rMu[m] * cArg - iMu[m] * sArg;
            iFhD[n] += iMu[m] * cArg + rMu[m] * sArg;
        }
    }
}

void fhd_compute_cpu(const float* rPhi, const float* iPhi,
                     const float* rD, const float* iD,
                     const float* kx, const float* ky, const float* kz,
                     const float* x, const float* y, const float* z,
                     float* rFhD, float* iFhD,
                     int M, int N) {
    // 分配临时 Mu 数组
    float* rMu = new float[M];
    float* iMu = new float[M];
    
    // 第一步：计算 Mu = Phi * D
    compute_mu_cpu(rPhi, iPhi, rD, iD, rMu, iMu, M);
    
    // 第二步：计算 F^H * Mu
    fhd_with_mu_cpu(rMu, iMu, kx, ky, kz, x, y, z, rFhD, iFhD, M, N);
    
    delete[] rMu;
    delete[] iMu;
}

// ============================================================================
// GPU 实现
// ============================================================================

void compute_mu_gpu(const float* d_rPhi, const float* d_iPhi,
                    const float* d_rD, const float* d_iD,
                    float* d_rMu, float* d_iMu, int M) {
    int num_blocks = (M + FHD_THREADS_PER_BLOCK - 1) / FHD_THREADS_PER_BLOCK;
    compute_mu_kernel<<<num_blocks, FHD_THREADS_PER_BLOCK>>>(
        d_rPhi, d_iPhi, d_rD, d_iD, d_rMu, d_iMu, M);
    CHECK_LAST_CUDA_ERROR();
}

void fhd_with_mu_gpu(const float* d_rMu, const float* d_iMu,
                     const float* d_kx, const float* d_ky, const float* d_kz,
                     const float* d_x, const float* d_y, const float* d_z,
                     float* d_rFhD, float* d_iFhD,
                     int M, int N) {
    int num_blocks = (N + FHD_THREADS_PER_BLOCK - 1) / FHD_THREADS_PER_BLOCK;
    fhd_kernel_basic<<<num_blocks, FHD_THREADS_PER_BLOCK>>>(
        d_rMu, d_iMu, d_kx, d_ky, d_kz, d_x, d_y, d_z, d_rFhD, d_iFhD, M, N);
    CHECK_LAST_CUDA_ERROR();
}

void fhd_compute_gpu(const float* d_rPhi, const float* d_iPhi,
                     const float* d_rD, const float* d_iD,
                     const float* d_kx, const float* d_ky, const float* d_kz,
                     const float* d_x, const float* d_y, const float* d_z,
                     float* d_rFhD, float* d_iFhD,
                     int M, int N) {
    // 分配临时 Mu 数组
    float *d_rMu, *d_iMu;
    cudaMalloc(&d_rMu, M * sizeof(float));
    cudaMalloc(&d_iMu, M * sizeof(float));
    
    // 第一步：计算 Mu
    compute_mu_gpu(d_rPhi, d_iPhi, d_rD, d_iD, d_rMu, d_iMu, M);
    
    // 第二步：计算 F^H * Mu
    fhd_with_mu_gpu(d_rMu, d_iMu, d_kx, d_ky, d_kz, d_x, d_y, d_z, d_rFhD, d_iFhD, M, N);
    
    cudaFree(d_rMu);
    cudaFree(d_iMu);
}

void fhd_compute_gpu_optimized(const float* d_rPhi, const float* d_iPhi,
                               const float* d_rD, const float* d_iD,
                               const float* d_kx, const float* d_ky, const float* d_kz,
                               const float* d_x, const float* d_y, const float* d_z,
                               float* d_rFhD, float* d_iFhD,
                               int M, int N) {
    // 分配临时 Mu 数组
    float *d_rMu, *d_iMu;
    cudaMalloc(&d_rMu, M * sizeof(float));
    cudaMalloc(&d_iMu, M * sizeof(float));
    
    // 第一步：计算 Mu
    compute_mu_gpu(d_rPhi, d_iPhi, d_rD, d_iD, d_rMu, d_iMu, M);
    
    // 第二步：使用优化版 kernel
    int num_blocks = (N + FHD_THREADS_PER_BLOCK - 1) / FHD_THREADS_PER_BLOCK;
    fhd_kernel_optimized<<<num_blocks, FHD_THREADS_PER_BLOCK>>>(
        d_rMu, d_iMu, d_kx, d_ky, d_kz, d_x, d_y, d_z, d_rFhD, d_iFhD, M, N);
    CHECK_LAST_CUDA_ERROR();
    
    cudaFree(d_rMu);
    cudaFree(d_iMu);
}
