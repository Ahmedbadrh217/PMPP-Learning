// ============================================================================
// solution.h - 第十七章练习2: F^H D 核心计算
// ============================================================================
// 实现 MRI 重建中的 F^H D（傅里叶变换的共轭转置乘以数据）计算
// 对应书中 Figure 17.4 的核心算法
// ============================================================================

#ifndef SOLUTION_H
#define SOLUTION_H

#include <cuda_runtime.h>

// ============================================================================
// 数据结构
// ============================================================================

// k 空间采样点结构
struct KSpaceSample {
    float kx, ky, kz;   // k 空间坐标
    float rPhi, iPhi;   // 相位（实部和虚部）
    float rD, iD;       // 数据（实部和虚部）
};

// ============================================================================
// F^H D 计算函数
// ============================================================================

/**
 * @brief CPU 串行版 F^H D 计算
 * 
 * 对应书中 Figure 17.4 的原始代码
 * 
 * @param rPhi, iPhi   相位矩阵（复数）[M]
 * @param rD, iD       k空间数据（复数）[M]
 * @param kx, ky, kz   k空间坐标 [M]
 * @param x, y, z      图像空间坐标 [N]
 * @param rFhD, iFhD   输出 F^H*D 结果（复数）[N]
 * @param M            k空间采样点数
 * @param N            图像像素数
 */
void fhd_compute_cpu(const float* rPhi, const float* iPhi,
                     const float* rD, const float* iD,
                     const float* kx, const float* ky, const float* kz,
                     const float* x, const float* y, const float* z,
                     float* rFhD, float* iFhD,
                     int M, int N);

/**
 * @brief GPU 基础版 F^H D 计算
 * 
 * 每个线程计算一个输出像素
 */
void fhd_compute_gpu(const float* d_rPhi, const float* d_iPhi,
                     const float* d_rD, const float* d_iD,
                     const float* d_kx, const float* d_ky, const float* d_kz,
                     const float* d_x, const float* d_y, const float* d_z,
                     float* d_rFhD, float* d_iFhD,
                     int M, int N);

/**
 * @brief GPU 优化版 F^H D 计算
 * 
 * 使用寄存器优化，对应书中 Figure 17.11
 * 将 x[n], y[n], z[n] 加载到寄存器
 */
void fhd_compute_gpu_optimized(const float* d_rPhi, const float* d_iPhi,
                               const float* d_rD, const float* d_iD,
                               const float* d_kx, const float* d_ky, const float* d_kz,
                               const float* d_x, const float* d_y, const float* d_z,
                               float* d_rFhD, float* d_iFhD,
                               int M, int N);

/**
 * @brief 预计算 Mu = Phi * D（循环分裂的第一部分）
 * 
 * 对应练习1中的循环分裂优化
 */
void compute_mu_cpu(const float* rPhi, const float* iPhi,
                    const float* rD, const float* iD,
                    float* rMu, float* iMu, int M);

void compute_mu_gpu(const float* d_rPhi, const float* d_iPhi,
                    const float* d_rD, const float* d_iD,
                    float* d_rMu, float* d_iMu, int M);

/**
 * @brief 使用预计算 Mu 的 F^H D 计算（循环分裂的第二部分）
 */
void fhd_with_mu_cpu(const float* rMu, const float* iMu,
                     const float* kx, const float* ky, const float* kz,
                     const float* x, const float* y, const float* z,
                     float* rFhD, float* iFhD,
                     int M, int N);

void fhd_with_mu_gpu(const float* d_rMu, const float* d_iMu,
                     const float* d_kx, const float* d_ky, const float* d_kz,
                     const float* d_x, const float* d_y, const float* d_z,
                     float* d_rFhD, float* d_iFhD,
                     int M, int N);

#endif // SOLUTION_H
