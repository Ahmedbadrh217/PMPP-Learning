// ============================================================================
// solution.h - 第十七章练习3: NUFFT Gridding
// ============================================================================
// 实现非均匀快速傅里叶变换的网格化操作
// 用于 MRI 重建中处理非笛卡尔采样轨迹
// ============================================================================

#ifndef SOLUTION_H
#define SOLUTION_H

#include <cuda_runtime.h>

// ============================================================================
// 数据结构
// ============================================================================

// 非均匀采样点
struct NonUniformSample {
    float kx, ky;       // k空间坐标（非规则）
    float real, imag;   // 复数数据
};

// ============================================================================
// Kaiser-Bessel 插值核
// ============================================================================

/**
 * @brief Kaiser-Bessel 窗函数
 * 
 * 用于 NUFFT 的插值核
 * 
 * @param x       到中心的距离
 * @param width   核宽度
 * @param beta    形状参数
 * @return 核函数值
 */
float kaiser_bessel(float x, float width, float beta);

// ============================================================================
// Gridding 操作
// ============================================================================

/**
 * @brief CPU 串行版 Gridding（Type-1 NUFFT 的第一步）
 * 
 * 将非均匀采样点散布到规则网格上
 * 
 * @param samples      非均匀采样数据 [num_samples × (kx, ky, real, imag)]
 * @param kx, ky       采样点 k 空间坐标 [num_samples]
 * @param data_real    采样点数据实部 [num_samples]
 * @param data_imag    采样点数据虚部 [num_samples]
 * @param grid_real    输出网格实部 [grid_size × grid_size]
 * @param grid_imag    输出网格虚部 [grid_size × grid_size]
 * @param num_samples  采样点数量
 * @param grid_size    网格大小
 * @param kernel_width 插值核宽度
 * @param beta         Kaiser-Bessel 参数
 */
void gridding_cpu(const float* kx, const float* ky,
                  const float* data_real, const float* data_imag,
                  float* grid_real, float* grid_imag,
                  int num_samples, int grid_size, float kernel_width, float beta);

/**
 * @brief GPU 并行版 Gridding
 * 
 * 每个线程处理一个采样点，使用 atomicAdd 写入网格
 */
void gridding_gpu(const float* d_kx, const float* d_ky,
                  const float* d_data_real, const float* d_data_imag,
                  float* d_grid_real, float* d_grid_imag,
                  int num_samples, int grid_size, float kernel_width, float beta);

// ============================================================================
// Degridding 操作（Type-2 NUFFT）
// ============================================================================

/**
 * @brief CPU 版 Degridding
 * 
 * 从规则网格插值到非均匀采样点
 */
void degridding_cpu(const float* grid_real, const float* grid_imag,
                    const float* kx, const float* ky,
                    float* data_real, float* data_imag,
                    int num_samples, int grid_size, float kernel_width, float beta);

/**
 * @brief GPU 版 Degridding
 */
void degridding_gpu(const float* d_grid_real, const float* d_grid_imag,
                    const float* d_kx, const float* d_ky,
                    float* d_data_real, float* d_data_imag,
                    int num_samples, int grid_size, float kernel_width, float beta);

// ============================================================================
// 密度补偿
// ============================================================================

/**
 * @brief 计算采样密度补偿权重
 */
void compute_density_compensation_cpu(const float* kx, const float* ky,
                                       float* weights, int num_samples);

void compute_density_compensation_gpu(const float* d_kx, const float* d_ky,
                                       float* d_weights, int num_samples);

#endif // SOLUTION_H
