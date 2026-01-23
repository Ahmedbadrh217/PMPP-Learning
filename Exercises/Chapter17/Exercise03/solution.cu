// ============================================================================
// solution.cu - 第十七章练习3: NUFFT Gridding
// ============================================================================
// 实现非均匀快速傅里叶变换的网格化操作
// 用于 MRI 重建中处理非笛卡尔采样轨迹
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
// Kaiser-Bessel 核函数
// ============================================================================

/**
 * @brief 修正的贝塞尔函数 I₀(x) 的近似计算
 */
__host__ __device__ float bessel_i0(float x) {
    float ax = fabsf(x);
    float ans;
    
    if (ax < 3.75f) {
        float y = x / 3.75f;
        y = y * y;
        ans = 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f
              + y * (0.2659732f + y * (0.0360768f + y * 0.0045813f)))));
    } else {
        float y = 3.75f / ax;
        ans = (expf(ax) / sqrtf(ax)) * (0.39894228f + y * (0.01328592f
              + y * (0.00225319f + y * (-0.00157565f + y * (0.00916281f
              + y * (-0.02057706f + y * (0.02635537f + y * (-0.01647633f
              + y * 0.00392377f))))))));
    }
    return ans;
}

/**
 * @brief Kaiser-Bessel 窗函数
 */
__host__ __device__ float kaiser_bessel_kernel(float dist, float width, float beta) {
    if (fabsf(dist) >= width / 2.0f) {
        return 0.0f;
    }
    
    float ratio = 2.0f * dist / width;
    float arg = beta * sqrtf(1.0f - ratio * ratio);
    return bessel_i0(arg) / bessel_i0(beta);
}

float kaiser_bessel(float x, float width, float beta) {
    return kaiser_bessel_kernel(x, width, beta);
}

// ============================================================================
// CUDA Kernel 定义
// ============================================================================

#define BLOCK_SIZE 256

/**
 * @brief Gridding kernel - 将非均匀点散布到网格
 * 
 * 每个线程处理一个采样点
 * 使用 atomicAdd 处理网格冲突
 */
__global__ void gridding_kernel(const float* kx, const float* ky,
                                 const float* data_real, const float* data_imag,
                                 float* grid_real, float* grid_imag,
                                 int num_samples, int grid_size,
                                 float kernel_width, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;
    
    // 获取采样点位置（假设 kx, ky 已归一化到 [0, grid_size)）
    float kx_val = kx[idx];
    float ky_val = ky[idx];
    float val_real = data_real[idx];
    float val_imag = data_imag[idx];
    
    // 计算影响的网格范围
    int half_width = (int)ceilf(kernel_width / 2.0f);
    int gx_center = (int)roundf(kx_val);
    int gy_center = (int)roundf(ky_val);
    
    int gx_start = max(0, gx_center - half_width);
    int gx_end = min(grid_size - 1, gx_center + half_width);
    int gy_start = max(0, gy_center - half_width);
    int gy_end = min(grid_size - 1, gy_center + half_width);
    
    // 散布到网格
    for (int gy = gy_start; gy <= gy_end; gy++) {
        for (int gx = gx_start; gx <= gx_end; gx++) {
            // 计算到采样点的距离
            float dx = (float)gx - kx_val;
            float dy = (float)gy - ky_val;
            float dist = sqrtf(dx * dx + dy * dy);
            
            // 计算核权重
            float weight = kaiser_bessel_kernel(dist, kernel_width, beta);
            
            if (weight > 0.0f) {
                int grid_idx = gy * grid_size + gx;
                atomicAdd(&grid_real[grid_idx], val_real * weight);
                atomicAdd(&grid_imag[grid_idx], val_imag * weight);
            }
        }
    }
}

/**
 * @brief Degridding kernel - 从网格插值到非均匀点
 */
__global__ void degridding_kernel(const float* grid_real, const float* grid_imag,
                                   const float* kx, const float* ky,
                                   float* data_real, float* data_imag,
                                   int num_samples, int grid_size,
                                   float kernel_width, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;
    
    float kx_val = kx[idx];
    float ky_val = ky[idx];
    
    int half_width = (int)ceilf(kernel_width / 2.0f);
    int gx_center = (int)roundf(kx_val);
    int gy_center = (int)roundf(ky_val);
    
    int gx_start = max(0, gx_center - half_width);
    int gx_end = min(grid_size - 1, gx_center + half_width);
    int gy_start = max(0, gy_center - half_width);
    int gy_end = min(grid_size - 1, gy_center + half_width);
    
    float sum_real = 0.0f;
    float sum_imag = 0.0f;
    float weight_sum = 0.0f;
    
    for (int gy = gy_start; gy <= gy_end; gy++) {
        for (int gx = gx_start; gx <= gx_end; gx++) {
            float dx = (float)gx - kx_val;
            float dy = (float)gy - ky_val;
            float dist = sqrtf(dx * dx + dy * dy);
            
            float weight = kaiser_bessel_kernel(dist, kernel_width, beta);
            
            if (weight > 0.0f) {
                int grid_idx = gy * grid_size + gx;
                sum_real += grid_real[grid_idx] * weight;
                sum_imag += grid_imag[grid_idx] * weight;
                weight_sum += weight;
            }
        }
    }
    
    // 归一化
    if (weight_sum > 0.0f) {
        data_real[idx] = sum_real / weight_sum;
        data_imag[idx] = sum_imag / weight_sum;
    } else {
        data_real[idx] = 0.0f;
        data_imag[idx] = 0.0f;
    }
}

/**
 * @brief 密度补偿计算 kernel
 */
__global__ void density_compensation_kernel(const float* kx, const float* ky,
                                             float* weights, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;
    
    // 简单的 Voronoi 近似：使用到原点的距离
    // 实际应用中应该用更复杂的方法
    float dist = sqrtf(kx[idx] * kx[idx] + ky[idx] * ky[idx]);
    weights[idx] = dist + 0.01f;  // 避免除以零
}

// ============================================================================
// CPU 实现
// ============================================================================

void gridding_cpu(const float* kx, const float* ky,
                  const float* data_real, const float* data_imag,
                  float* grid_real, float* grid_imag,
                  int num_samples, int grid_size, float kernel_width, float beta) {
    // 初始化网格
    memset(grid_real, 0, grid_size * grid_size * sizeof(float));
    memset(grid_imag, 0, grid_size * grid_size * sizeof(float));
    
    int half_width = (int)ceilf(kernel_width / 2.0f);
    
    for (int s = 0; s < num_samples; s++) {
        float kx_val = kx[s];
        float ky_val = ky[s];
        
        int gx_center = (int)roundf(kx_val);
        int gy_center = (int)roundf(ky_val);
        
        int gx_start = (gx_center - half_width > 0) ? gx_center - half_width : 0;
        int gx_end = (gx_center + half_width < grid_size - 1) ? gx_center + half_width : grid_size - 1;
        int gy_start = (gy_center - half_width > 0) ? gy_center - half_width : 0;
        int gy_end = (gy_center + half_width < grid_size - 1) ? gy_center + half_width : grid_size - 1;
        
        for (int gy = gy_start; gy <= gy_end; gy++) {
            for (int gx = gx_start; gx <= gx_end; gx++) {
                float dx = (float)gx - kx_val;
                float dy = (float)gy - ky_val;
                float dist = sqrtf(dx * dx + dy * dy);
                
                float weight = kaiser_bessel_kernel(dist, kernel_width, beta);
                
                if (weight > 0.0f) {
                    int grid_idx = gy * grid_size + gx;
                    grid_real[grid_idx] += data_real[s] * weight;
                    grid_imag[grid_idx] += data_imag[s] * weight;
                }
            }
        }
    }
}

void degridding_cpu(const float* grid_real, const float* grid_imag,
                    const float* kx, const float* ky,
                    float* data_real, float* data_imag,
                    int num_samples, int grid_size, float kernel_width, float beta) {
    int half_width = (int)ceilf(kernel_width / 2.0f);
    
    for (int s = 0; s < num_samples; s++) {
        float kx_val = kx[s];
        float ky_val = ky[s];
        
        int gx_center = (int)roundf(kx_val);
        int gy_center = (int)roundf(ky_val);
        
        int gx_start = (gx_center - half_width > 0) ? gx_center - half_width : 0;
        int gx_end = (gx_center + half_width < grid_size - 1) ? gx_center + half_width : grid_size - 1;
        int gy_start = (gy_center - half_width > 0) ? gy_center - half_width : 0;
        int gy_end = (gy_center + half_width < grid_size - 1) ? gy_center + half_width : grid_size - 1;
        
        float sum_real = 0.0f;
        float sum_imag = 0.0f;
        float weight_sum = 0.0f;
        
        for (int gy = gy_start; gy <= gy_end; gy++) {
            for (int gx = gx_start; gx <= gx_end; gx++) {
                float dx = (float)gx - kx_val;
                float dy = (float)gy - ky_val;
                float dist = sqrtf(dx * dx + dy * dy);
                
                float weight = kaiser_bessel_kernel(dist, kernel_width, beta);
                
                if (weight > 0.0f) {
                    int grid_idx = gy * grid_size + gx;
                    sum_real += grid_real[grid_idx] * weight;
                    sum_imag += grid_imag[grid_idx] * weight;
                    weight_sum += weight;
                }
            }
        }
        
        if (weight_sum > 0.0f) {
            data_real[s] = sum_real / weight_sum;
            data_imag[s] = sum_imag / weight_sum;
        } else {
            data_real[s] = 0.0f;
            data_imag[s] = 0.0f;
        }
    }
}

void compute_density_compensation_cpu(const float* kx, const float* ky,
                                       float* weights, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        float dist = sqrtf(kx[i] * kx[i] + ky[i] * ky[i]);
        weights[i] = dist + 0.01f;
    }
}

// ============================================================================
// GPU 实现
// ============================================================================

void gridding_gpu(const float* d_kx, const float* d_ky,
                  const float* d_data_real, const float* d_data_imag,
                  float* d_grid_real, float* d_grid_imag,
                  int num_samples, int grid_size, float kernel_width, float beta) {
    // 初始化网格
    cudaMemset(d_grid_real, 0, grid_size * grid_size * sizeof(float));
    cudaMemset(d_grid_imag, 0, grid_size * grid_size * sizeof(float));
    
    int num_blocks = (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gridding_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_kx, d_ky, d_data_real, d_data_imag,
        d_grid_real, d_grid_imag,
        num_samples, grid_size, kernel_width, beta);
    CHECK_LAST_CUDA_ERROR();
}

void degridding_gpu(const float* d_grid_real, const float* d_grid_imag,
                    const float* d_kx, const float* d_ky,
                    float* d_data_real, float* d_data_imag,
                    int num_samples, int grid_size, float kernel_width, float beta) {
    int num_blocks = (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    degridding_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_grid_real, d_grid_imag, d_kx, d_ky,
        d_data_real, d_data_imag,
        num_samples, grid_size, kernel_width, beta);
    CHECK_LAST_CUDA_ERROR();
}

void compute_density_compensation_gpu(const float* d_kx, const float* d_ky,
                                       float* d_weights, int num_samples) {
    int num_blocks = (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    density_compensation_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_kx, d_ky, d_weights, num_samples);
    CHECK_LAST_CUDA_ERROR();
}
