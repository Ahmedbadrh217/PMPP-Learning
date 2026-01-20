// ============================================================================
// solution.h - 第十六章: cuBLAS SGEMM 矩阵乘法包装
// ============================================================================
// 封装 cuBLAS 库的 SGEMM（单精度通用矩阵乘法）操作
// 对应参考仓库 cublas_wrapper.c 的功能
// ============================================================================

#ifndef SOLUTION_H
#define SOLUTION_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

// ============================================================================
// cuBLAS 句柄管理
// ============================================================================

/**
 * @brief 初始化 cuBLAS 库
 * @return 0 成功，-1 失败
 */
int init_cublas();

/**
 * @brief 清理 cuBLAS 资源
 * @return 0 成功，-1 失败
 */
int cleanup_cublas();

// ============================================================================
// SGEMM 矩阵乘法
// ============================================================================

/**
 * @brief 执行单精度矩阵乘法 C = A * B
 * 
 * 支持矩阵转置选项
 * 
 * @param A       矩阵 A [m × k] (行主序)
 * @param B       矩阵 B [k × n] (行主序)
 * @param C       结果矩阵 C [m × n] (行主序)
 * @param m       A 的行数 / C 的行数
 * @param n       B 的列数 / C 的列数
 * @param k       A 的列数 / B 的行数
 * @param transA  是否转置 A (0=不转置, 1=转置)
 * @param transB  是否转置 B (0=不转置, 1=转置)
 * @return 0 成功，-1 失败
 */
int sgemm_wrapper(float* A, float* B, float* C, 
                  int m, int n, int k, 
                  int transA, int transB);

/**
 * @brief 在 GPU 上执行 SGEMM (数据已在 GPU)
 * 
 * @param d_A     设备端矩阵 A
 * @param d_B     设备端矩阵 B
 * @param d_C     设备端结果矩阵 C
 * @param m, n, k 维度
 * @param transA, transB 转置选项
 * @return 0 成功，-1 失败
 */
int sgemm_device(float* d_A, float* d_B, float* d_C,
                 int m, int n, int k,
                 int transA, int transB);

// ============================================================================
// GPU 内存管理辅助函数
// ============================================================================

/**
 * @brief 分配 GPU 内存并可选复制数据
 * 
 * @param host_data 主机数据 (可为 NULL)
 * @param size      元素数量
 * @return 设备指针
 */
float* gpu_alloc(float* host_data, int size);

/**
 * @brief 从 GPU 复制数据到主机
 */
int gpu_to_host(float* dev_ptr, float* host_ptr, int size);

/**
 * @brief 释放 GPU 内存
 */
void gpu_free(float* dev_ptr);

// ============================================================================
// CPU 参考实现
// ============================================================================

/**
 * @brief CPU 矩阵乘法 C = A * B
 */
void sgemm_cpu(const float* A, const float* B, float* C,
               int m, int n, int k);

#endif // SOLUTION_H
