// ============================================================================
// solution.h - 第十七章练习1: 共轭梯度法 (Conjugate Gradient)
// ============================================================================
// 求解对称正定线性系统 Ax = b
// 对应参考仓库 cg_algo.py 的 C++/CUDA 实现
// ============================================================================

#ifndef SOLUTION_H
#define SOLUTION_H

#include <cuda_runtime.h>

// ============================================================================
// 共轭梯度求解器
// ============================================================================

/**
 * @brief CPU 串行版共轭梯度法
 * 
 * 求解 Ax = b，其中 A 是对称正定矩阵
 * 
 * @param A       系数矩阵 [n × n]（行主序）
 * @param b       右端项向量 [n]
 * @param x       解向量 [n]（输入初值，输出解）
 * @param n       系统大小
 * @param tol     收敛精度
 * @param max_iter 最大迭代次数
 * @return 实际迭代次数
 */
int cg_solve_cpu(const float* A, const float* b, float* x,
                 int n, float tol, int max_iter);

/**
 * @brief GPU 并行版共轭梯度法
 * 
 * 向量操作在 GPU 上并行执行
 * 
 * @param d_A     设备端系数矩阵 [n × n]
 * @param d_b     设备端右端项向量 [n]
 * @param d_x     设备端解向量 [n]
 * @param n       系统大小
 * @param tol     收敛精度
 * @param max_iter 最大迭代次数
 * @return 实际迭代次数
 */
int cg_solve_gpu(const float* d_A, const float* d_b, float* d_x,
                 int n, float tol, int max_iter);

// ============================================================================
// 向量操作（GPU 并行）
// ============================================================================

/**
 * @brief 向量内积 result = x · y
 */
float vector_dot_cpu(const float* x, const float* y, int n);
float vector_dot_gpu(const float* d_x, const float* d_y, int n);

/**
 * @brief 向量加法 y = alpha * x + y (AXPY)
 */
void vector_axpy_cpu(float alpha, const float* x, float* y, int n);
void vector_axpy_gpu(float alpha, const float* d_x, float* d_y, int n);

/**
 * @brief 向量复制 y = x
 */
void vector_copy_cpu(const float* x, float* y, int n);
void vector_copy_gpu(const float* d_x, float* d_y, int n);

/**
 * @brief 向量缩放加法 y = x + beta * y
 */
void vector_xpay_cpu(const float* x, float beta, float* y, int n);
void vector_xpay_gpu(const float* d_x, float beta, float* d_y, int n);

/**
 * @brief 矩阵-向量乘法 y = A * x
 */
void matvec_multiply_cpu(const float* A, const float* x, float* y, int n);
void matvec_multiply_gpu(const float* d_A, const float* d_x, float* d_y, int n);

/**
 * @brief 向量 L2 范数
 */
float vector_norm_cpu(const float* x, int n);
float vector_norm_gpu(const float* d_x, int n);

#endif // SOLUTION_H
