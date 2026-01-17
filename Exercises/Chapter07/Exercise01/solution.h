#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 第七章：卷积 - 2D卷积实现
 * 
 * 包含朴素版本和常量内存版本的2D卷积
 */

// 卷积参数定义
#define FILTER_RADIUS 1
#define BLOCK_SIZE 16

/**
 * 朴素2D卷积
 * @param h_N 输入矩阵
 * @param h_F 滤波器（大小为 (2*r+1) x (2*r+1)）
 * @param h_P 输出矩阵
 * @param r 滤波器半径
 * @param height 矩阵高度
 * @param width 矩阵宽度
 */
void conv2d_basic(float* h_N, float* h_F, float* h_P, int r, int height, int width);

/**
 * 使用常量内存的2D卷积
 * 滤波器存储在常量内存中，利用缓存加速访问
 */
void conv2d_const_memory(float* h_N, float* h_F, float* h_P, int r, int height, int width);

/**
 * CPU参考实现
 */
void conv2d_cpu(float* N, float* F, float* P, int r, int height, int width);

/**
 * 打印矩阵（调试用）
 */
void printMatrix(float* matrix, int rows, int cols);

#endif // SOLUTION_H
