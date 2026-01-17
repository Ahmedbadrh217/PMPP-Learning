#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 第七章：卷积 - Tiled 2D卷积实现
 * 
 * 包含 Tiled 版本和利用 L2 缓存的版本
 */

// 卷积参数定义
#define FILTER_RADIUS 1
#define BLOCK_SIZE 16
#define TILE_SIZE 16

// Tiled 卷积参数
#define IN_TILE_SIZE 32
#define OUT_TILE_SIZE (IN_TILE_SIZE - 2 * FILTER_RADIUS)

/**
 * Tiled 2D卷积（图7.12）
 * 使用共享内存存储输入 tile
 */
void conv2d_tiled(float* h_N, float* h_F, float* h_P, int r, int height, int width);

/**
 * Tiled 2D卷积 + L2 缓存利用（图7.15）
 * 内部 tile 使用共享内存，边界从 L2 缓存读取
 */
void conv2d_tiled_cached(float* h_N, float* h_F, float* h_P, int r, int height, int width);

/**
 * CPU参考实现
 */
void conv2d_cpu(float* N, float* F, float* P, int r, int height, int width);

/**
 * 打印矩阵（调试用）
 */
void printMatrix(float* matrix, int rows, int cols);

#endif // SOLUTION_H
