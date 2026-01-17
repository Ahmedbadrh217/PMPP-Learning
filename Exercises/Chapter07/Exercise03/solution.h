#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 第七章：卷积 - 3D卷积实现
 * 
 * 练习 8-10 的实现：
 * - 练习 8：基础3D卷积（扩展图7.7）
 * - 练习 9：常量内存3D卷积（扩展图7.9）
 * - 练习 10：Tiled 3D卷积（扩展图7.12）
 */

// 卷积参数
#define FILTER_RADIUS 1
#define BLOCK_SIZE 8  // 3D 需要更小的 block

// Tiled 3D 参数
#define IN_TILE_DIM 8
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

/**
 * 练习 8：基础3D卷积（图7.7扩展到3D）
 */
void conv3d_basic(float* h_N, float* h_F, float* h_P, int r, 
                  int width, int height, int depth);

/**
 * 练习 9：常量内存3D卷积（图7.9扩展到3D）
 */
void conv3d_const_memory(float* h_N, float* h_F, float* h_P, int r,
                         int width, int height, int depth);

/**
 * 练习 10：Tiled 3D卷积（图7.12扩展到3D）
 */
void conv3d_tiled(float* h_N, float* h_F, float* h_P, int r,
                  int width, int height, int depth);

/**
 * CPU 参考实现
 */
void conv3d_cpu(float* N, float* F, float* P, int r,
                int width, int height, int depth);

/**
 * 打印3D矩阵切片
 */
void print3DSlice(float* matrix, int width, int height, int depth, int slice);

#endif // SOLUTION_H
