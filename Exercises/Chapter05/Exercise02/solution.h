#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 计算最优 Tile 宽度
 * 基于硬件规格动态计算，而不是硬编码
 * 
 * @param m 矩阵 M 的行数
 * @param n 矩阵 M 的列数 / 矩阵 N 的行数
 * @param o 矩阵 N 的列数
 * @return 最优 Tile 宽度
 */
int calculateOptimalTileWidth(int m, int n, int o);

/**
 * 朴素矩阵乘法（无优化）
 */
void matrixMul(float* h_P, const float* h_M, const float* h_N, int m, int n, int o);

/**
 * 动态 Tile 大小的 Tiled 矩阵乘法
 * 使用动态共享内存，Tile 大小根据硬件规格计算
 */
void matrixMulTilingDynamic(float* h_P, const float* h_M, const float* h_N, int m, int n, int o);

#endif // SOLUTION_H
