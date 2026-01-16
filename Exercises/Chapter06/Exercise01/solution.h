#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 打印矩阵（调试用）
 */
void printMatrix(float* matrix, int rows, int cols);

/**
 * 原地矩阵转置
 * 注意：仅适用于小矩阵演示
 */
void inPlaceMatrixTranspose(float* h_M, int m, int n);

/**
 * 行主序 Tiled 矩阵乘法
 * 标准的 Tiled 实现
 */
void matrixMulTiledRowMajor(float* h_P, const float* h_M, const float* h_N, int m, int n, int o);

/**
 * 列主序 Tiled 矩阵乘法（Corner Turning）
 * N 矩阵以列主序存储，使用角转换技术保持合并访问
 * 
 * 关键思想：
 * - 不按行访问 N（会导致非合并访问）
 * - 按列访问 N（合并访问）
 * - 共享内存用于重排数据
 */
void matrixMulTiledColMajor(float* h_P, const float* h_M, const float* h_N_transposed, 
                             int m, int n, int o);

#endif // SOLUTION_H
