#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 第十四章：稀疏矩阵计算 - SpMV 实现
 * 
 * 包含多种稀疏矩阵格式和 SpMV 实现：
 * 1. COO (Coordinate) 格式 - 最简单的格式
 * 2. CSR (Compressed Sparse Row) 格式 - 最常用的格式
 * 3. ELL (ELLPACK) 格式 - GPU 友好的规则格式
 * 4. JDS (Jagged Diagonal Storage) 格式 - 练习5
 * 5. ELL-COO Hybrid 格式 - 练习4
 * 6. COO to CSR 转换 - 练习3
 * 
 * SpMV: y = A × x
 */

#include <cuda_runtime.h>

// ====================== 配置常量 ======================

#define BLOCK_SIZE 256

// ====================== 稀疏矩阵结构体 ======================

/**
 * COO 格式：坐标格式
 * 存储每个非零元素的 (行, 列, 值)
 */
struct COOMatrix {
    int numRows;     // 行数
    int numCols;     // 列数
    int nnz;         // 非零元素数
    int* rowIdx;     // 行索引数组 [nnz]
    int* colIdx;     // 列索引数组 [nnz]
    float* values;   // 值数组 [nnz]
};

/**
 * CSR 格式：压缩稀疏行格式
 * 压缩行索引，适合按行访问
 */
struct CSRMatrix {
    int numRows;     // 行数
    int numCols;     // 列数
    int nnz;         // 非零元素数
    int* rowPtrs;    // 行指针数组 [numRows + 1]
    int* colIdx;     // 列索引数组 [nnz]
    float* values;   // 值数组 [nnz]
};

/**
 * ELL 格式：ELLPACK 格式
 * 每行填充到相同长度，列主序存储
 */
struct ELLMatrix {
    int numRows;            // 行数
    int numCols;            // 列数
    int maxNnzPerRow;       // 每行最大非零元素数
    int* colIdx;            // 列索引数组 [numRows × maxNnzPerRow]（列主序）
    float* values;          // 值数组 [numRows × maxNnzPerRow]（列主序）
};

/**
 * JDS 格式：锯齿对角存储格式
 * 按行长度降序排列，减少填充浪费
 */
struct JDSMatrix {
    int numRows;     // 行数
    int numCols;     // 列数
    int numTiles;    // 对角线（迭代）数量
    int* colIdx;     // 列索引数组
    float* values;   // 值数组
    int* rowPerm;    // 行重排映射 [numRows]
    int* iterPtr;    // 每个对角线的起始位置 [numTiles + 1]
};

// ====================== SpMV 函数声明 ======================

/**
 * COO 格式 SpMV
 * 使用原子操作累加结果
 */
void spmv_coo(const COOMatrix& A, const float* x, float* y);

/**
 * CSR 格式 SpMV（每行一个线程）
 */
void spmv_csr(const CSRMatrix& A, const float* x, float* y);

/**
 * ELL 格式 SpMV
 * 列主序存储保证合并访问
 */
void spmv_ell(const ELLMatrix& A, const float* x, float* y);

/**
 * JDS 格式 SpMV（练习5）
 */
void spmv_jds(const JDSMatrix& A, const float* x, float* y);

/**
 * ELL-COO Hybrid 格式 SpMV（练习4）
 * ELL 处理规则部分，COO 处理溢出部分
 */
void spmv_hybrid(const ELLMatrix& ellPart, const COOMatrix& cooPart, 
                 const float* x, float* y);

// ====================== 格式转换函数 ======================

/**
 * COO 到 CSR 转换（练习3）
 * 使用并行直方图和前缀和
 */
void coo_to_csr(const COOMatrix& coo, CSRMatrix& csr);

// ====================== CPU 参考实现 ======================

/**
 * CPU 密集矩阵-向量乘法（用于验证）
 */
void spmv_dense_cpu(int m, int n, const float* A, const float* x, float* y);

#endif // SOLUTION_H
