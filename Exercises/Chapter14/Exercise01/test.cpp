/**
 * 第十四章：稀疏矩阵计算 - 测试程序
 * 
 * 参考：chapter-14/code/
 * 
 * 测试多种稀疏格式的 SpMV 实现
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "solution.h"

const float TOLERANCE = 1e-4f;

// 验证两个数组是否接近
bool allclose(const float* a, const float* b, int N, float tolerance = TOLERANCE) {
    for (int i = 0; i < N; i++) {
        float diff = fabs(a[i] - b[i]);
        float ref = fabs(b[i]) + 1e-6f;
        if (diff > tolerance * ref && diff > tolerance) {
            printf("  差异在索引 %d: %.4f vs %.4f (差: %.6f)\n", i, a[i], b[i], diff);
            return false;
        }
    }
    return true;
}

// 打印向量
void printVector(const char* name, const float* v, int n) {
    printf("%s: [", name);
    for (int i = 0; i < n; i++) {
        printf("%.2f%s", v[i], i < n - 1 ? ", " : "");
    }
    printf("]\n");
}

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("  第十四章：稀疏矩阵计算\n");
    printf("  Sparse Matrix-Vector Multiplication (SpMV)\n");
    printf("  参考: chapter-14/code/\n");
    printf("================================================================\n\n");

    // 测试矩阵（书中图14.2示例）
    // [1, 7, 0, 0]
    // [5, 0, 3, 9]
    // [0, 2, 8, 0]
    // [0, 0, 0, 6]
    
    const int numRows = 4;
    const int numCols = 4;
    const int nnz = 8;
    
    // COO 格式数据
    int h_coo_rowIdx[] = {0, 0, 1, 1, 1, 2, 2, 3};
    int h_coo_colIdx[] = {0, 1, 0, 2, 3, 1, 2, 3};
    float h_coo_values[] = {1, 7, 5, 3, 9, 2, 8, 6};
    
    // CSR 格式数据
    int h_csr_rowPtrs[] = {0, 2, 5, 7, 8};
    
    // ELL 格式数据（每行最多3个非零元素，列主序）
    // 行0: [1, 7, *], 行1: [5, 3, 9], 行2: [2, 8, *], 行3: [6, *, *]
    // 列主序: col0=[0,0,1,3], col1=[1,2,2,-1], col2=[-1,3,-1,-1]
    int h_ell_colIdx[] = {0, 0, 1, 3,    // t=0
                          1, 2, 2, -1,   // t=1
                          -1, 3, -1, -1}; // t=2
    float h_ell_values[] = {1, 5, 2, 6,   // t=0
                            7, 3, 8, 0,   // t=1
                            0, 9, 0, 0};  // t=2
    const int maxNnzPerRow = 3;
    
    // 输入向量
    float h_x[] = {1, 2, 3, 4};
    
    // 预期结果
    // y[0] = 1*1 + 7*2 = 15
    // y[1] = 5*1 + 3*3 + 9*4 = 50
    // y[2] = 2*2 + 8*3 = 28
    // y[3] = 6*4 = 24
    float h_y_expected[] = {15, 50, 28, 24};
    
    float h_y[numRows];
    
    printf("配置:\n");
    printf("  矩阵大小: %d × %d\n", numRows, numCols);
    printf("  非零元素: %d\n", nnz);
    printf("  稀疏度: %.1f%%\n", (1.0f - (float)nnz / (numRows * numCols)) * 100);
    printVector("输入向量 x", h_x, numCols);
    printVector("预期结果 y", h_y_expected, numRows);
    printf("\n");
    
    printf("=== 正确性验证 ===\n\n");
    
    // 分配设备内存
    int *d_coo_rowIdx, *d_coo_colIdx;
    float *d_coo_values, *d_x, *d_y;
    
    cudaMalloc(&d_coo_rowIdx, nnz * sizeof(int));
    cudaMalloc(&d_coo_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_coo_values, nnz * sizeof(float));
    cudaMalloc(&d_x, numCols * sizeof(float));
    cudaMalloc(&d_y, numRows * sizeof(float));
    
    cudaMemcpy(d_coo_rowIdx, h_coo_rowIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coo_colIdx, h_coo_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coo_values, h_coo_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, numCols * sizeof(float), cudaMemcpyHostToDevice);
    
    // 1. COO SpMV
    printf("1. COO SpMV...\n");
    {
        COOMatrix coo = {numRows, numCols, nnz, d_coo_rowIdx, d_coo_colIdx, d_coo_values};
        spmv_coo(coo, d_x, d_y);
        cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);
        
        if (allclose(h_y, h_y_expected, numRows)) {
            printf("   ✅ 结果正确！");
            printVector(" 输出", h_y, numRows);
        } else {
            printf("   ❌ 结果错误！\n");
        }
        printf("\n");
    }
    
    // 2. CSR SpMV
    printf("2. CSR SpMV...\n");
    {
        int* d_csr_rowPtrs;
        cudaMalloc(&d_csr_rowPtrs, (numRows + 1) * sizeof(int));
        cudaMemcpy(d_csr_rowPtrs, h_csr_rowPtrs, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        
        CSRMatrix csr = {numRows, numCols, nnz, d_csr_rowPtrs, d_coo_colIdx, d_coo_values};
        spmv_csr(csr, d_x, d_y);
        cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);
        
        if (allclose(h_y, h_y_expected, numRows)) {
            printf("   ✅ 结果正确！");
            printVector(" 输出", h_y, numRows);
        } else {
            printf("   ❌ 结果错误！\n");
        }
        printf("\n");
        
        cudaFree(d_csr_rowPtrs);
    }
    
    // 3. ELL SpMV
    printf("3. ELL SpMV...\n");
    {
        int* d_ell_colIdx;
        float* d_ell_values;
        int ellSize = numRows * maxNnzPerRow;
        
        cudaMalloc(&d_ell_colIdx, ellSize * sizeof(int));
        cudaMalloc(&d_ell_values, ellSize * sizeof(float));
        cudaMemcpy(d_ell_colIdx, h_ell_colIdx, ellSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ell_values, h_ell_values, ellSize * sizeof(float), cudaMemcpyHostToDevice);
        
        ELLMatrix ell = {numRows, numCols, maxNnzPerRow, d_ell_colIdx, d_ell_values};
        spmv_ell(ell, d_x, d_y);
        cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);
        
        if (allclose(h_y, h_y_expected, numRows)) {
            printf("   ✅ 结果正确！");
            printVector(" 输出", h_y, numRows);
        } else {
            printf("   ❌ 结果错误！\n");
        }
        printf("\n");
        
        cudaFree(d_ell_colIdx);
        cudaFree(d_ell_values);
    }
    
    // 4. COO to CSR 转换（练习3）
    printf("4. COO to CSR 转换 (练习3)...\n");
    {
        COOMatrix coo = {numRows, numCols, nnz, d_coo_rowIdx, d_coo_colIdx, d_coo_values};
        CSRMatrix csr;
        
        coo_to_csr(coo, csr);
        
        // 用转换后的 CSR 验证
        spmv_csr(csr, d_x, d_y);
        cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);
        
        if (allclose(h_y, h_y_expected, numRows)) {
            printf("   ✅ 转换正确，SpMV 结果匹配！\n");
        } else {
            printf("   ❌ 转换或 SpMV 结果错误！\n");
        }
        printf("\n");
        
        cudaFree(csr.rowPtrs);
        cudaFree(csr.colIdx);
        cudaFree(csr.values);
    }
    
    // 5. JDS SpMV（练习5）
    printf("5. JDS SpMV (练习5)...\n");
    {
        // JDS 格式：按行长度降序排列
        // 原矩阵行长度: 行0=2, 行1=3, 行2=2, 行3=1
        // 排序后顺序: 行1(3), 行0(2), 行2(2), 行3(1)
        // rowPerm 映射: [1, 0, 2, 3] (排序后位置 -> 原始行号)
        
        // JDS 数据（按排序后的行）
        // 迭代0: 所有行第1个元素 [5, 1, 2, 6] -> col [0, 0, 1, 3]
        // 迭代1: 前3行第2个元素 [3, 7, 8] -> col [2, 1, 2]
        // 迭代2: 第1行第3个元素 [9] -> col [3]
        int h_jds_colIdx[] = {0, 0, 1, 3,   // 迭代0
                              2, 1, 2,       // 迭代1
                              3};            // 迭代2
        float h_jds_values[] = {5, 1, 2, 6,  // 迭代0
                                3, 7, 8,     // 迭代1
                                9};          // 迭代2
        int h_jds_rowPerm[] = {1, 0, 2, 3};  // 排序后位置 -> 原始行号
        int h_jds_iterPtr[] = {0, 4, 7, 8};  // 每个迭代的起始位置
        int numTiles = 3;
        
        int* d_jds_colIdx;
        float* d_jds_values;
        int* d_jds_rowPerm;
        int* d_jds_iterPtr;
        
        cudaMalloc(&d_jds_colIdx, 8 * sizeof(int));
        cudaMalloc(&d_jds_values, 8 * sizeof(float));
        cudaMalloc(&d_jds_rowPerm, numRows * sizeof(int));
        cudaMalloc(&d_jds_iterPtr, (numTiles + 1) * sizeof(int));
        
        cudaMemcpy(d_jds_colIdx, h_jds_colIdx, 8 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jds_values, h_jds_values, 8 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jds_rowPerm, h_jds_rowPerm, numRows * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jds_iterPtr, h_jds_iterPtr, (numTiles + 1) * sizeof(int), cudaMemcpyHostToDevice);
        
        JDSMatrix jds = {numRows, numCols, numTiles, d_jds_colIdx, d_jds_values, d_jds_rowPerm, d_jds_iterPtr};
        spmv_jds(jds, d_x, d_y);
        cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);
        
        if (allclose(h_y, h_y_expected, numRows)) {
            printf("   ✅ 结果正确！");
            printVector(" 输出", h_y, numRows);
        } else {
            printf("   ❌ 结果错误！\n");
            printVector("   实际输出", h_y, numRows);
        }
        printf("\n");
        
        cudaFree(d_jds_colIdx);
        cudaFree(d_jds_values);
        cudaFree(d_jds_rowPerm);
        cudaFree(d_jds_iterPtr);
    }
    
    // 6. ELL-COO Hybrid SpMV（练习4）
    printf("6. ELL-COO Hybrid SpMV (练习4)...\n");
    {
        // ELL 部分：每行最多2个元素
        // 行0: [1, 7], 行1: [5, 3], 行2: [2, 8], 行3: [6, *]
        int h_ell2_colIdx[] = {0, 0, 1, 3,    // t=0
                               1, 2, 2, -1};   // t=1
        float h_ell2_values[] = {1, 5, 2, 6,   // t=0
                                 7, 3, 8, 0};  // t=1
        const int maxNnzPerRow2 = 2;
        
        // COO 溢出部分：行1的第3个元素 9
        int h_coo_overflow_rowIdx[] = {1};
        int h_coo_overflow_colIdx[] = {3};
        float h_coo_overflow_values[] = {9};
        
        int* d_ell2_colIdx;
        float* d_ell2_values;
        
        cudaMalloc(&d_ell2_colIdx, numRows * maxNnzPerRow2 * sizeof(int));
        cudaMalloc(&d_ell2_values, numRows * maxNnzPerRow2 * sizeof(float));
        cudaMemcpy(d_ell2_colIdx, h_ell2_colIdx, numRows * maxNnzPerRow2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ell2_values, h_ell2_values, numRows * maxNnzPerRow2 * sizeof(float), cudaMemcpyHostToDevice);
        
        ELLMatrix ellPart = {numRows, numCols, maxNnzPerRow2, d_ell2_colIdx, d_ell2_values};
        
        // COO 部分存储在主机上（Hybrid 模式下 COO 部分在 CPU 执行）
        COOMatrix cooPart;
        cooPart.numRows = numRows;
        cooPart.numCols = numCols;
        cooPart.nnz = 1;
        cooPart.rowIdx = h_coo_overflow_rowIdx;  // 主机指针
        cooPart.colIdx = h_coo_overflow_colIdx;
        cooPart.values = h_coo_overflow_values;
        
        spmv_hybrid(ellPart, cooPart, d_x, d_y);
        cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);
        
        if (allclose(h_y, h_y_expected, numRows)) {
            printf("   ✅ 结果正确！");
            printVector(" 输出", h_y, numRows);
        } else {
            printf("   ❌ 结果错误！\n");
            printVector("   实际输出", h_y, numRows);
        }
        printf("\n");
        
        cudaFree(d_ell2_colIdx);
        cudaFree(d_ell2_values);
    }
    
    printf("【关键概念】\n");
    printf("----------------------------------------------------------------\n");
    printf("• COO: 最简单，(行,列,值) 三元组，适合构建阶段\n");
    printf("• CSR: 最常用，压缩行指针，SpMV 按行并行\n");
    printf("• ELL: 列主序填充，适合规则稀疏矩阵，GPU 友好\n");
    printf("• JDS: 按行长度排序，减少填充浪费\n");
    printf("• Hybrid: ELL 处理规则部分，COO 处理溢出\n");
    printf("\n");
    
    // 清理
    cudaFree(d_coo_rowIdx);
    cudaFree(d_coo_colIdx);
    cudaFree(d_coo_values);
    cudaFree(d_x);
    cudaFree(d_y);
    
    printf("✅ 测试完成！\n\n");
    return 0;
}
