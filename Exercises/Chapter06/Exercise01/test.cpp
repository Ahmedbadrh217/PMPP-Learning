/**
 * 第六章：性能方面的考虑 - 列主序矩阵乘法测试
 * 
 * 参考：chapter-06/code/excercise1.cu
 * 
 * 本程序对比行主序和列主序矩阵乘法，演示 Corner Turning 技术
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "solution.h"

const float TOLERANCE = 1e-3f;

void initMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(i);
    }
}

bool verifyResults(const float* A, const float* B, int size, float tolerance = TOLERANCE) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > tolerance) {
            printf("Mismatch at index %d: %.6f vs %.6f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("\n");
    printf("================================================\n");
    printf("  第六章：性能方面的考虑\n");
    printf("  Column-Major Matrix Multiplication\n");
    printf("  (Corner Turning Technique)\n");
    printf("  参考: chapter-06/code/excercise1.cu\n");
    printf("================================================\n\n");

    // 使用小矩阵便于验证
    int m = 4;
    int n = 3;
    int o = 2;

    printf("矩阵大小: M(%d×%d) × N(%d×%d) = P(%d×%d)\n\n", m, n, n, o, m, o);

    // 分配内存
    float* M = new float[m * n];
    float* N = new float[n * o];
    float* N_transposed = new float[o * n];
    float* P1 = new float[m * o];
    float* P2 = new float[m * o];

    // 初始化
    initMatrix(M, m * n);
    for (int i = 0; i < n * o; ++i) {
        N[i] = i + 1;
    }
    memcpy(N_transposed, N, n * o * sizeof(float));

    // 打印输入矩阵
    printf("M:\n");
    printMatrix(M, m, n);
    printf("\n");

    printf("N:\n");
    printMatrix(N, n, o);
    printf("\n");

    // 转置 N
    inPlaceMatrixTranspose(N_transposed, n, o);
    printf("N transposed (列主序存储):\n");
    printMatrix(N_transposed, o, n);
    printf("\n");

    // 行主序矩阵乘法
    printf("=== 行主序矩阵乘法 ===\n");
    matrixMulTiledRowMajor(P1, M, N, m, n, o);
    printMatrix(P1, m, o);
    printf("\n");

    // 列主序矩阵乘法（Corner Turning）
    printf("=== 列主序矩阵乘法 (Corner Turning) ===\n");
    matrixMulTiledColMajor(P2, M, N_transposed, m, n, o);
    printMatrix(P2, m, o);
    printf("\n");

    // 验证
    if (verifyResults(P1, P2, m * o)) {
        printf("✅ 两种方法结果一致！\n\n");
    } else {
        printf("❌ 结果不一致！\n\n");
    }

    printf("【关键概念】\n");
    printf("------------------------------------------------\n");
    printf("• Corner Turning: 改变访问模式以保持合并访问\n");
    printf("• 列主序存储: N[col][row] = N[col * n + row]\n");
    printf("• 相邻线程访问相邻内存地址 → 合并访问\n");
    printf("• 共享内存用于重排数据进行计算\n");
    printf("\n");

    delete[] M;
    delete[] N;
    delete[] N_transposed;
    delete[] P1;
    delete[] P2;

    printf("✅ 测试完成！\n\n");
    return 0;
}
