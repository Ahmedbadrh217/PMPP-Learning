/**
 * 第七章：卷积 - 2D卷积测试
 * 
 * 参考：chapter-07/code/run_conv2d.cu
 * 
 * 测试朴素卷积和常量内存卷积的正确性和性能
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "solution.h"

const float TOLERANCE = 1e-4f;

/**
 * 初始化矩阵为随机值
 */
void initMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

/**
 * 初始化高斯滤波器
 */
void initGaussianFilter(float* filter, int r) {
    int size = 2 * r + 1;
    float sigma = r / 2.0f;
    float sum = 0.0f;
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float x = i - r;
            float y = j - r;
            float value = expf(-(x*x + y*y) / (2 * sigma * sigma));
            filter[i * size + j] = value;
            sum += value;
        }
    }
    
    // 归一化
    for (int i = 0; i < size * size; ++i) {
        filter[i] /= sum;
    }
}

/**
 * 验证结果
 */
bool verifyResults(const float* A, const float* B, int size, float tolerance = TOLERANCE) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > tolerance) {
            printf("Mismatch at index %d: %.6f vs %.6f (diff: %.6f)\n", 
                   i, A[i], B[i], fabs(A[i] - B[i]));
            return false;
        }
    }
    return true;
}

int main() {
    printf("\n");
    printf("================================================\n");
    printf("  第七章：卷积\n");
    printf("  2D Convolution - Basic & Constant Memory\n");
    printf("  参考: chapter-07/code/conv2d_kernels.cu\n");
    printf("================================================\n\n");
    
    // 测试参数
    int height = 64;
    int width = 64;
    int r = FILTER_RADIUS;
    int filterSize = 2 * r + 1;
    
    printf("矩阵大小: %d × %d\n", height, width);
    printf("滤波器大小: %d × %d (半径=%d)\n\n", filterSize, filterSize, r);
    
    // 分配内存
    float* N = new float[height * width];
    float* F = new float[filterSize * filterSize];
    float* P_cpu = new float[height * width];
    float* P_basic = new float[height * width];
    float* P_const = new float[height * width];
    
    // 初始化
    srand(42);
    initMatrix(N, height * width);
    initGaussianFilter(F, r);
    
    printf("高斯滤波器:\n");
    printMatrix(F, filterSize, filterSize);
    printf("\n");
    
    // === CPU 参考实现 ===
    printf("=== CPU 参考实现 ===\n");
    conv2d_cpu(N, F, P_cpu, r, height, width);
    printf("完成\n\n");
    
    // === 朴素 GPU 实现 ===
    printf("=== 朴素 GPU 实现 ===\n");
    conv2d_basic(N, F, P_basic, r, height, width);
    if (verifyResults(P_cpu, P_basic, height * width)) {
        printf("✅ 结果正确！\n\n");
    } else {
        printf("❌ 结果不正确！\n\n");
    }
    
    // === 常量内存 GPU 实现 ===
    printf("=== 常量内存 GPU 实现 ===\n");
    conv2d_const_memory(N, F, P_const, r, height, width);
    if (verifyResults(P_cpu, P_const, height * width)) {
        printf("✅ 结果正确！\n\n");
    } else {
        printf("❌ 结果不正确！\n\n");
    }
    
    // 小矩阵演示
    printf("================================================\n");
    printf("  小矩阵演示\n");
    printf("================================================\n\n");
    
    int demo_size = 4;
    float demo_N[16] = {1, 1, 1, 1, 
                        1, 1, 1, 1, 
                        1, 1, 1, 1, 
                        1, 1, 1, 1};
    float demo_F[9] = {1/16.f, 2/16.f, 1/16.f, 
                       2/16.f, 4/16.f, 2/16.f, 
                       1/16.f, 2/16.f, 1/16.f};
    float demo_P[16] = {0};
    
    printf("输入矩阵 (4×4 全1):\n");
    printMatrix(demo_N, demo_size, demo_size);
    printf("\n高斯滤波器 (3×3):\n");
    printMatrix(demo_F, 3, 3);
    
    conv2d_const_memory(demo_N, demo_F, demo_P, 1, demo_size, demo_size);
    
    printf("\n卷积结果:\n");
    printMatrix(demo_P, demo_size, demo_size);
    printf("\n");
    
    printf("【关键概念】\n");
    printf("------------------------------------------------\n");
    printf("• 朴素卷积：滤波器从全局内存读取\n");
    printf("• 常量内存：滤波器缓存在常量内存中\n");
    printf("• 常量内存优势：所有线程访问相同位置时广播\n");
    printf("• Ghost cells：边界外的元素视为 0\n");
    printf("\n");
    
    // 释放内存
    delete[] N;
    delete[] F;
    delete[] P_cpu;
    delete[] P_basic;
    delete[] P_const;
    
    printf("✅ 测试完成！\n\n");
    return 0;
}
