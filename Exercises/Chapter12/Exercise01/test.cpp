/**
 * 第十二章：归并 - 测试程序
 * 
 * 参考：chapter-12/code/merge.cu
 * 
 * 测试归并实现的正确性
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "solution.h"

const float TOLERANCE = 1e-5f;

// 验证两个数组是否接近
bool allclose(float* a, float* b, int N, float tolerance = TOLERANCE) {
    for (int i = 0; i < N; i++) {
        float allowed_error = tolerance + tolerance * fabs(b[i]);
        if (fabs(a[i] - b[i]) > allowed_error) {
            printf("  数组在索引 %d 处不匹配: %.2f vs %.2f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

// 创建有序数组
float* createSortedArray(int length, float start, float step) {
    float* array = new float[length];
    for (int i = 0; i < length; i++) {
        array[i] = start + i * step;
    }
    return array;
}

// 验证数组是否有序
bool isSorted(float* array, int length) {
    for (int i = 1; i < length; i++) {
        if (array[i] < array[i - 1]) {
            printf("  数组在索引 %d 处无序: %.2f > %.2f\n", i - 1, array[i - 1], array[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("  第十二章：归并\n");
    printf("  Merge Operations - Multiple Implementations\n");
    printf("  参考: chapter-12/code/merge.cu\n");
    printf("================================================================\n\n");

    // 测试参数
    const int m = 10283;
    const int n = 131131;
    const int total = m + n;

    printf("配置:\n");
    printf("  数组 A 长度: %d\n", m);
    printf("  数组 B 长度: %d\n", n);
    printf("  合并结果长度: %d\n", total);
    printf("  TILE_SIZE: %d\n\n", TILE_SIZE);

    // 创建有序数组
    float* A = createSortedArray(m, 1.0f, 0.3f);
    float* B = createSortedArray(n, 1.5f, 0.4f);
    float* C_ref = new float[total];
    float* C_basic = new float[total];
    float* C_tiled = new float[total];

    printf("=== 正确性验证 ===\n\n");

    // 1. CPU 顺序归并（参考）
    printf("1. CPU 顺序归并...\n");
    merge_sequential(A, m, B, n, C_ref);
    if (isSorted(C_ref, total)) {
        printf("   ✅ 结果有序\n\n");
    } else {
        printf("   ❌ 结果无序\n\n");
    }

    // 2. 基础并行归并
    printf("2. 基础并行归并 (图12.9)...\n");
    merge_basic_gpu(A, m, B, n, C_basic);
    if (allclose(C_basic, C_ref, total)) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }

    // 3. 分块并行归并
    printf("3. 分块并行归并 (图12.11-12.13)...\n");
    merge_tiled_gpu(A, m, B, n, C_tiled);
    if (allclose(C_tiled, C_ref, total)) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }

    // 显示部分结果
    printf("=== 合并结果示例 ===\n");
    printf("前10个元素: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", C_ref[i]);
    }
    printf("...\n");
    printf("后10个元素: ...");
    for (int i = total - 10; i < total; i++) {
        printf(" %.1f", C_ref[i]);
    }
    printf("\n\n");

    printf("【关键概念】\n");
    printf("----------------------------------------------------------------\n");
    printf("• co-rank：二分搜索找到归并位置，O(log n) 复杂度\n");
    printf("• 基础并行：每个线程独立计算 co-rank，全局内存访问\n");
    printf("• 分块归并：使用共享内存，Block 级别只需2次全局 co-rank\n");
    printf("• 工作分配：通过 co-rank 实现负载均衡的归并分区\n");
    printf("\n");

    // 清理
    delete[] A;
    delete[] B;
    delete[] C_ref;
    delete[] C_basic;
    delete[] C_tiled;

    printf("✅ 测试完成！\n\n");
    return 0;
}
