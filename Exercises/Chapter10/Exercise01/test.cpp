/**
 * 第十章：归约 - 测试程序
 * 
 * 参考：chapter-10/code/reduction_sum_2048.cu, reduction_sum.cu
 * 
 * 测试所有归约实现的正确性
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "solution.h"

const float TOLERANCE = 1e-3f;

// 验证两个结果是否接近
bool is_close(float a, float b, float tolerance = TOLERANCE) {
    float diff = fabs(a - b);
    float rel_tol = tolerance * fabs(b);
    return diff <= fmax(tolerance, rel_tol);
}

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("  第十章：归约\n");
    printf("  Reduction Operations - Multiple Implementations\n");
    printf("  参考: chapter-10/code/reduction_sum_2048.cu, reduction_sum.cu\n");
    printf("================================================================\n\n");

    // 测试参数
    const int small_length = 2 * BLOCK_DIM;  // 2048 元素（单 Block）
    const int large_length = 10000000;        // 1000万元素（多 Block）
    
    printf("配置:\n");
    printf("  BLOCK_DIM: %d\n", BLOCK_DIM);
    printf("  COARSE_FACTOR: %d\n", COARSE_FACTOR);
    printf("  小规模测试: %d 元素\n", small_length);
    printf("  大规模测试: %d 元素\n\n", large_length);

    // ==================== 小规模测试（2048 元素）====================
    printf("=== 小规模测试（单 Block，%d 元素）===\n\n", small_length);

    float* small_data = new float[small_length];
    for (int i = 0; i < small_length; i++) {
        small_data[i] = 1.0f;  // 全1，和应该等于长度
    }

    float expected_small = (float)small_length;

    printf("1. CPU 顺序归约...\n");
    float seq_result = reduction_sequential(small_data, small_length);
    printf("   结果: %.2f (期望 %.2f)\n\n", seq_result, expected_small);

    printf("2. 简单归约 (图10.6)...\n");
    float simple_result = reduction_simple(small_data, small_length);
    if (is_close(simple_result, expected_small)) {
        printf("   结果: %.2f ✅ 正确！\n\n", simple_result);
    } else {
        printf("   结果: %.2f ❌ 不正确（期望 %.2f）\n\n", simple_result, expected_small);
    }

    printf("3. 收敛归约 (图10.9)...\n");
    float conv_result = reduction_convergent(small_data, small_length);
    if (is_close(conv_result, expected_small)) {
        printf("   结果: %.2f ✅ 正确！\n\n", conv_result);
    } else {
        printf("   结果: %.2f ❌ 不正确（期望 %.2f）\n\n", conv_result, expected_small);
    }

    printf("4. 反向收敛归约 (练习3)...\n");
    float rev_result = reduction_convergent_reversed(small_data, small_length);
    if (is_close(rev_result, expected_small)) {
        printf("   结果: %.2f ✅ 正确！\n\n", rev_result);
    } else {
        printf("   结果: %.2f ❌ 不正确（期望 %.2f）\n\n", rev_result, expected_small);
    }

    printf("5. 共享内存归约 (图10.11)...\n");
    float shared_result = reduction_shared_memory(small_data, small_length);
    if (is_close(shared_result, expected_small)) {
        printf("   结果: %.2f ✅ 正确！\n\n", shared_result);
    } else {
        printf("   结果: %.2f ❌ 不正确（期望 %.2f）\n\n", shared_result, expected_small);
    }

    delete[] small_data;

    // ==================== 大规模测试（1000万元素）====================
    printf("=== 大规模测试（多 Block，%d 元素）===\n\n", large_length);

    float* large_data = new float[large_length];
    for (int i = 0; i < large_length; i++) {
        large_data[i] = 1.0f;
    }

    float expected_large = (float)large_length;

    printf("6. CPU 顺序归约...\n");
    float seq_large = reduction_sequential(large_data, large_length);
    printf("   结果: %.2f (期望 %.2f)\n\n", seq_large, expected_large);

    printf("7. 分段归约...\n");
    float seg_result = reduction_segmented(large_data, large_length);
    if (is_close(seg_result, expected_large)) {
        printf("   结果: %.2f ✅ 正确！\n\n", seg_result);
    } else {
        printf("   结果: %.2f ❌ 不正确（期望 %.2f）\n\n", seg_result, expected_large);
    }

    printf("8. 线程粗化归约 (图10.15)...\n");
    float coarse_result = reduction_coarsened(large_data, large_length);
    if (is_close(coarse_result, expected_large)) {
        printf("   结果: %.2f ✅ 正确！\n\n", coarse_result);
    } else {
        printf("   结果: %.2f ❌ 不正确（期望 %.2f）\n\n", coarse_result, expected_large);
    }

    // ==================== 最大值归约测试 ====================
    printf("=== 最大值归约测试 (练习4) ===\n\n");

    // 设置一些随机数据，其中最大值已知
    for (int i = 0; i < large_length; i++) {
        large_data[i] = (float)(i % 1000);  // 0-999
    }
    large_data[large_length / 2] = 9999.0f;  // 设置最大值
    float expected_max = 9999.0f;

    printf("9. CPU 顺序最大值归约...\n");
    float seq_max = max_reduction_sequential(large_data, large_length);
    printf("   结果: %.2f (期望 %.2f)\n\n", seq_max, expected_max);

    printf("10. 粗化最大值归约...\n");
    float gpu_max = max_reduction_coarsened(large_data, large_length);
    if (is_close(gpu_max, expected_max)) {
        printf("   结果: %.2f ✅ 正确！\n\n", gpu_max);
    } else {
        printf("   结果: %.2f ❌ 不正确（期望 %.2f）\n\n", gpu_max, expected_max);
    }

    delete[] large_data;

    printf("【关键概念】\n");
    printf("----------------------------------------------------------------\n");
    printf("• 简单归约：分歧严重，第5次迭代所有16个warp都有分歧\n");
    printf("• 收敛归约：消除分歧，只有前stride个线程活跃\n");
    printf("• 共享内存：减少全局内存访问\n");
    printf("• 分段归约：支持任意长度，使用 atomicAdd\n");
    printf("• 线程粗化：每线程处理多个元素，减少 Block 数量\n");
    printf("\n");

    printf("✅ 测试完成！\n\n");
    return 0;
}
