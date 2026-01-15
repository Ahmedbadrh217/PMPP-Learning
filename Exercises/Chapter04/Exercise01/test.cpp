/**
 * 第四章：计算架构和调度 - 设备属性查询测试
 * 
 * 本程序演示如何查询 CUDA 设备的硬件属性。
 * 了解这些属性对于优化 kernel 性能至关重要。
 */

#include <cstdio>
#include "solution.h"

int main() {
    printf("\n");
    printf("========================================\n");
    printf("  第四章：计算架构和调度\n");
    printf("  Device Properties Query\n");
    printf("========================================\n\n");

    // 打印设备属性
    printDeviceProperties();

    printf("✅ 设备属性查询完成！\n\n");

    // 打印关键概念提示
    printf("【关键概念回顾】\n");
    printf("-");
    for (int i = 0; i < 40; i++) printf("-");
    printf("\n");
    printf("  • Warp 大小: 32 个线程为一组，是 GPU 执行的基本单位\n");
    printf("  • SM (流式多处理器): GPU 的独立执行单元\n");
    printf("  • 占用率 = 活跃 Warp 数 / SM 最大 Warp 数\n");
    printf("  • 块大小建议: 128-256，且为 32 的倍数\n");
    printf("\n");

    return 0;
}
