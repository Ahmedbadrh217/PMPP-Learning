#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 打印所有 CUDA 设备的详细属性
 * 
 * 查询并打印以下信息：
 * - 设备名称和计算能力
 * - 全局内存、常量内存、共享内存大小
 * - SM 数量、每块最大线程数
 * - Warp 大小、寄存器数量
 * - 网格和块的维度限制
 * - 时钟频率和内存带宽
 * - L2 缓存大小
 */
void printDeviceProperties();

#endif // SOLUTION_H
