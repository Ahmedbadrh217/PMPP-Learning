#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 第十三章：排序 - 并行排序实现
 * 
 * 包含多种排序实现：
 * 1. 朴素三核基数排序（提取位、扫描、分散）
 * 2. 内存合并基数排序（练习1：共享内存优化）
 * 3. 多位基数排序（练习2：多位处理）
 * 4. 线程粗化基数排序（练习3：每线程处理多元素）
 * 5. 并行归并排序（练习4：使用第12章归并）
 * 
 * 核心算法：
 * - 基数排序：按位从低到高排序，利用前缀和确定输出位置
 * - 归并排序：Block内排序 + 跨Block归并
 */

#include <cuda_runtime.h>

// ====================== 配置常量 ======================

#define BLOCK_SIZE 1024       // 线程块大小
#define RADIX_BITS 4          // 多位基数：每次处理的位数
#define RADIX (1 << RADIX_BITS)  // 桶数量 (16)
#define COARSE_FACTOR 4       // 线程粗化因子

// ====================== 排序函数声明 ======================

/**
 * 1. 朴素并行基数排序（三核实现）
 * 书中图13.4基础实现
 * 
 * 流程：
 *   对于每一位（0-31）：
 *     1. 提取位 kernel
 *     2. 前缀和（使用 Thrust）
 *     3. 分散 kernel
 * 
 * @param d_input 设备端输入/输出数组
 * @param N 数组长度
 */
void gpuRadixSortNaive(unsigned int* d_input, int N);

/**
 * 2. 内存合并基数排序
 * 练习1：使用共享内存优化，实现合并写入
 * 
 * 优化点：
 *   - Block 内使用共享内存进行本地扫描
 *   - 分开计算0和1的偏移，实现合并写入
 * 
 * @param d_input 设备端输入/输出数组
 * @param N 数组长度
 */
void gpuRadixSortCoalesced(unsigned int* d_input, int N);

/**
 * 3. 多位基数排序
 * 练习2：同时处理多位（如4位），减少迭代次数
 * 
 * 优化点：
 *   - 32位整数只需 32/4=8 轮迭代
 *   - 使用直方图统计桶计数
 * 
 * @param d_input 设备端输入/输出数组
 * @param N 数组长度
 * @param r 每次处理的位数（默认4）
 */
void gpuRadixSortMultibit(unsigned int* d_input, int N, unsigned int r = RADIX_BITS);

/**
 * 4. 线程粗化基数排序
 * 练习3：每个线程处理多个元素
 * 
 * 优化点：
 *   - 减少线程数量，增加每线程工作量
 *   - 更好的寄存器利用率
 * 
 * @param d_input 设备端输入/输出数组
 * @param N 数组长度
 * @param r 每次处理的位数
 */
void gpuRadixSortCoarsened(unsigned int* d_input, int N, unsigned int r = RADIX_BITS);

/**
 * 5. 并行归并排序
 * 练习4：使用第12章的并行归并实现
 * 
 * 流程：
 *   1. Block 内排序（每个 Block 独立排序一段）
 *   2. 跨 Block 归并（多轮归并相邻有序段）
 * 
 * @param d_input 设备端输入/输出数组
 * @param N 数组长度
 */
void gpuMergeSort(unsigned int* d_input, int N);

// ====================== CPU 参考实现 ======================

/**
 * CPU 快速排序（使用 qsort）
 * 用于性能对比
 */
void cpuSort(unsigned int* arr, int N);

#endif // SOLUTION_H
