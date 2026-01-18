#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 第十章：归约 - 并行归约实现
 * 
 * 包含多种归约实现：
 * 1. 顺序实现（CPU参考）
 * 2. 简单归约（图10.6）- 分歧严重
 * 3. 收敛归约（图10.9）- 优化分歧
 * 4. 反向收敛归约（练习3）
 * 5. 共享内存归约（图10.11）
 * 6. 分段归约 - 支持任意长度
 * 7. 线程粗化归约（图10.15）
 * 
 * 另外包含最大值归约（练习4）
 */

// Block 尺寸和粗化因子
#define BLOCK_DIM 1024
#define COARSE_FACTOR 2

// ====================== 求和归约 ======================

/**
 * CPU 顺序归约（参考）
 */
float reduction_sequential(float* data, int length);

/**
 * 简单归约实现（图10.6）
 * 限制：仅支持 2048 元素
 * 问题：严重的控制分歧
 */
float reduction_simple(float* data, int length);

/**
 * 收敛归约实现（图10.9）
 * 限制：仅支持 2048 元素
 * 优化：消除控制分歧
 */
float reduction_convergent(float* data, int length);

/**
 * 反向收敛归约实现（练习3）
 * 限制：仅支持 2048 元素
 * 从右向左收敛
 */
float reduction_convergent_reversed(float* data, int length);

/**
 * 共享内存归约实现（图10.11）
 * 限制：仅支持 2048 元素
 * 优化：使用共享内存
 */
float reduction_shared_memory(float* data, int length);

/**
 * 分段归约实现
 * 支持任意长度，使用 atomicAdd 合并结果
 */
float reduction_segmented(float* data, int length);

/**
 * 线程粗化归约实现（图10.15）
 * 支持任意长度，每个线程处理多个元素
 */
float reduction_coarsened(float* data, int length);

// ====================== 最大值归约 ======================

/**
 * CPU 顺序最大值归约（参考）
 */
float max_reduction_sequential(float* data, int length);

/**
 * 最大值归约（练习4）
 * 使用粗化策略，支持任意长度
 */
float max_reduction_coarsened(float* data, int length);

#endif // SOLUTION_H
