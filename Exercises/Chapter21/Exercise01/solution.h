#ifndef CHAPTER_21_EXERCISE01_SOLUTION_H
#define CHAPTER_21_EXERCISE01_SOLUTION_H

#include <cuda_runtime.h>

// ============================================================================
// 常量定义
// ============================================================================

#define MAX_TESS_POINTS 32  // 最大细分点数

// ============================================================================
// 数据结构
// ============================================================================

/**
 * Bezier 曲线结构
 * - CP: 3 个控制点 [x0,y0,x1,y1,x2,y2]
 * - vertexPos: 细分后的顶点位置
 * - nVertices: 实际顶点数
 */
struct BezierLine {
    float CP[6];                           // 控制点（展平）
    float vertexPos[MAX_TESS_POINTS * 2];  // 细分顶点
    int nVertices;                         // 顶点数
};

// ============================================================================
// 函数声明
// ============================================================================

/**
 * 静态版本：使用循环处理每条曲线
 * @param lines    曲线数组
 * @param num_lines 曲线数量
 * @return 0=成功, <0=错误
 */
int tessellate_bezier_static(BezierLine* lines, int num_lines);

/**
 * 动态并行版本：父 kernel 为每条曲线启动子 kernel
 * @param lines    曲线数组
 * @param num_lines 曲线数量
 * @return 0=成功, <0=错误
 */
int tessellate_bezier_dynamic(BezierLine* lines, int num_lines);

#endif // CHAPTER_21_EXERCISE01_SOLUTION_H
