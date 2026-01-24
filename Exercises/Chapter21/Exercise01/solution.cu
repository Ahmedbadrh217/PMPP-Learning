#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "solution.h"

// ============================================================================
// 设备函数：计算曲率
// ============================================================================

__device__ float computeCurvature(float* cp) {
    // 计算首尾连线向量
    float dx = cp[4] - cp[0];  // cp[2].x - cp[0].x
    float dy = cp[5] - cp[1];  // cp[2].y - cp[0].y
    float line_length = sqrtf(dx * dx + dy * dy);
    
    if (line_length < 0.001f) {
        return 0.0f;
    }
    
    // 计算中点到直线的距离（曲率近似）
    float cross = fabsf((cp[2] - cp[0]) * dy - (cp[3] - cp[1]) * dx);
    return cross / line_length;
}

// ============================================================================
// 子 Kernel：计算细分点（动态并行版本使用）
// ============================================================================

__global__ void computeBezierLine_child(int lidx, BezierLine* bLines, int nTessPoints) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx < nTessPoints) {
        // 计算参数 u ∈ [0, 1]
        float u = (float)idx / (float)(nTessPoints - 1);
        float omu = 1.0f - u;
        
        // 二次 Bezier 基函数
        float B3u[3];
        B3u[0] = omu * omu;
        B3u[1] = 2.0f * u * omu;
        B3u[2] = u * u;
        
        // 计算点位置
        float pos_x = 0.0f, pos_y = 0.0f;
        for (int i = 0; i < 3; i++) {
            pos_x += B3u[i] * bLines[lidx].CP[i * 2];      // CP[i].x
            pos_y += B3u[i] * bLines[lidx].CP[i * 2 + 1];  // CP[i].y
        }
        
        bLines[lidx].vertexPos[idx * 2] = pos_x;
        bLines[lidx].vertexPos[idx * 2 + 1] = pos_y;
    }
}

// ============================================================================
// 父 Kernel：动态并行版本
// ============================================================================

__global__ void computeBezierLines_dynamic(BezierLine* bLines, int nLines) {
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (lidx < nLines) {
        // 根据曲率计算需要的顶点数
        float curvature = computeCurvature(bLines[lidx].CP);
        int nTessPoints = min(max((int)(curvature * 16.0f), 4), MAX_TESS_POINTS);
        bLines[lidx].nVertices = nTessPoints;
        
        // 动态启动子 kernel
        int childBlocks = (nTessPoints + 31) / 32;
        computeBezierLine_child<<<childBlocks, 32>>>(lidx, bLines, nTessPoints);
        
        // 子 kernel 在父 kernel 退出时自动同步
    }
}

// ============================================================================
// 静态版本：使用循环替代子 kernel
// ============================================================================

__global__ void computeBezierLines_static(BezierLine* bLines, int nLines) {
    int bidx = blockIdx.x;
    
    if (bidx < nLines) {
        // 根据曲率计算需要的顶点数
        float curvature = computeCurvature(bLines[bidx].CP);
        int nTessPoints = min(max((int)(curvature * 16.0f), 4), MAX_TESS_POINTS);
        bLines[bidx].nVertices = nTessPoints;
        
        // 使用循环代替子 kernel
        for (int inc = 0; inc < nTessPoints; inc += blockDim.x) {
            int idx = inc + threadIdx.x;
            if (idx < nTessPoints) {
                float u = (float)idx / (float)(nTessPoints - 1);
                float omu = 1.0f - u;
                
                float B3u[3];
                B3u[0] = omu * omu;
                B3u[1] = 2.0f * u * omu;
                B3u[2] = u * u;
                
                float pos_x = 0.0f, pos_y = 0.0f;
                for (int i = 0; i < 3; i++) {
                    pos_x += B3u[i] * bLines[bidx].CP[i * 2];
                    pos_y += B3u[i] * bLines[bidx].CP[i * 2 + 1];
                }
                
                bLines[bidx].vertexPos[idx * 2] = pos_x;
                bLines[bidx].vertexPos[idx * 2 + 1] = pos_y;
            }
        }
    }
}

// ============================================================================
// 主机包装函数：静态版本
// ============================================================================

int tessellate_bezier_static(BezierLine* lines, int num_lines) {
    BezierLine* d_lines;
    size_t size = num_lines * sizeof(BezierLine);
    
    // 分配设备内存
    if (cudaMalloc(&d_lines, size) != cudaSuccess) {
        return -1;
    }
    
    // 复制到设备
    if (cudaMemcpy(d_lines, lines, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_lines);
        return -2;
    }
    
    // 启动静态 kernel
    computeBezierLines_static<<<num_lines, 32>>>(d_lines, num_lines);
    cudaDeviceSynchronize();
    
    // 复制回主机
    if (cudaMemcpy(lines, d_lines, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_lines);
        return -3;
    }
    
    cudaFree(d_lines);
    return 0;
}

// ============================================================================
// 主机包装函数：动态并行版本
// ============================================================================

int tessellate_bezier_dynamic(BezierLine* lines, int num_lines) {
    BezierLine* d_lines;
    size_t size = num_lines * sizeof(BezierLine);
    
    // 分配设备内存
    if (cudaMalloc(&d_lines, size) != cudaSuccess) {
        return -1;
    }
    
    // 复制到设备
    if (cudaMemcpy(d_lines, lines, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_lines);
        return -2;
    }
    
    // 启动动态并行 kernel
    int blocks = (num_lines + 31) / 32;
    computeBezierLines_dynamic<<<blocks, 32>>>(d_lines, num_lines);
    cudaDeviceSynchronize();
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_lines);
        return -4;
    }
    
    // 复制回主机
    if (cudaMemcpy(lines, d_lines, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_lines);
        return -3;
    }
    
    cudaFree(d_lines);
    return 0;
}
