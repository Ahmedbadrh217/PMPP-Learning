// ============================================================================
// solution.cu - 第十六章: cuBLAS SGEMM 矩阵乘法包装
// ============================================================================
// 封装 cuBLAS 库的 SGEMM（单精度通用矩阵乘法）操作
// 对应参考仓库 cublas_wrapper.c 的功能
// ============================================================================

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cstdio>
#include <cstring>

// ============================================================================
// 全局 cuBLAS 句柄
// ============================================================================

static cublasHandle_t g_handle = nullptr;

// ============================================================================
// cuBLAS 句柄管理
// ============================================================================

int init_cublas() {
    cublasStatus_t status = cublasCreate(&g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS 初始化失败，错误码: %d\n", status);
        return -1;
    }
    printf("cuBLAS 初始化成功\n");
    return 0;
}

int cleanup_cublas() {
    if (g_handle != nullptr) {
        cublasStatus_t status = cublasDestroy(g_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS 清理失败，错误码: %d\n", status);
            return -1;
        }
        g_handle = nullptr;
        printf("cuBLAS 清理成功\n");
    }
    return 0;
}

// ============================================================================
// SGEMM 实现
// ============================================================================

/**
 * cuBLAS 使用列主序，而我们使用行主序
 * 利用公式: C^T = B^T * A^T
 * 所以 cuBLAS 计算 C^T，但因为我们读取时也是行主序，结果自动正确
 */
int sgemm_wrapper(float* A, float* B, float* C,
                  int m, int n, int k,
                  int transA, int transB) {
    if (g_handle == nullptr) {
        printf("错误: cuBLAS 未初始化\n");
        return -1;
    }
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // 分配设备内存
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    // 根据转置选项确定实际尺寸
    int sizeA = transA ? (k * m) : (m * k);
    int sizeB = transB ? (n * k) : (k * n);
    int sizeC = m * n;
    
    // 带错误处理的内存分配，失败时清理之前分配的内存
    if (cudaMalloc(&d_A, sizeA * sizeof(float)) != cudaSuccess) {
        printf("错误: 分配 d_A 失败\n");
        return -1;
    }
    if (cudaMalloc(&d_B, sizeB * sizeof(float)) != cudaSuccess) {
        printf("错误: 分配 d_B 失败\n");
        cudaFree(d_A);
        return -1;
    }
    if (cudaMalloc(&d_C, sizeC * sizeof(float)) != cudaSuccess) {
        printf("错误: 分配 d_C 失败\n");
        cudaFree(d_A);
        cudaFree(d_B);
        return -1;
    }
    
    if (cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_B, B, sizeB * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("错误: 数据拷贝失败\n");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }
    
    // 设置转置选项
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    // 计算 leading dimensions (行主序转列主序)
    int lda = transA ? m : k;
    int ldb = transB ? k : n;
    int ldc = n;
    
    // cuBLAS 使用列主序，所以我们交换 A 和 B 的顺序
    // C = A * B (行主序) => C^T = B^T * A^T (列主序)
    // 但由于我们存储也是行主序，相当于 cuBLAS 的 C = B * A
    cublasStatus_t status = cublasSgemm(g_handle,
                                         opB, opA,     // 注意顺序
                                         n, m, k,     // 注意顺序
                                         &alpha,
                                         d_B, ldb,
                                         d_A, lda,
                                         &beta,
                                         d_C, ldc);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS SGEMM 失败，错误码: %d\n", status);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }
    
    // 复制结果回主机
    CHECK_CUDA(cudaMemcpy(C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}

int sgemm_device(float* d_A, float* d_B, float* d_C,
                 int m, int n, int k,
                 int transA, int transB) {
    if (g_handle == nullptr) {
        printf("错误: cuBLAS 未初始化\n");
        return -1;
    }
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    int lda = transA ? m : k;
    int ldb = transB ? k : n;
    int ldc = n;
    
    cublasStatus_t status = cublasSgemm(g_handle,
                                         opB, opA,
                                         n, m, k,
                                         &alpha,
                                         d_B, ldb,
                                         d_A, lda,
                                         &beta,
                                         d_C, ldc);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS SGEMM 失败，错误码: %d\n", status);
        return -1;
    }
    
    return 0;
}

// ============================================================================
// GPU 内存管理辅助函数
// ============================================================================

float* gpu_alloc(float* host_data, int size) {
    float* dev_ptr;
    CHECK_CUDA(cudaMalloc(&dev_ptr, size * sizeof(float)));
    if (host_data != nullptr) {
        CHECK_CUDA(cudaMemcpy(dev_ptr, host_data, size * sizeof(float), cudaMemcpyHostToDevice));
    }
    return dev_ptr;
}

int gpu_to_host(float* dev_ptr, float* host_ptr, int size) {
    cudaError_t err = cudaMemcpy(host_ptr, dev_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

void gpu_free(float* dev_ptr) {
    if (dev_ptr != nullptr) {
        cudaFree(dev_ptr);
    }
}

// ============================================================================
// CPU 参考实现
// ============================================================================

void sgemm_cpu(const float* A, const float* B, float* C,
               int m, int n, int k) {
    // C = A * B
    // A: [m × k], B: [k × n], C: [m × n]
    memset(C, 0, m * n * sizeof(float));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}
