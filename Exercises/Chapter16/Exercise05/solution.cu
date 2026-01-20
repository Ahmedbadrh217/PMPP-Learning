// ============================================================================
// solution.cu - 第十六章练习5: cuDNN 封装实现
// ============================================================================
// 使用 NVIDIA cuDNN 库实现 Conv2D 和 MaxPool2D 操作
// 对应参考仓库的 legacy_cudnn_wrapper.cu 功能
// ============================================================================

#include "solution.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cstring>

// ============================================================================
// 错误检查宏
// ============================================================================

#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        cudaError_t err = (call);                                                     \
        if (err != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err));                                         \
            return -1;                                                                \
        }                                                                             \
    } while (0)

#define CHECK_CUDNN(call)                                                             \
    do {                                                                              \
        cudnnStatus_t status = (call);                                                \
        if (status != CUDNN_STATUS_SUCCESS) {                                         \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudnnGetErrorString(status));                                     \
            return -1;                                                                \
        }                                                                             \
    } while (0)

// ============================================================================
// 全局 cuDNN 句柄
// ============================================================================

static cudnnHandle_t g_cudnn_handle = nullptr;

// ============================================================================
// cuDNN 句柄管理
// ============================================================================

int init_cudnn() {
    cudnnStatus_t status = cudnnCreate(&g_cudnn_handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("cuDNN 初始化失败: %s\n", cudnnGetErrorString(status));
        return -1;
    }
    printf("cuDNN 初始化成功\n");
    return 0;
}

int cleanup_cudnn() {
    if (g_cudnn_handle != nullptr) {
        cudnnStatus_t status = cudnnDestroy(g_cudnn_handle);
        if (status != CUDNN_STATUS_SUCCESS) {
            printf("cuDNN 清理失败: %s\n", cudnnGetErrorString(status));
            return -1;
        }
        g_cudnn_handle = nullptr;
        printf("cuDNN 清理成功\n");
    }
    return 0;
}

// ============================================================================
// Conv2D Forward - cuDNN 实现
// ============================================================================

int conv2d_forward_cudnn(float* input, float* weights, float* bias, float* output,
                         int batch_size, int in_channels, int height, int width,
                         int out_channels, int kernel_h, int kernel_w,
                         int pad_h, int pad_w, int stride_h, int stride_w) {
    if (g_cudnn_handle == nullptr) {
        printf("错误: cuDNN 未初始化\n");
        return -1;
    }

    // 计算输出维度
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // 计算数据大小
    size_t input_bytes = batch_size * in_channels * height * width * sizeof(float);
    size_t weight_bytes = out_channels * in_channels * kernel_h * kernel_w * sizeof(float);
    size_t output_bytes = batch_size * out_channels * out_h * out_w * sizeof(float);
    size_t bias_bytes = out_channels * sizeof(float);

    // 分配设备内存
    float *d_input = nullptr, *d_weights = nullptr, *d_output = nullptr, *d_bias = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_weights, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias, bias_bytes));

    // 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, weights, weight_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias, bias_bytes, cudaMemcpyHostToDevice));

    // 创建描述符
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t weight_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&weight_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // 设置描述符
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            batch_size, in_channels, height, width));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                            out_channels, in_channels, kernel_h, kernel_w));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w,
                                                 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            batch_size, out_channels, out_h, out_w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            1, out_channels, 1, 1));

    // 选择卷积算法
    int returned_count = 0;
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(g_cudnn_handle, input_desc, weight_desc,
                                                        conv_desc, output_desc, 1, &returned_count, &algo_perf));
    cudnnConvolutionFwdAlgo_t algo = algo_perf.algo;

    // 获取工作空间大小
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(g_cudnn_handle, input_desc, weight_desc,
                                                         conv_desc, output_desc, algo, &workspace_size));
    void* d_workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
    }

    // 执行卷积前向
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(g_cudnn_handle, &alpha, input_desc, d_input,
                                         weight_desc, d_weights, conv_desc, algo,
                                         d_workspace, workspace_size, &beta, output_desc, d_output));

    // 添加偏置
    CHECK_CUDNN(cudnnAddTensor(g_cudnn_handle, &alpha, bias_desc, d_bias, &alpha, output_desc, d_output));

    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    // 清理
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(d_bias);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(weight_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return 0;
}

// ============================================================================
// Conv2D Backward - cuDNN 实现
// ============================================================================

int conv2d_backward_cudnn(float* input, float* weights, float* grad_output,
                          float* grad_input, float* grad_weights, float* grad_bias,
                          int batch_size, int in_channels, int height, int width,
                          int out_channels, int kernel_h, int kernel_w,
                          int pad_h, int pad_w, int stride_h, int stride_w) {
    if (g_cudnn_handle == nullptr) {
        printf("错误: cuDNN 未初始化\n");
        return -1;
    }

    // 计算输出维度
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // 计算数据大小
    size_t input_bytes = batch_size * in_channels * height * width * sizeof(float);
    size_t weight_bytes = out_channels * in_channels * kernel_h * kernel_w * sizeof(float);
    size_t output_bytes = batch_size * out_channels * out_h * out_w * sizeof(float);
    size_t bias_bytes = out_channels * sizeof(float);

    // 分配设备内存
    float *d_input, *d_weights, *d_grad_output;
    float *d_grad_input, *d_grad_weights, *d_grad_bias;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_weights, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_output, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_weights, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_bias, bias_bytes));

    // 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, weights, weight_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad_output, grad_output, output_bytes, cudaMemcpyHostToDevice));

    // 创建描述符
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t weight_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&weight_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // 设置描述符
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            batch_size, in_channels, height, width));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                            out_channels, in_channels, kernel_h, kernel_w));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w,
                                                 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            batch_size, out_channels, out_h, out_w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            1, out_channels, 1, 1));

    float alpha = 1.0f, beta = 0.0f;

    // ============ Backward Data (输入梯度) ============
    int bwd_data_count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(g_cudnn_handle, weight_desc, output_desc,
                                                             conv_desc, input_desc, 1, &bwd_data_count, &bwd_data_perf));
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo = bwd_data_perf.algo;

    size_t ws_data_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(g_cudnn_handle, weight_desc, output_desc,
                                                              conv_desc, input_desc, bwd_data_algo, &ws_data_size));
    void* d_ws_data = nullptr;
    if (ws_data_size > 0) CHECK_CUDA(cudaMalloc(&d_ws_data, ws_data_size));

    CHECK_CUDNN(cudnnConvolutionBackwardData(g_cudnn_handle, &alpha, weight_desc, d_weights,
                                              output_desc, d_grad_output, conv_desc, bwd_data_algo,
                                              d_ws_data, ws_data_size, &beta, input_desc, d_grad_input));

    // ============ Backward Filter (权重梯度) ============
    int bwd_filter_count = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf;
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(g_cudnn_handle, input_desc, output_desc,
                                                               conv_desc, weight_desc, 1, &bwd_filter_count, &bwd_filter_perf));
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo = bwd_filter_perf.algo;

    size_t ws_filter_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(g_cudnn_handle, input_desc, output_desc,
                                                                conv_desc, weight_desc, bwd_filter_algo, &ws_filter_size));
    void* d_ws_filter = nullptr;
    if (ws_filter_size > 0) CHECK_CUDA(cudaMalloc(&d_ws_filter, ws_filter_size));

    CHECK_CUDNN(cudnnConvolutionBackwardFilter(g_cudnn_handle, &alpha, input_desc, d_input,
                                                output_desc, d_grad_output, conv_desc, bwd_filter_algo,
                                                d_ws_filter, ws_filter_size, &beta, weight_desc, d_grad_weights));

    // ============ Backward Bias (偏置梯度) ============
    CHECK_CUDNN(cudnnConvolutionBackwardBias(g_cudnn_handle, &alpha, output_desc, d_grad_output,
                                              &beta, bias_desc, d_grad_bias));

    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(grad_input, d_grad_input, input_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_weights, d_grad_weights, weight_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_bias, d_grad_bias, bias_bytes, cudaMemcpyDeviceToHost));

    // 清理
    if (d_ws_data) cudaFree(d_ws_data);
    if (d_ws_filter) cudaFree(d_ws_filter);
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_grad_weights);
    cudaFree(d_grad_bias);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(weight_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return 0;
}

// ============================================================================
// MaxPool2D Forward - cuDNN 实现
// ============================================================================

int maxpool2d_forward_cudnn(float* input, float* output,
                            int batch_size, int channels, int height, int width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w) {
    if (g_cudnn_handle == nullptr) {
        printf("错误: cuDNN 未初始化\n");
        return -1;
    }

    int out_h = (height - kernel_h) / stride_h + 1;
    int out_w = (width - kernel_w) / stride_w + 1;

    size_t input_bytes = batch_size * channels * height * width * sizeof(float);
    size_t output_bytes = batch_size * channels * out_h * out_w * sizeof(float);

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    CHECK_CUDA(cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice));

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnPoolingDescriptor_t pool_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pool_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            batch_size, channels, height, width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            batch_size, channels, out_h, out_w));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                             kernel_h, kernel_w, 0, 0, stride_h, stride_w));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnPoolingForward(g_cudnn_handle, pool_desc, &alpha, input_desc, d_input,
                                     &beta, output_desc, d_output));

    CHECK_CUDA(cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyPoolingDescriptor(pool_desc);

    return 0;
}

// ============================================================================
// MaxPool2D Backward - cuDNN 实现
// ============================================================================

int maxpool2d_backward_cudnn(float* input, float* output, float* grad_output,
                             float* grad_input,
                             int batch_size, int channels, int height, int width,
                             int kernel_h, int kernel_w, int stride_h, int stride_w) {
    if (g_cudnn_handle == nullptr) {
        printf("错误: cuDNN 未初始化\n");
        return -1;
    }

    int out_h = (height - kernel_h) / stride_h + 1;
    int out_w = (width - kernel_w) / stride_w + 1;

    size_t input_bytes = batch_size * channels * height * width * sizeof(float);
    size_t output_bytes = batch_size * channels * out_h * out_w * sizeof(float);

    float *d_input, *d_output, *d_grad_output, *d_grad_input;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_output, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_input, input_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_output, output, output_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad_output, grad_output, output_bytes, cudaMemcpyHostToDevice));

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnPoolingDescriptor_t pool_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pool_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            batch_size, channels, height, width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            batch_size, channels, out_h, out_w));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                             kernel_h, kernel_w, 0, 0, stride_h, stride_w));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnPoolingBackward(g_cudnn_handle, pool_desc, &alpha, output_desc, d_output,
                                      output_desc, d_grad_output, input_desc, d_input,
                                      &beta, input_desc, d_grad_input));

    CHECK_CUDA(cudaMemcpy(grad_input, d_grad_input, input_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyPoolingDescriptor(pool_desc);

    return 0;
}

// ============================================================================
// CPU 参考实现 (用于验证)
// ============================================================================

void conv2d_forward_cpu(const float* input, const float* weights, const float* bias,
                        float* output,
                        int batch_size, int in_channels, int height, int width,
                        int out_channels, int kernel_h, int kernel_w,
                        int pad_h, int pad_w, int stride_h, int stride_w) {
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    for (int n = 0; n < batch_size; n++) {
        for (int c_out = 0; c_out < out_channels; c_out++) {
            for (int h_out = 0; h_out < out_h; h_out++) {
                for (int w_out = 0; w_out < out_w; w_out++) {
                    float val = bias[c_out];
                    
                    for (int c_in = 0; c_in < in_channels; c_in++) {
                        for (int kh = 0; kh < kernel_h; kh++) {
                            for (int kw = 0; kw < kernel_w; kw++) {
                                int h_in = h_out * stride_h - pad_h + kh;
                                int w_in = w_out * stride_w - pad_w + kw;
                                
                                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                                    int input_idx = ((n * in_channels + c_in) * height + h_in) * width + w_in;
                                    int weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
                                    val += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }
                    
                    int output_idx = ((n * out_channels + c_out) * out_h + h_out) * out_w + w_out;
                    output[output_idx] = val;
                }
            }
        }
    }
}

void maxpool2d_forward_cpu(const float* input, float* output,
                           int batch_size, int channels, int height, int width,
                           int kernel_h, int kernel_w, int stride_h, int stride_w) {
    int out_h = (height - kernel_h) / stride_h + 1;
    int out_w = (width - kernel_w) / stride_w + 1;

    for (int n = 0; n < batch_size; n++) {
        for (int c = 0; c < channels; c++) {
            for (int h_out = 0; h_out < out_h; h_out++) {
                for (int w_out = 0; w_out < out_w; w_out++) {
                    float max_val = -FLT_MAX;
                    
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int h_in = h_out * stride_h + kh;
                            int w_in = w_out * stride_w + kw;
                            
                            if (h_in < height && w_in < width) {
                                int input_idx = ((n * channels + c) * height + h_in) * width + w_in;
                                if (input[input_idx] > max_val) {
                                    max_val = input[input_idx];
                                }
                            }
                        }
                    }
                    
                    int output_idx = ((n * channels + c) * out_h + h_out) * out_w + w_out;
                    output[output_idx] = max_val;
                }
            }
        }
    }
}
