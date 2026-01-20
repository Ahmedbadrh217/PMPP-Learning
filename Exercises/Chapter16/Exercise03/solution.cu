// ============================================================================
// solution.cu - 第十六章: CNN 完整层实现
// ============================================================================
// 包含 Conv2D 和 MaxPool2D 的前向和反向传播完整实现
// 对应参考仓库 conv2d.cu 的完整功能
// ============================================================================

#include "solution.h"
#include "../../Common/utils.cuh"
#include <cfloat>
#include <cstring>

// ============================================================================
// GPU Kernel 定义
// ============================================================================

#define BLOCK_SIZE 256

// ============================================================================
// Conv2D Forward Kernel
// ============================================================================

__global__ void conv2d_forward_kernel(const float* input, const float* weights, const float* bias,
                                       float* output,
                                       int batch_size, int in_channels, int height, int width,
                                       int out_channels, int kernel_h, int kernel_w,
                                       int pad_h, int pad_w, int stride_h, int stride_w,
                                       int out_h, int out_w) {
    int b = blockIdx.z;
    int c_out = blockIdx.y;
    int hw_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = hw_out / out_w;
    int w_out = hw_out % out_w;

    if (b >= batch_size || c_out >= out_channels || h_out >= out_h || w_out >= out_w) {
        return;
    }

    float val = bias[c_out];

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((b * in_channels + c_in) * height + h_in) * width + w_in;
                    int weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
                    val += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    int output_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
    output[output_idx] = val;
}

// ============================================================================
// Conv2D Backward Input Kernel
// ============================================================================

__global__ void conv2d_backward_input_kernel(const float* weights, const float* grad_output,
                                              float* grad_input,
                                              int batch_size, int in_channels, int height, int width,
                                              int out_channels, int kernel_h, int kernel_w,
                                              int pad_h, int pad_w, int stride_h, int stride_w,
                                              int out_h, int out_w) {
    int b = blockIdx.z;
    int c_in = blockIdx.y;
    int hw_in = blockIdx.x * blockDim.x + threadIdx.x;
    int h_in = hw_in / width;
    int w_in = hw_in % width;

    if (b >= batch_size || c_in >= in_channels || h_in >= height || w_in >= width) {
        return;
    }

    float val = 0.0f;

    for (int c_out = 0; c_out < out_channels; ++c_out) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_out_offset = h_in + pad_h - kh;
                int w_out_offset = w_in + pad_w - kw;
                
                if (h_out_offset % stride_h == 0 && w_out_offset % stride_w == 0) {
                    int h_out = h_out_offset / stride_h;
                    int w_out = w_out_offset / stride_w;

                    if (h_out >= 0 && h_out < out_h && w_out >= 0 && w_out < out_w) {
                        int grad_output_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
                        val += grad_output[grad_output_idx] * weights[weight_idx];
                    }
                }
            }
        }
    }

    int grad_input_idx = ((b * in_channels + c_in) * height + h_in) * width + w_in;
    grad_input[grad_input_idx] = val;
}

// ============================================================================
// Conv2D Backward Weights Kernel
// ============================================================================

__global__ void conv2d_backward_weights_kernel(const float* input, const float* grad_output,
                                                float* grad_weights,
                                                int batch_size, int in_channels, int height, int width,
                                                int out_channels, int kernel_h, int kernel_w,
                                                int pad_h, int pad_w, int stride_h, int stride_w,
                                                int out_h, int out_w) {
    int c_out = blockIdx.z;
    int c_in = blockIdx.y;
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kh = k_idx / kernel_w;
    int kw = k_idx % kernel_w;

    if (c_out >= out_channels || c_in >= in_channels || kh >= kernel_h || kw >= kernel_w) {
        return;
    }

    float val = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
        for (int h_out = 0; h_out < out_h; ++h_out) {
            for (int w_out = 0; w_out < out_w; ++w_out) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((b * in_channels + c_in) * height + h_in) * width + w_in;
                    int grad_output_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
                    val += input[input_idx] * grad_output[grad_output_idx];
                }
            }
        }
    }

    int grad_weights_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
    grad_weights[grad_weights_idx] = val;
}

// ============================================================================
// Conv2D Backward Bias Kernel
// ============================================================================

__global__ void conv2d_backward_bias_kernel(const float* grad_output, float* grad_bias,
                                             int batch_size, int out_channels, int out_h, int out_w) {
    int c_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_out >= out_channels) {
        return;
    }

    float val = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
        for (int h_out = 0; h_out < out_h; ++h_out) {
            for (int w_out = 0; w_out < out_w; ++w_out) {
                int grad_output_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
                val += grad_output[grad_output_idx];
            }
        }
    }

    grad_bias[c_out] = val;
}

// ============================================================================
// MaxPool2D Forward Kernel (with indices)
// ============================================================================

__global__ void maxpool2d_forward_kernel(const float* input, float* output, int* indices,
                                          int batch_size, int channels, int height, int width,
                                          int kernel_h, int kernel_w, int stride_h, int stride_w,
                                          int out_h, int out_w) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int hw_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = hw_out / out_w;
    int w_out = hw_out % out_w;

    if (b >= batch_size || c >= channels || h_out >= out_h || w_out >= out_w) {
        return;
    }

    float max_val = -FLT_MAX;
    int max_idx = -1;

    int h_start = h_out * stride_h;
    int w_start = w_out * stride_w;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = h_start + kh;
            int w_in = w_start + kw;

            if (h_in < height && w_in < width) {
                int input_idx = ((b * channels + c) * height + h_in) * width + w_in;
                float val = input[input_idx];

                if (val > max_val) {
                    max_val = val;
                    max_idx = kh * kernel_w + kw;
                }
            }
        }
    }

    int output_idx = ((b * channels + c) * out_h + h_out) * out_w + w_out;
    output[output_idx] = max_val;
    indices[output_idx] = max_idx;
}

// ============================================================================
// MaxPool2D Backward Kernel
// ============================================================================

__global__ void maxpool2d_backward_kernel(const float* grad_output, const int* indices,
                                           float* grad_input,
                                           int batch_size, int channels, int height, int width,
                                           int kernel_h, int kernel_w, int stride_h, int stride_w,
                                           int out_h, int out_w) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int hw_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = hw_out / out_w;
    int w_out = hw_out % out_w;

    if (b >= batch_size || c >= channels || h_out >= out_h || w_out >= out_w) {
        return;
    }

    int output_idx = ((b * channels + c) * out_h + h_out) * out_w + w_out;
    float grad_val = grad_output[output_idx];
    int max_idx = indices[output_idx];

    int kh = max_idx / kernel_w;
    int kw = max_idx % kernel_w;

    int h_in = h_out * stride_h + kh;
    int w_in = w_out * stride_w + kw;

    if (h_in < height && w_in < width) {
        int input_idx = ((b * channels + c) * height + h_in) * width + w_in;
        atomicAdd(&grad_input[input_idx], grad_val);
    }
}

// ============================================================================
// CPU 串行实现
// ============================================================================

void conv2d_forward_cpu(const float* input, const float* weights, const float* bias,
                        float* output,
                        int N, int C_in, int H, int W, int C_out,
                        int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w) {
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;

    for (int n = 0; n < N; n++) {
        for (int c_out = 0; c_out < C_out; c_out++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
                    float val = bias[c_out];
                    
                    for (int c_in = 0; c_in < C_in; c_in++) {
                        for (int kh = 0; kh < K_h; kh++) {
                            for (int kw = 0; kw < K_w; kw++) {
                                int h_in = h_out * stride_h - pad_h + kh;
                                int w_in = w_out * stride_w - pad_w + kw;
                                
                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    int input_idx = ((n * C_in + c_in) * H + h_in) * W + w_in;
                                    int weight_idx = ((c_out * C_in + c_in) * K_h + kh) * K_w + kw;
                                    val += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }
                    
                    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                    output[output_idx] = val;
                }
            }
        }
    }
}

void conv2d_backward_input_cpu(const float* weights, const float* grad_output,
                               float* grad_input,
                               int N, int C_in, int H, int W, int C_out,
                               int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w) {
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    
    memset(grad_input, 0, N * C_in * H * W * sizeof(float));
    
    for (int n = 0; n < N; n++) {
        for (int c_out = 0; c_out < C_out; c_out++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
                    int grad_output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                    float grad = grad_output[grad_output_idx];
                    
                    for (int c_in = 0; c_in < C_in; c_in++) {
                        for (int kh = 0; kh < K_h; kh++) {
                            for (int kw = 0; kw < K_w; kw++) {
                                int h_in = h_out * stride_h - pad_h + kh;
                                int w_in = w_out * stride_w - pad_w + kw;
                                
                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    int grad_input_idx = ((n * C_in + c_in) * H + h_in) * W + w_in;
                                    int weight_idx = ((c_out * C_in + c_in) * K_h + kh) * K_w + kw;
                                    grad_input[grad_input_idx] += grad * weights[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void conv2d_backward_weights_cpu(const float* input, const float* grad_output,
                                 float* grad_weights,
                                 int N, int C_in, int H, int W, int C_out,
                                 int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w) {
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    
    memset(grad_weights, 0, C_out * C_in * K_h * K_w * sizeof(float));
    
    for (int c_out = 0; c_out < C_out; c_out++) {
        for (int c_in = 0; c_in < C_in; c_in++) {
            for (int kh = 0; kh < K_h; kh++) {
                for (int kw = 0; kw < K_w; kw++) {
                    float val = 0.0f;
                    
                    for (int n = 0; n < N; n++) {
                        for (int h_out = 0; h_out < H_out; h_out++) {
                            for (int w_out = 0; w_out < W_out; w_out++) {
                                int h_in = h_out * stride_h - pad_h + kh;
                                int w_in = w_out * stride_w - pad_w + kw;
                                
                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    int input_idx = ((n * C_in + c_in) * H + h_in) * W + w_in;
                                    int grad_output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                                    val += input[input_idx] * grad_output[grad_output_idx];
                                }
                            }
                        }
                    }
                    
                    int grad_weights_idx = ((c_out * C_in + c_in) * K_h + kh) * K_w + kw;
                    grad_weights[grad_weights_idx] = val;
                }
            }
        }
    }
}

void conv2d_backward_bias_cpu(const float* grad_output, float* grad_bias,
                              int N, int C_out, int H_out, int W_out) {
    memset(grad_bias, 0, C_out * sizeof(float));
    
    for (int c_out = 0; c_out < C_out; c_out++) {
        float val = 0.0f;
        for (int n = 0; n < N; n++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
                    int idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                    val += grad_output[idx];
                }
            }
        }
        grad_bias[c_out] = val;
    }
}

void maxpool2d_forward_cpu(const float* input, float* output, int* indices,
                           int N, int C, int H, int W,
                           int K_h, int K_w, int stride_h, int stride_w) {
    int H_out = (H - K_h) / stride_h + 1;
    int W_out = (W - K_w) / stride_w + 1;
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
                    float max_val = -FLT_MAX;
                    int max_idx = -1;
                    
                    for (int kh = 0; kh < K_h; kh++) {
                        for (int kw = 0; kw < K_w; kw++) {
                            int h_in = h_out * stride_h + kh;
                            int w_in = w_out * stride_w + kw;
                            
                            if (h_in < H && w_in < W) {
                                int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                                float val = input[input_idx];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = kh * K_w + kw;
                                }
                            }
                        }
                    }
                    
                    int output_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                    output[output_idx] = max_val;
                    indices[output_idx] = max_idx;
                }
            }
        }
    }
}

void maxpool2d_backward_cpu(const float* grad_output, const int* indices,
                            float* grad_input,
                            int N, int C, int H, int W,
                            int K_h, int K_w, int stride_h, int stride_w) {
    int H_out = (H - K_h) / stride_h + 1;
    int W_out = (W - K_w) / stride_w + 1;
    
    memset(grad_input, 0, N * C * H * W * sizeof(float));
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
                    int output_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                    float grad_val = grad_output[output_idx];
                    int max_idx = indices[output_idx];
                    
                    int kh = max_idx / K_w;
                    int kw = max_idx % K_w;
                    
                    int h_in = h_out * stride_h + kh;
                    int w_in = w_out * stride_w + kw;
                    
                    if (h_in < H && w_in < W) {
                        int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                        grad_input[input_idx] += grad_val;
                    }
                }
            }
        }
    }
}

// ============================================================================
// GPU 并行实现 - 调用 kernel
// ============================================================================

void conv2d_forward_gpu(const float* d_input, const float* d_weights, const float* d_bias,
                        float* d_output,
                        int N, int C_in, int H, int W, int C_out,
                        int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w) {
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    
    int hw_out = H_out * W_out;
    dim3 grid((hw_out + BLOCK_SIZE - 1) / BLOCK_SIZE, C_out, N);
    
    conv2d_forward_kernel<<<grid, BLOCK_SIZE>>>(
        d_input, d_weights, d_bias, d_output,
        N, C_in, H, W, C_out, K_h, K_w, pad_h, pad_w, stride_h, stride_w, H_out, W_out);
    
    CHECK_LAST_CUDA_ERROR();
}

void conv2d_backward_input_gpu(const float* d_weights, const float* d_grad_output,
                               float* d_grad_input,
                               int N, int C_in, int H, int W, int C_out,
                               int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w) {
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    
    int hw_in = H * W;
    dim3 grid((hw_in + BLOCK_SIZE - 1) / BLOCK_SIZE, C_in, N);
    
    conv2d_backward_input_kernel<<<grid, BLOCK_SIZE>>>(
        d_weights, d_grad_output, d_grad_input,
        N, C_in, H, W, C_out, K_h, K_w, pad_h, pad_w, stride_h, stride_w, H_out, W_out);
    
    CHECK_LAST_CUDA_ERROR();
}

void conv2d_backward_weights_gpu(const float* d_input, const float* d_grad_output,
                                 float* d_grad_weights,
                                 int N, int C_in, int H, int W, int C_out,
                                 int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w) {
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    
    int k_size = K_h * K_w;
    dim3 grid((k_size + BLOCK_SIZE - 1) / BLOCK_SIZE, C_in, C_out);
    
    conv2d_backward_weights_kernel<<<grid, BLOCK_SIZE>>>(
        d_input, d_grad_output, d_grad_weights,
        N, C_in, H, W, C_out, K_h, K_w, pad_h, pad_w, stride_h, stride_w, H_out, W_out);
    
    CHECK_LAST_CUDA_ERROR();
}

void conv2d_backward_bias_gpu(const float* d_grad_output, float* d_grad_bias,
                              int N, int C_out, int H_out, int W_out) {
    int num_blocks = (C_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    conv2d_backward_bias_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_grad_output, d_grad_bias, N, C_out, H_out, W_out);
    
    CHECK_LAST_CUDA_ERROR();
}

void maxpool2d_forward_gpu(const float* d_input, float* d_output, int* d_indices,
                           int N, int C, int H, int W,
                           int K_h, int K_w, int stride_h, int stride_w) {
    int H_out = (H - K_h) / stride_h + 1;
    int W_out = (W - K_w) / stride_w + 1;
    
    int hw_out = H_out * W_out;
    dim3 grid((hw_out + BLOCK_SIZE - 1) / BLOCK_SIZE, C, N);
    
    maxpool2d_forward_kernel<<<grid, BLOCK_SIZE>>>(
        d_input, d_output, d_indices,
        N, C, H, W, K_h, K_w, stride_h, stride_w, H_out, W_out);
    
    CHECK_LAST_CUDA_ERROR();
}

void maxpool2d_backward_gpu(const float* d_grad_output, const int* d_indices,
                            float* d_grad_input,
                            int N, int C, int H, int W,
                            int K_h, int K_w, int stride_h, int stride_w) {
    int H_out = (H - K_h) / stride_h + 1;
    int W_out = (W - K_w) / stride_w + 1;
    
    // 初始化梯度为0
    CHECK_CUDA(cudaMemset(d_grad_input, 0, N * C * H * W * sizeof(float)));
    
    int hw_out = H_out * W_out;
    dim3 grid((hw_out + BLOCK_SIZE - 1) / BLOCK_SIZE, C, N);
    
    maxpool2d_backward_kernel<<<grid, BLOCK_SIZE>>>(
        d_grad_output, d_indices, d_grad_input,
        N, C, H, W, K_h, K_w, stride_h, stride_w, H_out, W_out);
    
    CHECK_LAST_CUDA_ERROR();
}
