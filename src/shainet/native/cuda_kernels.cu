#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

// Device kernels
// Simple row-wise softmax kernel. This version runs one thread per row and
// performs the computation sequentially. It uses the row maximum for numerical
// stability.
__global__ void softmax_rows_kernel(float* out, const float* in, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = in + row * cols;
    float* row_out = out + row * cols;

    // Find the maximum value for numerical stability
    float max_val = row_in[0];
    for (int j = 1; j < cols; ++j) {
        float v = row_in[j];
        if (v > max_val) max_val = v;
    }

    // Compute exponentials and their sum
    float sum = 0.0;
    for (int j = 0; j < cols; ++j) {
        float e = expf(row_in[j] - max_val);
        row_out[j] = e;
        sum += e;
    }

    // Normalize
    for (int j = 0; j < cols; ++j) {
        row_out[j] /= sum;
    }
}

__global__ void relu_backward_kernel(float* output, const float* input, const float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = input[idx] > 0.0 ? grad[idx] : 0.0;
}

// Host wrapper functions
extern "C" {
void softmax_rows(float* out, const float* in, int rows, int cols) {
    softmax_rows_kernel<<<rows, 1>>>(out, in, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in softmax_rows: %s\n", cudaGetErrorString(err));
    }
}

void relu_backward(float* output, const float* input, const float* grad, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    relu_backward_kernel<<<blocks, threads_per_block>>>(output, input, grad, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in relu_backward: %s\n", cudaGetErrorString(err));
    }
}

__global__ void dropout_kernel(float* out, const float* in, int rows, int cols, double drop_p, unsigned long long seed) {
    int row = blockIdx.x;
    if(row >= rows) return;
    const float *row_in = in + row * cols;
    float *row_out = out + row * cols;
    if(drop_p >= 1.0) {
        for(int j=0;j<cols;++j) row_out[j] = 0.0;
        return;
    }
    if(drop_p <= 0.0) {
        for(int j=0;j<cols;++j) row_out[j] = row_in[j];
        return;
    }
    curandState state;
    curand_init(seed + row, 0, 0, &state);
    float scale = 1.0 / (1.0 - drop_p);
    for(int j=0;j<cols;++j){
        float r = curand_uniform(&state);
        row_out[j] = r < drop_p ? 0.0 : row_in[j] * scale;
    }
}

void dropout(float* out, const float* in, int rows, int cols, double drop_p, unsigned long long seed) {
    dropout_kernel<<<rows, 1>>>(out, in, rows, cols, drop_p, seed);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in dropout: %s\n", cudaGetErrorString(err));
    }
}

__global__ void gather_rows_kernel(float* out, const float* in, const int* ids, int rows, int cols) {
    int row = blockIdx.x;
    if(row >= rows) return;
    int id = ids[row];
    const float *row_in = in + id * cols;
    float *row_out = out + row * cols;
    for(int j=0;j<cols;++j){
        row_out[j] = row_in[j];
    }
}

void gather_rows(float* out, const float* in, const int* ids, int rows, int cols) {
    gather_rows_kernel<<<rows, 1>>>(out, in, ids, rows, cols);
    cudaDeviceSynchronize();
}

__global__ void row_mean_var_kernel(const float* in, float* mean, float* var,
                                    int rows, int cols) {
    int row = blockIdx.x;
    if(row >= rows) return;
    const float *row_in = in + row * cols;
    float sum = 0.0;
    float sq_sum = 0.0;
    for(int j=0;j<cols;++j){
        float v = row_in[j];
        sum += v;
        sq_sum += v*v;
    }
    float m = sum / cols;
    mean[row] = m;
    var[row] = sq_sum / cols - m*m;
}

void row_mean_var(const float* in, float* mean, float* var, int rows, int cols) {
    row_mean_var_kernel<<<rows, 1>>>(in, mean, var, rows, cols);
    cudaDeviceSynchronize();
}

__global__ void apply_layer_norm_kernel(float* out, const float* in,
                                        const float* mean, const float* var,
                                        int rows, int cols, double epsilon) {
    int row = blockIdx.x;
    if(row >= rows) return;
    const float *row_in = in + row * cols;
    float *row_out = out + row * cols;
    float m = mean[row];
    float denom = sqrtf(var[row] + epsilon);
    for(int j=0;j<cols;++j){
        row_out[j] = (row_in[j] - m) / denom;
    }
}

void apply_layer_norm(float* out, const float* in,
                      const float* mean, const float* var,
                      int rows, int cols, double epsilon) {
    apply_layer_norm_kernel<<<rows, 1>>>(out, in, mean, var, rows, cols, epsilon);
    cudaDeviceSynchronize();
}

__global__ void slice_cols_kernel(float* out, const float* in, int rows, int src_cols, int start, int len){
    int row = blockIdx.x;
    int col = threadIdx.x;
    if(row >= rows) return;
    for(; col < len; col += blockDim.x){
        out[row * len + col] = in[row * src_cols + start + col];
    }
}

void slice_cols(float* out, const float* in, int rows, int src_cols, int start, int len){
    int threads = len < 1024 ? len : 1024;
    slice_cols_kernel<<<rows, threads>>>(out, in, rows, src_cols, start, len);
    cudaDeviceSynchronize();
}

__global__ void set_cols_kernel(float* out, const float* in, int rows, int dst_cols, int start, int len){
    int row = blockIdx.x;
    int col = threadIdx.x;
    if(row >= rows) return;
    for(; col < len; col += blockDim.x){
        out[row * dst_cols + start + col] = in[row * len + col];
    }
}

void set_cols(float* out, const float* in, int rows, int dst_cols, int start, int len){
    int threads = len < 1024 ? len : 1024;
    set_cols_kernel<<<rows, threads>>>(out, in, rows, dst_cols, start, len);
    cudaDeviceSynchronize();
}

__global__ void count_token_pairs_kernel(const int* a, const int* b, const int* freq,
                                         int pair_count, int vocab_size, int* counts){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= pair_count) return;
    int offset = a[idx] * vocab_size + b[idx];
    atomicAdd(&counts[offset], freq[idx]);
}

void count_token_pairs(const int* a, const int* b, const int* freq,
                       int pair_count, int vocab_size, int* counts){
    int blocks = (pair_count + 255) / 256;
    count_token_pairs_kernel<<<blocks, 256>>>(a, b, freq, pair_count, vocab_size, counts);
    cudaDeviceSynchronize();
}

__global__ void layer_norm_backward_kernel(float* d_x, float* d_gamma, float* d_beta,
                                           const float* d_out, const float* x,
                                           const float* gamma, const float* mean,
                                           const float* var, const float* norm,
                                           int rows, int cols, double epsilon) {
    int row = blockIdx.x;
    if(row >= rows) return;

    const float *x_row = x + row * cols;
    const float *dout_row = d_out + row * cols;
    const float *norm_row = norm + row * cols;
    float *dx_row = d_x + row * cols;

    float m = mean[row];
    float v = var[row];
    float denom = sqrtf(v + epsilon);
    float inv = 1.0 / denom;
    float col_f = (float)cols;

    // Compute sum_dout_gamma and sum_dout_gamma_norm
    float sum_dout_gamma = 0.0;
    float sum_dout_gamma_norm = 0.0;
    for(int j = 0; j < cols; ++j) {
        float doutg = dout_row[j] * gamma[j];
        sum_dout_gamma += doutg;
        sum_dout_gamma_norm += doutg * (x_row[j] - m);

        // Accumulate gradients for gamma and beta
        atomicAdd(&d_gamma[j], dout_row[j] * norm_row[j]);
        atomicAdd(&d_beta[j], dout_row[j]);
    }

    // Compute d_x
    for(int j = 0; j < cols; ++j) {
        float xm = x_row[j] - m;
        float doutg = dout_row[j] * gamma[j];
        dx_row[j] = inv * (doutg - sum_dout_gamma/col_f - xm * inv*inv / col_f * sum_dout_gamma_norm);
    }
}

void layer_norm_backward(float* d_x, float* d_gamma, float* d_beta,
                         const float* d_out, const float* x,
                         const float* gamma, const float* mean,
                         const float* var, const float* norm,
                         int rows, int cols, double epsilon) {
    layer_norm_backward_kernel<<<rows, 1>>>(d_x, d_gamma, d_beta, d_out, x,
                                            gamma, mean, var, norm,
                                            rows, cols, epsilon);
    cudaDeviceSynchronize();
}

__global__ void sum_cols_kernel(float* out, const float* in, int rows, int cols) {
    int col = blockIdx.x;
    if(col >= cols) return;

    float sum = 0.0;
    for(int i = 0; i < rows; ++i) {
        sum += in[i * cols + col];
    }
    out[col] = sum;
}

void sum_cols(float* out, const float* in, int rows, int cols) {
    sum_cols_kernel<<<cols, 1>>>(out, in, rows, cols);
    cudaDeviceSynchronize();
}

__global__ void mul_row_vector_kernel(float* matrix, const float* vec, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int col = idx % cols;
    matrix[idx] *= vec[col];
}

void mul_row_vector(float* matrix, const float* vec, int rows, int cols) {
    int threads_per_block = 256;
    int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    mul_row_vector_kernel<<<blocks, threads_per_block>>>(matrix, vec, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in mul_row_vector: %s\n", cudaGetErrorString(err));
    }
}

__global__ void transpose_kernel(float* out, const float* in, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int row = idx / cols;
    int col = idx % cols;

    // Transpose: out[col][row] = in[row][col]
    // In row-major: out[col * rows + row] = in[row * cols + col]
    out[col * rows + row] = in[row * cols + col];
}

void transpose(float* out, const float* in, int rows, int cols) {
    int threads_per_block = 256;
    int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    transpose_kernel<<<blocks, threads_per_block>>>(out, in, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in transpose: %s\n", cudaGetErrorString(err));
    }
}

__global__ void sigmoid_forward_kernel(float* activations, float* derivatives, const float* linear, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float val = linear[idx];
    // Sigmoid: 1 / (1 + expf(-x))
    float exp_neg_val = expf(-val);
    float sigmoid_val = 1.0 / (1.0 + exp_neg_val);
    
    activations[idx] = sigmoid_val;
    // Sigmoid derivative: σ(x) * (1 - σ(x))
    derivatives[idx] = sigmoid_val * (1.0 - sigmoid_val);
}

void sigmoid_forward(float* activations, float* derivatives, const float* linear, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    sigmoid_forward_kernel<<<blocks, threads_per_block>>>(activations, derivatives, linear, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in sigmoid_forward: %s\n", cudaGetErrorString(err));
    }
}

// hidden = silu(gate) * up, the SwiGLU activation, fused so the FFN's gate and up
// projections never have to come back to the host to be combined. Uses expf
// rather than __expf: the fast intrinsic drifts far enough from the CPU path to
// break device/host parity, and this kernel is memory bound anyway.
__global__ void swiglu_forward_kernel(float* hidden, const float* gate, const float* up, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float g = gate[idx];
    hidden[idx] = (g / (1.0f + expf(-g))) * up[idx];
}

// Deliberately does NOT cudaDeviceSynchronize: this sits between two GEMV launches
// on the default stream, so stream ordering already guarantees the gate/up writes
// are visible. Syncing here would reintroduce the pipeline stall that keeping
// activations on the device exists to remove.
void swiglu_forward(float* hidden, const float* gate, const float* up, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    swiglu_forward_kernel<<<blocks, threads_per_block>>>(hidden, gate, up, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in swiglu_forward: %s\n", cudaGetErrorString(err));
    }
}

__global__ void apply_gradient_kernel(float* local_grad, const float* grad, const float* derivatives, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    local_grad[idx] = grad[idx] * derivatives[idx];
}

void apply_gradient(float* local_grad, const float* grad, const float* derivatives, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    apply_gradient_kernel<<<blocks, threads_per_block>>>(local_grad, grad, derivatives, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in apply_gradient: %s\n", cudaGetErrorString(err));
    }
}

__global__ void accumulate_bias_grad_kernel(float* bias_grad, const float* local_grad, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    float sum = 0.0;
    for (int row = 0; row < rows; row++) {
        sum += local_grad[row * cols + col];
    }
    atomicAdd(&bias_grad[col], sum);
}

void accumulate_bias_grad(float* bias_grad, const float* local_grad, int rows, int cols) {
    int threads_per_block = 256;
    int blocks = (cols + threads_per_block - 1) / threads_per_block;

    accumulate_bias_grad_kernel<<<blocks, threads_per_block>>>(bias_grad, local_grad, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in accumulate_bias_grad: %s\n", cudaGetErrorString(err));
    }
}

__global__ void row_sum_kernel(float* dst, const float* src, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    float sum = 0.0;
    for (int row = 0; row < rows; ++row) {
        sum += src[row * cols + col];
    }
    atomicAdd(&dst[col], sum);
}

void row_sum(float* dst, const float* src, int rows, int cols) {
    int threads_per_block = 256;
    int blocks = (cols + threads_per_block - 1) / threads_per_block;

    row_sum_kernel<<<blocks, threads_per_block>>>(dst, src, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in row_sum: %s\n", cudaGetErrorString(err));
    }
}

__global__ void zero_matrix_kernel(float* matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    matrix[idx] = 0.0;
}

void zero_matrix(float* matrix, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    zero_matrix_kernel<<<blocks, threads_per_block>>>(matrix, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in zero_matrix: %s\n", cudaGetErrorString(err));
    }
}

__global__ void fill_matrix_kernel(float* matrix, double value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    matrix[idx] = value;
}

void fill_matrix(float* matrix, double value, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    fill_matrix_kernel<<<blocks, threads_per_block>>>(matrix, value, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in fill_matrix: %s\n", cudaGetErrorString(err));
    }
}

__global__ void element_div_kernel(float* out, const float* a, const float* b, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;

    float denom = b[idx];
    out[idx] = denom == 0.0 ? 0.0 : a[idx] / denom;
}

void element_div(float* out, const float* a, const float* b, int size){
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    element_div_kernel<<<blocks, threads_per_block>>>(out, a, b, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in element_div: %s\n", cudaGetErrorString(err));
    }
}

__global__ void softmax_backward_kernel(float* output, const float* grad, const float* softmax_out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* grad_row = grad + row * cols;
    const float* softmax_row = softmax_out + row * cols;
    float* output_row = output + row * cols;
    
    // Compute sum of softmax * grad for this row
    float sum = 0.0;
    for (int j = 0; j < cols; j++) {
        sum += softmax_row[j] * grad_row[j];
    }
    
    // Compute softmax backward: softmax * (grad - sum)
    for (int j = 0; j < cols; j++) {
        output_row[j] = softmax_row[j] * (grad_row[j] - sum);
    }
}

void softmax_backward(float* output, const float* grad, const float* softmax_out, int rows, int cols) {
    softmax_backward_kernel<<<rows, 1>>>(output, grad, softmax_out, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in softmax_backward: %s\n", cudaGetErrorString(err));
    }
}


__global__ void element_log_kernel(float* out, const float* in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float val = in[idx];
    out[idx] = logf(val);
}

void element_log(float* out, const float* in, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    element_log_kernel<<<blocks, threads_per_block>>>(out, in, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in element_log: %s\n", cudaGetErrorString(err));
    }
}

__global__ void cross_entropy_loss_gradient_kernel(const float* pred, const float* target, float* grad, float* loss, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float p = pred[idx];
    float t = target[idx];
    grad[idx] = p - t;

    float contrib = -t * logf(fmaxf(p, 1e-15f));
    atomicAdd(loss, contrib);
}

__global__ void softmax_cross_entropy_label_kernel(const float* pred, const int* labels,
                                                    float* grad, float* loss,
                                                    int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_pred = pred + row * cols;
    float* row_grad = grad + row * cols;

    // Find maximum value in the row for numerical stability
    float max_val = row_pred[0];
    for (int j = 1; j < cols; ++j) {
        float v = row_pred[j];
        if (v > max_val) max_val = v;
    }

    // Compute exponentials and their sum
    float sum = 0.0;
    for (int j = 0; j < cols; ++j) {
        float e = expf(row_pred[j] - max_val);
        row_grad[j] = e;
        sum += e;
    }

    // Normalize to obtain probabilities
    for (int j = 0; j < cols; ++j) {
        row_grad[j] /= sum;
    }

    int label = labels[row];
    if (label >= 0 && label < cols) {
        float p = row_grad[label];
        row_grad[label] = p - 1.0;
        float contrib = -logf(fmaxf(p, 1e-15f));
        atomicAdd(loss, contrib);
    }
}

void cross_entropy_loss_gradient(float* pred, float* target,
                                 float* grad, float* loss,
                                 int rows, int cols) {
    int total = rows * cols;
    cudaMemset(loss, 0, sizeof(float));
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    cross_entropy_loss_gradient_kernel<<<blocks, threads>>>(pred, target, grad, loss, total);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in cross_entropy_loss_gradient: %s\n", cudaGetErrorString(err));

    }
}

void softmax_cross_entropy_label(float* pred, const int* labels,
                                 float* grad, float* loss,
                                 int rows, int cols) {
    cudaMemset(loss, 0, sizeof(float));
    softmax_cross_entropy_label_kernel<<<rows, 1>>>(pred, labels, grad, loss, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in softmax_cross_entropy_label: %s\n", cudaGetErrorString(err));
    }
}

// ---- Q8_0-style quantized matmul: y[M,N] = x[M,K] * dequant(W) ----
// W is quantized weights laid out row-major [N, K] (out-major) as int8 (q),
// with one fp32 scale per BLOCK (=32) contiguous K elements per output column,
// scales laid out [N, ceil(K/BLOCK)]. This reproduces row-major fp32 GEMM
// semantics result[m,n] = sum_k x[m,k] * (q[n,k] * scale[n, k/BLOCK]).
// One CUDA block computes one (m,n) output via threaded reduction over K.
#define Q8_BLK 32
__global__ void gemm_q8_f32_kernel(const float* __restrict__ x,
                                   const signed char* __restrict__ q,
                                   const float* __restrict__ scales,
                                   float* __restrict__ y,
                                   int M, int N, int K) {
    int n = blockIdx.x; // output column (0..N)
    int m = blockIdx.y; // activation row (0..M)
    if (n >= N || m >= M) return;

    int nblocks = (K + Q8_BLK - 1) / Q8_BLK;
    const float* xrow = x + (long)m * K;
    const signed char* qrow = q + (long)n * K;
    const float* srow = scales + (long)n * nblocks;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Each thread strides over the full K dimension so all threads do work
    // (the previous one-thread-per-32-block scheme left most threads idle for
    // small K). Scale is per-32-element block; srow is tiny and L1/L2 cached.
    float partial = 0.0f;
    for (int k = tid; k < K; k += nthreads) {
        partial += (float)qrow[k] * xrow[k] * srow[k >> 5];
    }

    extern __shared__ float sdata[];
    sdata[tid] = partial;
    __syncthreads();
    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[(long)m * N + n] = sdata[0];
}

void gemm_q8_f32(const float* x, const signed char* q, const float* scales,
                 float* y, int M, int N, int K) {
    int threads = 256;
    dim3 grid(N, M);
    size_t shmem = threads * sizeof(float);
    gemm_q8_f32_kernel<<<grid, threads, shmem>>>(x, q, scales, y, M, N, K);
    // Avoid a full device sync on every projection/lm_head call. Callers read
    // results back via a default-stream D2H memcpy, which is ordered after this
    // kernel and provides the needed synchronization. Just surface launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gemm_q8_f32: %s\n", cudaGetErrorString(err));
    }
}

// ---- Q4_0-style quantized GEMM (4-bit symmetric, block=32) ----
// Mirrors gemm_q8_f32 but weights are 4-bit. Two nibbles are packed per byte
// along K: byte q[n, k/2] holds k in the low nibble and k+1 in the high nibble,
// each stored as (value + 8), so value = nibble - 8. The host quantizer produces
// symmetric values in [-7, 7] (stored nibbles 1..15); the kernel decodes the
// full [-8, 7] nibble range so any unused/padding nibble is handled safely.
// One fp32 scale per BLOCK(=32) contiguous K elements per output column, scales
// laid out [N, ceil(K/32)]. Reproduces row-major fp32 GEMM semantics:
//   result[m,n] = sum_k x[m,k] * (value(n,k) * scale[n, k/32]).
#define Q4_BLK 32
__global__ void gemm_q4_f32_kernel(const float* __restrict__ x,
                                   const unsigned char* __restrict__ q,
                                   const float* __restrict__ scales,
                                   float* __restrict__ y,
                                   int M, int N, int K) {
    int n = blockIdx.x; // output column (0..N)
    int m = blockIdx.y; // activation row (0..M)
    if (n >= N || m >= M) return;

    int nblocks = (K + Q4_BLK - 1) / Q4_BLK;
    int kbytes = (K + 1) >> 1; // packed bytes per output column
    const float* xrow = x + (long)m * K;
    const unsigned char* qrow = q + (long)n * kbytes;
    const float* srow = scales + (long)n * nblocks;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Each thread strides over the full K dimension, unpacking its nibble.
    float partial = 0.0f;
    for (int k = tid; k < K; k += nthreads) {
        unsigned char byte = qrow[k >> 1];
        int nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
        partial += (float)(nib - 8) * xrow[k] * srow[k >> 5];
    }

    extern __shared__ float sdata[];
    sdata[tid] = partial;
    __syncthreads();
    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[(long)m * N + n] = sdata[0];
}

// Output columns handled per block, one warp each. Raising this increases how
// many columns share a single staged activation tile.
#define Q4_COLS_PER_BLOCK 8
// Activation elements staged in shared memory per iteration. 1024 = 32 scale
// blocks = exactly one block per warp lane, so no lane is idle on a full tile.
#define Q4_TILE 1024

// Vectorized Q4 GEMM.
//
// Improves on gemm_q4_f32_kernel in four ways:
//   * one WARP per output column instead of one block, so the reduction is a
//     __shfl_down_sync chain rather than an 8-step shared-memory tree with a
//     __syncthreads between every step;
//   * Q4_COLS_PER_BLOCK columns share one staged activation tile, so x is read
//     from global memory once per block instead of once per output column;
//   * weights move as uint4 (16 bytes = 32 nibbles = exactly one scale block),
//     so each lane issues one 16-byte load where the scalar kernel issued 32
//     single-byte loads, and consecutive lanes cover 512 contiguous bytes;
//   * the per-lane unpack loop is fully unrolled over the 16 bytes.
//
// Requires K % Q4_BLK == 0. That also makes every packed column row a multiple
// of 16 bytes (kbytes = K/2), which is what makes the uint4 loads legal; the
// caller falls back to the scalar kernel otherwise.
__global__ void gemm_q4_f32_vec_kernel(const float* __restrict__ x,
                                       const unsigned char* __restrict__ q,
                                       const float* __restrict__ scales,
                                       float* __restrict__ y,
                                       int M, int N, int K) {
    int m = blockIdx.y;
    if (m >= M) return; // uniform across the block, so no divergent barrier

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * Q4_COLS_PER_BLOCK + warp;

    int nblocks = K >> 5; // exact, K % 32 == 0
    int kbytes = K >> 1;
    const float* xrow = x + (long)m * K;
    const unsigned char* qrow = q + (long)n * kbytes;
    const float* srow = scales + (long)n * nblocks;

    extern __shared__ float xs[]; // Q4_TILE floats

    float acc = 0.0f;
    for (int ktile = 0; ktile < K; ktile += Q4_TILE) {
        int tile = min(Q4_TILE, K - ktile);
        // Whole block cooperates on the load, then every warp reads it back.
        for (int i = threadIdx.x; i < tile; i += blockDim.x) xs[i] = xrow[ktile + i];
        __syncthreads();

        if (n < N && lane * Q4_BLK < tile) {
            int kb = (ktile >> 5) + lane; // this lane's scale block
            uint4 packed = *reinterpret_cast<const uint4*>(qrow + (long)kb * 16);
            const unsigned char* b = reinterpret_cast<const unsigned char*>(&packed);
            const float* xb = xs + lane * Q4_BLK;
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                unsigned char byte = b[j];
                sum += (float)((int)(byte & 0x0F) - 8) * xb[2 * j];
                sum += (float)((int)(byte >> 4) - 8) * xb[2 * j + 1];
            }
            acc += sum * srow[kb];
        }
        __syncthreads(); // tile is reused next iteration
    }

    // Warp reduction. Every lane reaches this, including lanes whose column is
    // out of range, so the shuffle mask stays full.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0 && n < N) y[(long)m * N + n] = acc;
}

// Force the scalar Q4 kernel even on shapes the vectorized one supports. Exists
// so the two kernels can be A/B'd end to end on a real model, where every shape
// otherwise takes the vectorized path. Read once, on first use.
static int q4_scalar_forced = -1;
static inline bool q4_force_scalar() {
    if (q4_scalar_forced < 0) {
        const char* e = getenv("SHAINET_Q4_SCALAR");
        q4_scalar_forced = (e && e[0] == '1') ? 1 : 0;
    }
    return q4_scalar_forced == 1;
}

void gemm_q4_f32(const float* x, const unsigned char* q, const float* scales,
                 float* y, int M, int N, int K) {
    if (K % Q4_BLK == 0 && !q4_force_scalar()) {
        int threads = Q4_COLS_PER_BLOCK * 32;
        dim3 grid((N + Q4_COLS_PER_BLOCK - 1) / Q4_COLS_PER_BLOCK, M);
        size_t shmem = Q4_TILE * sizeof(float);
        gemm_q4_f32_vec_kernel<<<grid, threads, shmem>>>(x, q, scales, y, M, N, K);
    } else {
        // Packed rows are not 16-byte aligned for this K, so uint4 loads would
        // be illegal. Correctness first.
        int threads = 256;
        dim3 grid(N, M);
        size_t shmem = threads * sizeof(float);
        gemm_q4_f32_kernel<<<grid, threads, shmem>>>(x, q, scales, y, M, N, K);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gemm_q4_f32: %s\n", cudaGetErrorString(err));
    }
}

// ---- KV-cache attention (LLaMA decode/prefill) ----
// Device cache layout: [num_kv_heads, capacity, head_dim] for both K and V.
// Templates cannot carry C linkage, so the generic kernels live in a nested
// extern "C++" block; only the concrete wrappers below stay callable by name.
extern "C++" {

// Storage traits for the KV cache element type. The cache may be kept in fp16
// to halve its VRAM footprint (it is the dominant consumer at long context);
// all arithmetic still happens in fp32 after conversion, so only the stored
// precision changes.
template <typename KV> struct KVStore;
template <> struct KVStore<float> {
    __device__ static inline float load(const float* p, long i) { return p[i]; }
    __device__ static inline void store(float* p, long i, float v) { p[i] = v; }
};
template <> struct KVStore<__half> {
    __device__ static inline float load(const __half* p, long i) { return __half2float(p[i]); }
    __device__ static inline void store(__half* p, long i, float v) { p[i] = __float2half(v); }
};

// `staging` holds the newly appended rows, already RoPE'd (K) on the host:
//   [num_kv_heads * new_tokens * head_dim] K chunks (kv_head-major, then
//   token-major — i.e. each kv_head's tail is contiguous), followed by the
//   same layout for V. Scatter them into the cache at position start_pos.
// Staging is always fp32; the cache element type is the template parameter, so
// the fp16 cache is written through a conversion here.
template <typename KV>
__global__ void kv_cache_append_kernel_t(const float* __restrict__ staging,
                                         KV* __restrict__ kc,
                                         KV* __restrict__ vc,
                                         int new_tokens, int start_pos,
                                         int num_kv_heads, int head_dim,
                                         int capacity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = new_tokens * head_dim;       // floats per kv_head per tensor
    int total = num_kv_heads * chunk;        // floats per tensor (K or V)
    if (idx >= 2 * total) return;
    int is_v = idx >= total;
    int r = is_v ? idx - total : idx;
    int kv_h = r / chunk;
    int rem = r - kv_h * chunk;              // t * head_dim + d
    long dst = ((long)kv_h * capacity + start_pos) * head_dim + rem;
    if (is_v) KVStore<KV>::store(vc, dst, staging[idx]);
    else      KVStore<KV>::store(kc, dst, staging[idx]);
}

} // extern "C++"

void kv_cache_append_f32(const float* staging, float* kc, float* vc,
                         int new_tokens, int start_pos, int num_kv_heads,
                         int head_dim, int capacity) {
    int total = 2 * num_kv_heads * new_tokens * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_cache_append_kernel_t<float><<<blocks, threads>>>(staging, kc, vc, new_tokens,
                                                         start_pos, num_kv_heads,
                                                         head_dim, capacity);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in kv_cache_append_f32: %s\n", cudaGetErrorString(err));
    }
}

// fp16 cache variant. `kc`/`vc` are __half buffers passed from Crystal as
// UInt16 pointers (Crystal has no native half type); staging stays fp32.
void kv_cache_append_f16(const float* staging, unsigned short* kc, unsigned short* vc,
                         int new_tokens, int start_pos, int num_kv_heads,
                         int head_dim, int capacity) {
    int total = 2 * num_kv_heads * new_tokens * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_cache_append_kernel_t<__half><<<blocks, threads>>>(staging,
                                                          reinterpret_cast<__half*>(kc),
                                                          reinterpret_cast<__half*>(vc),
                                                          new_tokens, start_pos,
                                                          num_kv_heads, head_dim,
                                                          capacity);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in kv_cache_append_f16: %s\n", cudaGetErrorString(err));
    }
}

extern "C++" {

// Causal attention over the device KV cache. One block per (head, token):
//   scores[j] = scale * (q_h · K[kv_h][j])  for j < visible = start_pos+i+1
//   p = softmax(scores); out[h][d] = sum_j p[j] * V[kv_h][j][d]
// q/out are [new_tokens, num_heads*head_dim] row-major (q already RoPE'd).
// ws is a global scratch of at least num_heads*new_tokens*total_len floats.
// GQA: query head h reads kv head h / heads_per_kv.
// KV is the cache storage type (float or __half); loads convert to fp32 so the
// dot product, softmax and weighted sum are identical in both instantiations.
template <typename KV>
__global__ void attention_kv_kernel_t(const float* __restrict__ q,
                                      const KV* __restrict__ kc,
                                      const KV* __restrict__ vc,
                                      float* __restrict__ out,
                                      float* __restrict__ ws,
                                      int new_tokens, int start_pos,
                                      int num_heads, int heads_per_kv,
                                      int head_dim, int capacity, float scale) {
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= num_heads || i >= new_tokens) return;

    int kv_h = h / heads_per_kv;
    int visible = start_pos + i + 1;
    int total_len = start_pos + new_tokens;
    int d_model = num_heads * head_dim;

    const float* qv = q + (long)i * d_model + (long)h * head_dim;
    const KV* kh = kc + (long)kv_h * capacity * head_dim;
    const KV* vh = vc + (long)kv_h * capacity * head_dim;
    float* wsrow = ws + ((long)h * new_tokens + i) * total_len;

    int tid = threadIdx.x;
    int nt = blockDim.x;
    extern __shared__ float smem[];
    float* q_s = smem;            // head_dim
    float* red = smem + head_dim; // nt

    for (int d = tid; d < head_dim; d += nt) q_s[d] = qv[d];
    __syncthreads();

    // Pass 1: scores into ws, track max for stable softmax.
    float lmax = -INFINITY;
    for (int j = tid; j < visible; j += nt) {
        const KV* krow = kh + (long)j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q_s[d] * KVStore<KV>::load(krow, d);
        dot *= scale;
        wsrow[j] = dot;
        if (dot > lmax) lmax = dot;
    }
    red[tid] = lmax;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float gmax = red[0];
    __syncthreads();

    // Pass 2: exponentiate + sum.
    float lsum = 0.0f;
    for (int j = tid; j < visible; j += nt) {
        float e = expf(wsrow[j] - gmax);
        wsrow[j] = e;
        lsum += e;
    }
    red[tid] = lsum;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / red[0];

    // Pass 3: weighted sum of V rows. Threads cover head_dim, so adjacent
    // threads read adjacent V elements (coalesced per j).
    for (int d = tid; d < head_dim; d += nt) {
        float acc = 0.0f;
        for (int j = 0; j < visible; ++j) acc += wsrow[j] * KVStore<KV>::load(vh, (long)j * head_dim + d);
        out[(long)i * d_model + (long)h * head_dim + d] = acc * inv_sum;
    }
}

} // extern "C++"

void attention_kv_f32(const float* q, const float* kc, const float* vc,
                      float* out, float* ws, int new_tokens, int start_pos,
                      int num_heads, int heads_per_kv, int head_dim,
                      int capacity, float scale) {
    int threads = 128;
    dim3 grid(num_heads, new_tokens);
    size_t shmem = (head_dim + threads) * sizeof(float);
    attention_kv_kernel_t<float><<<grid, threads, shmem>>>(q, kc, vc, out, ws,
                                                           new_tokens, start_pos,
                                                           num_heads, heads_per_kv,
                                                           head_dim, capacity, scale);
    // No device sync: callers read `out` back via a default-stream D2H memcpy
    // which is ordered after this kernel. Just surface launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in attention_kv_f32: %s\n", cudaGetErrorString(err));
    }
}

// fp16 cache variant. Q, out and ws stay fp32; only the cache is half.
void attention_kv_f16(const float* q, const unsigned short* kc, const unsigned short* vc,
                      float* out, float* ws, int new_tokens, int start_pos,
                      int num_heads, int heads_per_kv, int head_dim,
                      int capacity, float scale) {
    int threads = 128;
    dim3 grid(num_heads, new_tokens);
    size_t shmem = (head_dim + threads) * sizeof(float);
    attention_kv_kernel_t<__half><<<grid, threads, shmem>>>(q,
                                                            reinterpret_cast<const __half*>(kc),
                                                            reinterpret_cast<const __half*>(vc),
                                                            out, ws,
                                                            new_tokens, start_pos,
                                                            num_heads, heads_per_kv,
                                                            head_dim, capacity, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in attention_kv_f16: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"
