#include <curand_kernel.h>
#include <cstdio>

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

} // extern "C"
