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

// One block per row, threads cooperating across the row. It was <<<rows, 1>>> with a
// serial column loop, which made it useless for batching a d_model-wide gather: one
// thread copying 2048 floats per row.
__global__ void gather_rows_kernel(float* out, const float* in, const int* ids, int rows, int cols) {
    int row = blockIdx.x;
    if(row >= rows) return;
    int id = ids[row];
    const float *row_in = in + (long)id * cols;
    float *row_out = out + (long)row * cols;
    for(int j = threadIdx.x; j < cols; j += blockDim.x){
        row_out[j] = row_in[j];
    }
}

// No cudaDeviceSynchronize: it is ordered on the default stream against whatever
// consumes the gathered batch, and the blocking sync per call was costing a device
// round trip on every gather.
void gather_rows(float* out, const float* in, const int* ids, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    int threads = cols < 256 ? 32 : 256;
    gather_rows_kernel<<<rows, threads>>>(out, in, ids, rows, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gather_rows: %s\n", cudaGetErrorString(err));
    }
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

// out = x / sqrt(mean(x^2) + eps) * gamma, one BLOCK per row so the sum of squares
// is a shared-memory reduction rather than a host loop.
//
// Until this existed there was no RMSNorm kernel at all, and the CudaMatrix path
// read the row back to the host, normalised it there and pushed it back, which made
// a device-resident block chain impossible: every norm broke the chain.
//
// Accumulates in float, not double. The host path sums in Float64, so results are
// close but not bit-identical; the specs assert parity to a tolerance and the real
// check is that greedy decoding still picks the same tokens.
__global__ void rms_norm_forward_kernel(float* out, const float* x, const float* gamma,
                                        int rows, int cols, float eps) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* xr = x + (size_t)row * (size_t)cols;
    float* orow = out + (size_t)row * (size_t)cols;

    float local = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = xr[j];
        local += v * v;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv = 1.0f / sqrtf(sdata[0] / (float)cols + eps);
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        orow[j] = xr[j] * inv * gamma[j];
    }
}

// dst[idx[r]] += w[r] * src[r], the routing weight applied as the batch is returned
// to token order.
//
// No atomics: a token selects a given expert at most once, so the rows of ONE launch
// never collide, and separate launches are ordered on the default stream.
__global__ void scatter_add_rows_kernel(float* __restrict__ dst,
                                        const float* __restrict__ src,
                                        const int* __restrict__ idx,
                                        const float* __restrict__ w,
                                        int n, int cols) {
    int r = blockIdx.x;
    if (r >= n) return;
    const float* s = src + (long)r * cols;
    float* d = dst + (long)idx[r] * cols;
    float wr = w[r];
    for (int c = threadIdx.x; c < cols; c += blockDim.x) d[c] += wr * s[c];
}

// No cudaDeviceSynchronize, same reasoning as swiglu_forward: ordered on the default
// stream between the launches that produce and consume the batch.
void scatter_add_rows(float* dst, const float* src, const int* idx,
                      const float* w, int n, int cols) {
    if (n <= 0 || cols <= 0) return;
    int threads = cols < 256 ? 32 : 256;
    scatter_add_rows_kernel<<<n, threads>>>(dst, src, idx, w, n, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in scatter_add_rows: %s\n", cudaGetErrorString(err));
    }
}

// No cudaDeviceSynchronize, same reasoning as swiglu_forward: it is ordered on the
// default stream between the launches that produce and consume the row.
void rms_norm_forward(float* out, const float* x, const float* gamma,
                      int rows, int cols, float eps) {
    int threads_per_block = 256;
    // Power-of-two thread count is required by the halving reduction above.
    size_t shmem = threads_per_block * sizeof(float);

    rms_norm_forward_kernel<<<rows, threads_per_block, shmem>>>(out, x, gamma, rows, cols, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rms_norm_forward: %s\n", cudaGetErrorString(err));
    }
}

// dst = dst + src, elementwise, for the block's residual adds. cuBLAS axpy could do
// this, but it needs a handle per call and the residual is on the hot path.
__global__ void add_inplace_kernel(float* dst, const float* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dst[idx] += src[idx];
}

void add_inplace(float* dst, const float* src, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    add_inplace_kernel<<<blocks, threads_per_block>>>(dst, src, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in add_inplace: %s\n", cudaGetErrorString(err));
    }
}

// Rotary position embedding, HF half-split layout, applied in place to a single
// token's row of `heads` heads each `head_dim` wide.
//
// Until this existed RoPE ran on the host, which is the whole reason q/k/v had to be
// read back from the device every layer. inv_freq is precomputed once per block and
// kept on the device, so a decode step passes only the position.
//
// One block per head, half the head_dim worth of threads doing the pair rotation.
// rot_dim is the number of leading dimensions that are rotated. Qwen3.5 sets
// partial_rotary_factor 0.25 on a head_dim of 256, so only the first 64 are rotated and the
// remaining 192 pass through untouched. rot_dim == head_dim is the ordinary full-rotary case.
__global__ void rope_forward_kernel(float* x, const float* inv_freq, int pos,
                                    int heads, int head_dim, int rot_dim) {
    int head = blockIdx.x;
    if (head >= heads) return;
    int half = rot_dim / 2;
    float* row = x + (size_t)head * (size_t)head_dim;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float angle = (float)pos * inv_freq[i];
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = row[i];
        float x1 = row[i + half];
        row[i] = x0 * c - x1 * s;
        row[i + half] = x1 * c + x0 * s;
    }
}

void rope_forward(float* x, const float* inv_freq, int pos, int heads, int head_dim, int rot_dim) {
    int threads = 128;
    if (rot_dim <= 0 || rot_dim > head_dim) rot_dim = head_dim;
    rope_forward_kernel<<<heads, threads>>>(x, inv_freq, pos, heads, head_dim, rot_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rope_forward: %s\n", cudaGetErrorString(err));
    }
}

// Qwen3 QK-norm: RMSNorm applied independently to each head's slice, over head_dim,
// in place. Same shape of reduction as rms_norm_forward but per head rather than per
// row, and gamma is shared across heads.
__global__ void head_rmsnorm_kernel(float* x, const float* gamma, int heads,
                                    int head_dim, float eps) {
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    if (head >= heads) return;
    float* row = x + (size_t)head * (size_t)head_dim;

    float local = 0.0f;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float v = row[j];
        local += v * v;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv = 1.0f / sqrtf(sdata[0] / (float)head_dim + eps);
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        row[j] = row[j] * inv * gamma[j];
    }
}

// --- Multi-row variants for device-resident PREFILL -------------------------------
//
// The single-row kernels above serve decode, where there is one token and one
// position. Prefill has `rows` tokens at consecutive positions, laid out
// [rows, heads * head_dim] row-major, so these take a row dimension and derive each
// token's position from base_pos. The math is identical per row, which is what makes
// parity with the host path assertable.

__global__ void rope_forward_rows_kernel(float* x, const float* inv_freq, int base_pos,
                                        int rows, int heads, int head_dim, int rot_dim) {
    int head = blockIdx.x;
    int r = blockIdx.y;
    if (head >= heads || r >= rows) return;
    int half = rot_dim / 2;
    int stride = heads * head_dim;
    float* row = x + (size_t)r * (size_t)stride + (size_t)head * (size_t)head_dim;
    float pos = (float)(base_pos + r);

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float angle = pos * inv_freq[i];
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = row[i];
        float x1 = row[i + half];
        row[i] = x0 * c - x1 * s;
        row[i + half] = x1 * c + x0 * s;
    }
}

void rope_forward_rows(float* x, const float* inv_freq, int base_pos,
                       int rows, int heads, int head_dim, int rot_dim) {
    if (rows <= 0 || heads <= 0) return;
    int threads = 128;
    if (rot_dim <= 0 || rot_dim > head_dim) rot_dim = head_dim;
    dim3 grid(heads, rows);
    rope_forward_rows_kernel<<<grid, threads>>>(x, inv_freq, base_pos, rows, heads, head_dim, rot_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rope_forward_rows: %s\n", cudaGetErrorString(err));
    }
}

__global__ void head_rmsnorm_rows_kernel(float* x, const float* gamma, int rows,
                                        int heads, int head_dim, float eps) {
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    int r = blockIdx.y;
    if (head >= heads || r >= rows) return;
    int stride = heads * head_dim;
    float* row = x + (size_t)r * (size_t)stride + (size_t)head * (size_t)head_dim;

    float local = 0.0f;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float v = row[j];
        local += v * v;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv = 1.0f / sqrtf(sdata[0] / (float)head_dim + eps);
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        row[j] = row[j] * inv * gamma[j];
    }
}

void head_rmsnorm_rows(float* x, const float* gamma, int rows, int heads,
                       int head_dim, float eps) {
    if (rows <= 0 || heads <= 0) return;
    int threads = 128; // power of two: the halving reduction above requires it
    size_t shmem = threads * sizeof(float);
    dim3 grid(heads, rows);
    head_rmsnorm_rows_kernel<<<grid, threads, shmem>>>(x, gamma, rows, heads, head_dim, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in head_rmsnorm_rows: %s\n", cudaGetErrorString(err));
    }
}

// x[r, c] += bias[c]: the projection biases are per output column, broadcast over
// tokens. Qwen2 has them, LLaMA does not, so this is a no-op path for some models.
__global__ void add_bias_rows_kernel(float* x, const float* bias, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float* row = x + (size_t)r * (size_t)cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) row[c] += bias[c];
}

void add_bias_rows(float* x, const float* bias, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    int threads = cols < 256 ? 32 : 256;
    add_bias_rows_kernel<<<rows, threads>>>(x, bias, rows, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in add_bias_rows: %s\n", cudaGetErrorString(err));
    }
}

// Repack token-major [rows, kv_heads * head_dim] into the kv-head-major layout the
// existing KV append expects: dst[kv_h][t][d]. This is what lets the projections stay
// on the device while append_kv and attend_kv keep working unchanged, instead of the
// host rebuilding the same blob from its mirror every chunk.
__global__ void pack_kv_heads_kernel(float* dst, const float* src,
                                    int rows, int kv_heads, int head_dim) {
    int kv_h = blockIdx.x;
    int r = blockIdx.y;
    if (kv_h >= kv_heads || r >= rows) return;
    int stride = kv_heads * head_dim;
    const float* s = src + (size_t)r * (size_t)stride + (size_t)kv_h * (size_t)head_dim;
    float* d = dst + (size_t)kv_h * (size_t)rows * (size_t)head_dim + (size_t)r * (size_t)head_dim;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) d[j] = s[j];
}

void pack_kv_heads(float* dst, const float* src, int rows, int kv_heads, int head_dim) {
    if (rows <= 0 || kv_heads <= 0) return;
    int threads = head_dim < 256 ? 32 : 256;
    dim3 grid(kv_heads, rows);
    pack_kv_heads_kernel<<<grid, threads>>>(dst, src, rows, kv_heads, head_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in pack_kv_heads: %s\n", cudaGetErrorString(err));
    }
}

void head_rmsnorm(float* x, const float* gamma, int heads, int head_dim, float eps) {
    int threads = 128;
    size_t shmem = threads * sizeof(float);
    head_rmsnorm_kernel<<<heads, threads, shmem>>>(x, gamma, heads, head_dim, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in head_rmsnorm: %s\n", cudaGetErrorString(err));
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
// Blocks sharing one fp32 super-block scale. Each block's own scale is a byte
// relative to it: effective = d[block / Q4_SUPER] * sub[block] / 255. Must match
// SHAInet::Q4CudaMatrix::SUPER.
#define Q4_SUPER 8
__global__ void gemm_q4_f32_kernel(const float* __restrict__ x,
                                   const unsigned char* __restrict__ q,
                                   const float* __restrict__ d,
                                   const unsigned char* __restrict__ sub,
                                   float* __restrict__ y,
                                   int M, int N, int K) {
    int n = blockIdx.x; // output column (0..N)
    int m = blockIdx.y; // activation row (0..M)
    if (n >= N || m >= M) return;

    int nblocks = (K + Q4_BLK - 1) / Q4_BLK;
    int nsupers = (nblocks + Q4_SUPER - 1) / Q4_SUPER;
    int kbytes = (K + 1) >> 1; // packed bytes per output column
    const float* xrow = x + (long)m * K;
    const unsigned char* qrow = q + (long)n * kbytes;
    const float* drow = d + (long)n * nsupers;
    const unsigned char* subrow = sub + (long)n * nblocks;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Each thread strides over the full K dimension, unpacking its nibble. The
    // effective block scale is d[block / Q4_SUPER] * sub[block] / 255.
    float partial = 0.0f;
    for (int k = tid; k < K; k += nthreads) {
        unsigned char byte = qrow[k >> 1];
        int nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
        int b = k >> 5;
        float eff = drow[b / Q4_SUPER] * (float)subrow[b] * (1.0f / 255.0f);
        partial += (float)(nib - 8) * xrow[k] * eff;
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
                                       const float* __restrict__ d,
                                       const unsigned char* __restrict__ sub,
                                       float* __restrict__ y,
                                       int M, int N, int K) {
    int m = blockIdx.y;
    if (m >= M) return; // uniform across the block, so no divergent barrier

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * Q4_COLS_PER_BLOCK + warp;

    int nblocks = K >> 5; // exact, K % 32 == 0
    int nsupers = (nblocks + Q4_SUPER - 1) / Q4_SUPER;
    int kbytes = K >> 1;
    const float* xrow = x + (long)m * K;
    const unsigned char* qrow = q + (long)n * kbytes;
    const float* drow = d + (long)n * nsupers;
    const unsigned char* subrow = sub + (long)n * nblocks;

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
            acc += sum * (drow[kb / Q4_SUPER] * (float)subrow[kb] * (1.0f / 255.0f));
        }
        __syncthreads(); // tile is reused next iteration
    }

    // Warp reduction. Every lane reaches this, including lanes whose column is
    // out of range, so the shuffle mask stays full.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0 && n < N) y[(long)m * N + n] = acc;
}

// Register-tiled Q4 GEMM. Addresses what the current kernel is actually limited by, which
// is NOT what it looks like from the source.
//
// Bisection on the vectorized kernel: separately removing the weight load, the activation
// reads, the nibble decode, and even the whole shared-memory staging each changed the time
// by under 3%. Nothing in the inner loop dominates. What does is the shape of the work: at
// the expert projection (M=128, K=2048, N=768) it launches 12288 blocks x 256 threads for
// only 201M MACs, so each thread performs 64 MACs and then pays two __syncthreads and a
// five-step warp-shuffle reduction. Overhead per thread is comparable to its useful work,
// and it reached 1004 GFLOP/s, about 3.5% of this card's fp32 peak.
//
// So this kernel gives each thread a TILE of outputs it owns across the whole K:
//   - 8192 MACs per thread instead of 64
//   - no cross-lane reduction at all, since a thread accumulates its own outputs
//   - the Q4 weights are dequantized into shared memory ONCE per block per k-tile,
//     instead of once per row-block
//
// Deliberately NOT the llama.cpp approach of int8 DP4A with quantized activations: that
// accelerates the decode and the MAC, which the bisection above shows are already free
// here, and it would cost activation precision for nothing.
#define MMQ_BM 32 // rows per block
#define MMQ_BN 32 // cols per block
#define MMQ_TM 2  // rows per thread
#define MMQ_TN 2  // cols per thread
// k per iteration is Q4_BLK: one k-tile is exactly one quantization block, so a column's
// scale is loaded once per tile rather than recomputed per element.

__global__ void gemm_q4_f32_tiled_kernel(const float* __restrict__ x,
                                         const unsigned char* __restrict__ q,
                                         const float* __restrict__ d,
                                         const unsigned char* __restrict__ sub,
                                         float* __restrict__ y,
                                         int M, int N, int K) {
    __shared__ float xs[MMQ_BM][Q4_BLK];
    __shared__ float ws[Q4_BLK][MMQ_BN];

    const int m0 = blockIdx.y * MMQ_BM;
    const int n0 = blockIdx.x * MMQ_BN;
    const int tid = threadIdx.x;

    // Thread's output tile: MMQ_TM rows x MMQ_TN cols.
    const int tm = (tid / (MMQ_BN / MMQ_TN)) * MMQ_TM;
    const int tn = (tid % (MMQ_BN / MMQ_TN)) * MMQ_TN;

    const int nblocks = K >> 5; // K % 32 == 0, checked by the caller
    const int nsupers = (nblocks + Q4_SUPER - 1) / Q4_SUPER;
    const int kbytes = K >> 1;

    float acc[MMQ_TM][MMQ_TN];
#pragma unroll
    for (int i = 0; i < MMQ_TM; ++i)
#pragma unroll
        for (int j = 0; j < MMQ_TN; ++j) acc[i][j] = 0.0f;

    for (int kb = 0; kb < nblocks; ++kb) {
        const int k0 = kb * Q4_BLK;

        // Stage the activation tile: MMQ_BM x 32 floats.
        for (int idx = tid; idx < MMQ_BM * Q4_BLK; idx += blockDim.x) {
            const int r = idx / Q4_BLK;
            const int c = idx - r * Q4_BLK;
            const int gm = m0 + r;
            xs[r][c] = (gm < M) ? x[(long)gm * K + k0 + c] : 0.0f;
        }

        // Dequantize this k-block for all MMQ_BN columns into shared memory. Each column's
        // 32 values are 16 packed bytes; four threads share a column, eight values each.
        for (int idx = tid; idx < MMQ_BN * 4; idx += blockDim.x) {
            const int col = idx & (MMQ_BN - 1);
            const int part = idx / MMQ_BN; // 0..3
            const int gn = n0 + col;
            if (gn < N) {
                const unsigned char* qrow = q + (long)gn * kbytes + (long)kb * 16;
                const float scale = d[(long)gn * nsupers + kb / Q4_SUPER] *
                                    (float)sub[(long)gn * nblocks + kb] * (1.0f / 255.0f);
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const unsigned char byte = qrow[part * 4 + b];
                    const int kk = (part * 4 + b) * 2;
                    ws[kk][col] = (float)((int)(byte & 0x0F) - 8) * scale;
                    ws[kk + 1][col] = (float)((int)(byte >> 4) - 8) * scale;
                }
            } else {
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const int kk = (part * 4 + b) * 2;
                    ws[kk][col] = 0.0f;
                    ws[kk + 1][col] = 0.0f;
                }
            }
        }
        __syncthreads();

        // The point: 128 MACs per thread per k-tile, from registers, no reduction.
#pragma unroll
        for (int kk = 0; kk < Q4_BLK; ++kk) {
            float a[MMQ_TM];
            float b[MMQ_TN];
#pragma unroll
            for (int i = 0; i < MMQ_TM; ++i) a[i] = xs[tm + i][kk];
#pragma unroll
            for (int j = 0; j < MMQ_TN; ++j) b[j] = ws[kk][tn + j];
#pragma unroll
            for (int i = 0; i < MMQ_TM; ++i)
#pragma unroll
                for (int j = 0; j < MMQ_TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MMQ_TM; ++i) {
        const int gm = m0 + tm + i;
        if (gm >= M) continue;
#pragma unroll
        for (int j = 0; j < MMQ_TN; ++j) {
            const int gn = n0 + tn + j;
            if (gn < N) y[(long)gm * N + gn] = acc[i][j];
        }
    }
}

// Force the one-row-per-block kernel at M > 1, so the register-tiled kernel can be A/B'd
// end to end on a real model. Read once, on first use.
static int q4_no_tile_forced = -1;
static inline bool q4_no_tile() {
    if (q4_no_tile_forced < 0) {
        const char* e = getenv("SHAINET_Q4_NO_TILE");
        q4_no_tile_forced = (e && e[0] == '1') ? 1 : 0;
    }
    return q4_no_tile_forced == 1;
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

void gemm_q4_f32(const float* x, const unsigned char* q, const float* d,
                 const unsigned char* sub, float* y, int M, int N, int K) {
    if (K % Q4_BLK == 0 && !q4_force_scalar()) {
        // Multi-row shapes go to the register-tiled kernel; M == 1 (decode) keeps the
        // GEMV-shaped kernel, where there is no output tile to amortize over.
        // SHAINET_Q4_NO_TILE=1 forces the old path so the two can be A/B'd end to end.
        if (M > 1 && !q4_no_tile()) {
            dim3 grid((N + MMQ_BN - 1) / MMQ_BN, (M + MMQ_BM - 1) / MMQ_BM);
            int threads = (MMQ_BM / MMQ_TM) * (MMQ_BN / MMQ_TN);
            gemm_q4_f32_tiled_kernel<<<grid, threads>>>(x, q, d, sub, y, M, N, K);
        } else {
            int threads = Q4_COLS_PER_BLOCK * 32;
            dim3 grid((N + Q4_COLS_PER_BLOCK - 1) / Q4_COLS_PER_BLOCK, M);
            size_t shmem = Q4_TILE * sizeof(float);
            gemm_q4_f32_vec_kernel<<<grid, threads, shmem>>>(x, q, d, sub, y, M, N, K);
        }
    } else {
        // Packed rows are not 16-byte aligned for this K, so uint4 loads would
        // be illegal. Correctness first.
        int threads = 256;
        dim3 grid(N, M);
        size_t shmem = threads * sizeof(float);
        gemm_q4_f32_kernel<<<grid, threads, shmem>>>(x, q, d, sub, y, M, N, K);
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

// ---------------------------------------------------------------------------
// Gated delta rule (Qwen3.5 linear attention), whole sequence in ONE launch.
//
// One CUDA block per VALUE head, with that head's [dk, dv] recurrent state resident in shared
// memory for the entire sequence, and the token loop INSIDE the kernel. That is the point: the
// recurrence is sequential, so a kernel per step would pay a launch per token per layer (29k
// launches for a 1216-token prefill over 24 layers), and the host implementation it replaces spent
// 4.4 s per layer -- 114 s for a prefill -- in scalar matrix products.
//
// The state is dk*dv floats = 64 KB at Qwen3.5's 128x128, which exceeds the 48 KB default limit,
// so the launcher opts in to the larger dynamic allocation (sm_80+ allows ~100 KB). Occupancy is
// one block per SM by construction; with only nv=32 blocks against 76 SMs that costs nothing.
//
// L2 normalization of q and k, and the 1/sqrt(dk) scale on q, are done HERE rather than on the
// host, which removes the per-head slicing and copying that cost as much as the arithmetic.
//
// q and k are indexed by KEY head (kh = h / heads_per_k), matching grouped-query attention: several
// value heads share one key head's projection, as the reference's repeat_interleave expresses.
__global__ void gated_delta_rule_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ alpha, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ out,
    int seq, int nv, int nk, int dk, int dv, int heads_per_k, float q_scale)
{
    extern __shared__ float sh[];
    const int h = blockIdx.x;
    if (h >= nv) return;
    const int kh = h / heads_per_k;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;

    float* S   = sh;                 // [dk, dv], row-major in dk
    float* qs  = S + (size_t)dk * dv;
    float* ks  = qs + dk;
    float* kvm = ks + dk;            // [dv] readout of the decayed state
    float* red = kvm + dv;           // [nthr] reduction scratch

    for (int idx = tid; idx < dk * dv; idx += nthr) {
        S[idx] = state[(size_t)h * dk * dv + idx];
    }
    __syncthreads();

    const size_t kstride = (size_t)nk * dk;
    const size_t vstride = (size_t)nv * dv;

    for (int t = 0; t < seq; ++t) {
        // Load this position's q/k for the shared key head.
        for (int i = tid; i < dk; i += nthr) {
            qs[i] = q[(size_t)t * kstride + (size_t)kh * dk + i];
            ks[i] = k[(size_t)t * kstride + (size_t)kh * dk + i];
        }
        __syncthreads();

        // L2 norms, one block reduction each.
        float lq = 0.0f, lk = 0.0f;
        for (int i = tid; i < dk; i += nthr) { lq += qs[i] * qs[i]; lk += ks[i] * ks[i]; }
        red[tid] = lq;
        __syncthreads();
        for (int off = nthr / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        const float nq = rsqrtf(fmaxf(red[0], 1e-12f));
        __syncthreads();
        red[tid] = lk;
        __syncthreads();
        for (int off = nthr / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        const float nk_inv = rsqrtf(fmaxf(red[0], 1e-12f));
        __syncthreads();

        for (int i = tid; i < dk; i += nthr) {
            qs[i] = qs[i] * nq * q_scale;
            ks[i] = ks[i] * nk_inv;
        }
        const float a = alpha[(size_t)t * nv + h];
        const float b = beta[(size_t)t * nv + h];
        __syncthreads();

        // Decay the state, then read it out with k. Order matters: the reference computes kv_mem
        // from the ALREADY DECAYED state.
        for (int idx = tid; idx < dk * dv; idx += nthr) S[idx] *= a;
        __syncthreads();

        for (int c = tid; c < dv; c += nthr) {
            float acc = 0.0f;
            for (int i = 0; i < dk; ++i) acc += S[(size_t)i * dv + c] * ks[i];
            kvm[c] = acc;
        }
        __syncthreads();

        // S += k (x) beta*(v - kv_mem), then out = q^T S.
        for (int c = tid; c < dv; c += nthr) {
            const float delta = (v[(size_t)t * vstride + (size_t)h * dv + c] - kvm[c]) * b;
            for (int i = 0; i < dk; ++i) S[(size_t)i * dv + c] += ks[i] * delta;
        }
        __syncthreads();

        for (int c = tid; c < dv; c += nthr) {
            float acc = 0.0f;
            for (int i = 0; i < dk; ++i) acc += S[(size_t)i * dv + c] * qs[i];
            out[(size_t)t * vstride + (size_t)h * dv + c] = acc;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < dk * dv; idx += nthr) {
        state[(size_t)h * dk * dv + idx] = S[idx];
    }
}

void gated_delta_rule(const float* q, const float* k, const float* v,
                      const float* alpha, const float* beta,
                      float* state, float* out,
                      int seq, int nv, int nk, int dk, int dv, int heads_per_k, float q_scale) {
    if (seq <= 0 || nv <= 0) return;
    int threads = 128;
    if (dv < threads) threads = dv < 32 ? 32 : dv;
    // Round to a power of two: the reduction loop halves nthr.
    int p = 32;
    while (p * 2 <= threads) p *= 2;
    threads = p;

    size_t shmem = ((size_t)dk * dv + 2 * (size_t)dk + (size_t)dv + (size_t)threads) * sizeof(float);
    // 64 KB of state exceeds the 48 KB default cap, so opt in explicitly. Without this the launch
    // fails with invalid argument rather than falling back.
    cudaFuncSetAttribute(gated_delta_rule_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    gated_delta_rule_kernel<<<nv, threads, shmem>>>(q, k, v, alpha, beta, state, out,
                                                    seq, nv, nk, dk, dv, heads_per_k, q_scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gated_delta_rule: %s (shmem=%zu threads=%d)\n",
               cudaGetErrorString(err), shmem, threads);
    }
}

// out[i] *= sigmoid(gate[i]). Qwen3.5's attention output gate.
//
// SIGMOID, not SiLU: HF applies `attn_output * torch.sigmoid(gate)`, and SiLU carries an extra
// factor of the unbounded pre-activation. Exists so a gated attention layer can keep the
// device prefill path -- applying the gate on the host instead meant declining that path, which
// measured 259.9 s per layer at a 1216-token prefill against 0.384 s on the device.
__global__ void mul_sigmoid_kernel(float* out, const float* gate, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = out[i] * (1.0f / (1.0f + __expf(-gate[i])));
    }
}

void mul_sigmoid(float* out, const float* gate, int size) {
    if (size <= 0) return;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mul_sigmoid_kernel<<<blocks, threads>>>(out, gate, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in mul_sigmoid: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"
