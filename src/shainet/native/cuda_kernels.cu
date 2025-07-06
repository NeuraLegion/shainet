#include <curand_kernel.h>
#include <cstdio>

// Device kernels
__global__ void softmax_rows_kernel(double* out, const double* in, int rows, int cols) {
    int row = blockIdx.x;
    if(row >= rows) return;
    const double *row_in = in + row * cols;
    double *row_out = out + row * cols;
    double sum = 0.0;
    for(int j=0;j<cols;++j){
        double e = exp(row_in[j]);
        row_out[j] = e;
        sum += e;
    }
    for(int j=0;j<cols;++j){
        row_out[j] /= sum;
    }
}

__global__ void relu_backward_kernel(double* output, const double* input, const double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = input[idx] > 0.0 ? grad[idx] : 0.0;
}

// Host wrapper functions
extern "C" {
void softmax_rows(double* out, const double* in, int rows, int cols) {
    softmax_rows_kernel<<<rows, 1>>>(out, in, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in softmax_rows: %s\n", cudaGetErrorString(err));
    }
}

void relu_backward(double* output, const double* input, const double* grad, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    relu_backward_kernel<<<blocks, threads_per_block>>>(output, input, grad, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in relu_backward: %s\n", cudaGetErrorString(err));
    }
}

__global__ void dropout_kernel(double* out, const double* in, int rows, int cols, double drop_p, unsigned long long seed) {
    int row = blockIdx.x;
    if(row >= rows) return;
    curandState state;
    curand_init(seed + row, 0, 0, &state);
    const double *row_in = in + row * cols;
    double *row_out = out + row * cols;
    for(int j=0;j<cols;++j){
        double r = curand_uniform_double(&state);
        row_out[j] = r < drop_p ? 0.0 : row_in[j];
    }
}

void dropout(double* out, const double* in, int rows, int cols, double drop_p, unsigned long long seed) {
    dropout_kernel<<<rows, 1>>>(out, in, rows, cols, drop_p, seed);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in dropout: %s\n", cudaGetErrorString(err));
    }
}

__global__ void gather_rows_kernel(double* out, const double* in, const int* ids, int rows, int cols) {
    int row = blockIdx.x;
    if(row >= rows) return;
    int id = ids[row];
    const double *row_in = in + id * cols;
    double *row_out = out + row * cols;
    for(int j=0;j<cols;++j){
        row_out[j] = row_in[j];
    }
}

void gather_rows(double* out, const double* in, const int* ids, int rows, int cols) {
    gather_rows_kernel<<<rows, 1>>>(out, in, ids, rows, cols);
    cudaDeviceSynchronize();
}

__global__ void row_mean_var_kernel(const double* in, double* mean, double* var,
                                    int rows, int cols) {
    int row = blockIdx.x;
    if(row >= rows) return;
    const double *row_in = in + row * cols;
    double sum = 0.0;
    double sq_sum = 0.0;
    for(int j=0;j<cols;++j){
        double v = row_in[j];
        sum += v;
        sq_sum += v*v;
    }
    double m = sum / cols;
    mean[row] = m;
    var[row] = sq_sum / cols - m*m;
}

void row_mean_var(const double* in, double* mean, double* var, int rows, int cols) {
    row_mean_var_kernel<<<rows, 1>>>(in, mean, var, rows, cols);
    cudaDeviceSynchronize();
}

__global__ void apply_layer_norm_kernel(double* out, const double* in,
                                        const double* mean, const double* var,
                                        int rows, int cols, double epsilon) {
    int row = blockIdx.x;
    if(row >= rows) return;
    const double *row_in = in + row * cols;
    double *row_out = out + row * cols;
    double m = mean[row];
    double denom = sqrt(var[row] + epsilon);
    for(int j=0;j<cols;++j){
        row_out[j] = (row_in[j] - m) / denom;
    }
}

void apply_layer_norm(double* out, const double* in,
                      const double* mean, const double* var,
                      int rows, int cols, double epsilon) {
    apply_layer_norm_kernel<<<rows, 1>>>(out, in, mean, var, rows, cols, epsilon);
    cudaDeviceSynchronize();
}

__global__ void slice_cols_kernel(double* out, const double* in, int rows, int src_cols, int start, int len){
    int row = blockIdx.x;
    int col = threadIdx.x;
    if(row >= rows || col >= len) return;
    out[row * len + col] = in[row * src_cols + start + col];
}

void slice_cols(double* out, const double* in, int rows, int src_cols, int start, int len){
    slice_cols_kernel<<<rows, len>>>(out, in, rows, src_cols, start, len);
    cudaDeviceSynchronize();
}

__global__ void set_cols_kernel(double* out, const double* in, int rows, int dst_cols, int start, int len){
    int row = blockIdx.x;
    int col = threadIdx.x;
    if(row >= rows || col >= len) return;
    out[row * dst_cols + start + col] = in[row * len + col];
}

void set_cols(double* out, const double* in, int rows, int dst_cols, int start, int len){
    set_cols_kernel<<<rows, len>>>(out, in, rows, dst_cols, start, len);
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

__global__ void layer_norm_backward_kernel(double* d_x, double* d_gamma, double* d_beta,
                                           const double* d_out, const double* x,
                                           const double* gamma, const double* mean,
                                           const double* var, const double* norm,
                                           int rows, int cols, double epsilon) {
    int row = blockIdx.x;
    if(row >= rows) return;

    const double *x_row = x + row * cols;
    const double *dout_row = d_out + row * cols;
    const double *norm_row = norm + row * cols;
    double *dx_row = d_x + row * cols;

    double m = mean[row];
    double v = var[row];
    double denom = sqrt(v + epsilon);
    double inv = 1.0 / denom;
    double col_f = (double)cols;

    // Compute sum_dout_gamma and sum_dout_gamma_norm
    double sum_dout_gamma = 0.0;
    double sum_dout_gamma_norm = 0.0;
    for(int j = 0; j < cols; ++j) {
        double doutg = dout_row[j] * gamma[j];
        sum_dout_gamma += doutg;
        sum_dout_gamma_norm += doutg * (x_row[j] - m);

        // Accumulate gradients for gamma and beta
        atomicAdd(&d_gamma[j], dout_row[j] * norm_row[j]);
        atomicAdd(&d_beta[j], dout_row[j]);
    }

    // Compute d_x
    for(int j = 0; j < cols; ++j) {
        double xm = x_row[j] - m;
        double doutg = dout_row[j] * gamma[j];
        dx_row[j] = inv * (doutg - sum_dout_gamma/col_f - xm * inv*inv / col_f * sum_dout_gamma_norm);
    }
}

void layer_norm_backward(double* d_x, double* d_gamma, double* d_beta,
                         const double* d_out, const double* x,
                         const double* gamma, const double* mean,
                         const double* var, const double* norm,
                         int rows, int cols, double epsilon) {
    layer_norm_backward_kernel<<<rows, 1>>>(d_x, d_gamma, d_beta, d_out, x,
                                            gamma, mean, var, norm,
                                            rows, cols, epsilon);
    cudaDeviceSynchronize();
}

__global__ void sum_cols_kernel(double* out, const double* in, int rows, int cols) {
    int col = blockIdx.x;
    if(col >= cols) return;

    double sum = 0.0;
    for(int i = 0; i < rows; ++i) {
        sum += in[i * cols + col];
    }
    out[col] = sum;
}

void sum_cols(double* out, const double* in, int rows, int cols) {
    sum_cols_kernel<<<cols, 1>>>(out, in, rows, cols);
    cudaDeviceSynchronize();
}

__global__ void mul_row_vector_kernel(double* matrix, const double* vec, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int col = idx % cols;
    matrix[idx] *= vec[col];
}

void mul_row_vector(double* matrix, const double* vec, int rows, int cols) {
    int threads_per_block = 256;
    int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    mul_row_vector_kernel<<<blocks, threads_per_block>>>(matrix, vec, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in mul_row_vector: %s\n", cudaGetErrorString(err));
    }
}

__global__ void transpose_kernel(double* out, const double* in, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int row = idx / cols;
    int col = idx % cols;

    // Transpose: out[col][row] = in[row][col]
    // In row-major: out[col * rows + row] = in[row * cols + col]
    out[col * rows + row] = in[row * cols + col];
}

void transpose(double* out, const double* in, int rows, int cols) {
    int threads_per_block = 256;
    int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    transpose_kernel<<<blocks, threads_per_block>>>(out, in, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in transpose: %s\n", cudaGetErrorString(err));
    }
}

__global__ void sigmoid_forward_kernel(double* activations, double* derivatives, const double* linear, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    double val = linear[idx];
    // Sigmoid: 1 / (1 + exp(-x))
    double exp_neg_val = exp(-val);
    double sigmoid_val = 1.0 / (1.0 + exp_neg_val);
    
    activations[idx] = sigmoid_val;
    // Sigmoid derivative: σ(x) * (1 - σ(x))
    derivatives[idx] = sigmoid_val * (1.0 - sigmoid_val);
}

void sigmoid_forward(double* activations, double* derivatives, const double* linear, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    sigmoid_forward_kernel<<<blocks, threads_per_block>>>(activations, derivatives, linear, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in sigmoid_forward: %s\n", cudaGetErrorString(err));
    }
}

__global__ void apply_gradient_kernel(double* local_grad, const double* grad, const double* derivatives, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    local_grad[idx] = grad[idx] * derivatives[idx];
}

void apply_gradient(double* local_grad, const double* grad, const double* derivatives, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    apply_gradient_kernel<<<blocks, threads_per_block>>>(local_grad, grad, derivatives, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in apply_gradient: %s\n", cudaGetErrorString(err));
    }
}

__global__ void accumulate_bias_grad_kernel(double* bias_grad, const double* local_grad, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    double sum = 0.0;
    for (int row = 0; row < rows; row++) {
        sum += local_grad[row * cols + col];
    }
    atomicAdd(&bias_grad[col], sum);
}

void accumulate_bias_grad(double* bias_grad, const double* local_grad, int rows, int cols) {
    int threads_per_block = 256;
    int blocks = (cols + threads_per_block - 1) / threads_per_block;

    accumulate_bias_grad_kernel<<<blocks, threads_per_block>>>(bias_grad, local_grad, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in accumulate_bias_grad: %s\n", cudaGetErrorString(err));
    }
}

__global__ void zero_matrix_kernel(double* matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    matrix[idx] = 0.0;
}

void zero_matrix(double* matrix, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    zero_matrix_kernel<<<blocks, threads_per_block>>>(matrix, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in zero_matrix: %s\n", cudaGetErrorString(err));
    }
}

__global__ void element_div_kernel(double* out, const double* a, const double* b, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;

    double denom = b[idx];
    out[idx] = denom == 0.0 ? 0.0 : a[idx] / denom;
}

void element_div(double* out, const double* a, const double* b, int size){
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    element_div_kernel<<<blocks, threads_per_block>>>(out, a, b, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in element_div: %s\n", cudaGetErrorString(err));
    }
}

__global__ void softmax_backward_kernel(double* output, const double* grad, const double* softmax_out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const double* grad_row = grad + row * cols;
    const double* softmax_row = softmax_out + row * cols;
    double* output_row = output + row * cols;
    
    // Compute sum of softmax * grad for this row
    double sum = 0.0;
    for (int j = 0; j < cols; j++) {
        sum += softmax_row[j] * grad_row[j];
    }
    
    // Compute softmax backward: softmax * (grad - sum)
    for (int j = 0; j < cols; j++) {
        output_row[j] = softmax_row[j] * (grad_row[j] - sum);
    }
}

void softmax_backward(double* output, const double* grad, const double* softmax_out, int rows, int cols) {
    softmax_backward_kernel<<<rows, 1>>>(output, grad, softmax_out, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in softmax_backward: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"
