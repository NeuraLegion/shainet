#include <curand_kernel.h>
extern "C" {
__global__ void softmax_rows(double* out, const double* in, int rows, int cols) {
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

__global__ void dropout(double* out, const double* in, int rows, int cols, double drop_p, unsigned long long seed) {
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
}
__global__ void slice_cols(double* out, const double* in, int rows, int src_cols, int start, int len){
    int row = blockIdx.x;
    int col = threadIdx.x;
    if(row >= rows || col >= len) return;
    out[row * len + col] = in[row * src_cols + start + col];
}

__global__ void set_cols(double* out, const double* in, int rows, int dst_cols, int start, int len){
    int row = blockIdx.x;
    int col = threadIdx.x;
    if(row >= rows || col >= len) return;
    out[row * dst_cols + start + col] = in[row * len + col];
}
}
