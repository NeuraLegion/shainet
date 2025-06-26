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

__global__ void gather_rows(double* out, const double* in, const int* ids, int rows, int cols) {
    int row = blockIdx.x;
    if(row >= rows) return;
    int id = ids[row];
    const double *row_in = in + id * cols;
    double *row_out = out + row * cols;
    for(int j=0;j<cols;++j){
        row_out[j] = row_in[j];
    }
}
}
