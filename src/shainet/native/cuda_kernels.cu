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

__global__ void row_mean_var(const double* in, double* mean, double* var,
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

__global__ void apply_layer_norm(double* out, const double* in,
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
}
