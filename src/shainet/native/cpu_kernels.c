/*
 * SHAInet CPU-side AVX2 kernels for k-quant GEMV.
 *
 * Q4_K and Q6_K dot products that dequant on-the-fly using AVX2 integer/FP
 * intrinsics, similar to llama.cpp's ggml-cpu-quants.c. No data is copied --
 * the weight pointer is the mmap'd GGUF file.
 *
 * Build:
 *   gcc -O3 -mavx2 -mfma -shared -fPIC -o libshainet_cpu_kernels.so \
 *       src/shainet/native/cpu_kernels.c
 *
 * Called from Crystal via lib binding (no FFI overhead beyond the call itself).
 */

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

/* ──────────────────────── Q4_K block layout ──────────────────────── */
/* 144 bytes per block of 256 values:
 *   [0..1]   f16 d       (super-block scale)
 *   [2..3]   f16 dmin    (super-block min)
 *   [4..15]  scales[12]  (packed 6-bit scale + 6-bit min per 32-val sub-block)
 *   [16..143] qs[128]    (4-bit quantized values, packed 2 per byte)
 */
#define Q4_K_BLOCK_SIZE   144
#define Q4_K_VALS_PER_BLK 256

/* ──────────────────────── Q6_K block layout ──────────────────────── */
/* 210 bytes per block of 256 values:
 *   [0..127]   ql[128]   (low 4 bits)
 *   [128..191] qh[64]    (high 2 bits)
 *   [192..207] scales[16] (int8 per 16-val group)
 *   [208..209] f16 d     (super-block scale)
 */
#define Q6_K_BLOCK_SIZE   210
#define Q6_K_VALS_PER_BLK 256

/* ──────────────────────── f16 → f32 ──────────────────────── */
static inline float f16_to_f32(uint16_t h) {
    __m128i v = _mm_cvtsi32_si128((int)h);
    __m128  f = _mm_cvtph_ps(v);          /* F16C */
    return _mm_cvtss_f32(f);
}

/* ──────────────────────── Q4_K scale/min decode ──────────────────── */
static inline void get_scale_min_k4(int j, const uint8_t *sc,
                                    uint8_t *scale, uint8_t *min) {
    if (j < 4) {
        *scale = sc[j] & 63;
        *min   = sc[j + 4] & 63;
    } else {
        *scale = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        *min   = (sc[j + 4] >>  4) | ((sc[j - 0] >> 6) << 4);
    }
}

/* ──────────────────────── Q4_K dot product (one row) ──────────────── */
/*
 * Compute dot(x[0..K-1], dequant(W_row[0..K-1])) for a single output row
 * where W_row is K values packed in Q4_K blocks.
 *
 * x:     float* input vector, length K
 * w:     uint8_t* to the first Q4_K block of this row
 * K:     number of values (input dimension)
 * returns: dot product as float
 */
static float dot_q4k_row(const float *x, const uint8_t *w, int K) {
    int nb = (K + Q4_K_VALS_PER_BLK - 1) / Q4_K_VALS_PER_BLK;
    __m256 acc = _mm256_setzero_ps();

    for (int blk = 0; blk < nb; blk++) {
        const uint8_t *block = w + blk * Q4_K_BLOCK_SIZE;
        float d    = f16_to_f32(*(const uint16_t *)(block + 0));
        float dmin = f16_to_f32(*(const uint16_t *)(block + 2));
        const uint8_t *sc = block + 4;
        const uint8_t *qs = block + 16;
        int base_k = blk * Q4_K_VALS_PER_BLK;
        const __m256i mask_lo = _mm256_set1_epi8(0x0F);

        /* 4 groups of 64 values each */
        for (int j64 = 0; j64 < 4; j64++) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(j64 * 2,     sc, &sc0, &m0);
            get_scale_min_k4(j64 * 2 + 1, sc, &sc1, &m1);
            float d1  = d * (float)sc0;
            float m1v = dmin * (float)m0;
            float d2  = d * (float)sc1;
            float m2v = dmin * (float)m1;
            const uint8_t *qp = qs + j64 * 32;
            int k0 = base_k + j64 * 64;

            /* First 32 values: low nibble */
            for (int l = 0; l < 32 && (k0 + l) < K; l += 8) {
                int rem = K - (k0 + l);
                if (rem >= 8) {
                    __m256 xv = _mm256_loadu_ps(x + k0 + l);
                    /* Dequant: d1 * (qs & 0xF) - m1v */
                    __m256 wv;
                    float wbuf[8];
                    for (int i = 0; i < 8; i++)
                        wbuf[i] = d1 * (float)(qp[l + i] & 0xF) - m1v;
                    wv = _mm256_loadu_ps(wbuf);
                    acc = _mm256_fmadd_ps(xv, wv, acc);
                } else {
                    for (int i = 0; i < rem; i++) {
                        float wval = d1 * (float)(qp[l + i] & 0xF) - m1v;
                        float xval = x[k0 + l + i];
                        acc = _mm256_add_ps(acc, _mm256_set1_ps(xval * wval));
                    }
                }
            }

            /* Next 32 values: high nibble */
            for (int l = 0; l < 32 && (k0 + 32 + l) < K; l += 8) {
                int rem = K - (k0 + 32 + l);
                if (rem >= 8) {
                    __m256 xv = _mm256_loadu_ps(x + k0 + 32 + l);
                    float wbuf[8];
                    for (int i = 0; i < 8; i++)
                        wbuf[i] = d2 * (float)(qp[l + i] >> 4) - m2v;
                    __m256 wv = _mm256_loadu_ps(wbuf);
                    acc = _mm256_fmadd_ps(xv, wv, acc);
                } else {
                    for (int i = 0; i < rem; i++) {
                        float wval = d2 * (float)(qp[l + i] >> 4) - m2v;
                        float xval = x[k0 + 32 + l + i];
                        acc = _mm256_add_ps(acc, _mm256_set1_ps(xval * wval));
                    }
                }
            }
        }
    }

    /* Horizontal sum */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

/* ──────────────────────── Q6_K dot product (one row) ──────────────── */
static float dot_q6k_row(const float *x, const uint8_t *w, int K) {
    int nb = (K + Q6_K_VALS_PER_BLK - 1) / Q6_K_VALS_PER_BLK;
    __m256 acc = _mm256_setzero_ps();

    for (int blk = 0; blk < nb; blk++) {
        const uint8_t *block = w + blk * Q6_K_BLOCK_SIZE;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t  *sc = (const int8_t *)(block + 192);
        float d = f16_to_f32(*(const uint16_t *)(block + 208));
        int base_k = blk * Q6_K_VALS_PER_BLK;

        /* 2 chunks of 128 values */
        for (int chunk = 0; chunk < 2; chunk++) {
            const uint8_t *ql_c = ql + chunk * 64;
            const uint8_t *qh_c = qh + chunk * 32;
            const int8_t  *sc_c = sc + chunk * 8;

            for (int l = 0; l < 32; l += 8) {
                /* Process 4 sets of values (l, l+32, l+64, l+96) for each l */
                for (int sub = 0; sub < 8 && (l + sub) < 32; sub++) {
                    int ll = l + sub;
                    int is = ll / 16;
                    int k0 = base_k + chunk * 128 + ll;
                    int k1 = k0 + 32;
                    int k2 = k0 + 64;
                    int k3 = k0 + 96;

                    int8_t q1 = ((ql_c[ll]      & 0xF) | (((qh_c[ll] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = ((ql_c[ll + 32]  & 0xF) | (((qh_c[ll] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = ((ql_c[ll]      >> 4)   | (((qh_c[ll] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = ((ql_c[ll + 32]  >> 4)  | (((qh_c[ll] >> 6) & 3) << 4)) - 32;

                    float s0 = d * (float)sc_c[is];
                    float s2 = d * (float)sc_c[is + 2];
                    float s4 = d * (float)sc_c[is + 4];
                    float s6 = d * (float)sc_c[is + 6];

                    float sum = 0.0f;
                    if (k0 < K) sum += x[k0] * s0 * (float)q1;
                    if (k1 < K) sum += x[k1] * s2 * (float)q2;
                    if (k2 < K) sum += x[k2] * s4 * (float)q3;
                    if (k3 < K) sum += x[k3] * s6 * (float)q4;
                    acc = _mm256_add_ps(acc, _mm256_set1_ps(sum));
                }
            }
        }
    }

    /* Horizontal sum */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

/* ──────────────────────── GEMV entry points ──────────────────────── */
/*
 * gemv_q4k_cpu: y[M, N] = x[M, K] * dequant(W[N, K])
 *
 * W is row-major with N rows of K values, each row packed in Q4_K blocks.
 * x is row-major [M, K] float32.
 * y is row-major [M, N] float32.
 *
 * Parallelism: caller handles M-parallelism (Crystal fibers or threads).
 * This function parallelizes over N with OpenMP if available.
 */
void gemv_q4k_cpu(const float *x, const uint8_t *W, float *y,
                  int M, int N, int K) {
    int bytes_per_row = ((K + Q4_K_VALS_PER_BLK - 1) / Q4_K_VALS_PER_BLK)
                        * Q4_K_BLOCK_SIZE;

    for (int m = 0; m < M; m++) {
        const float *xrow = x + m * K;
        float *yrow = y + m * N;

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            const uint8_t *wrow = W + (long long)n * bytes_per_row;
            yrow[n] = dot_q4k_row(xrow, wrow, K);
        }
    }
}

void gemv_q6k_cpu(const float *x, const uint8_t *W, float *y,
                  int M, int N, int K) {
    int bytes_per_row = ((K + Q6_K_VALS_PER_BLK - 1) / Q6_K_VALS_PER_BLK)
                        * Q6_K_BLOCK_SIZE;

    for (int m = 0; m < M; m++) {
        const float *xrow = x + m * K;
        float *yrow = y + m * N;

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            const uint8_t *wrow = W + (long long)n * bytes_per_row;
            yrow[n] = dot_q6k_row(xrow, wrow, K);
        }
    }
}

/* ──────────────────────── fp32 SGEMM (AVX2 + OpenMP) ──────────────── */
/*
 * sgemm_cpu: C[M, N] = A[M, K] * B[K, N], row-major fp32.
 *
 * Used for the dequanted fp32 Q+gate weights in full-attention layers
 * that otherwise go through Crystal's single-threaded matmul.
 */
void sgemm_cpu(const float *A, const float *B, float *C,
               int M, int N, int K) {
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float *arow = A + m * K;
        float *crow = C + m * N;
        memset(crow, 0, N * sizeof(float));

        for (int k = 0; k < K; k++) {
            float a_val = arow[k];
            __m256 va = _mm256_set1_ps(a_val);
            const float *brow = B + k * N;
            int n = 0;
            for (; n + 8 <= N; n += 8) {
                __m256 vc = _mm256_loadu_ps(crow + n);
                __m256 vb = _mm256_loadu_ps(brow + n);
                vc = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(crow + n, vc);
            }
            for (; n < N; n++) {
                crow[n] += a_val * brow[n];
            }
        }
    }
}
