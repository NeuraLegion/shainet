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
#include <stdlib.h>

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
    float acc = 0.0f;

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

            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int k0 = base_k + chunk * 128 + l;

                int8_t q1 = ((ql_c[l]      & 0xF) | (((qh_c[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = ((ql_c[l + 32]  & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = ((ql_c[l]      >> 4)   | (((qh_c[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = ((ql_c[l + 32]  >> 4)  | (((qh_c[l] >> 6) & 3) << 4)) - 32;

                float s0 = d * (float)sc_c[is];
                float s2 = d * (float)sc_c[is + 2];
                float s4 = d * (float)sc_c[is + 4];
                float s6 = d * (float)sc_c[is + 6];

                if (k0 < K)      acc += x[k0]      * s0 * (float)q1;
                if (k0+32 < K)   acc += x[k0+32]   * s2 * (float)q2;
                if (k0+64 < K)   acc += x[k0+64]   * s4 * (float)q3;
                if (k0+96 < K)   acc += x[k0+96]   * s6 * (float)q4;
            }
        }
    }

    return acc;
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
/* Dequantize one Q4_K row to fp32.
 *
 * The GEMV path below fuses dequant into the dot product, which is right for a single token: the
 * row is touched once either way. For M tokens the fused form re-dequantizes the SAME row M
 * times, so above M=1 it pays to unpack the row once and reuse it. */
static void dequant_q4k_row_f32(const uint8_t *w, float *out, int K) {
    int nb = (K + Q4_K_VALS_PER_BLK - 1) / Q4_K_VALS_PER_BLK;
    for (int blk = 0; blk < nb; blk++) {
        const uint8_t *block = w + blk * Q4_K_BLOCK_SIZE;
        float d    = f16_to_f32(*(const uint16_t *)(block + 0));
        float dmin = f16_to_f32(*(const uint16_t *)(block + 2));
        const uint8_t *sc = block + 4;
        const uint8_t *qs = block + 16;
        int base_k = blk * Q4_K_VALS_PER_BLK;

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

            for (int l = 0; l < 32; l++) {
                int ka = k0 + l;
                int kb = k0 + l + 32;
                if (ka < K) out[ka] = d1 * (float)(qp[l] & 0x0F) - m1v;
                if (kb < K) out[kb] = d2 * (float)(qp[l] >> 4)   - m2v;
            }
        }
    }
}

/* Dequantize one Q6_K row to fp32. Same reasoning as the Q4_K variant. */
static void dequant_q6k_row_f32(const uint8_t *w, float *out, int K) {
    int nb = (K + Q6_K_VALS_PER_BLK - 1) / Q6_K_VALS_PER_BLK;
    for (int blk = 0; blk < nb; blk++) {
        const uint8_t *block = w + blk * Q6_K_BLOCK_SIZE;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t  *sc = (const int8_t *)(block + 192);
        float d = f16_to_f32(*(const uint16_t *)(block + 208));
        int base_k = blk * Q6_K_VALS_PER_BLK;

        for (int chunk = 0; chunk < 2; chunk++) {
            const uint8_t *ql_c = ql + chunk * 64;
            const uint8_t *qh_c = qh + chunk * 32;
            const int8_t  *sc_c = sc + chunk * 8;
            int c0 = base_k + chunk * 128;
            for (int l = 0; l < 32; l++) {
                int q1 = (int)((ql_c[l]      & 0x0F) | (((qh_c[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql_c[l + 32] & 0x0F) | (((qh_c[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_c[l]      >> 4)   | (((qh_c[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_c[l + 32] >> 4)   | (((qh_c[l] >> 6) & 3) << 4)) - 32;
                int is = l / 16;
                if (c0 + l       < K) out[c0 + l]      = d * (float)sc_c[is + 0] * (float)q1;
                if (c0 + l + 32  < K) out[c0 + l + 32] = d * (float)sc_c[is + 2] * (float)q2;
                if (c0 + l + 64  < K) out[c0 + l + 64] = d * (float)sc_c[is + 4] * (float)q3;
                if (c0 + l + 96  < K) out[c0 + l + 96] = d * (float)sc_c[is + 6] * (float)q4;
            }
        }
    }
}

/* fp32 dot product, AVX2 with FMA. */
static inline float dot_f32(const float *a, const float *b, int K) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k),     _mm256_loadu_ps(b + k),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8), acc1);
    }
    for (; k + 8 <= K; k += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; k < K; k++) sum += a[k] * b[k];
    return sum;
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

/* Four dot products against a shared weight row, accumulated in separate registers.
 *
 * Doing the tokens one at a time re-reads the weight row from L1 per token; more importantly the
 * single-token loop re-streams the ACTIVATION for every output row, which is the dominant traffic
 * at prefill widths. Handling four tokens per pass loads each weight element once for four FMAs
 * and cuts the activation sweep to a quarter. */
static inline void dot4_f32(const float *a0, const float *a1, const float *a2, const float *a3,
                            const float *b, int K, float *out) {
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 vb = _mm256_loadu_ps(b + k);
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + k), vb, c0);
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + k), vb, c1);
        c2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + k), vb, c2);
        c3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + k), vb, c3);
    }
    out[0] = hsum256(c0); out[1] = hsum256(c1);
    out[2] = hsum256(c2); out[3] = hsum256(c3);
    for (; k < K; k++) {
        out[0] += a0[k] * b[k]; out[1] += a1[k] * b[k];
        out[2] += a2[k] * b[k]; out[3] += a3[k] * b[k];
    }
}

/* Shared tail for both k-quant types: the weight row is already unpacked into wbuf. */
static inline void apply_row(const float *x, float *y, const float *wbuf,
                             int M, int N, int K, int n) {
    int m = 0;
    float acc[4];
    for (; m + 4 <= M; m += 4) {
        dot4_f32(x + (long long)(m + 0) * K, x + (long long)(m + 1) * K,
                 x + (long long)(m + 2) * K, x + (long long)(m + 3) * K,
                 wbuf, K, acc);
        y[(long long)(m + 0) * N + n] = acc[0];
        y[(long long)(m + 1) * N + n] = acc[1];
        y[(long long)(m + 2) * N + n] = acc[2];
        y[(long long)(m + 3) * N + n] = acc[3];
    }
    for (; m < M; m++) {
        y[(long long)m * N + n] = dot_f32(x + (long long)m * K, wbuf, K);
    }
}

void gemv_q4k_cpu(const float *x, const uint8_t *W, float *y,
                  int M, int N, int K) {
    int bytes_per_row = ((K + Q4_K_VALS_PER_BLK - 1) / Q4_K_VALS_PER_BLK)
                        * Q4_K_BLOCK_SIZE;

    if (M == 1) {
        /* Fused dequant+dot: the row is read once regardless, so no buffer is worth its write. */
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            y[n] = dot_q4k_row(x, W + (long long)n * bytes_per_row, K);
        }
        return;
    }

    /* M > 1: unpack each weight row ONCE, then dot it against every token, four at a time.
     * Keeping `m` inside turns M dequant passes over the whole weight into one, which is the same
     * reuse the device path gets from doing a GEMM instead of M GEMVs. */
    #pragma omp parallel
    {
        float *wbuf = (float *)malloc((size_t)K * sizeof(float));
        if (wbuf) {
            #pragma omp for schedule(static)
            for (int n = 0; n < N; n++) {
                dequant_q4k_row_f32(W + (long long)n * bytes_per_row, wbuf, K);
                apply_row(x, y, wbuf, M, N, K, n);
            }
            free(wbuf);
        }
    }
}

void gemv_q6k_cpu(const float *x, const uint8_t *W, float *y,
                  int M, int N, int K) {
    int bytes_per_row = ((K + Q6_K_VALS_PER_BLK - 1) / Q6_K_VALS_PER_BLK)
                        * Q6_K_BLOCK_SIZE;

    if (M == 1) {
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            y[n] = dot_q6k_row(x, W + (long long)n * bytes_per_row, K);
        }
        return;
    }

    #pragma omp parallel
    {
        float *wbuf = (float *)malloc((size_t)K * sizeof(float));
        if (wbuf) {
            #pragma omp for schedule(static)
            for (int n = 0; n < N; n++) {
                dequant_q6k_row_f32(W + (long long)n * bytes_per_row, wbuf, K);
                apply_row(x, y, wbuf, M, N, K, n);
            }
            free(wbuf);
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
