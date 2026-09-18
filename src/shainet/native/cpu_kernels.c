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
#include <math.h>

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
/* ─────────── Q8 activation quantization + integer dot products ───────────
 *
 * This is llama.cpp's vec_dot_q4_K_q8_K / vec_dot_q6_K_q8_K strategy, and it is the reason
 * Ollama stays fast with layers offloaded to the host: instead of dequantizing the WEIGHT to
 * fp32 and doing fp32 dot products, quantize the ACTIVATION to int8 once and multiply against
 * the raw k-quant bytes with integer SIMD.
 *
 * Two wins over the fp32 path. Integer SIMD is 32 bytes wide against 8 floats, and the weight is
 * never materialized as fp32 at all, so the K-sized fp32 buffer per row disappears. It also helps
 * at M=1, where the fp32 path has nothing to amortize -- that is decode, where the gap to Ollama
 * was 9x.
 *
 * The trade is real: quantizing the activation to int8 costs roughly 0.4% relative accuracy, and
 * it is visible in llama.cpp's own numbers -- its MUL_MAT outputs differ from an fp32 reference by
 * 0.1-0.6%. That is the accuracy Ollama ships.
 *
 * Layout, per activation row: nb = K/256 blocks, each with an fp32 scale, 256 int8 quants, and 16
 * int16 group sums. The group sums are what make the asymmetric Q4_K min term cheap (it needs
 * sum(qa) per group, not a dot) and let Q6_K subtract its constant 32 offset without unpacking
 * signed values.
 */

typedef struct {
    float  *d;      /* [M * nb]        per-block activation scale */
    int8_t *qs;     /* [M * K]         int8 quants */
    int16_t *bsums; /* [M * nb * 16]   sums of qs over each group of 16 */
} q8k_act;

static inline int32_t hsum_epi32_128(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(v);
}

/* Sum 32 unsigned x signed byte products, split into the first and second 16.
 *
 * maddubs gives int16 lane k = bytes 2k,2k+1; madd_epi16 then gives int32 lane j = int16 lanes
 * 2j,2j+1 = bytes 4j..4j+3, within each 128-bit half. So int32 lanes 0-3 cover bytes 0-15 and
 * lanes 4-7 cover bytes 16-31, which is exactly the 16-value scale groups Q6_K needs; Q4_K just
 * adds the two halves. Bounds: maddubs peaks at 2*63*127 = 16002, inside int16. */
static inline void maddubs32_split(__m256i u, __m256i s, int32_t *lo16, int32_t *hi16) {
    __m256i p16 = _mm256_maddubs_epi16(u, s);
    __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
    *lo16 = hsum_epi32_128(_mm256_castsi256_si128(p32));
    *hi16 = hsum_epi32_128(_mm256_extracti128_si256(p32, 1));
}

/* Quantize one activation row to Q8_K. */
static void quantize_row_q8k(const float *x, int K, int nb,
                             float *d_out, int8_t *q_out, int16_t *bs_out) {
    for (int b = 0; b < nb; b++) {
        const float *xb = x + b * 256;
        float amax = 0.0f;
        for (int i = 0; i < 256; i++) {
            float a = fabsf(xb[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;
        d_out[b] = d;
        int8_t *qb = q_out + b * 256;
        for (int i = 0; i < 256; i++) {
            int v = (int)lrintf(xb[i] * inv);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            qb[i] = (int8_t)v;
        }
        int16_t *bs = bs_out + b * 16;
        for (int g = 0; g < 16; g++) {
            int s = 0;
            for (int i = 0; i < 16; i++) s += qb[g * 16 + i];
            bs[g] = (int16_t)s;
        }
    }
}

/* Q4_K weight row against a Q8_K activation row.
 *
 *   w_i  = d * sc[g] * q_i - dmin * m[g]        (g = i/32, q_i a 4-bit nibble)
 *   a_i  = da * qa_i
 *   dot  = da * ( d * SUM_g sc[g] * SUM_(i in g) q_i*qa_i
 *               - dmin * SUM_g m[g] * SUM_(i in g) qa_i )
 *
 * The second sum is two group-of-16 bsums, so the min term costs no multiplies over K. */
static float dot_q4k_q8k(const uint8_t *w, const float *da, const int8_t *qa,
                         const int16_t *bsums, int nb) {
    const __m256i lomask = _mm256_set1_epi8(0x0F);
    float sumf = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t *block = w + (size_t)b * Q4_K_BLOCK_SIZE;
        float d    = f16_to_f32(*(const uint16_t *)(block + 0));
        float dmin = f16_to_f32(*(const uint16_t *)(block + 2));
        const uint8_t *sc = block + 4;
        const uint8_t *qs = block + 16;
        const int8_t  *ap = qa + (size_t)b * 256;
        const int16_t *bs = bsums + (size_t)b * 16;

        int32_t main_acc = 0;
        int32_t min_acc = 0;

        for (int j64 = 0; j64 < 4; j64++) {
            uint8_t s0, m0, s1, m1;
            get_scale_min_k4(j64 * 2,     sc, &s0, &m0);
            get_scale_min_k4(j64 * 2 + 1, sc, &s1, &m1);

            __m256i qb = _mm256_loadu_si256((const __m256i *)(qs + j64 * 32));
            /* Group 2*j64 is the low nibbles (values j64*64 .. +32), group 2*j64+1 the high. */
            __m256i nlo = _mm256_and_si256(qb, lomask);
            __m256i nhi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), lomask);

            __m256i alo = _mm256_loadu_si256((const __m256i *)(ap + j64 * 64));
            __m256i ahi = _mm256_loadu_si256((const __m256i *)(ap + j64 * 64 + 32));

            int32_t l0, h0, l1, h1;
            maddubs32_split(nlo, alo, &l0, &h0);
            maddubs32_split(nhi, ahi, &l1, &h1);
            main_acc += (int32_t)s0 * (l0 + h0) + (int32_t)s1 * (l1 + h1);

            /* Each 32-value group spans two bsums groups of 16. */
            int g0 = j64 * 2, g1 = j64 * 2 + 1;
            min_acc += (int32_t)m0 * ((int32_t)bs[g0 * 2] + bs[g0 * 2 + 1]);
            min_acc += (int32_t)m1 * ((int32_t)bs[g1 * 2] + bs[g1 * 2 + 1]);
        }
        sumf += da[b] * (d * (float)main_acc - dmin * (float)min_acc);
    }
    return sumf;
}

/* Q6_K weight row against a Q8_K activation row.
 *
 *   w_i = d * sc[is] * (q6_i - 32)              (is = i/16, q6_i a 6-bit value 0..63)
 *   dot = da * d * SUM_is sc[is] * ( SUM_(i in is) q6_i*qa_i - 32 * bsums[is] )
 *
 * Keeping q6 unsigned and folding the -32 through bsums is what lets maddubs be used at all;
 * its unsigned operand cannot carry a signed weight. */
static float dot_q6k_q8k(const uint8_t *w, const float *da, const int8_t *qa,
                         const int16_t *bsums, int nb) {
    const __m256i lomask = _mm256_set1_epi8(0x0F);
    const __m256i m3 = _mm256_set1_epi8(3);
    float sumf = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t *block = w + (size_t)b * Q6_K_BLOCK_SIZE;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t  *sc = (const int8_t *)(block + 192);
        float d = f16_to_f32(*(const uint16_t *)(block + 208));
        const int8_t  *ap = qa + (size_t)b * 256;
        const int16_t *bs = bsums + (size_t)b * 16;

        int32_t main_acc = 0;

        for (int chunk = 0; chunk < 2; chunk++) {
            const uint8_t *ql_c = ql + chunk * 64;
            const uint8_t *qh_c = qh + chunk * 32;
            const int8_t  *sc_c = sc + chunk * 8;
            const int8_t  *ap_c = ap + chunk * 128;
            const int16_t *bs_c = bs + chunk * 8;

            __m256i qlo = _mm256_loadu_si256((const __m256i *)ql_c);        /* l = 0..31 */
            __m256i qhi = _mm256_loadu_si256((const __m256i *)(ql_c + 32)); /* l+32 */
            __m256i hbits = _mm256_loadu_si256((const __m256i *)qh_c);

            /* Four 32-value spans, matching dequantize_row_q6_K's q1..q4. */
            __m256i v0 = _mm256_or_si256(_mm256_and_si256(qlo, lomask),
                _mm256_slli_epi16(_mm256_and_si256(hbits, m3), 4));
            __m256i v1 = _mm256_or_si256(_mm256_and_si256(qhi, lomask),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hbits, 2), m3), 4));
            __m256i v2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlo, 4), lomask),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hbits, 4), m3), 4));
            __m256i v3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qhi, 4), lomask),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hbits, 6), m3), 4));

            const __m256i *vs[4] = {&v0, &v1, &v2, &v3};
            for (int sub = 0; sub < 4; sub++) {
                __m256i av = _mm256_loadu_si256((const __m256i *)(ap_c + sub * 32));
                int32_t lo, hi;
                maddubs32_split(*vs[sub], av, &lo, &hi);
                /* Within a 32-value span the scale changes at l=16: sc_c[sub*2], sc_c[sub*2+1]. */
                int is0 = sub * 2, is1 = sub * 2 + 1;
                main_acc += (int32_t)sc_c[is0] * (lo - 32 * (int32_t)bs_c[is0]);
                main_acc += (int32_t)sc_c[is1] * (hi - 32 * (int32_t)bs_c[is1]);
            }
        }
        sumf += da[b] * d * (float)main_acc;
    }
    return sumf;
}

/* Allocate and fill Q8_K activations for all M rows. Returns 0 on allocation failure. */
static int quantize_act_q8k(const float *x, int M, int K, int nb, q8k_act *out) {
    out->d = (float *)malloc((size_t)M * nb * sizeof(float));
    out->qs = (int8_t *)malloc((size_t)M * K);
    out->bsums = (int16_t *)malloc((size_t)M * nb * 16 * sizeof(int16_t));
    if (!out->d || !out->qs || !out->bsums) {
        free(out->d); free(out->qs); free(out->bsums);
        out->d = NULL; out->qs = NULL; out->bsums = NULL;
        return 0;
    }
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        quantize_row_q8k(x + (size_t)m * K, K, nb,
                         out->d + (size_t)m * nb,
                         out->qs + (size_t)m * K,
                         out->bsums + (size_t)m * nb * 16);
    }
    return 1;
}

static void free_act_q8k(q8k_act *a) {
    free(a->d); free(a->qs); free(a->bsums);
}

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

/* Set SHAINET_CPU_Q8=0 to force the fp32 path (for A/B testing the accuracy trade). */
static int q8_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("SHAINET_CPU_Q8");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached;
}

/* How many tokens to hold in the inner loop.
 *
 * With n outer and every token inner, the weight row is read once but the whole quantized
 * activation is re-read for each of the N rows -- at M=128, K=17408 that is 2.2 MB times N, about
 * 11 GB per matmul. Blocking the tokens so one block stays in L2 turns that into one pass over the
 * activation per block, at the cost of re-reading the weight once per block. ~384 KB per block
 * keeps it inside a typical L2 slice. */
static inline int q8_token_block(int K, int M) {
    int mb = 393216 / (K > 0 ? K : 1);
    if (mb < 1) mb = 1;
    if (mb > M) mb = M;
    return mb;
}

void gemv_q4k_cpu(const float *x, const uint8_t *W, float *y,
                  int M, int N, int K) {
    int bytes_per_row = ((K + Q4_K_VALS_PER_BLK - 1) / Q4_K_VALS_PER_BLK)
                        * Q4_K_BLOCK_SIZE;

    /* Q8 activation + integer dot: llama.cpp's strategy, and the fastest path at every M
     * including M=1. Requires whole 256-value blocks, which every weight in a k-quant GGUF has. */
    if (q8_enabled() && K % Q4_K_VALS_PER_BLK == 0) {
        int nb = K / Q4_K_VALS_PER_BLK;
        q8k_act act;
        if (quantize_act_q8k(x, M, K, nb, &act)) {
            int mblk = q8_token_block(K, M);
            for (int m0 = 0; m0 < M; m0 += mblk) {
                int mcount = (M - m0 < mblk) ? (M - m0) : mblk;
                #pragma omp parallel for schedule(static)
                for (int n = 0; n < N; n++) {
                    const uint8_t *wrow = W + (long long)n * bytes_per_row;
                    for (int mi = 0; mi < mcount; mi++) {
                        int m = m0 + mi;
                        y[(long long)m * N + n] = dot_q4k_q8k(wrow,
                            act.d + (size_t)m * nb,
                            act.qs + (size_t)m * K,
                            act.bsums + (size_t)m * nb * 16, nb);
                    }
                }
            }
            free_act_q8k(&act);
            return;
        }
    }

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

    if (q8_enabled() && K % Q6_K_VALS_PER_BLK == 0) {
        int nb = K / Q6_K_VALS_PER_BLK;
        q8k_act act;
        if (quantize_act_q8k(x, M, K, nb, &act)) {
            int mblk = q8_token_block(K, M);
            for (int m0 = 0; m0 < M; m0 += mblk) {
                int mcount = (M - m0 < mblk) ? (M - m0) : mblk;
                #pragma omp parallel for schedule(static)
                for (int n = 0; n < N; n++) {
                    const uint8_t *wrow = W + (long long)n * bytes_per_row;
                    for (int mi = 0; mi < mcount; mi++) {
                        int m = m0 + mi;
                        y[(long long)m * N + n] = dot_q6k_q8k(wrow,
                            act.d + (size_t)m * nb,
                            act.qs + (size_t)m * K,
                            act.bsums + (size_t)m * nb * 16, nb);
                    }
                }
            }
            free_act_q8k(&act);
            return;
        }
    }

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
