/*
 * Multi-threaded OpenMP Exact vs. Certified Modular FWHT Benchmark (up to n=32 qubits).
 *
 * Compile on 32-vCPU GCP VM (Debian/Ubuntu):
 *   gcc -O3 -march=native -fopenmp src/fwht_omp.c -o fwht_omp
 * Run:
 *   ./fwht_omp 28    # 28 qubits (268M amplitudes, p=69,  uint8  = 256 MB vs int32 = 1 GB)
 *   ./fwht_omp 32    # 32 qubits (4.29B amplitudes, p=137, uint16 = 8 GB  vs int32 = 16 GB)
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static uint32_t minimal_odd_8_15(int n) {
    int m = n / 4;
    uint64_t M = 1ULL << m;
    /* ceil(8*M / 15) */
    uint32_t p = (uint32_t)((8ULL * M + 14ULL) / 15ULL);
    if ((p & 1U) == 0U) p++;
    return p;
}

/* Maiorana-McFarland degree-4 bent function:
 * f(x, y) = popcount(x & y) mod 2  XOR  (y0 y1 y2) XOR (y0 y1 y2 y3)
 */
static inline uint8_t eval_mm_quartic(uint64_t idx, int k) {
    uint64_t mask = (1ULL << k) - 1ULL;
    uint64_t x = idx & mask;
    uint64_t y = (idx >> k) & mask;
    uint8_t parity = (uint8_t)(__builtin_parityll(x & y));
    uint8_t ccz  = (uint8_t)(((y & 0x7ULL) == 0x7ULL) ? 1 : 0);
    uint8_t cccz = (uint8_t)(((y & 0xFULL) == 0xFULL) ? 1 : 0);
    return (uint8_t)(parity ^ ccz ^ cccz);
}

static void run_modular_u8(int n, uint8_t p) {
    uint64_t N = 1ULL << n;
    int k = n / 2;
    uint8_t *w = (uint8_t *)malloc(N * sizeof(uint8_t));
    if (!w) { fprintf(stderr, "malloc uint8 failed\n"); exit(1); }

    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < N; i++) {
        w[i] = eval_mm_quartic(i, k) ? (uint8_t)(p - 1) : (uint8_t)1;
    }

    for (uint64_t h = 1; h < N; h <<= 1) {
        uint64_t step = h << 1;
        #pragma omp parallel for schedule(static)
        for (uint64_t base = 0; base < N; base += step) {
            for (uint64_t j = 0; j < h; j++) {
                uint8_t a = w[base + j];
                uint8_t b = w[base + h + j];
                uint8_t s = (uint8_t)(a + b);
                if (s >= p) s -= p;
                uint8_t d = (uint8_t)(a + (p - b));
                if (d >= p) d -= p;
                w[base + j] = s;
                w[base + h + j] = d;
            }
        }
    }

    uint64_t target = 1ULL << k;
    uint8_t tpos = (uint8_t)(target % p);
    uint8_t tneg = (uint8_t)((p - tpos) % p);
    uint64_t viol = 0;
    #pragma omp parallel for reduction(+:viol) schedule(static)
    for (uint64_t i = 0; i < N; i++) {
        if (w[i] != tpos && w[i] != tneg) viol++;
    }
    double dt = omp_get_wtime() - t0;
    printf("[n=%d | uint8 mod p=%u | %.2f GB] time=%.3f s | violations=%llu\n",
           n, p, (double)N / (1024.0 * 1024.0 * 1024.0), dt, (unsigned long long)viol);
    free(w);
}

static void run_modular_u16(int n, uint16_t p) {
    uint64_t N = 1ULL << n;
    int k = n / 2;
    uint16_t *w = (uint16_t *)malloc(N * sizeof(uint16_t));
    if (!w) { fprintf(stderr, "malloc uint16 failed\n"); exit(1); }

    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < N; i++) {
        w[i] = eval_mm_quartic(i, k) ? (uint16_t)(p - 1) : (uint16_t)1;
    }

    for (uint64_t h = 1; h < N; h <<= 1) {
        uint64_t step = h << 1;
        if (N / step >= 64ULL) {
            #pragma omp parallel for schedule(static)
            for (uint64_t base = 0; base < N; base += step) {
                for (uint64_t j = 0; j < h; j++) {
                    uint16_t a = w[base + j];
                    uint16_t b = w[base + h + j];
                    uint16_t s = (uint16_t)(a + b);
                    if (s >= p) s -= p;
                    uint16_t d = (uint16_t)(a + (p - b));
                    if (d >= p) d -= p;
                    w[base + j] = s;
                    w[base + h + j] = d;
                }
            }
        } else {
            for (uint64_t base = 0; base < N; base += step) {
                #pragma omp parallel for schedule(static)
                for (uint64_t j = 0; j < h; j++) {
                    uint16_t a = w[base + j];
                    uint16_t b = w[base + h + j];
                    uint16_t s = (uint16_t)(a + b);
                    if (s >= p) s -= p;
                    uint16_t d = (uint16_t)(a + (p - b));
                    if (d >= p) d -= p;
                    w[base + j] = s;
                    w[base + h + j] = d;
                }
            }
        }
    }

    uint64_t target = 1ULL << k;
    uint16_t tpos = (uint16_t)(target % p);
    uint16_t tneg = (uint16_t)((p - tpos) % p);
    uint64_t viol = 0;
    #pragma omp parallel for reduction(+:viol) schedule(static)
    for (uint64_t i = 0; i < N; i++) {
        if (w[i] != tpos && w[i] != tneg) viol++;
    }
    double dt = omp_get_wtime() - t0;
    printf("[n=%d | uint16 mod p=%u | %.2f GB] time=%.3f s | violations=%llu\n",
           n, p, (double)(N * 2ULL) / (1024.0 * 1024.0 * 1024.0), dt, (unsigned long long)viol);
    free(w);
}

static void run_exact_i32(int n) {
    uint64_t N = 1ULL << n;
    int k = n / 2;
    int32_t *w = (int32_t *)malloc(N * sizeof(int32_t));
    if (!w) { fprintf(stderr, "malloc int32 failed\n"); exit(1); }

    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < N; i++) {
        w[i] = eval_mm_quartic(i, k) ? -1 : 1;
    }

    for (uint64_t h = 1; h < N; h <<= 1) {
        uint64_t step = h << 1;
        #pragma omp parallel for schedule(static)
        for (uint64_t base = 0; base < N; base += step) {
            for (uint64_t j = 0; j < h; j++) {
                int32_t a = w[base + j];
                int32_t b = w[base + h + j];
                w[base + j] = a + b;
                w[base + h + j] = a - b;
            }
        }
    }

    int32_t target = (int32_t)(1ULL << k);
    uint64_t viol = 0;
    #pragma omp parallel for reduction(+:viol) schedule(static)
    for (uint64_t i = 0; i < N; i++) {
        int32_t v = w[i];
        if (v != target && v != -target) viol++;
    }
    double dt = omp_get_wtime() - t0;
    printf("[n=%d | EXACT  int32       | %.2f GB] time=%.3f s | violations=%llu\n",
           n, (double)(N * 4ULL) / (1024.0 * 1024.0 * 1024.0), dt, (unsigned long long)viol);
    free(w);
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 28;
    int mod_only = (argc > 2) ? atoi(argv[2]) : 0;
    if (n % 4 != 0 || n < 16 || n > 36) {
        fprintf(stderr, "Usage: %s <n in {16,20,24,28,32,36}> [mod_only=0|1]\n", argv[0]);
        return 1;
    }
    uint32_t p = minimal_odd_8_15(n);
    printf("=== OpenMP threads: %d | n=%d qubits | certified 8/15 modulus p=%u ===\n",
           omp_get_max_threads(), n, p);
    fflush(stdout);
    if (!mod_only) {
        run_exact_i32(n);
        fflush(stdout);
    }
    if (2U * p <= 255U) {
        run_modular_u8(n, (uint8_t)p);
    } else {
        run_modular_u16(n, (uint16_t)p);
    }
    return 0;
}
