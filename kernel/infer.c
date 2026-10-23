#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#define _GNU_SOURCE
#include <sched.h>
#endif

typedef struct {
    uint32_t n, h, d;
    float scales[6];
    int32_t ma, mf;
    int8_t *w1, *w2, *out;
    int32_t *b1, *b2;
    int32_t bias;
} Model;

static int read_exact(FILE *f, void *p, size_t size) { return fread(p, 1, size, f) == size; }

static int load_model(const char *path, Model *m) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char magic[4];
    int ok = read_exact(f, magic, 4) && memcmp(magic, "HWF2", 4) == 0;
    ok = ok && read_exact(f, &m->n, 4) && read_exact(f, &m->h, 4) && read_exact(f, &m->d, 4);
    ok = ok && read_exact(f, m->scales, 24) && read_exact(f, &m->ma, 4) && read_exact(f, &m->mf, 4);
    if (!ok || m->n > 100000 || m->h > 100000 || m->d > 100000) { fclose(f); return 0; }
    size_t nh = (size_t)m->n * m->h;
    size_t nhd = nh * m->d;
    m->w1 = malloc(nhd ? nhd : 1);
    m->b1 = malloc(nh * 4 ? nh * 4 : 1);
    m->w2 = malloc(nh ? nh : 1);
    m->b2 = malloc(m->n * 4 ? m->n * 4 : 1);
    m->out = malloc(m->n ? m->n : 1);
    ok = m->w1 && m->b1 && m->w2 && m->b2 && m->out;
    ok = ok && read_exact(f, m->w1, nhd) && read_exact(f, m->b1, nh * 4);
    ok = ok && read_exact(f, m->w2, nh) && read_exact(f, m->b2, m->n * 4);
    ok = ok && read_exact(f, m->out, m->n) && read_exact(f, &m->bias, 4);
    if (fgetc(f) != EOF) ok = 0;
    fclose(f);
    return ok;
}

static int8_t requant(int32_t x, int32_t mult, int nonnegative) {
    int64_t value = (int64_t)x * mult;
    int64_t q = value >= 0 ? (value + (1LL << 19)) >> 20 : -((-value + (1LL << 19)) >> 20);
    if (q > 127) q = 127;
    if (q < (nonnegative ? 0 : -128)) q = nonnegative ? 0 : -128;
    return (int8_t)q;
}

static int32_t predict_one(const Model *m, const int8_t *x) {
    int32_t total = m->bias;
    for (uint32_t neuron = 0; neuron < m->n; neuron++) {
        int32_t feat = m->b2[neuron];
        for (uint32_t hidden = 0; hidden < m->h; hidden++) {
            size_t idx = (size_t)neuron * m->h + hidden;
            int32_t z = m->b1[idx];
            for (uint32_t input = 0; input < m->d; input++)
                z += (int32_t)x[input] * m->w1[idx * m->d + input];
            int8_t a = requant(z > 0 ? z : 0, m->ma, 1);
            feat += (int32_t)a * m->w2[idx];
        }
        total += (int32_t)requant(feat, m->mf, 0) * m->out[neuron];
    }
    return total;
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int pin_first_core(int *core) {
#ifdef __linux__
    cpu_set_t allowed, only;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return 0;
    for (int i = 0; i < CPU_SETSIZE; i++) if (CPU_ISSET(i, &allowed)) {
        CPU_ZERO(&only); CPU_SET(i, &only);
        if (sched_setaffinity(0, sizeof(only), &only) == 0) { *core = i; return 1; }
    }
#endif
    *core = -1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4 || (strcmp(argv[1], "infer") && strcmp(argv[1], "bench"))) {
        fprintf(stderr, "usage: infer infer|bench model.hwi inputs.i8 [warmup iterations]\n");
        return 2;
    }
    Model m = {0};
    if (!load_model(argv[2], &m)) { fprintf(stderr, "invalid model\n"); return 2; }
    FILE *f = fopen(argv[3], "rb");
    uint32_t count = 0;
    if (!f || !read_exact(f, &count, 4) || count == 0 || count > 10000000) {
        fprintf(stderr, "invalid input count\n"); return 2;
    }
    size_t size = (size_t)count * m.d;
    int8_t *inputs = malloc(size ? size : 1);
    if (!inputs || !read_exact(f, inputs, size) || fgetc(f) != EOF) {
        fprintf(stderr, "invalid input payload\n"); return 2;
    }
    fclose(f);
    if (!strcmp(argv[1], "infer")) {
        for (uint32_t i = 0; i < count; i++) {
            int32_t value = predict_one(&m, inputs + (size_t)i * m.d);
            fwrite(&value, 4, 1, stdout);
        }
        return 0;
    }
    int warmup = argc > 4 ? atoi(argv[4]) : 100;
    int iterations = argc > 5 ? atoi(argv[5]) : 1000;
    if (warmup < 1 || iterations < 100) { fprintf(stderr, "warmup>=1 iterations>=100 required\n"); return 2; }
    int core = -1, pinned = pin_first_core(&core);
    volatile int32_t sink = 0;
    for (int i = 0; i < warmup; i++)
        for (uint32_t j = 0; j < count; j++) sink ^= predict_one(&m, inputs + (size_t)j * m.d);
    double *times = malloc((size_t)iterations * sizeof(double));
    double repeat_medians[5];
    double median = 0, p99 = 0;
    for (int rep = 0; rep < 5; rep++) {
        for (int i = 0; i < iterations; i++) {
            uint64_t start = now_ns();
            for (uint32_t j = 0; j < count; j++) sink ^= predict_one(&m, inputs + (size_t)j * m.d);
            times[i] = (double)(now_ns() - start) / count;
        }
        qsort(times, iterations, sizeof(double), compare_double);
        repeat_medians[rep] = times[iterations / 2];
        median += repeat_medians[rep] / 5;
        p99 += times[(iterations * 99 + 99) / 100 - 1] / 5;
    }
    double variance = 0;
    for (int i = 0; i < 5; i++) {
        double diff = repeat_medians[i] - median;
        variance += diff * diff / 5;
    }
    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname) - 1);
    printf("{\"median_ns_per_flow\":%.3f,\"p99_ns_per_flow\":%.3f,\"repeat_median_variance\":%.3f,\"warmup\":%d,\"iterations\":%d,\"repeats\":5,\"core\":%d,\"pinned\":%s,\"hostname\":\"%s\",\"sink\":%d}\n",
           median, p99, variance, warmup, iterations, core, pinned ? "true" : "false", hostname, sink);
    return 0;
}
