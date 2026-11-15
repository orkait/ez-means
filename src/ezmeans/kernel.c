#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { LANES = 16, CHUNK = 16, STREAMS = 4, MAX_PROBES = 64, ROW_CHUNK = 64, LABEL_CHUNK = 16 };
enum Status { DONE = 0, BAD_PROBES = 1, NO_MEMORY = 2, BAD_LABEL = 3, BAD_DIMS = 4, BAD_SIZE = 5 };

static const double PI = 3.14159265358979323846;
static const double UNIT = 0x1.0p-53;
static const uint64_t GOLDEN = 0x9E3779B97F4A7C15u;
static const uint64_t MIX_FIRST = 0xBF58476D1CE4E5B9u;
static const uint64_t MIX_SECOND = 0x94D049BB133111EBu;

typedef struct {
    int64_t batch;
    int64_t probes;
    int64_t check_every;
    int64_t group_iterations;
    uint64_t seed;
    double alpha0;
    double alpha_min_factor;
    double tau;
    double init_usage;
    double split_epsilon;
} Settings;

typedef struct {
    float *values;
    int64_t *ids;
    int64_t *first;
} Tiles;

static uint64_t next(uint64_t *state)
{
    uint64_t z = (*state += GOLDEN);
    z = (z ^ (z >> 30)) * MIX_FIRST;
    z = (z ^ (z >> 27)) * MIX_SECOND;
    return z ^ (z >> 31);
}

static int64_t bounded(uint64_t *state, int64_t bound)
{
    return (int64_t)(next(state) % (uint64_t)bound);
}

static void permute(int64_t *order, int64_t n, uint64_t *state)
{
    for (int64_t i = n - 1; i > 0; i--) {
        const int64_t j = bounded(state, i + 1);
        const int64_t held = order[i];
        order[i] = order[j];
        order[j] = held;
    }
}

static int64_t *identity(int64_t count)
{
    int64_t *order = malloc((size_t)count * sizeof *order);
    if (order)
        for (int64_t i = 0; i < count; i++)
            order[i] = i;
    return order;
}

static void release(Tiles *tiles)
{
    free(tiles->values);
    free(tiles->ids);
    free(tiles->first);
    *tiles = (Tiles){0};
}

static int tile(const float *source, const int64_t *order, const int64_t *offsets, int64_t sets, int64_t dims, Tiles *out)
{
    out->first = malloc((size_t)(sets + 1) * sizeof *out->first);
    if (!out->first)
        return NO_MEMORY;
    out->first[0] = 0;
    for (int64_t set = 0; set < sets; set++)
        out->first[set + 1] = out->first[set] + (offsets[set + 1] - offsets[set] + LANES - 1) / LANES;
    const int64_t tiles = out->first[sets];
    out->values = malloc((size_t)(tiles * dims * LANES) * sizeof *out->values);
    out->ids = malloc((size_t)(tiles * LANES) * sizeof *out->ids);
    if (!out->values || !out->ids)
        return NO_MEMORY;
    for (int64_t set = 0; set < sets; set++) {
        const int64_t count = offsets[set + 1] - offsets[set];
        const int64_t slots = (out->first[set + 1] - out->first[set]) * LANES;
        for (int64_t slot = 0; slot < slots; slot++) {
            const int64_t at = out->first[set] * LANES + slot;
            const int64_t id = slot < count ? order[offsets[set] + slot] : -1;
            float *lanes = out->values + (at / LANES) * dims * LANES + at % LANES;
            out->ids[at] = id;
            for (int64_t j = 0; j < dims; j++)
                lanes[j * LANES] = id < 0 ? INFINITY : source[id * dims + j];
        }
    }
    return DONE;
}

static int tile_all(const float *source, int64_t count, int64_t dims, Tiles *out)
{
    int64_t *order = identity(count);
    if (!order)
        return NO_MEMORY;
    const int64_t whole[2] = {0, count};
    const int status = tile(source, order, whole, 1, dims, out);
    free(order);
    return status;
}

static inline void square_into(float *acc, float value, const float *lanes)
{
#pragma omp simd
    for (int lane = 0; lane < LANES; lane++) {
        const float gap = value - lanes[lane];
        acc[lane] += gap * gap;
    }
}

static inline int scan(const float *row, const float *tile, int64_t dims, float bound, float *total, int64_t *work)
{
    float a0[LANES] = {0.0f}, a1[LANES] = {0.0f}, a2[LANES] = {0.0f}, a3[LANES] = {0.0f};
    int alive;
    int64_t j = 0;
    do {
        for (int64_t i = j; i < j + CHUNK; i += STREAMS) {
            square_into(a0, row[i], tile + i * LANES);
            square_into(a1, row[i + 1], tile + (i + 1) * LANES);
            square_into(a2, row[i + 2], tile + (i + 2) * LANES);
            square_into(a3, row[i + 3], tile + (i + 3) * LANES);
        }
        j += CHUNK;
        alive = 0;
#pragma omp simd reduction(| : alive)
        for (int lane = 0; lane < LANES; lane++) {
            total[lane] = (a0[lane] + a1[lane]) + (a2[lane] + a3[lane]);
            alive |= total[lane] < bound;
        }
    } while (alive && j < dims);
    *work += j * LANES;
    return alive;
}

static void insert(float *near, int64_t *which, int64_t probes, float value, int64_t id)
{
    if (!(value < near[probes - 1]))
        return;
    int64_t at = probes - 1;
    for (; at > 0 && value < near[at - 1]; at--) {
        near[at] = near[at - 1];
        which[at] = which[at - 1];
    }
    near[at] = value;
    which[at] = id;
}

static void probe(const float *row, const Tiles *groups, int64_t dims, int64_t probes, float *near, int64_t *which, int64_t *work)
{
    for (int64_t p = 0; p < probes; p++) {
        near[p] = INFINITY;
        which[p] = -1;
    }
    float total[LANES];
    for (int64_t t = 0; t < groups->first[1]; t++) {
        const float bound = near[probes - 1];
        if (!scan(row, groups->values + t * dims * LANES, dims, bound, total, work))
            continue;
        for (int lane = 0; lane < LANES; lane++)
            if (total[lane] < bound)
                insert(near, which, probes, total[lane], groups->ids[t * LANES + lane]);
    }
}

static void nearest(const float *row, const Tiles *cells, int64_t dims, const int64_t *which, int64_t probes, float *best,
                    int64_t *label, int64_t *work)
{
    float bound = INFINITY, total[LANES];
    int64_t found = -1;
    for (int64_t p = 0; p < probes && which[p] >= 0; p++)
        for (int64_t t = cells->first[which[p]]; t < cells->first[which[p] + 1]; t++) {
            if (!scan(row, cells->values + t * dims * LANES, dims, bound, total, work))
                continue;
            for (int lane = 0; lane < LANES; lane++)
                if (total[lane] < bound) {
                    bound = total[lane];
                    found = cells->ids[t * LANES + lane];
                }
        }
    *best = bound;
    *label = found;
}

static int64_t label_rows(const float *x, const int64_t *index, int64_t count, int64_t dims, const Tiles *groups, const Tiles *cells,
                          int64_t probes, float *best, int64_t *labels)
{
    int64_t total = 0;
#pragma omp parallel for schedule(dynamic, ROW_CHUNK) reduction(+ : total)
    for (int64_t r = 0; r < count; r++) {
        float near[MAX_PROBES];
        int64_t which[MAX_PROBES];
        const float *row = x + index[r] * dims;
        int64_t scanned = 0;
        probe(row, groups, dims, probes, near, which, &scanned);
        nearest(row, cells, dims, which, probes, best + r, labels + r, &scanned);
        total += scanned;
    }
    return total;
}

static int sum_rows(const float *x, const int64_t *index, int64_t count, int64_t dims, const int64_t *labels, int64_t k, float *sums,
                    float *counts)
{
    for (int64_t r = 0; r < count; r++)
        if (labels[r] < 0 || labels[r] >= k)
            return BAD_LABEL;
    int64_t *start = calloc((size_t)(k + 1), sizeof *start);
    int64_t *cursor = malloc((size_t)k * sizeof *cursor);
    int64_t *sorted = malloc((size_t)count * sizeof *sorted);
    if (!start || !cursor || !sorted) {
        free(start);
        free(cursor);
        free(sorted);
        return NO_MEMORY;
    }
    for (int64_t r = 0; r < count; r++)
        start[labels[r] + 1]++;
    for (int64_t c = 0; c < k; c++) {
        start[c + 1] += start[c];
        cursor[c] = start[c];
    }
    for (int64_t r = 0; r < count; r++)
        sorted[cursor[labels[r]]++] = r;
#pragma omp parallel for schedule(dynamic, LABEL_CHUNK)
    for (int64_t c = 0; c < k; c++) {
        float *sum = sums + c * dims;
        for (int64_t j = 0; j < dims; j++)
            sum[j] = 0.0f;
        for (int64_t i = start[c]; i < start[c + 1]; i++) {
            const float *row = x + index[sorted[i]] * dims;
            for (int64_t j = 0; j < dims; j++)
                sum[j] += row[j];
        }
        counts[c] = (float)(start[c + 1] - start[c]);
    }
    free(start);
    free(cursor);
    free(sorted);
    return DONE;
}

int assign(const float *rows, int64_t count, int64_t dims, const float *groups, int64_t group_count, const float *centroids,
           const int64_t *order, const int64_t *offsets, int64_t probes, float *best, int64_t *label, int64_t *work)
{
    if (probes < 1 || probes > MAX_PROBES || probes > group_count)
        return BAD_PROBES;
    if (dims < CHUNK || dims % CHUNK != 0)
        return BAD_DIMS;
    int64_t *index = identity(count);
    Tiles group_tiles = {0}, cell_tiles = {0};
    int status = index ? tile_all(groups, group_count, dims, &group_tiles) : NO_MEMORY;
    if (status == DONE)
        status = tile(centroids, order, offsets, group_count, dims, &cell_tiles);
    if (status == DONE)
        *work = label_rows(rows, index, count, dims, &group_tiles, &cell_tiles, probes, best, label);
    free(index);
    release(&group_tiles);
    release(&cell_tiles);
    return status;
}

int accumulate(const float *rows, int64_t count, int64_t dims, const int64_t *labels, int64_t k, float *sums, float *counts)
{
    int64_t *index = identity(count);
    const int status = index ? sum_rows(rows, index, count, dims, labels, k, sums, counts) : NO_MEMORY;
    free(index);
    return status;
}

static int regroup(const float *centroids, int64_t k, int64_t dims, const Settings *settings, float *groups, int64_t group_count,
                   int64_t *homes, int64_t *order, int64_t *offsets, int64_t *work)
{
    int64_t *pool = identity(k);
    double *sums = malloc((size_t)(group_count * dims) * sizeof *sums);
    int64_t *sizes = malloc((size_t)group_count * sizeof *sizes);
    Tiles tiles = {0};
    int status = pool && sums && sizes ? DONE : NO_MEMORY;
    uint64_t state = settings->seed;
    for (int64_t g = 0; status == DONE && g < group_count; g++) {
        const int64_t pick = g + bounded(&state, k - g);
        const int64_t held = pool[g];
        pool[g] = pool[pick];
        pool[pick] = held;
        memcpy(groups + g * dims, centroids + pool[g] * dims, (size_t)dims * sizeof *groups);
    }
    for (int64_t round = 0; status == DONE && round <= settings->group_iterations; round++) {
        status = tile_all(groups, group_count, dims, &tiles);
        if (status == DONE) {
            const int64_t only[1] = {0};
            int64_t total = 0;
#pragma omp parallel for schedule(dynamic, ROW_CHUNK) reduction(+ : total)
            for (int64_t c = 0; c < k; c++) {
                float best;
                nearest(centroids + c * dims, &tiles, dims, only, 1, &best, homes + c, &total);
            }
            *work += total;
        }
        release(&tiles);
        if (status != DONE || round == settings->group_iterations)
            break;
        memset(sums, 0, (size_t)(group_count * dims) * sizeof *sums);
        memset(sizes, 0, (size_t)group_count * sizeof *sizes);
        for (int64_t c = 0; c < k; c++) {
            sizes[homes[c]]++;
            for (int64_t j = 0; j < dims; j++)
                sums[homes[c] * dims + j] += centroids[c * dims + j];
        }
        for (int64_t g = 0; g < group_count; g++)
            if (sizes[g] > 0)
                for (int64_t j = 0; j < dims; j++)
                    groups[g * dims + j] = (float)(sums[g * dims + j] / (double)sizes[g]);
    }
    if (status == DONE) {
        memset(offsets, 0, (size_t)(group_count + 1) * sizeof *offsets);
        for (int64_t c = 0; c < k; c++)
            offsets[homes[c] + 1]++;
        for (int64_t g = 0; g < group_count; g++)
            offsets[g + 1] += offsets[g];
        memcpy(sizes, offsets, (size_t)group_count * sizeof *sizes);
        for (int64_t c = 0; c < k; c++)
            order[sizes[homes[c]]++] = c;
    }
    free(pool);
    free(sums);
    free(sizes);
    return status;
}

static void update(float *centroids, float *usage, const float *sums, const float *counts, int64_t k, int64_t dims, double alpha)
{
    const int decays = alpha != 1.0;
    const float rate = (float)alpha;
    const float keep = (float)(1.0 - alpha);
    for (int64_t c = 0; c < k; c++) {
        if (decays)
            usage[c] *= keep;
        if (!(counts[c] > 0.0f))
            continue;
        const float mass = (decays ? usage[c] : 0.0f) + rate * counts[c];
        const float weight = rate * counts[c] / mass;
        float *centroid = centroids + c * dims;
        for (int64_t j = 0; j < dims; j++) {
            const float mean = sums[c * dims + j] / counts[c];
            centroid[j] = weight == 1.0f ? mean : centroid[j] + weight * (mean - centroid[j]);
        }
        usage[c] = mass;
    }
}

typedef struct {
    double key;
    int64_t id;
} Ranked;

static int descending(const void *left, const void *right)
{
    const Ranked *a = left, *b = right;
    if (a->key != b->key)
        return a->key < b->key ? 1 : -1;
    return (a->id > b->id) - (a->id < b->id);
}

static int revive(float *centroids, float *usage, int64_t k, int64_t dims, const Settings *settings, uint64_t *state)
{
    double total = 0.0;
    for (int64_t c = 0; c < k; c++)
        total += usage[c];
    const float threshold = (float)total * (float)settings->tau / (float)k;
    int64_t unused = 0, donors = 0;
    for (int64_t c = 0; c < k; c++) {
        unused += usage[c] < threshold;
        donors += !(usage[c] < threshold) && usage[c] > 0.0f;
    }
    const int64_t count = unused < donors ? unused : donors;
    if (count == 0)
        return DONE;
    Ranked *ranked = malloc((size_t)k * sizeof *ranked);
    int64_t *targets = malloc((size_t)count * sizeof *targets);
    if (!ranked || !targets) {
        free(ranked);
        free(targets);
        return NO_MEMORY;
    }
    for (int64_t c = 0, found = 0; c < k; c++) {
        const double draw = fmax((double)(next(state) >> 11) * UNIT, DBL_MIN);
        const int spare = usage[c] < threshold;
        ranked[c] = (Ranked){!spare && usage[c] > 0.0f ? log((double)usage[c]) - log(-log(draw)) : -INFINITY, c};
        if (spare && found < count)
            targets[found++] = c;
    }
    qsort(ranked, (size_t)k, sizeof *ranked, descending);
    const float offset = (float)settings->split_epsilon;
    for (int64_t s = 0; s < count; s++) {
        const int64_t source = ranked[s].id, target = targets[s];
        for (int64_t j = 0; j < dims; j++) {
            const float signed_offset = j % 2 ? -offset : offset;
            const float picked = centroids[source * dims + j];
            centroids[source * dims + j] = picked * (1.0f - signed_offset);
            centroids[target * dims + j] = picked * (1.0f + signed_offset);
        }
        usage[source] /= 2.0f;
        usage[target] = usage[source];
    }
    free(ranked);
    free(targets);
    return DONE;
}

static double scheduled(const Settings *settings, int64_t step, int64_t total)
{
    const double done = (double)(step < total ? step : total) / (double)(total > 1 ? total : 1);
    const double cosine = 0.5 * (1.0 + cos(PI * done));
    return settings->alpha0 * (settings->alpha_min_factor + (1.0 - settings->alpha_min_factor) * cosine);
}

int train(const float *x, int64_t n, int64_t dims, int64_t k, const int64_t *plan, int64_t epochs, const Settings *settings,
          const float *init, int64_t init_count, float *out, int64_t *work)
{
    const int64_t rounded = llround(sqrt((double)k));
    const int64_t group_count = rounded > 1 ? rounded : 1;
    const int64_t width = (dims + CHUNK - 1) / CHUNK * CHUNK;
    const int copying = width != dims;
    if (dims < 1)
        return BAD_DIMS;
    if (settings->probes < 1 || settings->probes > MAX_PROBES || settings->probes > group_count)
        return BAD_PROBES;
    if (k < 1 || k > n || settings->batch < 1 || settings->check_every < 1 || epochs < 1 || (init_count != 0 && init_count != k))
        return BAD_SIZE;
    const size_t table = (size_t)(k * width);
    const int64_t batch = settings->batch;
    int64_t *order = identity(n);
    float *centroids = malloc(table * sizeof *centroids);
    float *usage = malloc((size_t)k * sizeof *usage);
    float *sums = malloc(table * sizeof *sums);
    float *counts = malloc((size_t)k * sizeof *counts);
    float *pass_sums = calloc(table, sizeof *pass_sums);
    double *pass_counts = calloc((size_t)k, sizeof *pass_counts);
    float *best = malloc((size_t)batch * sizeof *best);
    int64_t *labels = malloc((size_t)batch * sizeof *labels);
    float *groups = malloc((size_t)(group_count * width) * sizeof *groups);
    int64_t *homes = malloc((size_t)k * sizeof *homes);
    int64_t *cell_order = malloc((size_t)k * sizeof *cell_order);
    int64_t *offsets = malloc((size_t)(group_count + 1) * sizeof *offsets);
    float *working = copying ? malloc((size_t)n * (size_t)width * sizeof *working) : NULL;
    Tiles group_tiles = {0}, cell_tiles = {0};
    int status = order && centroids && usage && sums && counts && pass_sums && pass_counts && best && labels && groups && homes &&
                         cell_order && offsets && (!copying || working)
                     ? DONE
                     : NO_MEMORY;
    if (status == DONE && copying) {
#pragma omp parallel for schedule(static)
        for (int64_t r = 0; r < n; r++) {
            float *row = working + r * width;
            memcpy(row, x + r * dims, (size_t)dims * sizeof *row);
            memset(row + dims, 0, (size_t)(width - dims) * sizeof *row);
        }
    }
    const float *data = copying ? working : x;
    uint64_t state = settings->seed;
    uint64_t revival = next(&state);
    int64_t steps = 0, step = 0, searches = 0;
    *work = 0;
    if (status == DONE) {
        permute(order, n, &state);
        for (int64_t c = 0; c < k; c++) {
            float *centroid = centroids + c * width;
            if (init_count) {
                memcpy(centroid, init + c * dims, (size_t)dims * sizeof *centroid);
                memset(centroid + dims, 0, (size_t)(width - dims) * sizeof *centroid);
            } else {
                memcpy(centroid, data + order[c] * width, (size_t)width * sizeof *centroid);
            }
            usage[c] = (float)settings->init_usage;
        }
        for (int64_t e = 0; e < epochs; e++)
            steps += (plan[e] + batch - 1) / batch;
    }
    for (int64_t e = 0; status == DONE && e < epochs; e++) {
        const int last = e == epochs - 1;
        permute(order, n, &state);
        for (int64_t begin = 0; status == DONE && begin < plan[e]; begin += batch) {
            const int64_t count = plan[e] - begin < batch ? plan[e] - begin : batch;
            const int64_t *index = order + begin;
            if (searches % settings->check_every == 0) {
                release(&group_tiles);
                status = regroup(centroids, k, width, settings, groups, group_count, homes, cell_order, offsets, work);
                if (status == DONE)
                    status = tile_all(groups, group_count, width, &group_tiles);
            }
            searches++;
            if (status == DONE)
                status = tile(centroids, cell_order, offsets, group_count, width, &cell_tiles);
            if (status == DONE)
                *work += label_rows(data, index, count, width, &group_tiles, &cell_tiles, settings->probes, best, labels);
            if (status == DONE)
                status = sum_rows(data, index, count, width, labels, k, sums, counts);
            release(&cell_tiles);
            if (status != DONE)
                break;
            update(centroids, usage, sums, counts, k, width, scheduled(settings, step, steps));
            if (last) {
                for (int64_t c = 0; c < k; c++)
                    pass_counts[c] += counts[c];
                for (size_t i = 0; i < table; i++)
                    pass_sums[i] += sums[i];
            }
            if ((step + 1) % settings->check_every == 0)
                status = revive(centroids, usage, k, width, settings, &revival);
            step++;
        }
    }
    if (status == DONE) {
        for (int64_t c = 0; c < k; c++) {
            float *mean = sums + c * width;
            for (int64_t j = 0; j < width; j++)
                mean[j] = pass_counts[c] > 0.0 ? pass_sums[c * width + j] / (float)pass_counts[c] : centroids[c * width + j];
            memcpy(out + c * dims, mean, (size_t)dims * sizeof *out);
        }
    }
    release(&group_tiles);
    free(working);
    free(order);
    free(centroids);
    free(usage);
    free(sums);
    free(counts);
    free(pass_sums);
    free(pass_counts);
    free(best);
    free(labels);
    free(groups);
    free(homes);
    free(cell_order);
    free(offsets);
    return status;
}
