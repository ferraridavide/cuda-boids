#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CANVAS_SIZE 400.0f
#define NEIGHBOR_RADIUS 5.0f
#define SEPARATION_RADIUS 2.0f
#define MAX_SPEED 20.0f
#define MAX_FORCE 50.0f
#define ALIGNMENT 1.0f
#define COHESION 0.5f
#define SEPARATION 1.5f
#define DT 0.01f

// Grid dimensions
// #define GRID_RES 8
// #define NUM_CELLS (GRID_RES * GRID_RES)
// #define CELL_SIZE (CANVAS_SIZE / GRID_RES)

__device__ __host__ inline float2 f2(float x, float y)
{
    float2 v{x, y};
    return v;
}

// Inline: replace this function call with its code body -> Less overhead
__device__ __host__ inline float2 f2add(float2 a, float2 b) { return f2(a.x + b.x, a.y + b.y); }
__device__ __host__ inline float2 f2sub(float2 a, float2 b) { return f2(a.x - b.x, a.y - b.y); }
__device__ __host__ inline float2 f2mul(float2 a, float s) { return f2(a.x * s, a.y * s); }
__device__ __host__ inline float f2dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
__device__ __host__ inline float f2len(float2 a) { return sqrtf(f2dot(a, a)); }
__device__ inline float2 clamp_vector(float2 v, float max)
{
    float len2 = f2dot(v, v);
    float max2 = max * max;
    if (len2 > max2)
    {
        float invL = rsqrtf(len2);
        v = f2mul(v, max * invL);
    }
    return v;
}
// Wrap coordinate into [0, CANVAS_SIZE)
__device__ inline float warp_coord(float coord)
{
    float r = fmodf(coord, CANVAS_SIZE);
    if (r < 0.0f)
        r += CANVAS_SIZE;
    return r;
}

// Wrap coordinates around toroidal space [0, CANVAS_SIZE)
__device__ inline float2 min_image(float2 r)
{
    const float half = CANVAS_SIZE * 0.5f;
    if (r.x > half)
        r.x -= CANVAS_SIZE;
    if (r.x < -half)
        r.x += CANVAS_SIZE;
    if (r.y > half)
        r.y -= CANVAS_SIZE;
    if (r.y < -half)
        r.y += CANVAS_SIZE;
    return r;
}

__global__ void hash_boids(const float2* pos, int* cell_idx, const int cell_size, const int grid_res, const int boids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= boids) return;

    float2 p = pos[i];
    int grid_x = (int)(p.x / cell_size);
    int grid_y = (int)(p.y / cell_size);
    cell_idx[i] = grid_y * grid_res + grid_x;
}

__global__ void find_cell_bounds(const int* cell_idx, int* cell_starts, int* cell_ends, int boids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= boids) return;

    // Find cell start and end indices
    if (i == 0) {
        cell_starts[cell_idx[i]] = 0;
    } else {
        if (cell_idx[i] != cell_idx[i-1]) {
            cell_starts[cell_idx[i]] = i;
            cell_ends[cell_idx[i-1]] = i;
        }
    }

    if (i == boids - 1) {
        cell_ends[cell_idx[i]] = boids;
    }
}

__global__ void update_boids(
    const float2 *pos_in,
    const float2 *vel_in,
    float2 *pos_out,
    float2 *vel_out,
    const int* cell_starts,
    const int* cell_ends,
    const int grid_res,
    const int cell_size,
    const int boids)
{
    int myself = blockIdx.x * blockDim.x + threadIdx.x;
    if (myself >= boids)
        return;

    float2 my_pos = pos_in[myself];
    float2 my_vel = vel_in[myself];

    float2 sum_vel = f2(0.0f, 0.0f);
    float2 sum_pos = f2(0.0f, 0.0f);
    float2 sum_sep = f2(0.0f, 0.0f);
    int countAlign = 0;
    int countCoh = 0;

    const float neighR2 = NEIGHBOR_RADIUS * NEIGHBOR_RADIUS;
    const float sepR2 = SEPARATION_RADIUS * SEPARATION_RADIUS;

    // Determine my cell
    int my_grid_x = (int)(my_pos.x / cell_size);
    int my_grid_y = (int)(my_pos.y / cell_size);

    // Loop over my cell and neighbor cells
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int cell_y = (my_grid_y + dy + grid_res) % grid_res;
            int cell_x = (my_grid_x + dx + grid_res) % grid_res;
            int cell_id = cell_y * grid_res + cell_x;

            int start = cell_starts[cell_id];
            int end = cell_ends[cell_id];

            if (start == -1) continue; // Skip empty cells

            for (int other = start; other < end; ++other)
            {
                if (other == myself)
                    continue;

                float2 r = f2sub(pos_in[other], my_pos);
                r = min_image(r);
                float d2 = f2dot(r, r);

                if (d2 < neighR2)
                {
                    sum_vel = f2add(sum_vel, vel_in[other]);
                    sum_pos = f2add(sum_pos, f2add(pos_in[myself], r));
                    countAlign++;
                    countCoh++;
                }

                if (d2 < sepR2)
                {
                    float invDist = rsqrtf(d2 + 1e-6f);
                    float2 away = f2mul(r, -invDist); // -r/|r|
                    sum_sep = f2add(sum_sep, away);
                }
            }
        }
    }

    // Alignment
    float2 align = f2(0, 0);
    if (countAlign > 0)
    {
        align = f2mul(sum_vel, 1.0f / (float)countAlign);
        align = f2sub(align, my_vel);
    }
    // Cohesion
    float2 coh = f2(0, 0);
    if (countCoh > 0)
    {
        float2 center = f2mul(sum_pos, 1.0f / (float)countCoh);
        coh = min_image(f2sub(center, my_pos));
    }
    // Combine forces
    float2 accel = f2(0, 0);
    accel = f2add(accel, f2mul(align, ALIGNMENT));
    accel = f2add(accel, f2mul(coh, COHESION));
    accel = f2add(accel, f2mul(sum_sep, SEPARATION));
    accel = clamp_vector(accel, MAX_FORCE);
    // Integrate
    float2 new_vel = f2add(my_vel, f2mul(accel, DT));
    new_vel = clamp_vector(new_vel, MAX_SPEED);
    float2 new_pos = f2add(my_pos, f2mul(new_vel, DT));

    new_pos.x = warp_coord(new_pos.x);
    new_pos.y = warp_coord(new_pos.y);

    pos_out[myself] = new_pos;
    vel_out[myself] = new_vel;
}

static inline float frand() { return (float)rand() / (float)RAND_MAX; }
static void initialize_boids(float2 *pos, float2 *vel, int count)
{
    srand(1234);
    for (int i = 0; i < count; ++i)
    {
        pos[i] = f2(frand() * CANVAS_SIZE, frand() * CANVAS_SIZE);

        float angle = frand() * (float)(2.0 * M_PI);
        float intensity = frand() * MAX_SPEED;

        vel[i] = f2(intensity * cosf(angle), intensity * sinf(angle));
    }
}
int main(int argc, char **argv)
{
    int boids = 2048; // default
    int steps = 100;
    int grid_res = 8;

    int TPB = 1024;
    
    if (argc > 1)
    boids = atoi(argv[1]);
    if (argc > 2)
    steps = atoi(argv[2]);
    if (argc > 3)
    grid_res = atoi(argv[3]);
    
    int cell_size = CANVAS_SIZE / grid_res;
    size_t bytes =           (size_t)boids * sizeof(float2);
    size_t boids_int_bytes = (size_t)boids * sizeof(int);
    size_t grid_int_bytes = (size_t)(grid_res * grid_res) * sizeof(int);

    // Host buffers
    float2 *h_pos = (float2 *)malloc(bytes);
    float2 *h_vel = (float2 *)malloc(bytes);
    int *h_cell_idx = (int *)malloc(boids_int_bytes);

    initialize_boids(h_pos, h_vel, boids);
    // Device buffers (double-buffered)
    float2 *d_pos_in, *d_vel_in;
    float2 *d_pos_out, *d_vel_out;
    int *d_cell_idx;
    int *d_cell_starts, *d_cell_ends;

    cudaMalloc((void **)&d_pos_in, bytes);
    cudaMalloc((void **)&d_vel_in, bytes);
    cudaMalloc((void **)&d_pos_out, bytes);
    cudaMalloc((void **)&d_vel_out, bytes);

    cudaMalloc((void **)&d_cell_idx, boids_int_bytes);
    cudaMalloc((void **)&d_cell_starts, grid_int_bytes);
    cudaMalloc((void **)&d_cell_ends, grid_int_bytes);

    cudaMemcpy(d_pos_in, h_pos, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_in, h_vel, bytes, cudaMemcpyHostToDevice);
    dim3 block(TPB);
    dim3 grid((boids + TPB - 1) / TPB);
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int s = 0; s < steps; ++s)
    {
        hash_boids<<<grid, block>>>(d_pos_in, d_cell_idx, cell_size, grid_res, boids);

        // Sort boids by cell index to improve memory access patterns
        thrust::device_ptr<int> d_cell_idx_ptr(d_cell_idx);
        thrust::device_ptr<float2> d_pos_in_ptr(d_pos_in);
        thrust::device_ptr<float2> d_vel_in_ptr(d_vel_in);
        auto d_values_ptr = thrust::make_zip_iterator(thrust::make_tuple(d_pos_in_ptr, d_vel_in_ptr));
        thrust::sort_by_key(d_cell_idx_ptr, d_cell_idx_ptr + boids, d_values_ptr);

        // Find cell boundaries
        cudaMemset(d_cell_starts, -1, grid_int_bytes);
        cudaMemset(d_cell_ends, -1, grid_int_bytes);
        find_cell_bounds<<<grid, block>>>(d_cell_idx, d_cell_starts, d_cell_ends, boids);

        update_boids<<<grid, block>>>(d_pos_in, d_vel_in, d_pos_out, d_vel_out, d_cell_starts, d_cell_ends, grid_res, cell_size, boids);
        // Swap buffers
        float2 *tmp_pos = d_pos_in;
        d_pos_in = d_pos_out;
        d_pos_out = tmp_pos;
        float2 *tmp_vel = d_vel_in;
        d_vel_in = d_vel_out;
        d_vel_out = tmp_vel;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Hash final positions
    hash_boids<<<grid, block>>>(d_pos_in, d_cell_idx, cell_size, grid_res, boids);

    // Copy back final state
    cudaMemcpy(h_pos, d_pos_in, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel, d_vel_in, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cell_idx, d_cell_idx, boids_int_bytes, cudaMemcpyDeviceToHost);
    printf("Naive 2D boids (double-buffered): %d boids × %d steps in %.3f ms (%.3f ms/step)\n",
           boids, steps, ms, ms / (float)steps);

    // printf("%.3f",ms);
    FILE *fp = fopen("boids_fast.csv", "w");
    if (!fp)
    {
        fprintf(stderr, "Failed to open boids_fast.csv for writing\n");
    }
    else
    {
        fprintf(fp, "id,x,y,dir_x,dir_y,cell_idx\n");
        for (int i = 0; i < boids; ++i)
        {
            float x = h_pos[i].x;
            float y = h_pos[i].y;
            float vx = h_vel[i].x;
            float vy = h_vel[i].y;
            float speed = f2len(f2(vx, vy));
            float dx = 0.0f, dy = 0.0f;
            if (speed > 1e-6f)
            {
                float invSpeed = 1.0f / speed;
                dx = vx * invSpeed;
                dy = vy * invSpeed;
            }
            fprintf(fp, "%d,%.6f,%.6f,%.6f,%.6f,%d\n", i, x, y, dx, dy, h_cell_idx[i]);
        }
        fclose(fp);
        printf("Wrote boids_fast.csv\n");
    }
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_pos_in);
    cudaFree(d_vel_in);
    cudaFree(d_pos_out);
    cudaFree(d_vel_out);

    cudaFree(d_cell_idx);
    cudaFree(d_cell_starts);
    cudaFree(d_cell_ends);

    free(h_pos);
    free(h_vel);
    free(h_cell_idx);
    return 0;
}