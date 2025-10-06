#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CANVAS_SIZE 100.0f
#define NEIGHBOR_RADIUS 5.0f
#define SEPARATION_RADIUS 2.0f
#define MAX_SPEED 20.0f
#define MAX_FORCE 50.0f
#define ALIGNMENT 1.0f
#define COHESION 0.5f
#define SEPARATION 1.5f
#define DT 0.01f

typedef struct
{
    float x;
    float y;
} float2;

float2 f2(float x, float y) { float2 v = {x, y}; return v; }
float2 f2add(float2 a, float2 b) { return f2(a.x + b.x, a.y + b.y); }
float2 f2sub(float2 a, float2 b) { return f2(a.x - b.x, a.y - b.y); }
float2 f2mul(float2 a, float s) { return f2(a.x * s, a.y * s); }
float f2dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
float f2len(float2 a) { return sqrtf(f2dot(a, a)); }
float frand() { return (float)rand() / RAND_MAX; }

float2 clamp_vector(float2 v, float max)
{
    float len2 = f2dot(v, v);
    float max2 = max * max;
    if (len2 > max2)
    {
        float invL = 1.0f / sqrtf(len2);
        v.x = v.x * max * invL;
        v.y = v.y * max * invL;
    }
    return v;
}

// Wrap coordinates around toroidal space
static inline float2 min_image(float2 r)
{
    const float half = CANVAS_SIZE * 0.5f;
    if (r.x > half) r.x -= CANVAS_SIZE;
    if (r.x < -half) r.x += CANVAS_SIZE;
    if (r.y > half) r.y -= CANVAS_SIZE;
    if (r.y < -half) r.y += CANVAS_SIZE;
    return r;
}

void initialize_boids(float2 *pos, float2 *vel, int count)
{
    srand(0);
    for (int i = 0; i < count; ++i)
    {
        pos[i] = f2(frand() * CANVAS_SIZE, frand() * CANVAS_SIZE);

        float angle = frand() * (float)(2.0 * M_PI);
        float intensity = frand() * MAX_SPEED;

        vel[i] = f2(intensity * cosf(angle), intensity * sinf(angle));
    }
}

float warp_coord(float coord)
{
    float r = fmodf(coord, CANVAS_SIZE);
    if (r < 0.0f) r += CANVAS_SIZE;
    return r;
}

void update_boids(float2 *pos, float2 *vel, int count)
{
    for (int myself = 0; myself < count; ++myself)
    {

        float2 sum_vel = f2(0.0f, 0.0f);
        float2 sum_pos = f2(0.0f, 0.0f);
        float2 sum_sep = f2(0.0f, 0.0f);
        int countAlign = 0;
        int countCoh = 0;

        const float neighR2 = NEIGHBOR_RADIUS * NEIGHBOR_RADIUS;
        const float sepR2   = SEPARATION_RADIUS * SEPARATION_RADIUS;

        for (int other = 0; other < count; ++other)
        {
            if (other == myself)
                continue;

            // Wrap around coordinates 
            float2 diff = f2(pos[other].x - pos[myself].x, pos[other].y - pos[myself].y);
            diff = min_image(diff);
            float d2 = diff.x * diff.x + diff.y * diff.y;

            if (d2 < neighR2)
            {
                sum_vel = f2add(sum_vel, vel[other]);
                sum_pos = f2add(sum_pos, pos[other]);

                countAlign++;
                countCoh++;
            }

            if (d2 < sepR2)
            {
                // Epsilon guard to avoid divide-by-zero when overlapping
                float invDist = 1.0f / sqrtf(d2 + 1e-6f);
                float2 away = f2(diff.x * -invDist, diff.y * -invDist);
                sum_sep = f2add(sum_sep, away);
            }
        }

        // Alignment
        float2 align = f2(0, 0);
        if (countAlign > 0)
        {
            align = f2mul(sum_vel, 1.0f / (float)countAlign);
            align = f2sub(align, vel[myself]);
        }

        // Cohesion
        float2 coh = f2(0, 0);
        if (countCoh > 0)
        {
            float2 center = f2mul(sum_pos, 1.0f / (float)countCoh);
            float2 disp = f2(center.x - pos[myself].x, center.y - pos[myself].y);
            coh = min_image(disp);
        }

        // Combine forces and clamp acceleration
        float2 accel = f2(0, 0);
        accel = f2add(accel, f2mul(align, ALIGNMENT));
        accel = f2add(accel, f2mul(coh, COHESION));
        accel = f2add(accel, f2mul(sum_sep, SEPARATION));
        accel = clamp_vector(accel, MAX_FORCE);

        // Integrate
        vel[myself] = f2add(vel[myself], f2mul(accel, DT));
        vel[myself] = clamp_vector(vel[myself], MAX_SPEED);
        pos[myself] = f2add(pos[myself], f2mul(vel[myself], DT));

        pos[myself].x = warp_coord(pos[myself].x);
        pos[myself].y = warp_coord(pos[myself].y);
    }
}

int main(int argc, char **argv)
{
    printf("Simulation start\n");
    fflush(stdout);
    int boids = 2048;
    int steps = 100;

    if (argc > 1)
        boids = atoi(argv[1]);
    if (argc > 2)
        steps = atoi(argv[2]);

    float2 *pos = (float2 *)malloc(boids * sizeof(float2));
    float2 *vel = (float2 *)malloc(boids * sizeof(float2));

    initialize_boids(pos, vel, boids);

   

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int step = 0; step < steps; step++)
    {

        update_boids(pos, vel, boids);

    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt_sec = (double)(t1.tv_sec - t0.tv_sec) +
                    (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("Simulation took %.6f seconds for %d boids and %d steps\n", dt_sec, boids, steps);
    printf("Average time per step: %.6f ms\n", (dt_sec / steps) * 1e3);


    FILE *fp = fopen("boids.csv", "w");
    if (!fp)
    {
        fprintf(stderr, "Failed to open boids.csv for writing\n");
    }
    else
    {
        fprintf(fp, "id,x,y,dir_x,dir_y\n");
        for (int i = 0; i < boids; ++i)
        {
            float x = pos[i].x;
            float y = pos[i].y;
            float vx = vel[i].x;
            float vy = vel[i].y;
            float speed = f2len(f2(vx, vy));
            float dx = 0.0f, dy = 0.0f;
            if (speed > 1e-6f)
            {
                float invSpeed = 1.0f / speed;
                dx = vx * invSpeed;
                dy = vy * invSpeed;
            }
            fprintf(fp, "%d,%.6f,%.6f,%.6f,%.6f\n", i, x, y, dx, dy);
        }
        fclose(fp);
        printf("Wrote boids.csv\n");
    }

    return 0;
}
