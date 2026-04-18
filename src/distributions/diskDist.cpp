#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>

void generate(
    int SPAWN_RANGE,
    int NUM_BODIES,
    std::vector<float> x, 
    std::vector<float> y, 
    std::vector<float> z, 
    std::vector<float> mass,
    std::vector<float> vx,
    std::vector<float> vy,
    std::vector<float> vz,
    int massBlackHole,
    int G) 
    /*override*/{ 

    for (int i = 1; i < NUM_BODIES; i++)
    {
        float angle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float u = (float)rand() / RAND_MAX;
        float radius = sqrt(u) * SPAWN_RANGE;

        x[i] = radius * cos(angle);
        y[i] = radius * sin(angle);
        z[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f * SPAWN_RANGE;

        float bhMass = massBlackHole;

        float r = sqrt(x[i] * x[i] + y[i] * y[i]);
        if (r < 1.0f)
        {
            r = 1.0f;
        }

        float orbitalVelocity = sqrt(G * bhMass / r);

        vx[i] = sin(angle) * orbitalVelocity;
        vy[i] = -cos(angle) * orbitalVelocity;
        vz[i] = 0.0f;
    }
}