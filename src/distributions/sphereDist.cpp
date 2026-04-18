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
        float u = (float)rand() / RAND_MAX;
        float v = (float)rand() / RAND_MAX;

        float theta = 2.0f * M_PI * u;
        float phi   = acos(2.0f * v - 1.0f);

        float r = cbrt((float)rand() / RAND_MAX) * SPAWN_RANGE;

        x[i] = r * sin(phi) * cos(theta);
        y[i] = r * sin(phi) * sin(theta);
        z[i] = r * cos(phi);

        //TODO velocity
    }
}