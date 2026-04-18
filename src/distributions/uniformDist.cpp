#include <iostream>
#include <vector>

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
    int massBlackHole) 
    /*override*/{ 

    for (int i = 1; i < NUM_BODIES; i++)
    {
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * SPAWN_RANGE;
        y[i] = ((float)rand() / RAND_MAX - 0.5f) * SPAWN_RANGE;
        z[i] = ((float)rand() / RAND_MAX - 0.5f) * SPAWN_RANGE;

        vx[i] = vy[i] = vz[i] = 0;
    }
}