
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable

#define NUMBER_OF_CELLS 8

__attribute__((reqd_work_group_size(THREADS, 1, 1)))
__kernel void summarizeTreeKernel(
    __global float*          x,
    __global float*          y,
    __global float*          z,
    __global volatile float* mass,
    __global volatile int*   child,
    __global int*            nodeCount,
    __global int*            bottom)
{

    if (get_global_id(0) != 0) return;

    const int btm = *bottom;

    for (int node = btm; node <= NUMBER_OF_NODES; node++)
    {

        if (mass[node] >= 0.0f) continue;

        float totalMass = 0.0f;
        float cx = 0.0f, cy = 0.0f, cz = 0.0f;
        int count = 0;

        for (int i = 0; i < NUMBER_OF_CELLS; i++)
        {
            int c = child[node * NUMBER_OF_CELLS + i];
            if (c < 0) continue;

            float m = mass[c];
            totalMass += m;
            cx += x[c] * m;
            cy += y[c] * m;
            cz += z[c] * m;
            count += (c < NUM_BODIES) ? 1 : nodeCount[c];
        }

        nodeCount[node] = count;

        if (totalMass > 0.0f)
        {
            float inv = 1.0f / totalMass;
            x[node] = cx * inv;
            y[node] = cy * inv;
            z[node] = cz * inv;
        }

        mem_fence(CLK_GLOBAL_MEM_FENCE);
        mass[node] = totalMass;
    }
}