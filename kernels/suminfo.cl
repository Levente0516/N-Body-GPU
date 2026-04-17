
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable

#define NUMBER_OF_CELLS 8

__attribute__((reqd_work_group_size(THREADS, 1, 1)))
__kernel void summarizeTreeKernel(
    __global float* x,   
    __global float* y, 
    __global float* z,
    __global volatile float* mass,
    __global int* child,
    __global int* nodeCount,
    __global int* bottom,
    __global int* numNodes)
{
    __local volatile int lcChild[THREADS * NUMBER_OF_CELLS];

    const int stepSize = get_local_size(0) * get_num_groups(0);
    const int localId = get_local_id(0);
    const int btm = *bottom;

    int node = (btm & (-WARPSIZE)) + get_global_id(0);
    if (node < btm)
    {
        node += stepSize;
    } 

    int   missing  = 0;
    int   cbCount  = 0;
    float cellMass = 0.0f;
    float cx = 0.0f; 
    float cy = 0.0f; 
    float cz = 0.0f;

    while (node <= NUMBER_OF_NODES) 
    {
        if (missing == 0) 
        {
            cbCount = 0; 
            cellMass = 0.0f; 
            cx = cy = cz = 0.0f;
            int usedIdx = 0;

#pragma unroll NUMBER_OF_CELLS
            for (int i = 0; i < NUMBER_OF_CELLS; i++) 
            {
                int c = child[node * NUMBER_OF_CELLS + i];

                if (c >= 0)
                {
                    if (i != usedIdx) 
                    {
                        child[NUMBER_OF_CELLS * node + i] = EMPTY;
                        child[NUMBER_OF_CELLS * node + usedIdx] = c;
                    }

                    lcChild[THREADS * missing + localId] = c;

                    float m = mass[c];
                    missing++;

                    if (m >= 0.0f) 
                    {
                        missing--;
                        if (c >= NUM_BODIES)
                        {
                            cbCount += nodeCount[c] - 1;
                        } 
                        cellMass += m;
                        cx += x[c] * m; 
                        cy += y[c] * m; 
                        cz += z[c] * m;
                    }
                    ++usedIdx;
                }
            }

            cbCount += usedIdx;
        }

        if (missing != 0) 
        {
            float last_m;
            do 
            {
                int c  = lcChild[(missing - 1) * THREADS + localId];

                last_m = mass[c];

                if (last_m >= 0.0f)
                {
                    missing--;
                    if (c >= NUM_BODIES)
                    {
                        cbCount += nodeCount[c] - 1;
                    } 
                    cellMass += last_m;
                    cx += x[c] * last_m; 
                    cy += y[c] * last_m; 
                    cz += z[c] * last_m;
                }
            } while (last_m >= 0.0f && missing != 0);
        }

        if (missing == 0) 
        {
            nodeCount[node] = cbCount;

            if (cellMass > 0.0f)
            {
                float inv = 1.0f / cellMass;
                x[node] = cx * inv;
                y[node] = cy * inv;
                z[node] = cz * inv;
            }

            mem_fence(CLK_GLOBAL_MEM_FENCE);

            mass[node] = cellMass;

            node += stepSize;
        }
    }
}