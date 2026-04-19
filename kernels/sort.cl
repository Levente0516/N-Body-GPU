
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable

#define NUMBER_OF_CELLS 8

__attribute__((reqd_work_group_size(THREADS, 1, 1)))
__kernel void sortKernel(
    __global int* child,
    __global int* nodeCount,
    __global int* start,
    __global int* sorted,
    __global int* bottom,
    __global int* numNodes)
{

    if (get_global_id(0) != 0) return;

    const int btm = *bottom;

    for (int cell = NUMBER_OF_NODES; cell >= btm; cell--)
    {
        int s = start[cell];
        if (s < 0) continue;  

        for (int i = 0; i < NUMBER_OF_CELLS; i++)
        {
            int c = child[NUMBER_OF_CELLS * cell + i];

            if (c >= NUM_BODIES)
            {

                start[c] = s;
                s += nodeCount[c];
            }
            else if (c >= 0)
            {

                sorted[s++] = c;
            }
        }
    }
}