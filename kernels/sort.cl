
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
    const int stepSize = get_local_size(0) * get_num_groups(0);
    const int btm = *bottom;

    int cell = NUMBER_OF_NODES + 1 - stepSize + get_global_id(0);

    while (cell >= btm) 
    {
        int s = start[cell];
        if (s >= 0) 
        {
#pragma unroll NUMBER_OF_CELLS
            for (int i = 0; i < NUMBER_OF_CELLS; i++) 
            {
                int c = child[NUMBER_OF_CELLS * cell + i];

                if (c >= NUM_BODIES) 
                {
                    start[c] = s;
                    mem_fence(CLK_GLOBAL_MEM_FENCE);
                    s += nodeCount[c];
                } 
                else if (c >= 0)
                {
                    sorted[s] = c;
                    s++;
                }
            }
            cell -= stepSize;
        }

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}
