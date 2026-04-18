
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
    // Only thread 0 runs — parent_index > child_index always (atom_dec guarantee).
    // Processing high→low ensures parent start[] is set before child reads it.
    if (get_global_id(0) != 0) return;

    const int btm = *bottom;

    for (int cell = NUMBER_OF_NODES; cell >= btm; cell--)
    {
        int s = start[cell];
        if (s < 0) continue;  // cell wasn't reached by tree (shouldn't happen)

        for (int i = 0; i < NUMBER_OF_CELLS; i++)
        {
            int c = child[NUMBER_OF_CELLS * cell + i];

            if (c >= NUM_BODIES)
            {
                // Internal cell — pass start index down
                start[c] = s;
                s += nodeCount[c];
            }
            else if (c >= 0)
            {
                // Leaf body — record in sorted array
                sorted[s++] = c;
            }
        }
    }
}