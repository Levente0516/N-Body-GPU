
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable

#define NUMBER_OF_CELLS 8

// Launch with: globalSize = THREADS, localSize = THREADS (single workgroup)
// Child index is ALWAYS < parent index due to atom_dec allocation order.
// Processing btm→root in batches of THREADS with barriers guarantees
// all children are written before any parent is read. No polling needed.
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
    // Only thread 0 executes — single-threaded sequential scan.
    // Correctness guarantee: atom_dec allocation ensures
    // parent_index > child_index always. Iterating low→high
    // means every child is processed before its parent.
    // No barriers, no polling, no deadlock possible.
    if (get_global_id(0) != 0) return;

    const int btm = *bottom;

    for (int node = btm; node <= NUMBER_OF_NODES; node++)
    {
        // Skip body slots and nodes that don't need computation
        if (mass[node] >= 0.0f) continue;

        float totalMass = 0.0f;
        float cx = 0.0f, cy = 0.0f, cz = 0.0f;
        int count = 0;

        for (int i = 0; i < NUMBER_OF_CELLS; i++)
        {
            int c = child[node * NUMBER_OF_CELLS + i];
            if (c < 0) continue;

            // c < node always (atom_dec guarantee), so mass[c] >= 0 here.
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