
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
    __global volatile int* child,
    __global int* parentArray, // You'll need to store parent pointers during Build
    __global int* readyCounter,
    __global int* nodeCount)
{
    int curr = get_global_id(0);
    if (curr >= NUMBER_OF_NODES) return;

    // Start from a leaf (body)
    int parent = parentArray[curr];

    while (parent != -1) // While not at the root
    {
        // Increment the parent's "children finished" counter
        int finished = atom_inc(&readyCounter[parent]);

        if (finished < 7) {
            return; 
        }

        // --- I AM THE LAST CHILD ---
        // Now I safely calculate the Parent's Center of Mass
        float m_total = 0.0f;
        float tx = 0;
        float ty = 0;
        float tz = 0;
        int count = 0;

        for (int i = 0; i < 8; i++) {
            int c = child[parent * 8 + i];
            if (c != -1) {
                float m = mass[c];
                if (m > 0.0f)
                {
                    m_total += m;
                    tx += x[c] * m;
                    ty += y[c] * m;
                    tz += z[c] * m;
                    count += (c >= NUMBER_OF_NODES) ? nodeCount[c] : 1;
                }
            }
        }

        if (m_total > 0.0f) 
        {
            float invM = 1.0f / m_total;
            x[parent] = tx * invM;
            y[parent] = ty * invM;
            z[parent] = tz * invM;
            nodeCount[parent] = count;
        }

        // Move up to the next level
        curr = parent;
        parent = parentArray[curr];

        // Ensure memory is visible to the next atomic check
        mem_fence(CLK_GLOBAL_MEM_FENCE);
    }
}