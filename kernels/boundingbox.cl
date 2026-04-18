
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable

#define NUMBER_OF_CELLS 8
#define EMPTY -1
#define LOCKED -2

__attribute__((reqd_work_group_size(THREADS, 1, 1)))
__kernel void boundingBoxKernel(
    __global int* step, 
    __global volatile float* pos_x,
    __global volatile float* pos_y, 
    __global volatile float* pos_z,
    __global int* blockCount, 
    __global int* bottom,
    __global float* mass, 
    __global int* numNodes,
    __global float* MinX,
    __global float* MinY,
    __global float* MinZ,
    __global float* MaxX,
    __global float* MaxY,
    __global float* MaxZ,
    __global int* child,
    __global int* start,
    __global float* nodeSize,
    __global int* maxDepth)
{
    __local volatile float sMinX[THREADS];
    __local volatile float sMaxX[THREADS];
    __local volatile float sMinY[THREADS];
    __local volatile float sMaxY[THREADS];
    __local volatile float sMinZ[THREADS];
    __local volatile float sMaxZ[THREADS];

    int localId = get_local_id(0);
    int groupId = get_group_id(0);
    int localSize = get_local_size(0);
    int numGroup = get_num_groups(0);
    
    if (localId == 0)
    {
        sMinX[0] = pos_x[0];
        sMinY[0] = pos_y[0];
        sMinZ[0] = pos_z[0];
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    sMinX[localId] = sMaxX[localId] = sMinX[0];
    sMinY[localId] = sMaxY[localId] = sMinY[0];
    sMinZ[localId] = sMaxZ[localId] = sMinZ[0];
    
    for (int i = localId + groupId * localSize; i < NUM_BODIES; i += (localSize * numGroup))
    {
        sMinX[localId] = fmin(sMinX[localId], pos_x[i]);
        sMaxX[localId] = fmax(sMaxX[localId], pos_x[i]);

        sMinY[localId] = fmin(sMinY[localId], pos_y[i]);
        sMaxY[localId] = fmax(sMaxY[localId], pos_y[i]);

        sMinZ[localId] = fmin(sMinZ[localId], pos_z[i]);
        sMaxZ[localId] = fmax(sMaxZ[localId], pos_z[i]);
    }

    for (int i = localSize / 2; i > 0; i >>= 1)
    {
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

        if (localId < i)
        {
            sMinX[localId] = fmin(sMinX[localId], sMinX[localId + i]);
            sMaxX[localId] = fmax(sMaxX[localId], sMaxX[localId + i]);

            sMinY[localId] = fmin(sMinY[localId], sMinY[localId + i]);
            sMaxY[localId] = fmax(sMaxY[localId], sMaxY[localId + i]);

            sMinZ[localId] = fmin(sMinZ[localId], sMinZ[localId + i]);
            sMaxZ[localId] = fmax(sMaxZ[localId], sMaxZ[localId + i]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0)
    {
        MinX[groupId] = sMinX[0];
        MinY[groupId] = sMinY[0];
        MinZ[groupId] = sMinZ[0];
        MaxX[groupId] = sMaxX[0];
        MaxY[groupId] = sMaxY[0];
        MaxZ[groupId] = sMaxZ[0];

        mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

        if (numGroup - 1 == atomic_inc(blockCount))
        {
            for (int i = 0; i < numGroup; i++)
            {
                sMinX[0] = fmin(sMinX[0], MinX[i]);
                sMaxX[0] = fmax(sMaxX[0], MaxX[i]);

                sMinY[0] = fmin(sMinY[0], MinY[i]);
                sMaxY[0] = fmax(sMaxY[0], MaxY[i]);

                sMinZ[0] = fmin(sMinZ[0], MinZ[i]);
                sMaxZ[0] = fmax(sMaxZ[0], MaxZ[i]);
            }

            const float rootX = 0.5f * (sMinX[0] + sMaxX[0]); 
            const float rootY = 0.5f * (sMinY[0] + sMaxY[0]);
            const float rootZ = 0.5f * (sMinZ[0] + sMaxZ[0]);
            const float radius = 0.5f * fmax(fmax(sMaxX[0]-sMinX[0], sMaxY[0]-sMinY[0]), sMaxZ[0]-sMinZ[0]);

            *bottom = NUMBER_OF_NODES;
            *blockCount = 0;
            *maxDepth = 1;

            pos_x[NUMBER_OF_NODES] = rootX;
            pos_y[NUMBER_OF_NODES] = rootY;
            pos_z[NUMBER_OF_NODES] = rootZ;
            mass[NUMBER_OF_NODES] = -1.0f;
            start[NUMBER_OF_NODES] = 0;
            nodeSize[NUMBER_OF_NODES] = radius;

#pragma unroll NUMBER_OF_CELLS
            for (int i = 0; i < NUMBER_OF_CELLS; i++)
            {
                child[NUMBER_OF_NODES * NUMBER_OF_CELLS + i] = EMPTY;
            }

            (*step)++;
        }
    }
}