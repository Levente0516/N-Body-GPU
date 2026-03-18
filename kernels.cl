void atomicMinFloat(__global float* addr, float val)
{
    __global int* addr_i = (__global int*)addr;
    int old, newVal;
    do {
        old = *addr_i;
        float oldFloat = as_float(old);
        newVal = as_int(fmin(oldFloat, val));
    } while (atomic_cmpxchg(addr_i, old, newVal) != old);
}

void atomicMaxFloat(__global float* addr, float val)
{
    __global int* addr_i = (__global int*)addr;
    int old, newVal;
    do {
        old = *addr_i;
        float oldFloat = as_float(old);
        newVal = as_int(fmax(oldFloat, val));
    } while (atomic_cmpxchg(addr_i, old, newVal) != old);
}

__kernel void resetBBoxKernel(__global float* bbox)
{
    bbox[0] =  1e30f;   // minX
    bbox[1] = -1e30f;   // maxX
    bbox[2] =  1e30f;   // minY
    bbox[3] = -1e30f;   // maxY
    bbox[4] =  1e30f;   // minZ
    bbox[5] = -1e30f;   // maxZ
}

__kernel void boundingBoxKernel(__global float* bbox, __global float* buf_x, __global float* buf_y, __global float* buf_z)
{
    __local float sMinX[THREADS];
    __local float sMaxX[THREADS];
    __local float sMinY[THREADS];
    __local float sMaxY[THREADS];
    __local float sMinZ[THREADS];
    __local float sMaxZ[THREADS];

    int tid = get_local_id(0);  
    int gid = get_global_id(0);  

    if (gid < NUM_BODIES)
    {
        sMinX[tid] = buf_x[gid];
        sMaxX[tid] = buf_x[gid];
        sMinY[tid] = buf_y[gid];
        sMaxY[tid] = buf_y[gid];
        sMinZ[tid] = buf_z[gid];
        sMaxZ[tid] = buf_z[gid];
    }
    else
    {
        sMinX[tid] =  1e30f;
        sMaxX[tid] = -1e30f;
        sMinY[tid] =  1e30f;
        sMaxY[tid] = -1e30f;
        sMinZ[tid] =  1e30f;
        sMaxZ[tid] = -1e30f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sMinX[tid] = fmin(sMinX[tid], sMinX[tid + stride]);
            sMaxX[tid] = fmax(sMaxX[tid], sMaxX[tid + stride]);
            sMinY[tid] = fmin(sMinY[tid], sMinY[tid + stride]);
            sMaxY[tid] = fmax(sMaxY[tid], sMaxY[tid + stride]);
            sMinZ[tid] = fmin(sMinZ[tid], sMinZ[tid + stride]);
            sMaxZ[tid] = fmax(sMaxZ[tid], sMaxZ[tid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
    {
        atomicMinFloat(&bbox[0], sMinX[0]);
        atomicMaxFloat(&bbox[1], sMaxX[0]);
        atomicMinFloat(&bbox[2], sMinY[0]);
        atomicMaxFloat(&bbox[3], sMaxY[0]);
        atomicMinFloat(&bbox[4], sMinZ[0]);
        atomicMaxFloat(&bbox[5], sMaxZ[0]);
    }
}