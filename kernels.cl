#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable


void atomicMinFloat(__global float* addr, float val)
{
    volatile __global int* addr_i = (volatile __global int*)addr;
    int old, newVal;
    do {
        old = *addr_i;
        float oldFloat = as_float(old);
        newVal = as_int(fmin(oldFloat, val));
    } while (atomic_cmpxchg(addr_i, old, newVal) != old);
}

void atomicMaxFloat(__global float* addr, float val)
{
    volatile __global int* addr_i = (volatile __global int*)addr;
    int old, newVal;
    do {
        old = *addr_i;
        float oldFloat = as_float(old);
        newVal = as_int(fmax(oldFloat, val));
    } while (atomic_cmpxchg(addr_i, old, newVal) != old);
}

__kernel void boundingBoxKernel(
    __global float* bbox, __global float* buf_x,
    __global float* buf_y, __global float* buf_z)
{
    __local float sMinX[THREADS];
    __local float sMaxX[THREADS];
    __local float sMinY[THREADS];
    __local float sMaxY[THREADS];
    __local float sMinZ[THREADS];
    __local float sMaxZ[THREADS];

    int tid = get_local_id(0);
    int gid = get_global_id(0);

    sMinX[tid] = buf_x[gid];
    sMaxX[tid] = buf_x[gid];
    sMinY[tid] = buf_y[gid];
    sMaxY[tid] = buf_y[gid];
    sMinZ[tid] = buf_z[gid];
    sMaxZ[tid] = buf_z[gid];
    
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

/*
__kernel void initTreeKernel(
    __global int*   child,
    __global float* nodeX,    __global float* nodeY,    __global float* nodeZ,
    __global float* nodeMass, __global int*   nodeCount, __global float* nodeSize,
    __global int*   nextNode,
    __global float* bbox)
{
    int gid = get_global_id(0);

    if (gid < MAX_NODE)
    {
        for (int i = 0; i < 8; i++)
        {
            child[gid * 8 + i] = EMPTY;
        }

        nodeX[gid] = nodeY[gid] = nodeZ[gid] = 0.0f;
        nodeMass[gid]  = 0.0f;
        nodeCount[gid] = 0;
        nodeSize[gid]  = 0.0f;
    }

    if (gid == 0)
    {
        float minX = bbox[0], maxX = bbox[1];
        float minY = bbox[2], maxY = bbox[3];
        float minZ = bbox[4], maxZ = bbox[5];

        nodeX[0] = (minX + maxX) * 0.5f;
        nodeY[0] = (minY + maxY) * 0.5f;
        nodeZ[0] = (minZ + maxZ) * 0.5f;

        float span = fmax(fmax(maxX - minX, maxY - minY), maxZ - minZ);
        nodeSize[0] = span * 0.5f + 1.0f;

        nextNode[0] = 1;
    }

}

__kernel void insertBodiesKernel(
    __global int*   child,
    __global float* nodeX,    __global float* nodeY,    __global float* nodeZ,
    __global float* nodeMass, __global int*   nodeCount, __global float* nodeSize,
    __global int*   nextNode,
    __global float* x, __global float* y, __global float* z,
    __global float* mass)
{
    int bodyIdx = get_global_id(0);
    if (bodyIdx >= NUM_BODIES) return;

    float bx = x[bodyIdx];
    float by = y[bodyIdx];
    float bz = z[bodyIdx];

    bool inserted = false;

    while (!inserted)
    {
        int  node    = 0;
        bool restart = false;

        while (!restart)
        {
            float cx   = nodeX[node];
            float cy   = nodeY[node];
            float cz   = nodeZ[node];
            float size = nodeSize[node];

            int oct = 0;
            if (bx >= cx) oct |= 1;
            if (by >= cy) oct |= 2;
            if (bz >= cz) oct |= 4;

            volatile __global int* slot = (volatile __global int*)&child[node * 8 + oct];
            int c = *slot;

            if (c == LOCKED) 
            { 
                restart = true; 
                break;
            }

            if (c == EMPTY)
            {
                if (atomic_cmpxchg(slot, -1, bodyIdx) == EMPTY)
                    inserted = true;
                continue;
            }
            if (c >= NUM_BODIES)
            {
                node = c - NUM_BODIES;
                continue;
            }

            if (atomic_cmpxchg(slot, c, -2) != c)
            {
                continue;
            }

            int newNode = atomic_add((volatile __global int*)nextNode, 1);
            if (newNode >= MAX_NODE)
            {
                atomic_dec((volatile __global int*)nextNode);
                *slot = c;
                inserted = true;
                break;
            }

            float halfSize = size * 0.5f;
            float newCX = cx + ((oct & 1) ? halfSize : -halfSize);
            float newCY = cy + ((oct & 2) ? halfSize : -halfSize);
            float newCZ = cz + ((oct & 4) ? halfSize : -halfSize);

            nodeX[newNode]     = newCX;
            nodeY[newNode]     = newCY;
            nodeZ[newNode]     = newCZ;
            nodeSize[newNode]  = halfSize;
            nodeMass[newNode]  = -1.0f;
            nodeCount[newNode] = 0;

            for (int i = 0; i < 8; i++)
            {
                child[newNode * 8 + i] = EMPTY;
            }

            float ex = x[c], ey = y[c], ez = z[c];
            int existOct = 0;
            if (ex >= newCX) existOct |= 1;
            if (ey >= newCY) existOct |= 2;
            if (ez >= newCZ) existOct |= 4;
            child[newNode * 8 + existOct] = c;

            mem_fence(CLK_GLOBAL_MEM_FENCE);

            *slot = newNode + NUM_BODIES;

            node = newNode;
        }
    }
}
__kernel void computeCOMKernel(
    __global int*   child,
    __global float* nodeX,    __global float* nodeY,    __global float* nodeZ,
    __global float* nodeMass, __global int*   nodeCount,
    __global float* x,        __global float* y,        __global float* z,
    __global float* mass,
    __global int*   nextNode,
    int             startNode,
    int             endNode)
{
    int node = get_global_id(0);
    if (node >= endNode) return;

    float totalMass = 0.0f;
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    int   count = 0;
    bool  ready = false;

    while (!ready)
    {
        totalMass = 0.0f;
        cx = 0.0f; cy = 0.0f; cz = 0.0f;
        count = 0;
        ready = true;

        for (int i = 0; i < 8; i++)
        {
            int c = child[node * 8 + i];
            if (c == EMPTY) continue;

            if (c < NUM_BODIES)
            {
                float m = mass[c];
                totalMass += m;
                cx += x[c] * m;
                cy += y[c] * m;
                cz += z[c] * m;
                count++;
            }
            else
            {
                int cn = c - NUM_BODIES;

                float m = ((volatile __global float*)nodeMass)[cn];

                if (m < 0.0f)
                {
                    ready = false;
                    break;  // break inner loop, retry outer while
                }

                totalMass += m;
                cx += nodeX[cn] * m;
                cy += nodeY[cn] * m;
                cz += nodeZ[cn] * m;
                count += nodeCount[cn];
            }
        }
    }

    if (totalMass > 0.0f)
    {
        nodeX[node]    = cx / totalMass;
        nodeY[node]    = cy / totalMass;
        nodeZ[node]    = cz / totalMass;
        nodeCount[node]= count;
    }

    mem_fence(CLK_GLOBAL_MEM_FENCE);
    nodeMass[node] = (totalMass > 0.0f) ? totalMass : 0.0f;
}

__kernel void forceAndIntegrationKernel(
    __global int*   child,
    __global float* nodeX,    __global float* nodeY,    __global float* nodeZ,
    __global float* nodeMass, __global float* nodeSize,
    __global int*   nextNode,
    __global float* x,  __global float* y,  __global float* z,
    __global float* vx, __global float* vy, __global float* vz,
    __global float* mass)
{
       
    int bodyIdx = get_global_id(0);
    if (bodyIdx >= NUM_BODIES) return;

    float bx = x[bodyIdx];
    float by = y[bodyIdx];
    float bz = z[bodyIdx];
    float bm = mass[bodyIdx];

    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    int stack[1024];
    int stackTop = 0;
    stack[stackTop++] = 0; 

    while (stackTop > 0)
    {
        int node = stack[--stackTop];

        for (int i = 0; i < 8; i++)
        {
            int c = child[node * 8 + i];
            if (c == EMPTY) continue;

            float dx, dy, dz, dist2, dist, force;

            if (c < NUM_BODIES)
            {
                if (c == bodyIdx) continue;
                dx = x[c] - bx;
                dy = y[c] - by;
                dz = z[c] - bz;
                dist2 = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
                dist  = native_sqrt(dist2);
                force = G * bm * mass[c] / dist2;
                ax += force * (dx / dist);
                ay += force * (dy / dist);
                az += force * (dz / dist);
            }
            else
            {
                int childNode = c - NUM_BODIES;
                dx = nodeX[childNode] - bx;
                dy = nodeY[childNode] - by;
                dz = nodeZ[childNode] - bz;
                dist2 = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
                dist  = native_sqrt(dist2);

                if ((nodeSize[childNode] / dist) < THETA)
                {
                    force = G * bm * nodeMass[childNode] / dist2;
                    ax += force * (dx / dist);
                    ay += force * (dy / dist);
                    az += force * (dz / dist);
                }
                else if (stackTop < 511)
                {
                    stack[stackTop++] = childNode;
                }
            }
        }
    }

    float inv_m = 1.0f / bm;
    vx[bodyIdx] += ax * inv_m * DT;
    vy[bodyIdx] += ay * inv_m * DT;
    vz[bodyIdx] += az * inv_m * DT;
    x[bodyIdx]  += vx[bodyIdx] * DT;
    y[bodyIdx]  += vy[bodyIdx] * DT;
    z[bodyIdx]  += vz[bodyIdx] * DT;
}
*/

__kernel void forceAndIntegrationKernel(
    __global const float* nodeCOMX,
    __global const float* nodeCOMY,
    __global const float* nodeMass,
    __global const float* nodeHalfSize,
    __global const int*   nodeChildren, 
    __global const int*   nodeBodyIdx,   
    __global       float* x,
    __global       float* y,
    __global       float* vx,
    __global       float* vy,
    __global const float* mass,
    int numNodes)
{
    int i = get_global_id(0);
    if (i >= NUM_BODIES) return;

    float bx = x[i];
    float by = y[i];
    float ax = 0.0f, ay = 0.0f;

    int stack[256];
    int top = 0;
    if (numNodes > 0) stack[top++] = 0;

    while (top > 0)
    {
        int node = stack[--top];
        int bodyJ = nodeBodyIdx[node];

        if (bodyJ >= 0)
        {
            if (bodyJ == i) continue;

            float dx    = x[bodyJ] - bx;
            float dy    = y[bodyJ] - by;
            float dist2 = dx*dx + dy*dy + SOFTENING*SOFTENING;
            float invD  = native_rsqrt(dist2);
            float invD3 = invD * invD * invD;
            ax += G * nodeMass[node] * dx * invD3;
            ay += G * nodeMass[node] * dy * invD3;
        }
        else
        {
            float dx    = nodeCOMX[node] - bx;
            float dy    = nodeCOMY[node] - by;
            float dist2 = dx*dx + dy*dy + SOFTENING*SOFTENING;
            float dist  = native_sqrt(dist2);
            float s     = nodeHalfSize[node] * 2.0f;

            if (s < THETA * dist)
            {
                float invD  = native_rsqrt(dist2);
                float invD3 = invD * invD * invD;
                ax += G * nodeMass[node] * dx * invD3;
                ay += G * nodeMass[node] * dy * invD3;
            }
            else
            {
                for (int c = 0; c < 4; c++)
                {
                    int child = nodeChildren[node * 4 + c];
                    if (child >= 0 && top < 255)
                        stack[top++] = child;
                }
            }
        }
    }

    vx[i] += ax * DT;
    vy[i] += ay * DT;
    x[i]  += vx[i] * DT;
    y[i]  += vy[i] * DT;
}

__kernel void writePositionsInterleaved(
    __global float* x, __global float* y,
    __global float* out)
{
    int i = get_global_id(0);
    if (i >= NUM_BODIES) return;

    out[i * 3 + 0] = x[i];
    out[i * 3 + 1] = y[i];
    out[i * 3 + 2] = 0.0f;
}