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

__kernel void resetBBoxKernel(__global float* bbox)
{
    bbox[0] =  1e30f;
    bbox[1] = -1e30f;
    bbox[2] =  1e30f;
    bbox[3] = -1e30f;
    bbox[4] =  1e30f;
    bbox[5] = -1e30f;
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
            child[gid * 8 + i] = EMPTY;

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

        while (!inserted && !restart)
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

            if (c == LOCKED) { restart = true; break; }

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
                continue;

            int newNode = atomic_add((volatile __global int*)nextNode, 1);
            if (newNode >= MAX_NODE)
            {
                *slot    = c;
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
            nodeMass[newNode]  = 0.0f;
            nodeCount[newNode] = 0;

            for (int i = 0; i < 8; i++)
                child[newNode * 8 + i] = EMPTY;

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
    __global int*   nextNode)
{
    if (get_global_id(0) != 0) return;

    int totalNodes = nextNode[0];

    // Bottom-up: children always have higher indices than parents
    for (int node = totalNodes - 1; node >= 0; node--)
    {
        float totalMass = 0.0f;
        float cx = 0.0f, cy = 0.0f, cz = 0.0f;
        int   count = 0;

        for (int i = 0; i < 8; i++)
        {
            int c = child[node * 8 + i];

            if (c == EMPTY) continue;

            if (c < NUM_BODIES)
            {
                // Leaf — use body data directly
                float m = mass[c];
                totalMass += m;
                cx += x[c] * m;
                cy += y[c] * m;
                cz += z[c] * m;
                count++;
            }
            else
            {
                // Internal node — already computed (higher index processed first)
                int childNode = c - NUM_BODIES;
                float m = nodeMass[childNode];
                totalMass += m;
                cx += nodeX[childNode] * m;
                cy += nodeY[childNode] * m;
                cz += nodeZ[childNode] * m;
                count += nodeCount[childNode];
            }
        }

        if (totalMass > 0.0f)
        {
            nodeX[node]     = cx / totalMass;
            nodeY[node]     = cy / totalMass;
            nodeZ[node]     = cz / totalMass;
            nodeMass[node]  = totalMass;
            nodeCount[node] = count;
        }
    }
}

__kernel void forceKernel(
    __global int*   child,
    __global float* nodeX,    __global float* nodeY,    __global float* nodeZ,
    __global float* nodeMass, __global float* nodeSize,
    __global int*   nextNode,
    __global float* x,  __global float* y,  __global float* z,
    __global float* fx, __global float* fy, __global float* fz,
    __global float* mass)
{
    int bodyIdx = get_global_id(0);
    if (bodyIdx >= NUM_BODIES) return;

    float bx = x[bodyIdx];
    float by = y[bodyIdx];
    float bz = z[bodyIdx];
    float bm = mass[bodyIdx];

    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    // Private stack — one per thread, lives in registers
    int stack[64];
    int stackTop = 0;
    stack[stackTop++] = 0; // start at root

    while (stackTop > 0)
    {
        int node = stack[--stackTop];

        for (int i = 0; i < 8; i++)
        {
            int c = child[node * 8 + i];

            if (c == EMPTY) continue;

            float cx, cy, cz, cm;

            if (c < NUM_BODIES)
            {
                if (c == bodyIdx) continue;

                cx = x[c];
                cy = y[c];
                cz = z[c];
                cm = mass[c];

                float dx    = cx - bx;
                float dy    = cy - by;
                float dz    = cz - bz;
                float dist2 = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
                float dist  = sqrt(dist2);
                float force = G * bm * cm / dist2;

                ax += force * dx / dist;
                ay += force * dy / dist;
                az += force * dz / dist;
            }
            else
            {
                int childNode = c - NUM_BODIES;

                cx = nodeX[childNode];
                cy = nodeY[childNode];
                cz = nodeZ[childNode];
                cm = nodeMass[childNode];

                float dx   = cx - bx;
                float dy   = cy - by;
                float dz   = cz - bz;
                float dist = sqrt(dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING);
                float size = nodeSize[childNode];

                if ((size / dist) < THETA)
                {
                    float force = G *  mass[bodyIdx] * cm / (dist * dist);

                    ax += force * dx / dist;
                    ay += force * dy / dist;
                    az += force * dz / dist;
                }
                else
                {
                    if (stackTop < 64)
                        stack[stackTop++] = childNode;
                }
            }
        }
    }

    fx[bodyIdx] = ax;
    fy[bodyIdx] = ay;
    fz[bodyIdx] = az;
}

__kernel void integrationKernel(
    __global float* x,  __global float* y,  __global float* z,
    __global float* vx, __global float* vy, __global float* vz,
    __global float* fx, __global float* fy, __global float* fz,
    __global float* mass)
{
    int i = get_global_id(0);
    if (i >= NUM_BODIES) return;

    float inv_mass = 1.0f / mass[i];

    // Acceleration from force
    float ax = fx[i] * inv_mass;
    float ay = fy[i] * inv_mass;
    float az = fz[i] * inv_mass;

    // Update velocity
    vx[i] += ax * DT;
    vy[i] += ay * DT;
    vz[i] += az * DT;

    // Update position
    x[i] += vx[i] * DT;
    y[i] += vy[i] * DT;
    z[i] += vz[i] * DT;

    float speed = sqrt(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
    float maxSpeed = 50000.0f; // tune this based on your units

    if (speed > maxSpeed)
    {
        float scale = maxSpeed / speed;
        vx[i] *= scale;
        vy[i] *= scale;
        vz[i] *= scale;
    }
}

__kernel void writePositionsInterleaved(
    __global float* x, __global float* y, __global float* z,
    __global float* out)  // interleaved x,y,z for Vulkan
{
    int i = get_global_id(0);
    if (i >= NUM_BODIES) return;

    out[i * 3 + 0] = x[i];
    out[i * 3 + 1] = y[i];
    out[i * 3 + 2] = z[i];
}