
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable

#define NUMBER_OF_CELLS 8

__kernel void forceKernel(
    __global float* x,   
    __global float* y,   
    __global float* z,
    __global float* mass,
    __global int*   child,
    __global volatile float* nodeSize,
    __global int*   sorted,
    __global float* accX, 
    __global float* accY, 
    __global float* accZ,
    __global int*   numNodes)
{
    const int bodyIdx = get_global_id(0);
    if (bodyIdx >= NUM_BODIES)
    {
        return;
    }

    const int si = sorted[bodyIdx];
    const float bx = x[si]; 
    const float by = y[si];
    const float bz = z[si];
    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    float f;
    int stk[64];
    int top = 0;
    stk[top++] = NUMBER_OF_NODES;

    while (top > 0) 
    {
        const int node = stk[--top];

        for (int i = 0; i < NUMBER_OF_CELLS; i++)
        {
            const int c = child[node * NUMBER_OF_CELLS + i];

            if (c < 0)
            {
                break;
            }

            const float dx = x[c] - bx;
            const float dy = y[c] - by;
            const float dz = z[c] - bz;
            const float dist2 = dx*dx + dy*dy + dz*dz + SOFTENING * SOFTENING;
            const float invD  = native_rsqrt(dist2);

            if (c < NUM_BODIES) 
            {
                if (c == si)
                {
                    continue;
                } 

                f = G * mass[c] * invD * invD * invD;



                ax += dx * f; 
                ay += dy * f; 
                az += dz * f;
            } 
            else 
            {
                if (nodeSize[c] * invD < THETA) 
                {
                    f = G * mass[c] * invD * invD * invD;

                    
                    ax += dx * f; 
                    ay += dy * f; 
                    az += dz * f;
                } 
                else if (top < 63) 
                {
                    stk[top++] = c;
                }
            }
        }
    }



    accX[si] = ax;
    accY[si] = ay;
    accZ[si] = az;
}