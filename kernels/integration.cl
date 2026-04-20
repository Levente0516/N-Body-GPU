
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable

#define NUMBER_OF_CELLS 8

__attribute__ ((reqd_work_group_size(THREADS, 1, 1)))
__kernel void integrateKernel(
    __global float* x,   
    __global float* y,   
    __global float* z,
    __global float* vx,  
    __global float* vy, 
    __global float* vz,
    __global float* accX, 
    __global float* accY, 
    __global float* accZ,
    __global float* mass,
    float DT)
{
    int stepSize = get_local_size(0) * get_num_groups(0);

    for (int i = get_global_id(0); i < NUM_BODIES; i += stepSize)
    {
        //if (i == 0) continue;  // black hole stays fixed

        vx[i] += accX[i] * DT;
        vy[i] += accY[i] * DT;
        vz[i] += accZ[i] * DT;

        x[i] += vx[i] * DT;
        y[i] += vy[i] * DT;
        z[i] += vz[i] * DT;

    }
}

