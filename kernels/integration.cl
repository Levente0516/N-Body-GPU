
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
    __global float* accZ)
{
    int stepSize = get_local_size(0) * get_num_groups(0);

    for (int i = get_global_id(0); i < NUM_BODIES; i += stepSize)
    {
        float delta_vx = accX[i] * DT * 0.5f;
        float delta_vy = accY[i] * DT * 0.5f;
        float delta_vz = accZ[i] * DT * 0.5f;

        float velX = vx[i] + delta_vx;
        float velY = vy[i] + delta_vy;
        float velZ = vz[i] + delta_vz;

        x[i]  += velX * DT;
        y[i]  += velY * DT;
        z[i]  += velZ * DT;

        vx[i] += velX + delta_vx;
        vy[i] += velY + delta_vy;
        vz[i] += velZ + delta_vz;
    }
}

