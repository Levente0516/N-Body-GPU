
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable

#define NUMBER_OF_CELLS 8

__kernel void writePositionsInterleaved(
    __global float* x, 
    __global float* y, 
    __global float* z,
    __global float* out)
{
    const int i = get_global_id(0);

    if (i >= NUM_BODIES) 
    {    
        return;
    }

    out[i * 3 + 0] = x[i];
    out[i * 3 + 1] = y[i];
    out[i * 3 + 2] = z[i];
}