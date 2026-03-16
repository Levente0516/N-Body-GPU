__kernel void resetBBoxKernel(__global float* bbox)
{
    bbox[0] =  1e30f;   // minX
    bbox[1] = -1e30f;   // maxX
    bbox[2] =  1e30f;   // minY
    bbox[3] = -1e30f;   // maxY
    bbox[4] =  1e30f;   // minZ
    bbox[5] = -1e30f;   // maxZ
}