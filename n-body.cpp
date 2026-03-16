#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "variables.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main()
{
    srand(time(NULL));

#pragma region PLATFORM_AND_DEVICE

    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

#pragma endregion

#pragma region CONTEXT

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

#pragma endregion

#pragma region KERNEL__AND_HEADER_FOR_KERNEL

    FILE *fConfig = fopen("variables.h", "rb");
    fseek(fConfig, 0, SEEK_END);
    size_t configLen = ftell(fConfig);
    rewind(fConfig);
    char *configSrc = (char *)malloc(configLen + 1);
    memset(configSrc, 0, configLen + 1);
    fread(configSrc, 1, configLen, fConfig);
    configSrc[configLen] = '\0';
    fclose(fConfig);

    FILE *fKernel = fopen("kernels.cl", "rb");
    fseek(fKernel, 0, SEEK_END);
    size_t kernelLen = ftell(fKernel);
    rewind(fKernel);
    char *kernelSrc = (char *)malloc(kernelLen + 1);
    memset(kernelSrc, 0, kernelLen + 1);
    fread(kernelSrc, 1, kernelLen, fKernel);
    kernelSrc[kernelLen] = '\0';
    fclose(fKernel);

    const char *sources[2] = {configSrc, kernelSrc};
    size_t lengths[2] = {configLen, kernelLen};

    cl_program program = clCreateProgramWithSource(context, 2, sources, lengths, NULL);
    free(kernelSrc);
    free(configSrc);

    cl_int err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t logLen;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logLen);
        char *log = (char *)malloc(logLen);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logLen, log, NULL);
        printf("Build error:\n%s\n", log);
        free(log);
        return 1;
    }

    cl_kernel resetBBoxKernel = clCreateKernel(program, "resetBBoxKernel", NULL);
    cl_kernel boundingBoxKernel = clCreateKernel(program, "boundingBoxKernel", NULL);
    cl_kernel initTreeKernel = clCreateKernel(program, "initTreeKernel", NULL);
    cl_kernel insertKernel = clCreateKernel(program, "insertBodiesKernel", NULL);
    cl_kernel comKernel = clCreateKernel(program, "computeCOMKernel", NULL);
    cl_kernel forceKernel = clCreateKernel(program, "forceKernel", NULL);
    cl_kernel integKernel = clCreateKernel(program, "integrationKernel", NULL);

#pragma endregion

#pragma region GPU_BUFFERS

#pragma region BODY_BUFFERS

    cl_mem buf_x = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, NULL);
    cl_mem buf_y = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, NULL);
    cl_mem buf_z = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, NULL);

    cl_mem buf_vx = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, NULL);
    cl_mem buf_vy = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, NULL);
    cl_mem buf_vz = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, NULL);

    cl_mem buf_fx = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, NULL);
    cl_mem buf_fy = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, NULL);
    cl_mem buf_fz = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, NULL);

    cl_mem buf_mass = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, NULL);

#pragma endregion

#pragma region OCTREE

    cl_mem buf_child = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_NODE * 8 * sizeof(int), NULL, NULL);

    cl_mem buf_nodeX = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_NODE * sizeof(float), NULL, NULL);
    cl_mem buf_nodeY = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_NODE * sizeof(float), NULL, NULL);
    cl_mem buf_nodeZ = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_NODE * sizeof(float), NULL, NULL);

    cl_mem buf_nodeMass = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_NODE * sizeof(float), NULL, NULL);

    cl_mem buf_nodeCount = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_NODE * sizeof(int), NULL, NULL);

    cl_mem buf_nodeSize = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_NODE * sizeof(float), NULL, NULL);

    cl_mem buf_nextNode = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

#pragma endregion

    // Bounding box [minX, maxX, minY, maxY, minZ, maxZ]
    cl_mem buf_bbox = clCreateBuffer(context, CL_MEM_READ_WRITE, 6 * sizeof(float), NULL, NULL);

    // Insertion retry flag
    cl_mem buf_flag = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

#pragma endregion

#pragma region INIT

    // Initialise bodies on CPU
    float h_x[N], h_y[N], h_z[N];
    float h_vx[N], h_vy[N], h_vz[N];
    float h_fx[N], h_fy[N], h_fz[N];
    float h_mass[N];

    for (int i = 0; i < N; i++)
    {
        h_x[i] = (float)(rand() % 10000);
        h_y[i] = (float)(rand() % 10000);
        h_z[i] = (float)(rand() % 10000);

        h_vx[i] = 0.0f;
        h_vy[i] = 0.0f;
        h_vz[i] = 0.0f;

        h_fx[i] = 0.0f;
        h_fy[i] = 0.0f;
        h_fz[i] = 0.0f;

        h_mass[i] = (float)((rand() % 4000) + 10000);
    }

    // Upload to GPU
    clEnqueueWriteBuffer(queue, buf_x, CL_TRUE, 0, N * sizeof(float), h_x, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_y, CL_TRUE, 0, N * sizeof(float), h_y, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_z, CL_TRUE, 0, N * sizeof(float), h_z, 0, NULL, NULL);

    clEnqueueWriteBuffer(queue, buf_vx, CL_TRUE, 0, N * sizeof(float), h_vx, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_vy, CL_TRUE, 0, N * sizeof(float), h_vy, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_vz, CL_TRUE, 0, N * sizeof(float), h_vz, 0, NULL, NULL);

    clEnqueueWriteBuffer(queue, buf_fx, CL_TRUE, 0, N * sizeof(float), h_fx, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_fy, CL_TRUE, 0, N * sizeof(float), h_fy, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_fz, CL_TRUE, 0, N * sizeof(float), h_fz, 0, NULL, NULL);

    clEnqueueWriteBuffer(queue, buf_mass, CL_TRUE, 0, N * sizeof(float), h_mass, 0, NULL, NULL);

#pragma endregion

#pragma region LOOP

    size_t globalSize = N;
    size_t localSize = THREADS;
    size_t globalSizeTree = MAX_NODE;
    size_t one = 1;

    for (int step = 0; step < 100; step++)
    {
        // 1. Reset bounding box
        // clSetKernelArg + clEnqueueNDRangeKernel for resetBBoxKernel
        clSetKernelArg(resetBBoxKernel, 0, sizeof(cl_mem), &buf_bbox);
        clEnqueueNDRangeKernel(queue, resetBBoxKernel, 1, NULL, &one, &one, 0, NULL, NULL);
        clFinish(queue);

        // 2. Compute bounding box
        // clSetKernelArg + clEnqueueNDRangeKernel for boundingBoxKernel
        clSetKernelArg(boundingBoxKernel, 0, sizeof(cl_mem), &buf_bbox);
        clSetKernelArg(boundingBoxKernel, 1, sizeof(cl_mem), &buf_x);
        clSetKernelArg(boundingBoxKernel, 2, sizeof(cl_mem), &buf_y);
        clSetKernelArg(boundingBoxKernel, 3, sizeof(cl_mem), &buf_z);
        clEnqueueNDRangeKernel(queue, boundingBoxKernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
        clFinish(queue);

        if (step == 0)
        {
            float h_bbox[6];
            clEnqueueReadBuffer(queue, buf_bbox, CL_TRUE, 0, 6 * sizeof(float), h_bbox, 0, NULL, NULL);
            printf("Bounding box:\n");
            printf("  X: %.2f -> %.2f\n", h_bbox[0], h_bbox[1]);
            printf("  Y: %.2f -> %.2f\n", h_bbox[2], h_bbox[3]);
            printf("  Z: %.2f -> %.2f\n", h_bbox[4], h_bbox[5]);
        }

        // 3. Init tree
        // clSetKernelArg + clEnqueueNDRangeKernel for initTreeKernel

        // 4. Insert bodies (retry loop)
        // int h_flag = 1;
        // while (h_flag) { ... insertKernel ... }

        // 5. Compute COM (multiple passes)
        // for (int p = 0; p < 24; p++) { ... comKernel ... }

        // 6. Forces
        // clSetKernelArg + clEnqueueNDRangeKernel for forceKernel

        // 7. Integration
        // clSetKernelArg + clEnqueueNDRangeKernel for integKernel

        clFinish(queue);
    }

#pragma endregion

#pragma region READBACK

    float h_x_out[N], h_y_out[N], h_z_out[N];
    clEnqueueReadBuffer(queue, buf_x, CL_TRUE, 0, N * sizeof(float), h_x_out, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, buf_y, CL_TRUE, 0, N * sizeof(float), h_y_out, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, buf_z, CL_TRUE, 0, N * sizeof(float), h_z_out, 0, NULL, NULL);

    printf("\nFirst 10 final positions:\n");
    for (int i = 0; i < N; i++)
    {
        printf("  Body %d: (%.2f, %.2f, %.2f)\n", i, h_x_out[i], h_y_out[i], h_z_out[i]);
    }

#pragma endregion

#pragma region CLEANUP

    clReleaseMemObject(buf_x);
    clReleaseMemObject(buf_y);
    clReleaseMemObject(buf_z);
    clReleaseMemObject(buf_vx);
    clReleaseMemObject(buf_vy);
    clReleaseMemObject(buf_vz);
    clReleaseMemObject(buf_fx);
    clReleaseMemObject(buf_fy);
    clReleaseMemObject(buf_fz);
    clReleaseMemObject(buf_mass);
    clReleaseMemObject(buf_child);
    clReleaseMemObject(buf_nodeX);
    clReleaseMemObject(buf_nodeY);
    clReleaseMemObject(buf_nodeZ);
    clReleaseMemObject(buf_nodeMass);
    clReleaseMemObject(buf_nodeCount);
    clReleaseMemObject(buf_nodeSize);
    clReleaseMemObject(buf_nextNode);
    clReleaseMemObject(buf_bbox);
    clReleaseMemObject(buf_flag);

    clReleaseKernel(resetBBoxKernel);
    clReleaseKernel(boundingBoxKernel);
    clReleaseKernel(initTreeKernel);
    clReleaseKernel(insertKernel);
    clReleaseKernel(comKernel);
    clReleaseKernel(forceKernel);
    clReleaseKernel(integKernel);

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

#pragma endregion

    return 0;
}