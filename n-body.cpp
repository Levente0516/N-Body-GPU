#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include "variables.hpp"

#define CL_HPP_ENABLE_EXCEPTIONS  

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#pragma region FILE_READ_HELPER

std::string loadFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Cannot open file: " << path << std::endl;
        exit(1);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

#pragma endregion

int main()
{
    srand(time(NULL));

#pragma region PLATFORM_AND_DEVICE

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    // Get GPU device
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

#pragma endregion

#pragma region CONTEXT

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

#pragma endregion

#pragma region KERNEL__AND_HEADER_FOR_KERNEL

    std::string configSrc = loadFile("variables.hpp");
    std::string kernelSrc = loadFile("kernels.cl");

    cl::Program::Sources sources;
    sources.push_back(configSrc);
    sources.push_back(kernelSrc);

    cl::Program program(context, sources);

    try
    {
        program.build({ device });
    }
    catch (const cl::Error&)
    {
        std::cerr << "Build error:\n"
                    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
                    << std::endl;
        return 1;
    }

    cl::Kernel resetBBoxKernel(program, "resetBBoxKernel");
    cl::Kernel boundingBoxKernel(program, "boundingBoxKernel");
    /*
    cl::Kernel initTreeKernel(program, "initTreeKernel");
    cl::Kernel insertKernel(program, "insertBodiesKernel");
    cl::Kernel comKernel(program, "computeCOMKernel");
    cl::Kernel forceKernel(program, "forceKernel");
    cl::Kernel integKernel(program, "integrationKernel");
    */

#pragma endregion

#pragma region GPU_BUFFERS

#pragma region BODY_BUFFERS

    cl::Buffer buf_x(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_y(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_z(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_vx(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_vy(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_vz(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_fx(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_fy(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_fz(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_mass(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));

#pragma endregion

#pragma region OCTREE

    cl::Buffer buf_child    (context, CL_MEM_READ_WRITE, MAX_NODE*8*sizeof(int)  );
    cl::Buffer buf_nodeX    (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float)  );
    cl::Buffer buf_nodeY    (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float)  );
    cl::Buffer buf_nodeZ    (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float)  );
    cl::Buffer buf_nodeMass (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float)  );
    cl::Buffer buf_nodeCount(context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(int)    );
    cl::Buffer buf_nodeSize (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float)  );
    cl::Buffer buf_nextNode (context, CL_MEM_READ_WRITE, sizeof(int)             );

#pragma endregion

    cl::Buffer buf_bbox(context, CL_MEM_READ_WRITE, 6*sizeof(float));
    cl::Buffer buf_flag(context, CL_MEM_READ_WRITE, sizeof(int)    );

#pragma endregion

#pragma region INIT

    // Initialise bodies on CPU
    std::vector<float> h_x(NUM_BODIES), h_y(NUM_BODIES), h_z(NUM_BODIES);
    std::vector<float> h_vx(NUM_BODIES), h_vy(NUM_BODIES), h_vz(NUM_BODIES);
    std::vector<float> h_fx(NUM_BODIES), h_fy(NUM_BODIES), h_fz(NUM_BODIES);
    std::vector<float> h_mass(NUM_BODIES);

    for (int i = 0; i < NUM_BODIES; i++)
    {
        h_x[i] = (float)((rand() % 10000) - 5000);
        h_y[i] = (float)((rand() % 10000) - 5000);
        h_z[i] = (float)((rand() % 10000) - 5000);

        h_vx[i] = 0.0f;
        h_vy[i] = 0.0f;
        h_vz[i] = 0.0f;

        h_fx[i] = 0.0f;
        h_fy[i] = 0.0f;
        h_fz[i] = 0.0f;

        h_mass[i] = (float)((rand() % 4000) + 10000);
    }

    // Upload to GPU
    queue.enqueueWriteBuffer(buf_x,    CL_TRUE, 0, NUM_BODIES*sizeof(float), h_x.data()   );
    queue.enqueueWriteBuffer(buf_y,    CL_TRUE, 0, NUM_BODIES*sizeof(float), h_y.data()   );
    queue.enqueueWriteBuffer(buf_z,    CL_TRUE, 0, NUM_BODIES*sizeof(float), h_z.data()   );
    queue.enqueueWriteBuffer(buf_vx,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vx.data()  );
    queue.enqueueWriteBuffer(buf_vy,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vy.data()  );
    queue.enqueueWriteBuffer(buf_vz,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vz.data()  );
    queue.enqueueWriteBuffer(buf_fx,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_fx.data()  );
    queue.enqueueWriteBuffer(buf_fy,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_fy.data()  );
    queue.enqueueWriteBuffer(buf_fz,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_fz.data()  );
    queue.enqueueWriteBuffer(buf_mass, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_mass.data());

#pragma endregion

#pragma region LOOP

    cl::NDRange global(NUM_BODIES);
    cl::NDRange local(THREADS);
    cl::NDRange globalTree(MAX_NODE);
    cl::NDRange one(1);

     for (int step = 0; step < 10; step++)
    {
        // 1. Reset bounding box
        resetBBoxKernel.setArg(0, buf_bbox);
        queue.enqueueNDRangeKernel(resetBBoxKernel, cl::NullRange, one, one);
        queue.finish();

        // 2. Compute bounding box
        boundingBoxKernel.setArg(0, buf_bbox);
        boundingBoxKernel.setArg(1, buf_x);
        boundingBoxKernel.setArg(2, buf_y);
        boundingBoxKernel.setArg(3, buf_z);
        queue.enqueueNDRangeKernel(boundingBoxKernel, cl::NullRange, global, local);
        queue.finish();

        // 3. Init tree
        // TODO

        // 4. Insert bodies
        // TODO

        // 5. Compute COM
        // TODO

        // 6. Forces
        // TODO

        // 7. Integration
        // TODO

        queue.finish();
    }

    std::vector<float> h_bbox(6);
    queue.enqueueReadBuffer(buf_bbox, CL_TRUE, 0, 6*sizeof(float), h_bbox.data());
    std::cout << "Bounding box:" << std::endl;
    std::cout << "  X: " << h_bbox[0] << " -> " << h_bbox[1] << std::endl;
    std::cout << "  Y: " << h_bbox[2] << " -> " << h_bbox[3] << std::endl;
    std::cout << "  Z: " << h_bbox[4] << " -> " << h_bbox[5] << std::endl;

#pragma endregion

#pragma region READBACK

    std::vector<float> h_x_out(NUM_BODIES), h_y_out(NUM_BODIES), h_z_out(NUM_BODIES);
    queue.enqueueReadBuffer(buf_x, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_x_out.data());
    queue.enqueueReadBuffer(buf_y, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_y_out.data());
    queue.enqueueReadBuffer(buf_z, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_z_out.data());

    std::cout << "\nFirst 10 final positions:" << std::endl;
    for (int i = 0; i < NUM_BODIES; i++)
    {
        std::cout << "  Body " << i << ": ("
                    << h_x_out[i] << ", "
                    << h_y_out[i] << ", "
                    << h_z_out[i] << ")" << std::endl;
    }

#pragma endregion

    return 0;
}