#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
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
    cl::Kernel initTreeKernel(program, "initTreeKernel");
    cl::Kernel insertKernel(program, "insertBodiesKernel");
    cl::Kernel comKernel(program, "computeCOMKernel");
    cl::Kernel forceKernel(program, "forceKernel");
    cl::Kernel integKernel(program, "integrationKernel");

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

    std::vector<float> h_x_out(NUM_BODIES), h_y_out(NUM_BODIES), h_z_out(NUM_BODIES); // testing

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

     for (int step = 0; step < 100; step++)
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
        initTreeKernel.setArg(0, buf_child);
        initTreeKernel.setArg(1, buf_nodeX);
        initTreeKernel.setArg(2, buf_nodeY);
        initTreeKernel.setArg(3, buf_nodeZ);
        initTreeKernel.setArg(4, buf_nodeMass);
        initTreeKernel.setArg(5, buf_nodeCount);
        initTreeKernel.setArg(6, buf_nodeSize);
        initTreeKernel.setArg(7, buf_nextNode);
        initTreeKernel.setArg(8, buf_bbox);
        queue.enqueueNDRangeKernel(initTreeKernel, cl::NullRange, globalTree, local);
        queue.finish();

        // 4. Insert bodies
        insertKernel.setArg(0, buf_child);
        insertKernel.setArg(1, buf_nodeX);
        insertKernel.setArg(2, buf_nodeY);
        insertKernel.setArg(3, buf_nodeZ);
        insertKernel.setArg(4, buf_nodeMass);
        insertKernel.setArg(5, buf_nodeCount);
        insertKernel.setArg(6, buf_nodeSize);
        insertKernel.setArg(7, buf_nextNode);
        insertKernel.setArg(8, buf_x);
        insertKernel.setArg(9, buf_y);
        insertKernel.setArg(10, buf_z);
        insertKernel.setArg(11, buf_mass);
        queue.enqueueNDRangeKernel(insertKernel, cl::NullRange, global, local);
        queue.finish();

        // 5. Compute COM
        comKernel.setArg(0, buf_child);
        comKernel.setArg(1, buf_nodeX);
        comKernel.setArg(2, buf_nodeY);
        comKernel.setArg(3, buf_nodeZ);
        comKernel.setArg(4, buf_nodeMass);
        comKernel.setArg(5, buf_nodeCount);
        comKernel.setArg(6, buf_x);
        comKernel.setArg(7, buf_y);
        comKernel.setArg(8, buf_z);
        comKernel.setArg(9, buf_mass);
        comKernel.setArg(10, buf_nextNode);
        queue.enqueueNDRangeKernel(comKernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
        queue.finish();

        // 6. Forces
        forceKernel.setArg(0, buf_child);
        forceKernel.setArg(1, buf_nodeX);
        forceKernel.setArg(2, buf_nodeY);
        forceKernel.setArg(3, buf_nodeZ);
        forceKernel.setArg(4, buf_nodeMass);
        forceKernel.setArg(5, buf_nodeSize);
        forceKernel.setArg(6, buf_nextNode);
        forceKernel.setArg(7, buf_x);
        forceKernel.setArg(8, buf_y);
        forceKernel.setArg(9, buf_z);
        forceKernel.setArg(10, buf_fx);
        forceKernel.setArg(11, buf_fy);
        forceKernel.setArg(12, buf_fz);
        forceKernel.setArg(13, buf_mass);
        queue.enqueueNDRangeKernel(forceKernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
        queue.finish();

        // 7. Integration
        integKernel.setArg(0, buf_x);
        integKernel.setArg(1, buf_y);
        integKernel.setArg(2, buf_z);
        integKernel.setArg(3, buf_vx);
        integKernel.setArg(4, buf_vy);
        integKernel.setArg(5, buf_vz);
        integKernel.setArg(6, buf_fx);
        integKernel.setArg(7, buf_fy);
        integKernel.setArg(8, buf_fz);
        integKernel.setArg(9, buf_mass);
        queue.enqueueNDRangeKernel(integKernel, cl::NullRange, global, local);
        queue.finish();

        queue.enqueueReadBuffer(buf_x, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_x_out.data());
        queue.enqueueReadBuffer(buf_y, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_y_out.data());
        queue.enqueueReadBuffer(buf_z, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_z_out.data());

        std::cout << "Step " << step << " | Body 0: ("
                << h_x_out[0] << ", " << h_y_out[0] << ", " << h_z_out[0] << ")\n";
    }

#pragma endregion

#pragma region READBACK

    std::vector<float> h_bbox(6);
    queue.enqueueReadBuffer(buf_bbox, CL_TRUE, 0, 6*sizeof(float), h_bbox.data());
    std::cout << "Bounding box:" << std::endl;
    std::cout << "  X: " << h_bbox[0] << " -> " << h_bbox[1] << std::endl;
    std::cout << "  Y: " << h_bbox[2] << " -> " << h_bbox[3] << std::endl;
    std::cout << "  Z: " << h_bbox[4] << " -> " << h_bbox[5] << std::endl;

    //std::vector<float> h_x_out(NUM_BODIES), h_y_out(NUM_BODIES), h_z_out(NUM_BODIES);
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

    std::vector<int>   h_child(MAX_NODE * 8);
    std::vector<float> h_nodeX(MAX_NODE);
    std::vector<float> h_nodeY(MAX_NODE);
    std::vector<float> h_nodeZ(MAX_NODE);
    std::vector<float> h_nodeSize(MAX_NODE);
    std::vector<int>   h_nextNode(1);

    queue.enqueueReadBuffer(buf_child, CL_TRUE, 0, sizeof(int)*MAX_NODE*8, h_child.data());
    queue.enqueueReadBuffer(buf_nodeX, CL_TRUE, 0, sizeof(float)*MAX_NODE, h_nodeX.data());
    queue.enqueueReadBuffer(buf_nodeY, CL_TRUE, 0, sizeof(float)*MAX_NODE, h_nodeY.data());
    queue.enqueueReadBuffer(buf_nodeZ, CL_TRUE, 0, sizeof(float)*MAX_NODE, h_nodeZ.data());
    queue.enqueueReadBuffer(buf_nodeSize, CL_TRUE, 0, sizeof(float)*MAX_NODE, h_nodeSize.data());
    queue.enqueueReadBuffer(buf_nextNode, CL_TRUE, 0, sizeof(int), h_nextNode.data());

    std::cout << "\nRoot node:\n";
    std::cout << "Center: (" 
            << h_nodeX[0] << ", "
            << h_nodeY[0] << ", "
            << h_nodeZ[0] << ")\n";
    std::cout << "Size: " << h_nodeSize[0] << "\n";

    std::cout << "Children:\n";
    for (int i = 0; i < 8; i++)
    {
        std::cout << "  child[" << i << "] = " << h_child[i] << "\n";
    }

    std::cout << "Total nodes used: " << h_nextNode[0] << std::endl;

    int bodyRefs = 0;

    for (int n = 0; n < h_nextNode[0]; n++)
    {
        for (int i = 0; i < 8; i++)
        {
            int c = h_child[n * 8 + i];
            if (c >= 0 && c < NUM_BODIES)
                bodyRefs++;
        }
    }

    std::cout << "Body references in tree: " << bodyRefs << std::endl;

    for (int n = 0; n < h_nextNode[0]; n++)
    {
        if (h_nodeSize[n] <= 0.0f)
        {
            std::cout << "ERROR: node " << n << " has invalid size!\n";
        }
    }

    for (int n = 0; n < h_nextNode[0]; n++)
    {
        for (int i = 0; i < 8; i++)
        {
            int c = h_child[n * 8 + i];

            if (c >= NUM_BODIES)
            {
                int childNode = c - NUM_BODIES;

                if (childNode >= h_nextNode[0])
                {
                    std::cout << "ERROR: invalid child node index!\n";
                }
            }
        }
    }

    // COM verification readback
    std::vector<float> h_nodeMass(MAX_NODE);
    std::vector<int>   h_nodeCount(MAX_NODE);

    queue.enqueueReadBuffer(buf_nodeMass,  CL_TRUE, 0, sizeof(float)*MAX_NODE, h_nodeMass.data());
    queue.enqueueReadBuffer(buf_nodeCount, CL_TRUE, 0, sizeof(int)*MAX_NODE,   h_nodeCount.data());

    std::cout << "\n--- COM Verification ---\n";

    // Root node should contain everything
    std::cout << "Root node:\n";
    std::cout << "  COM position : ("
            << h_nodeX[0] << ", "
            << h_nodeY[0] << ", "
            << h_nodeZ[0] << ")\n";
    std::cout << "  Total mass   : " << h_nodeMass[0]  << "\n";
    std::cout << "  Body count   : " << h_nodeCount[0] << " (expected " << NUM_BODIES << ")\n";

    if (h_nodeCount[0] != NUM_BODIES)
        std::cout << "  WARNING: body count mismatch!\n";

    // Compute expected total mass from CPU side for cross-check
    float expectedMass = 0.0f;
    for (int i = 0; i < NUM_BODIES; i++)
        expectedMass += h_mass[i];

    std::cout << "  Expected mass: " << expectedMass << "\n";

    float massError = fabs(h_nodeMass[0] - expectedMass) / expectedMass;
    std::cout << "  Mass error   : " << massError * 100.0f << "%\n";
    if (massError > 0.001f)
        std::cout << "  WARNING: mass error too large!\n";

    // Print a few internal nodes to sanity check
    std::cout << "\nSample internal nodes:\n";
    int printed = 0;
    for (int n = 0; n < h_nextNode[0] && printed < 5; n++)
    {
        if (h_nodeCount[n] > 1)
        {
            std::cout << "  Node " << n << ":"
                    << "  COM=(" << h_nodeX[n] << ", " << h_nodeY[n] << ", " << h_nodeZ[n] << ")"
                    << "  mass="  << h_nodeMass[n]
                    << "  count=" << h_nodeCount[n]
                    << "  size="  << h_nodeSize[n] << "\n";
            printed++;
        }
    }

    // Check for any nodes with mass but zero count or vice versa
    int badNodes = 0;
    for (int n = 0; n < h_nextNode[0]; n++)
    {
        bool hasMass  = h_nodeMass[n]  > 0.0f;
        bool hasCount = h_nodeCount[n] > 0;
        if (hasMass != hasCount)
        {
            std::cout << "  ERROR: node " << n << " has inconsistent mass/count!\n";
            badNodes++;
        }
    }
    if (badNodes == 0)
        std::cout << "\nAll " << h_nextNode[0] << " nodes passed mass/count consistency check.\n";


#pragma endregion

    return 0;
}