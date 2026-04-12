#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <functional>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include "variables.hpp"
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/opencl.hpp>
#endif
#include <CL/cl_gl.h>
#include <wingdi.h>

int current  = 0;

static uint32_t expandBits(uint32_t v)
{
    v &= 0x0000ffff;
    v = (v ^ (v <<  8)) & 0x00ff00ff;
    v = (v ^ (v <<  4)) & 0x0f0f0f0f;
    v = (v ^ (v <<  2)) & 0x33333333;
    v = (v ^ (v <<  1)) & 0x55555555;
    return v;
}

static uint32_t morton2D(uint32_t ix, uint32_t iy)
{
    return (expandBits(ix) << 1) | expandBits(iy);
}

struct QNode
{
    float comX, comY;   
    float mass;
    float halfSize;   
    float cx, cy;       
    int   children[4]; 
    int   bodyIdx;      
};

int buildQuadtree(
    std::vector<QNode>&       nodes,
    std::vector<int>&         order,
    std::vector<int>&         scratch,
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& mass,
    int start, int count,
    float cx, float cy, float halfSize,
    int depth)
{
    if (count == 0)
    {
        return -1;
    } 

    int idx = (int)nodes.size();

    nodes.push_back({});

    float totalMass = 0.0f, comXacc = 0.0f, comYacc = 0.0f;
    for (int k = start; k < start + count; k++) 
    {
        int b = order[k];
        totalMass += mass[b];
        comXacc   += x[b] * mass[b];
        comYacc   += y[b] * mass[b];
    }

    nodes[idx].cx       = cx;
    nodes[idx].cy       = cy;
    nodes[idx].halfSize = halfSize;
    nodes[idx].mass     = totalMass;
    nodes[idx].comX     = (totalMass > 0.0f) ? comXacc / totalMass : cx;
    nodes[idx].comY     = (totalMass > 0.0f) ? comYacc / totalMass : cy;
    nodes[idx].bodyIdx  = -1;
    nodes[idx].children[0] = nodes[idx].children[1] =
    nodes[idx].children[2] = nodes[idx].children[3] = -1;

    if (count == 1 || depth >= 24) 
    {
        nodes[idx].bodyIdx = order[start];
        return idx;
    }


    int qcount[4] = {0, 0, 0, 0};
    for (int k = start; k < start + count; k++) 
    {
        int b = order[k];
        int q = ((x[b] >= cx) ? 1 : 0) | ((y[b] >= cy) ? 2 : 0);
        qcount[q]++;
    }

    int qoff[4];
    qoff[0] = 0;
    qoff[1] = qcount[0];
    qoff[2] = qcount[0] + qcount[1];
    qoff[3] = qcount[0] + qcount[1] + qcount[2];

    int qpos[4] = { qoff[0], qoff[1], qoff[2], qoff[3] };
    for (int k = start; k < start + count; k++) 
    {
        int b = order[k];
        int q = ((x[b] >= cx) ? 1 : 0) | ((y[b] >= cy) ? 2 : 0);
        scratch[qpos[q]++] = b;
    }

    for (int k = 0; k < count; k++)
    {
        order[start + k] = scratch[k];
    }

    float h    = halfSize * 0.5f;
    float ccx[4] = { cx - h, cx + h, cx - h, cx + h };
    float ccy[4] = { cy - h, cy - h, cy + h, cy + h };

    int pos = start;
    for (int q = 0; q < 4; q++) 
    {
        if (qcount[q] > 0) 
        {
            int childIdx = buildQuadtree(nodes, order, scratch,
                                         x, y, mass,
                                         pos, qcount[q],
                                         ccx[q], ccy[q], h,
                                         depth + 1);
            nodes[idx].children[q] = childIdx;
        }
        pos += qcount[q];
    }

    return idx;
}

// ─── OpenGL shader helpers ────────────────────────────────────────────────────

std::string loadFile(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) { std::cerr << "Cannot open: " << path << std::endl; exit(1); }
    std::ostringstream ss; ss << file.rdbuf(); return ss.str();
}

GLuint compileShader(GLenum type, const std::string& src)
{
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader error: " << log << std::endl; exit(1);
    }
    return s;
}

GLuint createProgram(const std::string& vertSrc, const std::string& fragSrc)
{
    GLuint v = compileShader(GL_VERTEX_SHADER,   vertSrc);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// ─── Simulation class (OpenGL rendering) ─────────────────────────────────────

class Simulation
{
public:
    float camZoom        = SPAWN_RANGE * CAMERAZOOM * 2.0f;
    float camX           = 0.0f;
    float camY           = 0.0f;
    float camZ           = 0.0f;
    bool  dragging       = false;
    double dragStartX    = 0.0;
    double dragStartY    = 0.0;
    double dragStartZ    = 0.0;
    float  dragCamStartX = 0.0f;
    float  dragCamStartY = 0.0f;
    float  dragCamStartZ = 0.0f;    
    glm::vec3 camTarget  = glm::vec3(0.0f);; 
    float camYaw         = 0.0f;
    float camPitch       = glm::radians(65.0f);

    GLFWwindow* window   = nullptr;
    GLuint      vbo[2];
    GLuint      vao      = 0;
    GLuint      program  = 0;

    void init()
    {
        initWindow();
        initGL();
    }

    void loop(std::function<void()> simulateStep)
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            simulateStep();
            drawFrame();
        }
        cleanup();
    }

    GLuint getVBO(int i) { return vbo[i]; }

private:
    void initWindow()
    {
        if (!glfwInit()) exit(1);

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        std::cout << "GLFW  init\n";
        
        std::cout << camZoom << std::endl; 

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);

        std::cout << mode->width << " " << mode->height << std::endl;

        window = glfwCreateWindow(mode->width, mode->height, "N-Body", nullptr, nullptr);
        glfwMakeContextCurrent(window);
        glfwSwapInterval(0);

        gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

        std::cout << "GLAD created\n";

        glfwSetWindowUserPointer(window, this);

        glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int, int action, int)
        {
            if (action == GLFW_PRESS && (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE))
            {
                glfwSetWindowShouldClose(w, GLFW_TRUE);
            }
        });

        glfwSetScrollCallback(window, [](GLFWwindow* w, double, double yOffset)
        {
            auto* sim = reinterpret_cast<Simulation*>(glfwGetWindowUserPointer(w));

            double mouseX, mouseY;
            glfwGetCursorPos(w, &mouseX, &mouseY);
            
            int winW, winH;
            glfwGetWindowSize(w, &winW, &winH);
            
            float worldX = sim->camX + (mouseX / winW - 0.5f) * (SPAWN_RANGE / sim->camZoom);
            float worldY = sim->camY + (0.5f - mouseY / winH) * (SPAWN_RANGE / sim->camZoom);
            
            float oldZoom = sim->camZoom;
            float factor = (yOffset > 0) ? 0.9f : 1.1f;
            sim->camZoom *= factor;
            
            float newWorldX = sim->camX + (mouseX / winW - 0.5f) * (SPAWN_RANGE / sim->camZoom);
            float newWorldY = sim->camY + (0.5f - mouseY / winH) * (SPAWN_RANGE / sim->camZoom);

            sim->camX += worldX - newWorldX;
            sim->camY += worldY - newWorldY;
        });

        glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int button, int action, int)
        {
            auto* sim = reinterpret_cast<Simulation*>(glfwGetWindowUserPointer(w));
            if (button == GLFW_MOUSE_BUTTON_LEFT)
            {
                if (action == GLFW_PRESS)
                {
                    sim->dragging = true;
                    glfwGetCursorPos(w, &sim->dragStartX, &sim->dragStartY);
                    sim->dragCamStartX = sim->camX;
                    sim->dragCamStartY = sim->camY;
                }
                else 
                {
                    sim->dragging = false;
                }
            }
        });

        glfwSetCursorPosCallback(window, [](GLFWwindow* w, double xpos, double ypos)
        {
            auto* sim = reinterpret_cast<Simulation*>(glfwGetWindowUserPointer(w));
            if (!sim->dragging) return;

            double dx = xpos - sim->dragStartX;
            double dy = ypos - sim->dragStartY;

            float sensitivity = 0.005f;

            sim->camYaw   += dx * sensitivity;
            sim->camPitch += dy * sensitivity;

            sim->camPitch = glm::clamp(
                sim->camPitch,
                glm::radians(-89.0f),
                glm::radians(89.0f)
            );

            sim->dragStartX = xpos;
            sim->dragStartY = ypos;
        });
    }

    void initGL()
    {
        glGenBuffers(2, vbo);

        for (int i = 0; i < 2; i++)
        {
            glBindBuffer(GL_ARRAY_BUFFER, vbo[i]);
            glBufferData(GL_ARRAY_BUFFER,
                sizeof(float) * 3 * NUM_BODIES,
                nullptr,
                GL_STREAM_DRAW);
        }

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, (void*)0);
        glEnableVertexAttribArray(0);

        glBindVertexArray(0);

        std::string vertSrc = loadFile("shader.vert");
        std::string fragSrc = loadFile("shader.frag");
        program = createProgram(vertSrc, fragSrc);
        
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        std::cout << "OpenGL initialized" << std::endl;
        std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    }

    void drawFrame()
    {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = (float)w / (float)h;

        glm::mat4 projection = glm::perspective(
            glm::radians(45.0f),
            aspect,
            100.0f,
            SPAWN_RANGE * 100.0f  
        );

        glm::vec3 direction;
        direction.x = cos(camPitch) * sin(camYaw);
        direction.y = sin(camPitch);
        direction.z = -cos(camPitch) * cos(camYaw);

        glm::vec3 position = camTarget - direction * camZoom;

        glm::mat4 view = glm::lookAt(
            position,
            camTarget,
            glm::vec3(0, 1, 0)
        );

        glUseProgram(program);
        glUniformMatrix4fv(
            glGetUniformLocation(program, "projection"),
            1, GL_FALSE, glm::value_ptr(projection)
        );

        glUniformMatrix4fv(
            glGetUniformLocation(program, "view"),
            1, GL_FALSE, glm::value_ptr(view)
        );

        int drawBuf = 1 - current;

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo[drawBuf]);
        
        GLint blackHoleLoc = glGetUniformLocation(program, "blackHoleIndex");
        glUniform1i(blackHoleLoc, 0);

        glDrawArrays(GL_POINTS, 0, NUM_BODIES);

        glBindVertexArray(0);

        glfwSwapBuffers(window);

    }

    void cleanup()
    {
        glDeleteBuffers(1, &vbo[0]);
        glDeleteBuffers(1, &vbo[1]);
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(program);
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

// ─── main ─────────────────────────────────────────────────────────────────────

int main()
{
    srand(time(NULL));

    Simulation sim;
    sim.init();

    // ── OpenCL platform/device ──────────────────────────────────────────────
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    std::string exts = device.getInfo<CL_DEVICE_EXTENSIONS>();

    bool hasGLSharing = exts.find("cl_khr_gl_sharing") != std::string::npos;

    // ── Create OpenCL context sharing with OpenGL ───────────────────────────
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
        CL_GL_CONTEXT_KHR,   (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR,      (cl_context_properties)wglGetCurrentDC(),
        0
    };

    cl::Context context;

    if (hasGLSharing)
    {
        context = cl::Context(device, props);
    }
    else
    {
        context = cl::Context(device);
    }

    cl::CommandQueue queue(context, device, cl::QueueProperties::None);

    // ── Build kernels ───────────────────────────────────────────────────────
    std::string configSrc = loadFile("variables.hpp");
    std::string kernelSrc = loadFile("kernels.cl");

    cl::Program::Sources sources;

    sources.push_back(configSrc);
    sources.push_back(kernelSrc);

    std::string buildOptions = "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros";

    cl::Program program(context, sources);
    try 
    {     
        program.build({ device }, 
        buildOptions.c_str()); 
    }
    catch (const cl::Error&) 
    {
        std::cerr << "Build error:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }

    // ── Kernels ────────────────────────────────────────────────────────────
    
    /*
    cl::Kernel boundingBoxKernel    (program, "boundingBoxKernel");
    cl::Kernel initTreeKernel       (program, "initTreeKernel");
    cl::Kernel insertKernel         (program, "insertBodiesKernel");
    cl::Kernel comKernel            (program, "computeCOMKernel");
    */
    cl::Kernel forceAndIntKernel    (program, "forceAndIntegrationKernel");
    cl::Kernel writePositionsKernel (program, "writePositionsInterleaved");

    // ── OpenCL buffers ──────────────────────────────────────────────────────

    cl::Buffer buf_x    (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_y    (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_z    (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));

    cl::Buffer buf_vx   (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_vy   (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_vz   (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));

    cl::Buffer buf_mass (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));

    cl::Buffer buf_child    (context, CL_MEM_READ_WRITE, MAX_NODE*8*sizeof(int));

    cl::Buffer buf_nodeX    (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float));
    cl::Buffer buf_nodeY    (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float));
    cl::Buffer buf_nodeZ    (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float));

    //cl::Buffer buf_nodeMass (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float));

    cl::Buffer buf_nodeCount(context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(int));

    cl::Buffer buf_nodeSize (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float));

    cl::Buffer buf_nextNode (context, CL_MEM_READ_WRITE, sizeof(int));

    cl::Buffer buf_bbox     (context, CL_MEM_READ_WRITE, 6*sizeof(float));

    cl::Buffer buf_nodeCOMX    (context, CL_MEM_READ_ONLY, MAX_NODE*sizeof(float));
    cl::Buffer buf_nodeCOMY    (context, CL_MEM_READ_ONLY, MAX_NODE*sizeof(float));
    cl::Buffer buf_nodeMass    (context, CL_MEM_READ_ONLY, MAX_NODE*sizeof(float));
    cl::Buffer buf_nodeHalfSize(context, CL_MEM_READ_ONLY, MAX_NODE*sizeof(float));
    cl::Buffer buf_nodeChildren(context, CL_MEM_READ_ONLY, MAX_NODE*4*sizeof(int));
    cl::Buffer buf_nodeBodyIdx (context, CL_MEM_READ_ONLY, MAX_NODE*sizeof(int));

    // ── Shared VBO buffer ───────────────────────────────────────────────────

    cl::BufferGL buf_pos_gl[2];
    bool usingGLShare = false;
    if (hasGLSharing) {
        try {
            for (int i = 0; i < 2; i++)
                buf_pos_gl[i] = cl::BufferGL(context, CL_MEM_READ_WRITE, sim.getVBO(i));
            usingGLShare = true;
            std::cout << "GL interop active\n";
        } catch (...) {
            std::cerr << "BufferGL failed — GL interop unavailable\n";
        }
    }
    if (!hasGLSharing) {
        std::cerr << "ERROR: GL sharing required for this build\n";
        return 1;
    }

    // ── Init bodies ─────────────────────────────────────────────────────────

    std::vector<float> h_x(NUM_BODIES), h_y(NUM_BODIES), h_z(NUM_BODIES);
    std::vector<float> h_vx(NUM_BODIES), h_vy(NUM_BODIES), h_vz(NUM_BODIES);
    std::vector<float> h_mass(NUM_BODIES);
    
    //Optional blackhole xd
    
    h_x[0] = 0;
    h_y[0] = 0;
    h_z[0] = 0;
    h_mass[0] = 20000000.0f; // 200000000.0f;
    h_vx[0] = 0.0f;
    h_vy[0] = 0.0f;
    h_vz[0] = 0.0f; 

    for (int i = 1; i < NUM_BODIES; i++)
    {
        float massFactor = pow(((float)rand() / RAND_MAX), 2.0f);

        h_mass[i] = 5000.0f + massFactor * 50000.0f;
    }

    for (int i = 1; i < NUM_BODIES; i++)
    {
        float angle = ((float)rand() / RAND_MAX) * 2.0f * 3.14159f;

        float radius = 0.0f;

        for (int j = 0; j < CAMERAZOOM; j++) 
        { 
            radius += (((float)rand() / RAND_MAX) * SPAWN_RANGE);
        }

        //float z = ((float)rand() / RAND_MAX - 0.5f) * (SPAWN_RANGE / 2.0f);

        h_x[i] = radius * cos(angle);
        h_y[i] = radius * sin(angle);
        h_z[i] = 0;

        float bhMass = h_mass[0];

        float r = sqrt(h_x[i]*h_x[i] + h_y[i]*h_y[i]);

        if (r < 1.0f)
        {
            r = 1.0f;
        } 
            

        float orbitalVelocity = sqrt(G * bhMass / r);
        
        h_vx[i] =  sin(angle) * orbitalVelocity;
        h_vy[i] = -cos(angle) * orbitalVelocity;
        h_vz[i] = 0.0f;
        
        /*
        h_vx[i] = 0.0f;
        h_vy[i] = 0.0f;
        h_vz[i] = 0.0f;
        */

    }

    // ── Copying buffers to the GPU memory ─────────────────────────────────────────────────

    queue.enqueueWriteBuffer(buf_x,    CL_TRUE, 0, NUM_BODIES*sizeof(float), h_x.data());
    queue.enqueueWriteBuffer(buf_y,    CL_TRUE, 0, NUM_BODIES*sizeof(float), h_y.data());
    queue.enqueueWriteBuffer(buf_z,    CL_TRUE, 0, NUM_BODIES*sizeof(float), h_z.data());
    queue.enqueueWriteBuffer(buf_vx,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vx.data());
    queue.enqueueWriteBuffer(buf_vy,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vy.data());
    queue.enqueueWriteBuffer(buf_vz,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vz.data());
    queue.enqueueWriteBuffer(buf_mass, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_mass.data());


    std::vector<uint32_t> mortonCodes(NUM_BODIES);
    std::vector<int>      sortedOrder(NUM_BODIES);
    std::vector<int>      scratchBuf(NUM_BODIES);
    std::vector<QNode>    nodes;
    nodes.reserve(MAX_NODE);

    std::vector<float> h_nodeCOMX(MAX_NODE), h_nodeCOMY(MAX_NODE);
    std::vector<float> h_nodeMass(MAX_NODE), h_nodeHalfSize(MAX_NODE);
    std::vector<int>   h_nodeChildren(MAX_NODE * 4);
    std::vector<int>   h_nodeBodyIdx(MAX_NODE);

    // ── Simulate lambda ─────────────────────────────────────────────────────

    cl::NDRange global(NUM_BODIES);
    cl::NDRange local(THREADS);
    cl::NDRange globalTree(MAX_NODE);
    cl::NDRange one(1);
    cl::NDRange localBBox(64);

    /*
    int startNode = 1;
    queue.enqueueWriteBuffer(buf_nextNode, CL_TRUE, 0, sizeof(int), &startNode);
    
    boundingBoxKernel.setArg(0, buf_bbox);
    boundingBoxKernel.setArg(1, buf_x);
    boundingBoxKernel.setArg(2, buf_y);
    boundingBoxKernel.setArg(3, buf_z);
    
    initTreeKernel.setArg(0, buf_child);
    initTreeKernel.setArg(1, buf_nodeX);
    initTreeKernel.setArg(2, buf_nodeY);
    initTreeKernel.setArg(3, buf_nodeZ);
    initTreeKernel.setArg(4, buf_nodeMass);
    initTreeKernel.setArg(5, buf_nodeCount);
    initTreeKernel.setArg(6, buf_nodeSize);
    initTreeKernel.setArg(7, buf_nextNode);
    initTreeKernel.setArg(8, buf_bbox);
    
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
    comKernel.setArg(11, 0); 
    comKernel.setArg(12, MAX_NODE);
    
    forceAndIntKernel .setArg(0, buf_child);
    forceAndIntKernel .setArg(1, buf_nodeX);
    forceAndIntKernel .setArg(2, buf_nodeY);
    forceAndIntKernel .setArg(3, buf_nodeZ);
    forceAndIntKernel .setArg(4, buf_nodeMass);
    forceAndIntKernel .setArg(5, buf_nodeSize);
    forceAndIntKernel .setArg(6, buf_nextNode);
    forceAndIntKernel .setArg(7, buf_x);
    forceAndIntKernel .setArg(8, buf_y);
    forceAndIntKernel .setArg(9, buf_z);
    forceAndIntKernel .setArg(10, buf_vx);
    forceAndIntKernel .setArg(11, buf_vy);
    forceAndIntKernel .setArg(12, buf_vz);
    forceAndIntKernel .setArg(13, buf_mass);
    
    writePositionsKernel.setArg(0, buf_x);
    writePositionsKernel.setArg(1, buf_y);
    writePositionsKernel.setArg(2, buf_z);
    //writePositionsKernel.setArg(3, buf_pos_gl);
    */
    
    
    auto simulateStep = [&]()
    {
        /*
        try 
        {
            int writeBuf = current;
            int readBuf  = 1 - current;
            
            std::vector<cl::Memory> shared = { buf_pos_gl[writeBuf] };
            
            writePositionsKernel.setArg(3, buf_pos_gl[writeBuf]);

            float bboxInit[] = {1e30f,-1e30f, 1e30f,-1e30f, 1e30f,-1e30f};
            queue.enqueueWriteBuffer(buf_bbox, CL_FALSE, 0, 6*sizeof(float), bboxInit);
            
            // At the start of simulateStep, before boundingBoxKernel:
            int startNode = 1;
            queue.enqueueWriteBuffer(buf_nextNode, CL_FALSE, 0, sizeof(int), &startNode);
            queue.enqueueNDRangeKernel(boundingBoxKernel, cl::NullRange, global, localBBox);
            queue.enqueueNDRangeKernel(initTreeKernel, cl::NullRange, globalTree, local);

            queue.enqueueNDRangeKernel(insertKernel, cl::NullRange, global, local);

            int actualNodes;
            queue.enqueueReadBuffer(buf_nextNode, CL_TRUE, 0, sizeof(int), &actualNodes);
            actualNodes = std::min(actualNodes, MAX_NODE);
            int comThreads = ((actualNodes + THREADS - 1) / THREADS) * THREADS;
            cl::NDRange globalCOM(comThreads);

            queue.enqueueNDRangeKernel(comKernel, cl::NullRange, globalCOM, local);

            queue.finish();

            queue.enqueueNDRangeKernel(forceAndIntKernel , cl::NullRange, global, local);

            queue.enqueueAcquireGLObjects(&shared);

            queue.enqueueNDRangeKernel(writePositionsKernel, cl::NullRange, global, local);

            queue.enqueueReleaseGLObjects(&shared);

            queue.finish();

            current = 1 - current;
        }
        catch (cl::Error& e) 
        {
            std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
            exit(1);
        }
            
        */


        queue.enqueueReadBuffer(buf_x, CL_FALSE, 0, NUM_BODIES*sizeof(float), h_x.data());
        queue.enqueueReadBuffer(buf_y, CL_TRUE,  0, NUM_BODIES*sizeof(float), h_y.data());

        float minX = h_x[0], maxX = h_x[0], minY = h_y[0], maxY = h_y[0];
        for (int i = 1; i < NUM_BODIES; i++) {
            if (h_x[i] < minX) minX = h_x[i];
            if (h_x[i] > maxX) maxX = h_x[i];
            if (h_y[i] < minY) minY = h_y[i];
            if (h_y[i] > maxY) maxY = h_y[i];
        }
        float rangeX = std::max(maxX - minX, 1.0f);
        float rangeY = std::max(maxY - minY, 1.0f);

        for (int i = 0; i < NUM_BODIES; i++) {
            uint32_t ix = std::min((uint32_t)((h_x[i] - minX) / rangeX * 1023.0f), 1023u);
            uint32_t iy = std::min((uint32_t)((h_y[i] - minY) / rangeY * 1023.0f), 1023u);
            mortonCodes[i] = morton2D(ix, iy);
        }
        std::iota(sortedOrder.begin(), sortedOrder.end(), 0);
        std::sort(sortedOrder.begin(), sortedOrder.end(),
                  [&](int a, int b){ return mortonCodes[a] < mortonCodes[b]; });

        float treeHalf = std::max(rangeX, rangeY) * 0.5f + 1.0f;
        float treeCX   = (minX + maxX) * 0.5f;
        float treeCY   = (minY + maxY) * 0.5f;

        nodes.clear();
        buildQuadtree(nodes, sortedOrder, scratchBuf,
                      h_x, h_y, h_mass,
                      0, NUM_BODIES,
                      treeCX, treeCY, treeHalf,
                      0);

        int numNodes = (int)nodes.size();

        for (int k = 0; k < numNodes; k++) {
            h_nodeCOMX[k]     = nodes[k].comX;
            h_nodeCOMY[k]     = nodes[k].comY;
            h_nodeMass[k]     = nodes[k].mass;
            h_nodeHalfSize[k] = nodes[k].halfSize;
            h_nodeBodyIdx[k]  = nodes[k].bodyIdx;
            for (int c = 0; c < 4; c++)
                h_nodeChildren[k * 4 + c] = nodes[k].children[c];
        }

        queue.enqueueWriteBuffer(buf_nodeCOMX,     CL_FALSE, 0, numNodes*sizeof(float),   h_nodeCOMX.data());
        queue.enqueueWriteBuffer(buf_nodeCOMY,     CL_FALSE, 0, numNodes*sizeof(float),   h_nodeCOMY.data());
        queue.enqueueWriteBuffer(buf_nodeMass,     CL_FALSE, 0, numNodes*sizeof(float),   h_nodeMass.data());
        queue.enqueueWriteBuffer(buf_nodeHalfSize, CL_FALSE, 0, numNodes*sizeof(float),   h_nodeHalfSize.data());
        queue.enqueueWriteBuffer(buf_nodeChildren, CL_FALSE, 0, numNodes*4*sizeof(int),   h_nodeChildren.data());
        queue.enqueueWriteBuffer(buf_nodeBodyIdx,  CL_TRUE,  0, numNodes*sizeof(int),     h_nodeBodyIdx.data());

        forceAndIntKernel.setArg(0,  buf_nodeCOMX);
        forceAndIntKernel.setArg(1,  buf_nodeCOMY);
        forceAndIntKernel.setArg(2,  buf_nodeMass);
        forceAndIntKernel.setArg(3,  buf_nodeHalfSize);
        forceAndIntKernel.setArg(4,  buf_nodeChildren);
        forceAndIntKernel.setArg(5,  buf_nodeBodyIdx);
        forceAndIntKernel.setArg(6,  buf_x);
        forceAndIntKernel.setArg(7,  buf_y);
        forceAndIntKernel.setArg(8,  buf_vx);
        forceAndIntKernel.setArg(9,  buf_vy);
        forceAndIntKernel.setArg(10, buf_mass);
        forceAndIntKernel.setArg(11, numNodes);
        queue.enqueueNDRangeKernel(forceAndIntKernel, cl::NullRange, global, local);

        std::vector<cl::Memory> shared = { buf_pos_gl[current] };
        writePositionsKernel.setArg(0, buf_x);
        writePositionsKernel.setArg(1, buf_y);
        writePositionsKernel.setArg(2, buf_pos_gl[current]);
        queue.enqueueAcquireGLObjects(&shared);
        queue.enqueueNDRangeKernel(writePositionsKernel, cl::NullRange, global, local);
        queue.enqueueReleaseGLObjects(&shared);
        queue.finish();

        current = 1 - current;
    };

    sim.loop(simulateStep);
    return 0;
}