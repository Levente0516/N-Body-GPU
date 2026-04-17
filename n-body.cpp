#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
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

int current = 0;

int calcNumNodes()
{
    int numNodes = NUM_BODIES * 2;
    if (numNodes < 1024 * 32)
    {
        numNodes = 1024 * 32;
    }
    while ((numNodes & (WARPSIZE - 1)) != 0)
    {
        ++numNodes;
    }

    return numNodes;
}

// ─── OpenGL shader helpers ────────────────────────────────────────────────────

std::string loadFile(const std::string &path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Cannot open: " << path << std::endl;
        exit(1);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

GLuint compileShader(GLenum type, const std::string &src)
{
    GLuint s = glCreateShader(type);
    const char *c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader error: " << log << std::endl;
        exit(1);
    }
    return s;
}

GLuint createProgram(const std::string &vertSrc, const std::string &fragSrc)
{
    GLuint v = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// ─── Simulation class (OpenGL rendering) ─────────────────────────────────────

class Simulation
{
public:
    float camZoom = SPAWN_RANGE * CAMERAZOOM * 2.0f;
    float camX = 0.0f;
    float camY = 0.0f;
    float camZ = 0.0f;
    bool dragging = false;
    double dragStartX = 0.0;
    double dragStartY = 0.0;
    double dragStartZ = 0.0;
    float dragCamStartX = 0.0f;
    float dragCamStartY = 0.0f;
    float dragCamStartZ = 0.0f;
    glm::vec3 camTarget = glm::vec3(0.0f);
    ;
    float camYaw = 0.0f;
    float camPitch = glm::radians(65.0f);

    GLFWwindow *window = nullptr;
    GLuint vbo[2];
    GLuint vao = 0;
    GLuint program = 0;

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
        if (!glfwInit())
            exit(1);

        // OpenGL context hints
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        std::cout << "GLFW  init\n";

        std::cout << camZoom << std::endl;

        GLFWmonitor *monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode *mode = glfwGetVideoMode(monitor);

        std::cout << mode->width << " " << mode->height << std::endl;
        // std::cout << "Zoom: " << camZoom << "\n CamX: " << camX << "\n CamY: " << camY << std::endl;

        window = glfwCreateWindow(mode->width, mode->height, "N-Body", nullptr, nullptr);
        glfwMakeContextCurrent(window);
        glfwSwapInterval(0);

        gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

        std::cout << "GLAD created\n";

        glfwSetWindowUserPointer(window, this);

        glfwSetKeyCallback(window, [](GLFWwindow *w, int key, int, int action, int)
                           {
            if (action == GLFW_PRESS && (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE))
            {
                glfwSetWindowShouldClose(w, GLFW_TRUE);
            } });

        glfwSetScrollCallback(window, [](GLFWwindow *w, double, double yOffset)
        {
            auto *sim = reinterpret_cast<Simulation *>(glfwGetWindowUserPointer(w));

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

            // std::cout << "Zoo: " << sim->camZoom << "\nCamX: " << sim->camX << "\nCamY: " << sim->camY << std::endl;
        });

        glfwSetMouseButtonCallback(window, [](GLFWwindow *w, int button, int action, int)
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
            } });

        glfwSetCursorPosCallback(window, [](GLFWwindow *w, double xpos, double ypos)
                                 {
            auto* sim = reinterpret_cast<Simulation*>(glfwGetWindowUserPointer(w));
            if (!sim->dragging) return;

            double dx = xpos - sim->dragStartX;
            double dy = ypos - sim->dragStartY;

            float sensitivity = 0.005f;

            sim->camYaw   += dx * sensitivity;
            sim->camPitch += dy * sensitivity;

            // clamp pitch so you don't flip
            sim->camPitch = glm::clamp(
                sim->camPitch,
                glm::radians(-89.0f),
                glm::radians(89.0f)
            );

            sim->dragStartX = xpos;
            sim->dragStartY = ypos; });
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

        // VAO
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        // bind first buffer just to define layout
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void *)0);
        glEnableVertexAttribArray(0);

        glBindVertexArray(0);

        // Shaders
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
            glm::radians(45.0f), // Field of view
            aspect,
            100.0f,              // Near plane (increase this)
            SPAWN_RANGE * 100.0f // Far plane
        );

        glm::vec3 direction;
        direction.x = cos(camPitch) * sin(camYaw);
        direction.y = sin(camPitch);
        direction.z = -cos(camPitch) * cos(camYaw);

        glm::vec3 position = camTarget - direction * camZoom;

        glm::mat4 view = glm::lookAt(
            position,
            camTarget,
            glm::vec3(0, 1, 0));

        glUseProgram(program);
        glUniformMatrix4fv(
            glGetUniformLocation(program, "projection"),
            1, GL_FALSE, glm::value_ptr(projection));

        glUniformMatrix4fv(
            glGetUniformLocation(program, "view"),
            1, GL_FALSE, glm::value_ptr(view));

        int drawBuf = 1 - current;

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo[drawBuf]);

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

    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::string exts = device.getInfo<CL_DEVICE_EXTENSIONS>();
    bool hasGLSharing = exts.find("cl_khr_gl_sharing") != std::string::npos;
    std::cout << "cl_khr_gl_sharng: " << (hasGLSharing ? "YES" : "NO") << std::endl;

    // ── Create OpenCL context sharing with OpenGL ───────────────────────────
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
        0};

    cl::Context context;
    if (hasGLSharing)
    {
        context = cl::Context(device, props);
        std::cout << "OpenCL context created with OpenGL sharing" << std::endl;
    }
    else
    {
        context = cl::Context(device);
        std::cout << "WARNING: No GL sharing — falling back to readback" << std::endl;
    }

    cl::CommandQueue queue(context, device);

    // ── Build kernels ───────────────────────────────────────────────────────
    std::string configSrc = loadFile("variables.hpp");
    std::string kernelSrc = loadFile("kernels/boundingbox.cl") +
                            loadFile("kernels/buildtree.cl") +
                            loadFile("kernels/suminfo.cl") +
                            loadFile("kernels/sort.cl") +
                            loadFile("kernels/force.cl") +
                            loadFile("kernels/integration.cl") +
                            loadFile("kernels/writepos.cl");

    cl::Program::Sources sources;
    sources.push_back(configSrc);
    sources.push_back(kernelSrc);

    int numNodes = calcNumNodes();

    std::string buildOptions = std::string("-cl-mad-enable ") +
                                "-D NUMBER_OF_NODES=" + std::to_string(numNodes);

    cl::Program program(context, sources);
    try
    {
        program.build({device}, buildOptions.c_str());
    }
    catch (const cl::Error &)
    {
        std::cerr << "Build error:\n"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }

    cl::Kernel boundingBoxKernel  (program, "boundingBoxKernel");
    cl::Kernel buildTreeKernel    (program, "buildTreeKernel");
    cl::Kernel sumInfoKernel(program, "summarizeTreeKernel");
    cl::Kernel sortKernel         (program, "sortKernel");
    cl::Kernel forceKernel        (program, "forceKernel");
    cl::Kernel integrateKernel    (program, "integrateKernel");
    cl::Kernel writePositionsKernel(program, "writePositionsInterleaved");

    // ── OpenCL buffers ──────────────────────────────────────────────────────

    std::cout << "Num nodes:" << numNodes << std::endl;

    cl::Buffer buf_x(context, CL_MEM_READ_WRITE, (numNodes+1) * sizeof(float));
    cl::Buffer buf_y(context, CL_MEM_READ_WRITE, (numNodes+1) * sizeof(float));
    cl::Buffer buf_z(context, CL_MEM_READ_WRITE, (numNodes+1) * sizeof(float));
    cl::Buffer buf_mass(context, CL_MEM_READ_WRITE, (numNodes+1) * sizeof(float));

    cl::Buffer buf_nodeSize(context, CL_MEM_READ_WRITE, (numNodes+1) * sizeof(float));
    cl::Buffer buf_nodeCount(context, CL_MEM_READ_WRITE, (numNodes+1) * sizeof(int));

    cl::Buffer buf_vx(context, CL_MEM_READ_WRITE, NUM_BODIES * sizeof(float));
    cl::Buffer buf_vy(context, CL_MEM_READ_WRITE, NUM_BODIES * sizeof(float));
    cl::Buffer buf_vz(context, CL_MEM_READ_WRITE, NUM_BODIES * sizeof(float));
    cl::Buffer buf_accX(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_accY(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_accZ(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    


    cl::Buffer buf_nextNode(context, CL_MEM_READ_WRITE, sizeof(int));

    int blockCount = 0;
    int maxD = 1;
    int bottomInit = 0;

    cl::Buffer buf_blockCount(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &blockCount);
    cl::Buffer buf_bottom    (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &bottomInit);
    cl::Buffer buf_numNodes  (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &numNodes);
    cl::Buffer buf_maxDepth  (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &maxD);

    cl::Buffer buf_minX(context, CL_MEM_READ_WRITE, NUM_BODIES * sizeof(float));
    cl::Buffer buf_minY(context, CL_MEM_READ_WRITE, NUM_BODIES * sizeof(float));
    cl::Buffer buf_minZ(context, CL_MEM_READ_WRITE, NUM_BODIES * sizeof(float));

    cl::Buffer buf_maxX(context, CL_MEM_READ_WRITE, NUM_BODIES * sizeof(float));
    cl::Buffer buf_maxY(context, CL_MEM_READ_WRITE, NUM_BODIES * sizeof(float));
    cl::Buffer buf_maxZ(context, CL_MEM_READ_WRITE, NUM_BODIES * sizeof(float));

    std::vector<int> child(8 * (numNodes + 1), 0);
    std::vector<int> start((numNodes + 1), 0);
    std::vector<int> h_sorted(numNodes + 1, 0);


    cl::Buffer buf_sorted(context, CL_MEM_READ_WRITE, (numNodes+1)*sizeof(int));

    int step = -1;

    cl::Buffer buf_child(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * child.size(), child.data());
    cl::Buffer buf_start(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * start.size(), start.data());
    cl::Buffer buf_step(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &step);

    // ── Shared VBO buffer ───────────────────────────────────────────────────
    cl::BufferGL buf_pos_gl[2];
    cl::Buffer buf_pos_fallback;
    bool usingGLSharing = false;

    if (hasGLSharing)
    {
        try
        {
            for (int i = 0; i < 2; i++)
            {
                buf_pos_gl[i] = cl::BufferGL(context, CL_MEM_READ_WRITE, sim.getVBO(i));
            }
            usingGLSharing = true;
            std::cout << "Shared OpenCL-OpenGL VBO created — true GPU interop active" << std::endl;
        }
        catch (...)
        {
            std::cerr << "BufferGL creation failed — falling back to readback" << std::endl;
        }
    }

    if (!usingGLSharing)
    {
        buf_pos_fallback = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * NUM_BODIES);
        std::cout << "Using fallback readback path" << std::endl;
    }

    // ── Init bodies ─────────────────────────────────────────────────────────
    std::vector<float> h_x(NUM_BODIES), h_y(NUM_BODIES), h_z(NUM_BODIES);
    std::vector<float> h_vx(NUM_BODIES), h_vy(NUM_BODIES), h_vz(NUM_BODIES);
    std::vector<float> h_fx(NUM_BODIES), h_fy(NUM_BODIES), h_fz(NUM_BODIES);
    std::vector<float> h_mass(NUM_BODIES);

    // Optional blackhole xd

    h_x[0] = 0;
    h_y[0] = 0;
    h_z[0] = 0;
    h_mass[0] = 200000000.0f;
    h_vx[0] = 0.0f;
    h_vy[0] = 0.0f;
    h_vz[0] = 0.0f;

    float totalSystemMass = 0.00f;

    for (int i = 1; i < NUM_BODIES; i++)
    {
        float massFactor = pow(((float)rand() / RAND_MAX), 2.0f);
        h_mass[i] = 5000.0f + massFactor * 50000.0f;
        totalSystemMass += h_mass[i];
    }

    for (int i = 1; i < NUM_BODIES; i++)
    {
        float angle = ((float)rand() / RAND_MAX) * 2.0f * 3.14159f;
        float radius = 0.0f;
        for (int j = 0; j < CAMERAZOOM; j++)
        {
            radius += (((float)rand() / RAND_MAX) * SPAWN_RANGE);
        }

        float z = ((float)rand() / RAND_MAX - 0.5f) * (SPAWN_RANGE / 2.0f);

        h_x[i] = radius * cos(angle);
        h_y[i] = radius * sin(angle);
        h_z[i] = z;

        float bhMass = h_mass[0]; // 1e21

        float r = sqrt(h_x[i] * h_x[i] + h_y[i] * h_y[i]);
        if (r < 1.0f)
            r = 1.0f;

        float enclosedMass = bhMass;

        float orbitalVelocity = sqrt(G * enclosedMass / r);

        h_vx[i] = sin(angle) * orbitalVelocity;
        h_vy[i] = -cos(angle) * orbitalVelocity;
        h_vz[i] = 0.0f;

        //h_vx[i] = 0.0f;
        //h_vy[i] = 0.0f;
    }

    queue.enqueueWriteBuffer(buf_x, CL_TRUE, 0, NUM_BODIES * sizeof(float), h_x.data());
    queue.enqueueWriteBuffer(buf_y, CL_TRUE, 0, NUM_BODIES * sizeof(float), h_y.data());
    queue.enqueueWriteBuffer(buf_z, CL_TRUE, 0, NUM_BODIES * sizeof(float), h_z.data());
    queue.enqueueWriteBuffer(buf_vx, CL_TRUE, 0, NUM_BODIES * sizeof(float), h_vx.data());
    queue.enqueueWriteBuffer(buf_vy, CL_TRUE, 0, NUM_BODIES * sizeof(float), h_vy.data());
    queue.enqueueWriteBuffer(buf_vz, CL_TRUE, 0, NUM_BODIES * sizeof(float), h_vz.data());
    queue.enqueueWriteBuffer(buf_mass, CL_TRUE, 0, NUM_BODIES * sizeof(float), h_mass.data());

    // ── Simulate lambda ─────────────────────────────────────────────────────

    boundingBoxKernel.setArg(0, buf_step);
    boundingBoxKernel.setArg(1, buf_x);
    boundingBoxKernel.setArg(2, buf_y);
    boundingBoxKernel.setArg(3, buf_z);
    boundingBoxKernel.setArg(4, buf_blockCount);
    boundingBoxKernel.setArg(5, buf_bottom);
    boundingBoxKernel.setArg(6, buf_mass);
    boundingBoxKernel.setArg(7, buf_numNodes);
    boundingBoxKernel.setArg(8, buf_minX);
    boundingBoxKernel.setArg(9, buf_minY);
    boundingBoxKernel.setArg(10, buf_minZ);
    boundingBoxKernel.setArg(11, buf_maxX);
    boundingBoxKernel.setArg(12, buf_maxY);
    boundingBoxKernel.setArg(13, buf_maxZ);
    boundingBoxKernel.setArg(14, buf_child);
    boundingBoxKernel.setArg(15, buf_start);
    boundingBoxKernel.setArg(16, buf_nodeSize);
    boundingBoxKernel.setArg(17, buf_maxDepth);
    
    buildTreeKernel.setArg(0, buf_x);         
    buildTreeKernel.setArg(1, buf_y);
    buildTreeKernel.setArg(2, buf_z);
    buildTreeKernel.setArg(3, buf_mass);
    buildTreeKernel.setArg(4, buf_child);
    buildTreeKernel.setArg(5, buf_start);
    buildTreeKernel.setArg(6, buf_nodeSize);
    buildTreeKernel.setArg(7, buf_bottom);
    buildTreeKernel.setArg(8, buf_maxDepth);
    buildTreeKernel.setArg(9, buf_numNodes);

    // summarizeTree
    sumInfoKernel.setArg(0, buf_x);         
    sumInfoKernel.setArg(1, buf_y);
    sumInfoKernel.setArg(2, buf_z);
    sumInfoKernel.setArg(3, buf_mass);
    sumInfoKernel.setArg(4, buf_child);
    sumInfoKernel.setArg(5, buf_nodeCount);
    sumInfoKernel.setArg(6, buf_bottom);
    sumInfoKernel.setArg(7, buf_numNodes);

    // sort
    sortKernel.setArg(0, buf_child);
    sortKernel.setArg(1, buf_nodeCount);
    sortKernel.setArg(2, buf_start);
    sortKernel.setArg(3, buf_sorted);
    sortKernel.setArg(4, buf_bottom);
    sortKernel.setArg(5, buf_numNodes);

    // force
    forceKernel.setArg(0, buf_x);         
    forceKernel.setArg(1, buf_y);
    forceKernel.setArg(2, buf_z);
    forceKernel.setArg(3, buf_mass);
    forceKernel.setArg(4, buf_child);
    forceKernel.setArg(5, buf_nodeSize);
    forceKernel.setArg(6, buf_sorted);
    forceKernel.setArg(7, buf_accX);       
    forceKernel.setArg(8, buf_accY);
    forceKernel.setArg(9, buf_accZ);
    forceKernel.setArg(10, buf_numNodes);

    // integrate
    integrateKernel.setArg(0, buf_x);   
    integrateKernel.setArg(1, buf_y);
    integrateKernel.setArg(2, buf_z);
    integrateKernel.setArg(3, buf_vx);  
    integrateKernel.setArg(4, buf_vy);
    integrateKernel.setArg(5, buf_vz);
    integrateKernel.setArg(6, buf_accX); 
    integrateKernel.setArg(7, buf_accY);
    integrateKernel.setArg(8, buf_accZ);

    // writePositions — arg 3 (VBO) set per-frame in simulate lambda
    writePositionsKernel.setArg(0, buf_x);
    writePositionsKernel.setArg(1, buf_y);
    writePositionsKernel.setArg(2, buf_z);

    cl::NDRange globalN    (NUM_BODIES);
    cl::NDRange globalNodes(numNodes);
    cl::NDRange local      (THREADS);

    auto simulateStep = [&]()
    {
        try 
        {
            int zero = 0, one_val = 1;
            // blockCount and maxDepth are reset by host; step/bottom reset inside boundingBox
            queue.enqueueWriteBuffer(buf_blockCount, CL_FALSE, 0, sizeof(int), &zero);
            queue.enqueueWriteBuffer(buf_maxDepth,   CL_FALSE, 0, sizeof(int), &one_val);

            int writeBuf = current;
            std::vector<cl::Memory> shared = { buf_pos_gl[writeBuf] };
            writePositionsKernel.setArg(3, buf_pos_gl[writeBuf]);

            // Each kernel must fully complete before the next starts
            queue.enqueueNDRangeKernel(boundingBoxKernel, cl::NullRange, globalN,local);
            queue.enqueueBarrierWithWaitList();
            
            queue.enqueueNDRangeKernel(buildTreeKernel, cl::NullRange, globalN,local);
            queue.enqueueBarrierWithWaitList();
            
            
            /*
            queue.enqueueNDRangeKernel(sumInfoKernel, cl::NullRange, globalNodes, local);
            queue.enqueueBarrierWithWaitList();
            */
            /*
            queue.enqueueNDRangeKernel(sortKernel, cl::NullRange, globalNodes, local);
            queue.enqueueBarrierWithWaitList();
           
            queue.enqueueNDRangeKernel(forceKernel, cl::NullRange, globalN, local);
            queue.enqueueBarrierWithWaitList();
            
            queue.enqueueNDRangeKernel(integrateKernel, cl::NullRange, globalN, local);
            queue.enqueueBarrierWithWaitList();
            */

            queue.enqueueAcquireGLObjects(&shared);
            queue.enqueueNDRangeKernel(writePositionsKernel, cl::NullRange, globalN, local);
            queue.enqueueReleaseGLObjects(&shared);

            queue.finish();
            current = 1 - current;
        }
        catch (cl::Error& e) {
            std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
            exit(1);
        }
    };

    sim.loop(simulateStep);
    return 0;
}