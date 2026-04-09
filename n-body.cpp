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

int         current  = 0;

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

        // OpenGL context hints
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        std::cout << "GLFW  init\n";
        
        std::cout << camZoom << std::endl; 

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);

        std::cout << mode->width << " " << mode->height << std::endl;
        //std::cout << "Zoom: " << camZoom << "\n CamX: " << camX << "\n CamY: " << camY << std::endl; 

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
            
            //std::cout << "Zoo: " << sim->camZoom << "\nCamX: " << sim->camX << "\nCamY: " << sim->camY << std::endl;
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

            // clamp pitch so you don't flip
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

        // VAO
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        // bind first buffer just to define layout
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, (void*)0);
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
            glm::radians(45.0f),  // Field of view
            aspect,
            100.0f,              // Near plane (increase this)
            SPAWN_RANGE * 100.0f   // Far plane
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
        CL_GL_CONTEXT_KHR,   (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR,      (cl_context_properties)wglGetCurrentDC(),
        0
    };

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
    std::string kernelSrc = loadFile("kernels.cl");

    cl::Program::Sources sources;
    sources.push_back(configSrc);
    sources.push_back(kernelSrc);

    std::string buildOptions = "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros";

    cl::Program program(context, sources);
    try {     
        program.build({ device }, buildOptions.c_str()); 
    }
    catch (const cl::Error&) {
        std::cerr << "Build error:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }

    cl::Kernel boundingBoxKernel    (program, "boundingBoxKernel");
    cl::Kernel initTreeKernel       (program, "initTreeKernel");
    cl::Kernel insertKernel         (program, "insertBodiesKernel");
    cl::Kernel comKernel            (program, "computeCOMKernel");
    cl::Kernel forceAndIntKernel    (program, "forceAndIntegrationKernel");
    cl::Kernel writePositionsKernel (program, "writePositionsInterleaved");

    // ── OpenCL buffers ──────────────────────────────────────────────────────
    cl::Buffer buf_x    (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_y    (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_z    (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_vx   (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_vy   (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_vz   (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_fx   (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_fy   (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_fz   (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_mass (context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_child    (context, CL_MEM_READ_WRITE, MAX_NODE*8*sizeof(int));
    cl::Buffer buf_nodeX    (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float));
    cl::Buffer buf_nodeY    (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float));
    cl::Buffer buf_nodeZ    (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float));
    cl::Buffer buf_nodeMass (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float));
    cl::Buffer buf_nodeCount(context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(int));
    cl::Buffer buf_nodeSize (context, CL_MEM_READ_WRITE, MAX_NODE*sizeof(float));
    cl::Buffer buf_nextNode (context, CL_MEM_READ_WRITE, sizeof(int));
    cl::Buffer buf_bbox     (context, CL_MEM_READ_WRITE, 6*sizeof(float));
    cl::Buffer buf_flag     (context, CL_MEM_READ_WRITE, sizeof(int));

    // ── Shared VBO buffer ───────────────────────────────────────────────────
    cl::BufferGL buf_pos_gl[2];
    cl::Buffer   buf_pos_fallback;
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
        buf_pos_fallback = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float)*3*NUM_BODIES);
        std::cout << "Using fallback readback path" << std::endl;
    }

    // ── Init bodies ─────────────────────────────────────────────────────────
    std::vector<float> h_x(NUM_BODIES), h_y(NUM_BODIES), h_z(NUM_BODIES);
    std::vector<float> h_vx(NUM_BODIES), h_vy(NUM_BODIES), h_vz(NUM_BODIES);
    std::vector<float> h_fx(NUM_BODIES), h_fy(NUM_BODIES), h_fz(NUM_BODIES);
    std::vector<float> h_mass(NUM_BODIES);
    
    //Optional blackhole xd
    
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
        for (int j = 0; j < CAMERAZOOM; j++) { 
            radius += (((float)rand() / RAND_MAX) * SPAWN_RANGE);
        }

        float z = ((float)rand() / RAND_MAX - 0.5f) * (SPAWN_RANGE / 2.0f);

        h_x[i] = radius * cos(angle);
        h_y[i] = radius * sin(angle);
        h_z[i] = 0;

        float bhMass = h_mass[0]; // 1e21

        float r = sqrt(h_x[i]*h_x[i] + h_y[i]*h_y[i]);
        if (r < 1.0f) r = 1.0f;

        float enclosedMass = bhMass;

        float orbitalVelocity = sqrt(G * enclosedMass / r);

        h_vx[i] =  sin(angle) * orbitalVelocity;
        h_vy[i] = -cos(angle) * orbitalVelocity;
        h_vz[i] = 0.0f;

    }

    queue.enqueueWriteBuffer(buf_x,    CL_TRUE, 0, NUM_BODIES*sizeof(float), h_x.data());
    queue.enqueueWriteBuffer(buf_y,    CL_TRUE, 0, NUM_BODIES*sizeof(float), h_y.data());
    queue.enqueueWriteBuffer(buf_z,    CL_TRUE, 0, NUM_BODIES*sizeof(float), h_z.data());
    queue.enqueueWriteBuffer(buf_vx,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vx.data());
    queue.enqueueWriteBuffer(buf_vy,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vy.data());
    queue.enqueueWriteBuffer(buf_vz,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vz.data());
    queue.enqueueWriteBuffer(buf_fx,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_fx.data());
    queue.enqueueWriteBuffer(buf_fy,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_fy.data());
    queue.enqueueWriteBuffer(buf_fz,   CL_TRUE, 0, NUM_BODIES*sizeof(float), h_fz.data());
    queue.enqueueWriteBuffer(buf_mass, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_mass.data());

    // ── Simulate lambda ─────────────────────────────────────────────────────
    cl::NDRange global(NUM_BODIES);
    cl::NDRange local(THREADS);
    cl::NDRange globalTree(MAX_NODE);
    // cl::NDRange one(1);
    cl::NDRange localBBox(64);
    
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


    auto simulateStep = [&]()
    {
        try {

            int writeBuf = current;
            int readBuf  = 1 - current;

            std::vector<cl::Memory> shared = { buf_pos_gl[writeBuf] };

            writePositionsKernel.setArg(3, buf_pos_gl[writeBuf]);


            queue.enqueueNDRangeKernel(boundingBoxKernel, cl::NullRange, global, localBBox);
            queue.enqueueNDRangeKernel(initTreeKernel, cl::NullRange, globalTree, local);
            queue.enqueueNDRangeKernel(insertKernel, cl::NullRange, global, local);

            for (int pass = 0; pass < 6; pass++)
            {
                queue.enqueueNDRangeKernel(comKernel, cl::NullRange, globalTree, local);
            }

            queue.enqueueNDRangeKernel(forceAndIntKernel , cl::NullRange, global, local);

            //glFinish();
            queue.enqueueAcquireGLObjects(&shared);
            queue.enqueueNDRangeKernel(writePositionsKernel, cl::NullRange, global, local);
            queue.enqueueReleaseGLObjects(&shared);

            queue.flush();

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