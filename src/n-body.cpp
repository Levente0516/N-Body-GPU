#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#define _USE_MATH_DEFINES
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
#define CL_HPP_TARGET_OPENCL_VERSION 210 
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/opencl.hpp>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#endif
#include <CL/cl_gl.h>
#include <wingdi.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

const int NUM_BODIES = 32768; //131072 //32768 //
const int THREADS = 64;
const int WARPSIZE = 64;
const float SPAWN_RANGE = 200000.0f;
const float THETA = 0.5f;
const float G = 5.0f;
const float SOFTENING = 50.0f;
const float DT = 0.5f;
const int CAMERAZOOM = 2;
const int MAXDEPTH = 64;

enum class DistributionType
{
    UNIFORM,
    DISK,
    SPHERE,
    GAUSSIAN,
    RING
};

struct SimParams {
    float g          = 5.0f;
    float dt         = 1.0f;
    float theta      = 0.5f;
    float softening  = 50.0f;
    int   numBodies  = 32768;
    int   distType   = 0;  // 0=disk, 1=uniform, 2=sphere, 3=ring
    bool  restart    = false;
};

int current = 0;

int calcNumNodes()
{
    int numNodes = NUM_BODIES * 2;
    if (numNodes < 1024 * 7)
    {
        numNodes = 1024 * 7;
    }
    while ((numNodes & (WARPSIZE - 1)) != 0)
    {
        ++numNodes;
    }

    return numNodes;
}


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

class SimulationRender
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
        GLuint spriteTexture = 0;
        GLuint massVBO;

        SimParams* simParams = nullptr; 

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

        GLuint getMassVBO() { return massVBO; }

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
                if (ImGui::GetIO().WantCaptureMouse)
                {
                    return;
                } 
                auto *sim = reinterpret_cast<SimulationRender*>(glfwGetWindowUserPointer(w));

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
                if (ImGui::GetIO().WantCaptureMouse)
                {
                    return;
                } 
                auto* sim = reinterpret_cast<SimulationRender*>(glfwGetWindowUserPointer(w));
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

            glfwSetCursorPosCallback(window, [](GLFWwindow *w, double xpos, double ypos)
            {
                if (ImGui::GetIO().WantCaptureMouse)
                {
                    return;
                } 
                auto* sim = reinterpret_cast<SimulationRender*>(glfwGetWindowUserPointer(w));
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
                glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * NUM_BODIES, nullptr, GL_STREAM_DRAW);
            }

            glGenBuffers(1, &massVBO);
            glBindBuffer(GL_ARRAY_BUFFER, massVBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * NUM_BODIES, nullptr, GL_STREAM_DRAW);

            // VAO
            glGenVertexArrays(1, &vao);
            glBindVertexArray(vao);

            // bind first buffer just to define layout
            glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void *)0);
            glEnableVertexAttribArray(0);

            glBindBuffer(GL_ARRAY_BUFFER, massVBO);
            glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(1);

            glBindVertexArray(0);



            // Shaders
            std::string vertSrc = loadFile("shaders/shader.vert");
            std::string fragSrc = loadFile("shaders/shader.frag");
            program = createProgram(vertSrc, fragSrc);

            glEnable(GL_PROGRAM_POINT_SIZE);
            glEnable(GL_BLEND);
            glDisable(GL_DEPTH_TEST);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);

            auto loadTexture = [](const char* path) -> GLuint {
                int w, h, ch;
                // Flip vertically because OpenGL UV origin is bottom-left
                stbi_set_flip_vertically_on_load(false);
                unsigned char* data = stbi_load(path, &w, &h, &ch, 4); // force RGBA
                if (!data) {
                    std::cerr << "Failed to load texture: " << path << std::endl;
                    return 0;
                }
                GLuint tex;
                glGenTextures(1, &tex);
                glBindTexture(GL_TEXTURE_2D, tex);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                stbi_image_free(data);
                return tex;
            };

            spriteTexture = loadTexture("texture/spotlight_7.png");

            std::cout << "OpenGL initialized" << std::endl;
            std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;

            // At the END of initGL(), after everything else:
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGuiIO& io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
            ImGui::StyleColorsDark();
            ImGui_ImplGlfw_InitForOpenGL(window, true);
            ImGui_ImplOpenGL3_Init("#version 330");
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

            GLint blackHoleLoc = glGetUniformLocation(program, "blackHoleIndex");
            glUniform1i(blackHoleLoc, 0);

            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo[drawBuf]);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, (void*)0);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, spriteTexture);
            glUniform1i(glGetUniformLocation(program, "uSprite"), 0);

            glDrawArrays(GL_POINTS, 0, NUM_BODIES);

            glBindVertexArray(0);

            // Add this block just before glfwSwapBuffers(window):
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.75f);
            ImGui::Begin("N-Body Controls", nullptr,
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoCollapse);

            ImGui::Text("Bodies: %d", NUM_BODIES);
            ImGui::Separator();

            ImGui::SliderFloat("G",         &simParams->g,         0.1f, 50.0f);
            ImGui::SliderFloat("DT",        &simParams->dt,        0.01f, 5.0f);
            ImGui::SliderFloat("Theta",     &simParams->theta,     0.1f, 1.5f);
            ImGui::SliderFloat("Softening", &simParams->softening, 1.0f, 500.0f);

            ImGui::Separator();
            const char* distributions[] = { "Disk", "Uniform", "Sphere", "Ring" };
            ImGui::Combo("Distribution", &simParams->distType, distributions, 4);

            ImGui::Separator();
            if (ImGui::Button("Restart Simulation", ImVec2(-1, 30)))
                simParams->restart = true;

            ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
            ImGui::End();

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
        }

        void cleanup()
        {
            glDeleteBuffers(1, &vbo[0]);
            glDeleteBuffers(1, &vbo[1]);
            glDeleteVertexArrays(1, &vao);
            glDeleteTextures(1, &spriteTexture);
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            glDeleteProgram(program);
            glfwDestroyWindow(window);
            glfwTerminate();
        }
};

// ─── main ─────────────────────────────────────────────────────────────────────

/*
class Simulation
{
    public:
    private:
};
*/



int main()
{
    srand(time(NULL));

    SimParams params;
    SimulationRender sim;
    sim.simParams = &params;
    sim.init();


    // ── OpenCL platform/device ──────────────────────────────────────────────
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // 2. Find a platform that supports at least 2.1
    auto platIt = std::find_if(platforms.begin(), platforms.end(),
        [](const cl::Platform& p) {
            std::string version = p.getInfo<CL_PLATFORM_VERSION>();
            // This checks if "2.1" or "3.0" is present
            return version.find("OpenCL 2.1") != std::string::npos || 
                version.find("OpenCL 2.0") != std::string::npos;
        });

    if (platIt == platforms.end()) {
        // If we didn't find 2.1, just take the first one available
        if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found.");
        platIt = platforms.begin();
    }

    // 3. Set it as default so cl::Context::getDefault() works
    cl::Platform::setDefault(*platIt);

    // 4. Verification print
    std::cout << "Platform: " << platIt->getInfo<CL_PLATFORM_NAME>() << "\n";
    std::cout << "Version:  " << platIt->getInfo<CL_PLATFORM_VERSION>() << "\n";

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::string exts = device.getInfo<CL_DEVICE_EXTENSIONS>();
    bool hasGLSharing = exts.find("cl_khr_gl_sharing") != std::string::npos;
    std::cout << "cl_khr_gl_sharng: " << (hasGLSharing ? "YES" : "NO") << std::endl;

    // ── Create OpenCL context sharing with OpenGL ───────────────────────────
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0](),
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
        std::cout << "WARNING: No GL sharing — falling back to readbak" << std::endl;
    }

    cl::CommandQueue queue(context, device);

    // ── Build kernels ───────────────────────────────────────────────────────

    std::string kernelSrc = loadFile("kernels/boundingbox.cl") +
                            loadFile("kernels/buildtree.cl") +
                            loadFile("kernels/suminfo.cl") +
                            loadFile("kernels/sort.cl") +
                            loadFile("kernels/force.cl") +
                            loadFile("kernels/integration.cl") +
                            loadFile("kernels/writepos.cl");

    cl::Program::Sources sources;
    sources.push_back(kernelSrc);

    int numNodes = calcNumNodes();

    std::string buildOptions = std::string("-cl-mad-enable ") +
                                " -D NUMBER_OF_NODES=" + std::to_string(numNodes) +
                                " -D THREADS=" + std::to_string(THREADS) +
                                " -D WARPSIZE=" + std::to_string(WARPSIZE) +
                                " -D NUM_BODIES=" + std::to_string(NUM_BODIES) +
                                " -D DT=" + std::to_string(DT) +
                                " -D SOFTENING=" + std::to_string(SOFTENING) +
                                " -D G=" + std::to_string(G) +
                                " -D THETA=" + std::to_string(THETA);

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

    cl::Buffer buf_vx(context, CL_MEM_READ_WRITE, NUM_BODIES * sizeof(float));
    cl::Buffer buf_vy(context, CL_MEM_READ_WRITE, NUM_BODIES * sizeof(float));
    cl::Buffer buf_vz(context, CL_MEM_READ_WRITE, NUM_BODIES * sizeof(float));

    cl::Buffer buf_accX(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_accY(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));
    cl::Buffer buf_accZ(context, CL_MEM_READ_WRITE, NUM_BODIES*sizeof(float));

    cl::Buffer buf_mass(context, CL_MEM_READ_WRITE, (numNodes+1) * sizeof(float));

    cl::Buffer buf_nodeSize(context, CL_MEM_READ_WRITE, (numNodes+1) * sizeof(float));
    cl::Buffer buf_nodeCount(context, CL_MEM_READ_WRITE, (numNodes+1) * sizeof(int));

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
    //std::vector<int> h_sorted(numNodes + 1, 0);


    cl::Buffer buf_sorted(context, CL_MEM_READ_WRITE, (numNodes+1)*sizeof(int));

    int step = -1;
    int zero = 0;

    cl::Buffer buf_child(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * child.size(), child.data());
    cl::Buffer buf_start(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * start.size(), start.data());
    cl::Buffer buf_step(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &step);

    cl::Buffer buf_errorFlag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &zero);

    cl::Buffer buf_ready(context, CL_MEM_READ_WRITE, sizeof(int) * (numNodes + 1));

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
        std::cout << "Using fallback readback pat" << std::endl;
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
    h_mass[0] = 1e6f;
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
        float angle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float u = (float)rand() / RAND_MAX;
        float radius = sqrt(u) * SPAWN_RANGE;

        float z = ((float)rand() / RAND_MAX - 0.5f) * 0.1f * (SPAWN_RANGE);

        h_x[i] = radius * cos(angle);
        h_y[i] = radius * sin(angle);
        h_z[i] = z;

        float bhMass = h_mass[0]; // 1e21

        float r = sqrt(h_x[i] * h_x[i] + h_y[i] * h_y[i]);
        if (r < 1.0f)
        {
            r = 1.0f;
        }

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

    std::vector<int> h_sorted(numNodes + 1, 0);
    for (int i = 0; i < NUM_BODIES; i++) h_sorted[i] = i;
    queue.enqueueWriteBuffer(buf_sorted, CL_TRUE, 0, (numNodes+1)*sizeof(int), h_sorted.data());

    std::vector<float> h_zeros(NUM_BODIES, 0.0f);
    queue.enqueueWriteBuffer(buf_accX, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_zeros.data());
    queue.enqueueWriteBuffer(buf_accY, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_zeros.data());
    queue.enqueueWriteBuffer(buf_accZ, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_zeros.data());

    glBindBuffer(GL_ARRAY_BUFFER, sim.getMassVBO());
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*NUM_BODIES, h_mass.data());

    // ── Simulate lambda ─────────────────────────────────────────────────────
#pragma region kernelArguments

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

    // summarizeTree
    sumInfoKernel.setArg(0, buf_x);         
    sumInfoKernel.setArg(1, buf_y);
    sumInfoKernel.setArg(2, buf_z);
    sumInfoKernel.setArg(3, buf_mass);
    sumInfoKernel.setArg(4, buf_child);
    sumInfoKernel.setArg(5, buf_nodeCount);
    sumInfoKernel.setArg(6, buf_bottom);
    

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

#pragma endregion    


    int maxComputeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

    std::cout << "Max compute units:" << maxComputeUnits << std::endl;

    cl::NDRange global (NUM_BODIES);
    cl::NDRange safe (maxComputeUnits * THREADS);
    cl::NDRange local (THREADS);

    //std::cout << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;

    int j = 0;


    auto simulateStep = [&]()
    {
        try 
        {
            if (params.restart)
            {
                params.restart = false;
                 queue.finish(); 
                // Re-generate bodies with current params
                // (same init code as original, but use params.g etc.)
                // Re-upload to GPU:
                queue.enqueueWriteBuffer(buf_x,  CL_TRUE, 0, NUM_BODIES*sizeof(float), h_x.data());
                queue.enqueueWriteBuffer(buf_y,  CL_TRUE, 0, NUM_BODIES*sizeof(float), h_y.data());
                queue.enqueueWriteBuffer(buf_z,  CL_TRUE, 0, NUM_BODIES*sizeof(float), h_z.data());
                queue.enqueueWriteBuffer(buf_vx, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vx.data());
                queue.enqueueWriteBuffer(buf_vy, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vy.data());
                queue.enqueueWriteBuffer(buf_vz, CL_TRUE, 0, NUM_BODIES*sizeof(float), h_vz.data());
                queue.enqueueWriteBuffer(buf_mass,CL_TRUE,0, NUM_BODIES*sizeof(float), h_mass.data());
            }
            int zero = 0, one_val = 1;
            cl_int empty_val = -1;
            float mass_neg = -1.0f;
            cl_float zero_f = 0.0f;

            queue.enqueueWriteBuffer(buf_blockCount, CL_FALSE, 0, sizeof(int), &zero);
            queue.enqueueWriteBuffer(buf_maxDepth,   CL_FALSE, 0, sizeof(int), &one_val);
            
            queue.enqueueFillBuffer(buf_accX, zero_f, 0, NUM_BODIES * sizeof(cl_float));
            queue.enqueueFillBuffer(buf_accY, zero_f, 0, NUM_BODIES * sizeof(cl_float));
            queue.enqueueFillBuffer(buf_accZ, zero_f, 0, NUM_BODIES * sizeof(cl_float));
            queue.enqueueFillBuffer(buf_child, empty_val, 0, sizeof(cl_int) * 8 * (numNodes + 1));
            queue.enqueueFillBuffer(buf_start, empty_val, 0, sizeof(cl_int) * (numNodes + 1));
            queue.enqueueFillBuffer(buf_mass, mass_neg, sizeof(float) * NUM_BODIES, sizeof(float) * (numNodes + 1 - NUM_BODIES));


            int writeBuf = current;
            std::vector<cl::Memory> shared = { buf_pos_gl[writeBuf] };
            writePositionsKernel.setArg(3, buf_pos_gl[writeBuf]);


            queue.enqueueNDRangeKernel(boundingBoxKernel, cl::NullRange, global, local);
            queue.enqueueBarrierWithWaitList();
            
            queue.enqueueNDRangeKernel(buildTreeKernel, cl::NullRange, safe, local);
            queue.enqueueBarrierWithWaitList();
            
            queue.enqueueNDRangeKernel(sumInfoKernel, cl::NullRange, local, local);
            queue.enqueueBarrierWithWaitList();

            queue.enqueueNDRangeKernel(sortKernel, cl::NullRange, local, local);
            queue.enqueueBarrierWithWaitList();
            
            queue.enqueueNDRangeKernel(forceKernel, cl::NullRange, global, local);
            queue.enqueueBarrierWithWaitList();
            
            
            queue.enqueueNDRangeKernel(integrateKernel, cl::NullRange, global, local);
            queue.enqueueBarrierWithWaitList();

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