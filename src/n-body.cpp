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

const int THREADS = 64;
const int WARPSIZE = 64;
const int CAMERAZOOM = 1;
const int MAXDEPTH = 64;

struct SimParams {
    float g = 4.0f;
    float dt = 0.5f;
    float theta = 0.5f;
    float softening = 10.0f;
    int   numBodies = 16384;
    int   numBodiesIdx = 8;  
    float bhMass = 2e9f;
    float spawnRange = 100000.0f;
    float spawnRangeScale = 1.0f;
    int   distType = 0;  // 0=disk, 1=uniform, 2=sphere
    bool  restart = false;
    bool  resetCamera = false;
    bool reParam = false;
    bool secondBlackHole = false;
};

static const char* BODY_COUNT_LABELS[] = {
    "64","128","256","512",
    "1024","2048","4096","8192","16384","32768","65536"
};
static const int BODY_COUNT_VALUES[] = {
    64,128,256,512,1024,2048,4096,8192,16384,32768,65536
};

int calcNumNodes(SimParams& p)
{
    int numNodes = p.numBodies * 2;
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
        float camZoom = 0.0f;
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
        float camYaw = 0.0f;
        float camPitch = glm::radians(65.0f);

        GLFWwindow *window = nullptr;
        GLuint vbo[2];
        GLuint vao = 0;
        GLuint program = 0;
        GLuint spriteTexture = 0;
        GLuint massVBO;

        SimParams* simParams = nullptr;
        SimParams defaultParams;
        int* current   = nullptr; 

        void init(SimParams* p, int* cur)
        {
            simParams = p;
            defaultParams = *p;
            current = cur;
            camZoom = (float)simParams->spawnRange * CAMERAZOOM * 2.0f;
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

        void resetView()
        {
            camZoom  = (float)simParams->spawnRange * 2.5f;
            camYaw   = 0.0f;
            camPitch = glm::radians(65.0f);
            camTarget = glm::vec3(0.0f);
        }

    private:
        void initWindow()
        {
            if (!glfwInit())
                exit(1);

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
                } 
            });

            glfwSetScrollCallback(window, [](GLFWwindow *w, double, double yOffset)
            {
                if (ImGui::GetIO().WantCaptureMouse)
                {
                    return;
                } 
                auto *s = reinterpret_cast<SimulationRender*>(glfwGetWindowUserPointer(w));

                s->camZoom = std::max(s->camZoom * ((yOffset > 0) ? 0.9f : 1.1f), 100.0f);

                /*
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
                */

                // std::cout << "Zoo: " << sim->camZoom << "\nCamX: " << sim->camX << "\nCamY: " << sim->camY << std::endl;
            });

            glfwSetMouseButtonCallback(window, [](GLFWwindow *w, int button, int action, int)
            {
                if (ImGui::GetIO().WantCaptureMouse)
                {
                    return;
                } 
                auto* s = reinterpret_cast<SimulationRender*>(glfwGetWindowUserPointer(w));
                if (button == GLFW_MOUSE_BUTTON_LEFT)
                {
                    s->dragging = (action == GLFW_PRESS);
                    if (s->dragging)
                    {
                        glfwGetCursorPos(w, &s->dragStartX, &s->dragStartY);
                    } 
                }
                /*
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
                */
            });

            glfwSetCursorPosCallback(window, [](GLFWwindow *w, double xpos, double ypos)
            {
                if (ImGui::GetIO().WantCaptureMouse)
                {
                    return;
                } 
                auto* s = reinterpret_cast<SimulationRender*>(glfwGetWindowUserPointer(w));
                
                if (!s->dragging) return;

                double dx = xpos - s->dragStartX;
                double dy = ypos - s->dragStartY;
                s->dragStartX = xpos; s->dragStartY = ypos;

                bool ctrl = (glfwGetKey(w, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);

                if (ctrl) 
                {
                    s->camYaw   += (float)dx * 0.005f;
                    s->camPitch += (float)dy * 0.005f;
                    s->camPitch  = glm::clamp(s->camPitch, glm::radians(-89.f), glm::radians(89.f));
                }
                else
                {
                    glm::vec3 dir;
                    dir.x = std::cos(s->camPitch) * std::sin(s->camYaw);
                    dir.y = std::sin(s->camPitch);
                    dir.z = -std::cos(s->camPitch) * std::cos(s->camYaw);
                    glm::vec3 right   = glm::normalize(glm::cross(dir, glm::vec3(0,1,0)));
                    glm::vec3 flatFwd = glm::normalize(glm::cross(glm::vec3(0,1,0), right));
                    float speed = s->camZoom * 0.0010f;
                    s->camTarget -= right   * (float)dx * speed;
                    s->camTarget += flatFwd * (float)dy * speed;
                }

                /*
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
                */
            });
        }

        void initGL()
        {
            glGenBuffers(2, vbo);

            for (int i = 0; i < 2; i++)
            {
                glBindBuffer(GL_ARRAY_BUFFER, vbo[i]);
                glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * simParams->numBodies, nullptr, GL_STREAM_DRAW);
            }

            glGenBuffers(1, &massVBO);
            glBindBuffer(GL_ARRAY_BUFFER, massVBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * simParams->numBodies, nullptr, GL_STREAM_DRAW);

            glGenVertexArrays(1, &vao);
            glBindVertexArray(vao);

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
            if (simParams->resetCamera) 
            {
                simParams->resetCamera = false;
                resetView();
            }

            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            glViewport(0, 0, w, h);
            glClearColor(0, 0, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            float aspect = (float)w / (float)h;

            glm::mat4 projection = glm::perspective(
                glm::radians(45.0f), // Field of view
                aspect,
                100.0f,              // Near plane
                simParams->spawnRange * 100.0f // Far plane
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
            glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
            glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_FALSE, glm::value_ptr(view));

            int drawBuf = 1 - *current;

            int halfPos = simParams->numBodies / 2;

            glUniform1i(glGetUniformLocation(program, "blackHoleIndex1"), 0);
            if (simParams->secondBlackHole)
            {
                glUniform1i(glGetUniformLocation(program, "blackHoleIndex2"), halfPos);
            }
            else
            {
                 glUniform1i(glGetUniformLocation(program, "blackHoleIndex2"), -1);
            }
            glUniform1f(glGetUniformLocation(program, "uSpawnRange"), (float)simParams->spawnRange);

            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo[drawBuf]);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, (void*)0);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, spriteTexture);
            glUniform1i(glGetUniformLocation(program, "uSprite"), 0);

            glDrawArrays(GL_POINTS, 0, simParams->numBodies);

            glBindVertexArray(0);

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(400, 0), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.75f);
            ImGui::Begin("N-Body Controls", nullptr,
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoResize);

            ImGui::Text("Bodies: %d", simParams->numBodies);
            ImGui::TextDisabled("Drag = move around  Ctrl+Drag = rotate  Scroll = zoom");
            ImGui::Separator();
            ImGui::TextDisabled("Important Note: Chaning a value");
            ImGui::TextDisabled("results in the restart of the simulation");
            ImGui::Separator();
            
            if (ImGui::CollapsingHeader("Simulation variables", ImGuiTreeNodeFlags_NoAutoOpenOnLog))
            {
                ImGui::PushID("G");
                ImGui::SliderFloat("G", &simParams->g, 0.1f, 50.0f);
                if (ImGui::IsItemDeactivatedAfterEdit())
                {
                    simParams->reParam = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset"))
                {
                    simParams->g = defaultParams.g;
                    simParams->reParam = true;
                }
                ImGui::PopID();
    
                ImGui::PushID("DT");
                ImGui::SliderFloat("DT",&simParams->dt,0.01f, 5.0f);
                if (ImGui::IsItemDeactivatedAfterEdit())
                {
                    simParams->reParam = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset"))
                {
                    simParams->dt = defaultParams.dt;
                    simParams->reParam = true;
                }
                ImGui::PopID();
    
                ImGui::PushID("Theta");
                ImGui::SliderFloat("Theta",     &simParams->theta,     0.1f, 1.5f);
                if (ImGui::IsItemDeactivatedAfterEdit())
                {
                    simParams->reParam = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset"))
                {
                    simParams->theta = defaultParams.theta;
                    simParams->reParam = true;
                }
                ImGui::PopID();
    
                ImGui::PushID("Softening");
                ImGui::SliderFloat("Softening", &simParams->softening, 0.01f, 5000.0f);
                if (ImGui::IsItemDeactivatedAfterEdit())
                {
                    simParams->reParam = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset"))
                {
                    simParams->softening = defaultParams.softening;
                    simParams->reParam = true;
                }
                ImGui::PopID();

                if (ImGui::Button("Reset everything", {-1, 0}))
                {
                    simParams->softening = defaultParams.softening;
                    simParams->theta = defaultParams.theta;
                    simParams->dt = defaultParams.dt;
                    simParams->g = defaultParams.g;
                }
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Initial Conditions", ImGuiTreeNodeFlags_NoAutoOpenOnLog)) {

                // Body count — index maps to actual power-of-two value
                if (ImGui::Combo("Body count", &simParams->numBodiesIdx, BODY_COUNT_LABELS, 11))
                {
                    simParams->numBodies = BODY_COUNT_VALUES[simParams->numBodiesIdx];
                    simParams->restart = true;
                }

                // Spawn range scale
                ImGui::PushID("Spawn range scale");
                ImGui::SliderFloat("S.R. scale", &simParams->spawnRangeScale, 0.1f, 10.0f);
                if (ImGui::IsItemDeactivatedAfterEdit())
                {
                    simParams->restart = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset"))
                {
                    simParams->spawnRangeScale= defaultParams.spawnRangeScale;
                    simParams->restart = true;
                }
                ImGui::PopID();

                ImGui::PushID("BH Mass");
                ImGui::SliderFloat("BH Mass",  &simParams->bhMass,    1e6f,   1e12f);
                if (ImGui::IsItemDeactivatedAfterEdit())
                {
                    simParams->restart = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset"))
                {
                    simParams->bhMass = defaultParams.bhMass;
                    simParams->restart = true;
                }
                ImGui::PopID();
                    
                // Distribution
                const char* distNames[] = {
                    "Disk","Uniform","Sphere","Ring","Gaussian","Plummer","NFW (dark matter)", "Two galaxy"
                };
                if (ImGui::Combo("Distribution", &simParams->distType, distNames, 8))
                {
                    simParams->restart = true;

                    if (simParams->distType == 7)
                    {
                        simParams->secondBlackHole = true;
                    }
                    else
                    {
                        simParams->secondBlackHole = false;
                    }
                }
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_NoAutoOpenOnLog))
            {
                ImGui::Text("Yaw: %.2f  Pitch: %.2f  Zoom: %.0f", camYaw, glm::degrees(camPitch), camZoom);
                if (ImGui::Button("Reset Camera View", {-1, 0}))
                {
                    simParams->resetCamera = true;
                }
            }

            ImGui::Separator();
            if (ImGui::Button("Restart Simulation", ImVec2(-1, 35)))
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

class Simulation
{
    public:

        int numNodes = 0;
        int current = 0;

        std::vector<float> h_x, h_y, h_z;
        std::vector<float> h_vx, h_vy, h_vz;
        std::vector<float> h_mass;

        void init(SimulationRender& render, SimParams& p)
        {
            numNodes = calcNumNodes(p);
            initOpenGl(render);
            buildKernels(p);
            allocateBuffers(p);
            generateBodies(p);
            uploadBodies(p);
            setKernelArgs(p);
            initNDRanges(p);
        }

        void step(SimulationRender& render, SimParams& p)
        {
            if (p.restart) 
            {
                numNodes = calcNumNodes(p);
                p.restart = false;
                queue.finish();
                buildKernels(p);
                allocateBuffers(p);
                generateBodies(p);
                uploadBodies(p);
                initNDRanges(p);
                setKernelArgs(p);   

                glBindBuffer(GL_ARRAY_BUFFER, render.getMassVBO());
                glBufferData(GL_ARRAY_BUFFER, sizeof(float)*p.numBodies, nullptr, GL_STREAM_DRAW);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*p.numBodies, h_mass.data());
            }

            if (p.reParam)
            {
                p.reParam = false;
                queue.finish();
                setKernelArgs(p);
            }

            try 
            {
                int zero = 0, one_val = 1;
                cl_int empty_val = -1;
                float mass_neg = -1.0f;
                cl_float zero_f = 0.0f;

                
                queue.enqueueWriteBuffer(buf_blockCount, CL_FALSE, 0, sizeof(int), &zero);
                queue.enqueueWriteBuffer(buf_maxDepth,   CL_FALSE, 0, sizeof(int), &one_val);
                
                queue.enqueueFillBuffer(buf_accX, zero_f, 0, p.numBodies * sizeof(cl_float));
                queue.enqueueFillBuffer(buf_accY, zero_f, 0, p.numBodies * sizeof(cl_float));
                queue.enqueueFillBuffer(buf_accZ, zero_f, 0, p.numBodies * sizeof(cl_float));
                queue.enqueueFillBuffer(buf_child, empty_val, 0, sizeof(cl_int) * 8 * (numNodes + 1));
                queue.enqueueFillBuffer(buf_start, empty_val, 0, sizeof(cl_int) * (numNodes + 1));
                queue.enqueueFillBuffer(buf_mass, mass_neg, sizeof(float) * p.numBodies, sizeof(float) * (numNodes + 1 - p.numBodies));
                


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

                glFinish();
                queue.enqueueAcquireGLObjects(&shared);
                queue.enqueueNDRangeKernel(writePositionsKernel, cl::NullRange, global, local);
                queue.enqueueReleaseGLObjects(&shared);

                queue.finish();

                current = 1 - current;
            }
            catch (cl::Error& e) {
                std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
                exit(1);
            }
        }

    private:

        cl::Context      context;
        cl::CommandQueue queue;
        cl::Program      program;
        cl::Device       device;

        cl::Kernel boundingBoxKernel, buildTreeKernel, sumInfoKernel;
        cl::Kernel sortKernel, forceKernel, integrateKernel, writePositionsKernel;

        cl::Buffer buf_x, buf_y, buf_z;
        cl::Buffer buf_vx, buf_vy, buf_vz;
        cl::Buffer buf_accX, buf_accY, buf_accZ;
        cl::Buffer buf_mass, buf_nodeSize, buf_nodeCount;
        cl::Buffer buf_child, buf_start, buf_sorted;
        cl::Buffer buf_blockCount, buf_bottom, buf_numNodes, buf_maxDepth, buf_step;
        cl::Buffer buf_minX, buf_minY, buf_minZ, buf_maxX, buf_maxY, buf_maxZ;
        cl::BufferGL buf_pos_gl[2];

        cl::NDRange global, safe, local;

        void initOpenGl(SimulationRender& render)
        {
            std::vector<cl::Platform> platforms;

            cl::Platform::get(&platforms);
            
            if (platforms.empty())
            {
                throw std::runtime_error("No OpenCL platforms");
            }

            auto it = std::find_if(platforms.begin(), platforms.end(), [](const cl::Platform& p)
            {
                std::string v = p.getInfo<CL_PLATFORM_VERSION>();

                return v.find("2.1") != std::string::npos || v.find("2.0") != std::string::npos;
            });

            if (it == platforms.end())
            {
                it = platforms.begin();
            } 
            
            cl::Platform platform = *it;

            std::vector<cl::Device> devices;
            
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            
            if (devices.empty())
            {
                throw std::runtime_error("No GPU devices");
            } 
            
            device = devices[0];

            std::string exts = device.getInfo<CL_DEVICE_EXTENSIONS>();
            
            bool hasSharing  = exts.find("cl_khr_gl_sharing") != std::string::npos;

            std::cout << "Platform: "  << platform.getInfo<CL_PLATFORM_NAME>()    << "\n";
            std::cout << "Device:   "  << device.getInfo<CL_DEVICE_NAME>()        << "\n";
            std::cout << "GL sharing: "<< (hasSharing ? "YES" : "NO")             << "\n";
            std::cout << "Compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";

            cl_context_properties props[] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
                CL_GL_CONTEXT_KHR,   (cl_context_properties)wglGetCurrentContext(),
                CL_WGL_HDC_KHR,      (cl_context_properties)wglGetCurrentDC(),
                0
            };
            
            context = hasSharing ? cl::Context(device, props) : cl::Context(device);
            
            queue   = cl::CommandQueue(context, device);

            if (hasSharing)
            {
                try 
                {
                    for (int i = 0; i < 2; i++)
                    {
                        buf_pos_gl[i] = cl::BufferGL(context, CL_MEM_READ_WRITE, render.getVBO(i));
                    }
                    
                    std::cout << "GL interop: active\n";
                } 
                catch (...) 
                {
                    std::cerr << "BufferGL failed\n"; exit(1);
                }
            } 
            else
            {
                std::cerr << "No GL sharing — cannot continue\n"; exit(1);
            }
        }

        static std::string fstr(float v)
        {
            std::ostringstream ss;
            ss.imbue(std::locale::classic());
            ss << v;
            return ss.str();
        }

        void buildKernels(const SimParams& p)
        {
            std::string kernelSrc = 
                loadFile("kernels/boundingbox.cl") +
                loadFile("kernels/buildtree.cl") +
                loadFile("kernels/suminfo.cl") +
                loadFile("kernels/sort.cl") +
                loadFile("kernels/force.cl") +
                loadFile("kernels/integration.cl") +
                loadFile("kernels/writepos.cl");

            cl::Program::Sources sources;
            sources.push_back(kernelSrc);

            std::string opts =
                std::string("-cl-mad-enable ") +
                std::string("-cl-fast-relaxed-math ") +
                " -D NUMBER_OF_NODES=" + std::to_string(numNodes) +
                " -D THREADS="         + std::to_string(THREADS)  +
                " -D WARPSIZE="        + std::to_string(WARPSIZE) +
                " -D NUM_BODIES="      + std::to_string(p.numBodies);
            
            program = cl::Program(context, sources);

            try 
            { 
                program.build({device}, opts.c_str());
            }
            catch (const cl::Error&)
            {
                std::cerr << "Build error:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
                exit(1);
            }

            boundingBoxKernel = cl::Kernel(program, "boundingBoxKernel");
            buildTreeKernel = cl::Kernel(program, "buildTreeKernel");
            sumInfoKernel = cl::Kernel(program, "summarizeTreeKernel");
            sortKernel = cl::Kernel(program, "sortKernel");
            forceKernel = cl::Kernel(program, "forceKernel");
            integrateKernel = cl::Kernel(program, "integrateKernel");
            writePositionsKernel = cl::Kernel(program, "writePositionsInterleaved");

            std::cout << "Kernels compiled\n";
        }

        void allocateBuffers(SimParams& p)
        {
            buf_x         = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (numNodes+1));
            buf_y         = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (numNodes+1));
            buf_z         = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (numNodes+1));
            buf_mass      = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (numNodes+1));
            buf_nodeSize  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (numNodes+1));
            buf_nodeCount = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * (numNodes+1));
            buf_vx        = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (p.numBodies));
            buf_vy        = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (p.numBodies));
            buf_vz        = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (p.numBodies));
            buf_accX      = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (p.numBodies));
            buf_accY      = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (p.numBodies));
            buf_accZ      = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (p.numBodies));
            buf_sorted    = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * (numNodes+1));
            buf_minX      = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (p.numBodies));
            buf_minY      = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (p.numBodies));
            buf_minZ      = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (p.numBodies));
            buf_maxX      = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (p.numBodies));
            buf_maxY      = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (p.numBodies));
            buf_maxZ      = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (p.numBodies));

            int v0 = 0;
            int v1 = 1; 
            int vm1 = -1;
            std::vector<int> child(8 * (numNodes + 1), 0);
            std::vector<int> start((numNodes + 1), 0);
            buf_child     = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * (8*(numNodes+1)), child.data());
            buf_start     = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * (numNodes+1), start.data());
            buf_blockCount = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &v0);
            buf_bottom     = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &v0);
            buf_numNodes   = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &numNodes);
            buf_maxDepth   = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &v1);
            buf_step       = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &vm1);
        }

        void setKernelArgs(SimParams& p)
        {
            boundingBoxKernel.setArg( 0, buf_step);
            boundingBoxKernel.setArg( 1, buf_x);
            boundingBoxKernel.setArg( 2, buf_y);
            boundingBoxKernel.setArg( 3, buf_z);
            boundingBoxKernel.setArg( 4, buf_blockCount);
            boundingBoxKernel.setArg( 5, buf_bottom);
            boundingBoxKernel.setArg( 6, buf_mass);
            boundingBoxKernel.setArg( 7, buf_numNodes);
            boundingBoxKernel.setArg( 8, buf_minX);
            boundingBoxKernel.setArg( 9, buf_minY);
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
            buildTreeKernel.setArg(9, p.softening);

            sumInfoKernel.setArg(0, buf_x);
            sumInfoKernel.setArg(1, buf_y);
            sumInfoKernel.setArg(2, buf_z);
            sumInfoKernel.setArg(3, buf_mass);
            sumInfoKernel.setArg(4, buf_child);
            sumInfoKernel.setArg(5, buf_nodeCount);
            sumInfoKernel.setArg(6, buf_bottom);

            sortKernel.setArg(0, buf_child);
            sortKernel.setArg(1, buf_nodeCount);
            sortKernel.setArg(2, buf_start);
            sortKernel.setArg(3, buf_sorted);
            sortKernel.setArg(4, buf_bottom);
            sortKernel.setArg(5, buf_numNodes);

            forceKernel.setArg( 0, buf_x);
            forceKernel.setArg( 1, buf_y);
            forceKernel.setArg( 2, buf_z);
            forceKernel.setArg( 3, buf_mass);
            forceKernel.setArg( 4, buf_child);
            forceKernel.setArg( 5, buf_nodeSize);
            forceKernel.setArg( 6, buf_sorted);
            forceKernel.setArg( 7, buf_accX);
            forceKernel.setArg( 8, buf_accY);
            forceKernel.setArg( 9, buf_accZ);
            forceKernel.setArg(10, buf_numNodes);
            forceKernel.setArg(11, p.softening);
            forceKernel.setArg(12, p.g);
            forceKernel.setArg(13, p.theta);

            integrateKernel.setArg(0, buf_x);
            integrateKernel.setArg(1, buf_y);
            integrateKernel.setArg(2, buf_z);
            integrateKernel.setArg(3, buf_vx);
            integrateKernel.setArg(4, buf_vy);
            integrateKernel.setArg(5, buf_vz);
            integrateKernel.setArg(6, buf_accX);
            integrateKernel.setArg(7, buf_accY);
            integrateKernel.setArg(8, buf_accZ);
            integrateKernel.setArg(9, buf_mass);
            integrateKernel.setArg(10, p.dt);

            writePositionsKernel.setArg(0, buf_x);
            writePositionsKernel.setArg(1, buf_y);
            writePositionsKernel.setArg(2, buf_z);
        }
        
        void initNDRanges(SimParams& p)
        {
            int maxComputeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            global = cl::NDRange(p.numBodies);
            safe = cl::NDRange(/*maxComputeUnits * */THREADS);
            local = cl::NDRange(THREADS);
        }

        void generateBodies(const SimParams& p)
        {
            h_x.resize(p.numBodies); 
            h_y.resize(p.numBodies); 
            h_z.resize(p.numBodies);
            h_vx.resize(p.numBodies); 
            h_vy.resize(p.numBodies);
            h_vz.resize(p.numBodies);
            h_mass.resize(p.numBodies);

            h_x[0] = h_y[0] = h_z[0] = 0.0f;
            h_vx[0] = h_vy[0] = h_vz[0] = 0.0f;
            h_mass[0] = p.bhMass;

            for (int i = 1; i < p.numBodies; i++) 
            {
                float f = std::pow((float)rand()/RAND_MAX, 2.0f);
                h_mass[i] = 5000.0f + f * 50000.0f;
            }

            switch (p.distType) 
            {
                case 0: generateDisk(p);    break;
                case 1: generateUniform(p); break;
                case 2: generateSphere(p);  break;
                case 3: generateRing(p);     break;
                case 4: generateGaussian(p); break;
                case 5: generatePlummer(p);  break;
                case 6: generateNFW(p);      break;
                case 7: generateTwoDisk(p); break;
                default: generateDisk(p);   break;
            }

            //std::cout << "Bodies generated (distribution = " << p.distType << ")\n";
        }

        float circVel(float r, const SimParams& p) {
            float r2    = r * r;
            float dist2 = r2 + p.softening * p.softening;
            float dist  = std::sqrt(dist2);
            return std::sqrt(p.g * p.bhMass * r2 / (dist2 * dist));
        }

        void generateDisk(const SimParams& p)
        {
            float safeT   = 50.0f * p.dt;
            float r_min   = std::cbrt((safeT / (2.0f*(float)M_PI)) *
                                    (safeT / (2.0f*(float)M_PI)) *
                                    p.g * p.bhMass);
            float r_max   = (float)p.spawnRange;
            float eps = p.softening;
            for (int i = 1; i < p.numBodies; i++) 
            {
                float angle  = ((float)rand()/RAND_MAX) * 2.0f * (float)M_PI;
                float u      = (float)rand()/RAND_MAX;
                float radius = r_min + (r_max - r_min) * std::sqrt(u);

                h_x[i] = radius * std::cos(angle);
                h_y[i] = radius * std::sin(angle);
                h_z[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.05f * r_max;

                float r2 = h_x[i]*h_x[i] + h_y[i]*h_y[i];
                float r  = std::max(std::sqrt(r2), r_min);
                
                float dist2 = r2 + eps*eps;
                float dist  = std::sqrt(dist2);
                float v     = std::sqrt(p.g * p.bhMass * r2 / (dist2 * dist));

                h_vx[i] =  std::sin(angle) * v;
                h_vy[i] = -std::cos(angle) * v;
                h_vz[i] = 0.0f;
            }
        }

        void generateTwoDisk(const SimParams& p)
        {
            int   half   = p.numBodies / 2;
            float offSet = p.spawnRange * 0.5f;
            float tilt   = glm::radians(30.0f);
            float c = std::cos(tilt), s = std::sin(tilt);

            float safeT = 50.0f * p.dt;
            float r_min = std::cbrt((safeT/(2.0f*(float)M_PI)) *
                                    (safeT/(2.0f*(float)M_PI)) * p.g * p.bhMass);
            float r_max = offSet * 0.8f;
            float eps   = p.softening;

            h_x[0] = -offSet; h_y[0] = 0.0f; h_z[0] = 0.0f;
            h_vx[0] = h_vy[0] = h_vz[0] = 0.0f;
            h_mass[0] = p.bhMass;

            h_x[half] = +offSet; h_y[half] = 0.0f; h_z[half] = 0.0f;
            h_vx[half] = h_vy[half] = h_vz[half] = 0.0f;
            h_mass[half] = p.bhMass;

            for (int i = 1; i < half; i++)
            {
                float angle  = ((float)rand()/RAND_MAX) * 2.0f * (float)M_PI;
                float radius = r_min + (r_max - r_min) * std::sqrt((float)rand()/RAND_MAX);

                float lx = radius * std::cos(angle);
                float ly = radius * std::sin(angle);
                float lz = ((float)rand()/RAND_MAX - 0.5f) * 0.05f * r_max;
                h_x[i] = lx - offSet;
                h_y[i] = ly;
                h_z[i] = lz;

                float r2    = lx*lx + ly*ly;
                float r     = std::max(std::sqrt(r2), r_min);
                float dist2 = r2 + eps*eps;
                float dist  = std::sqrt(dist2);
                float v     = std::sqrt(p.g * p.bhMass * r2 / (dist2 * dist));
                h_vx[i] =  std::sin(angle) * v;
                h_vy[i] = -std::cos(angle) * v;
                h_vz[i] = 0.0f;
            }

            for (int i = half + 1; i < p.numBodies; i++)
            {
                float angle  = ((float)rand()/RAND_MAX) * 2.0f * (float)M_PI;
                float radius = r_min + (r_max - r_min) * std::sqrt((float)rand()/RAND_MAX);

                float lx = radius * std::cos(angle);
                float ly = radius * std::sin(angle);
                float lz = ((float)rand()/RAND_MAX - 0.5f) * 0.05f * r_max;

                float r2    = lx*lx + ly*ly;
                float r     = std::max(std::sqrt(r2), r_min);
                float dist2 = r2 + eps*eps;
                float dist  = std::sqrt(dist2);
                float v     = std::sqrt(p.g * p.bhMass * r2 / (dist2 * dist));
                float vx = std::sin(angle) * v;
                float vy = -std::cos(angle) * v;
                float vz = 0.0f;

                h_x[i] = lx + offSet;
                h_y[i] = ly*c - lz*s;
                h_z[i] = ly*s + lz*c;

                h_vx[i] = vx;
                h_vy[i] = vy*c - vz*s;
                h_vz[i] = vy*s + vz*c;
            }
        }

        void generateUniform(const SimParams& p)
        {
            for (int i = 1; i < p.numBodies; i++) {
                h_x[i] = ((float)rand()/RAND_MAX - 0.5f) * p.spawnRange;
                h_y[i] = ((float)rand()/RAND_MAX - 0.5f) * p.spawnRange;
                h_z[i] = ((float)rand()/RAND_MAX - 0.5f) * p.spawnRange;
                h_vx[i] = h_vy[i] = h_vz[i] = 0.0f;
            }
        }

        void generateSphere(const SimParams& p)
        {
            for (int i = 1; i < p.numBodies; i++) {
                float u = (float)rand()/RAND_MAX;
                float v = (float)rand()/RAND_MAX;
                float theta = 2.0f * (float)M_PI * u;
                float phi   = std::acos(2.0f * v - 1.0f);
                float r     = std::cbrt((float)rand()/RAND_MAX) * p.spawnRange;
                h_x[i] = r * std::sin(phi) * std::cos(theta);
                h_y[i] = r * std::sin(phi) * std::sin(theta);
                h_z[i] = r * std::cos(phi);

                float rl = std::max(std::sqrt(h_x[i]*h_x[i] + h_y[i]*h_y[i]), 1.0f);
                float vt = std::sqrt(p.g * p.bhMass / rl) * 0.7f;
                h_vx[i] =  h_y[i] / rl * vt;
                h_vy[i] = -h_x[i] / rl * vt;
                h_vz[i] = 0.0f;
            }
        }

        void generateRing(const SimParams& p)
        {
            float r_inner = (float)p.spawnRange * 0.35f;
            float r_outer = (float)p.spawnRange * 0.65f;
            float h_thick = (float)p.spawnRange * 0.02f;

            for (int i = 1; i < p.numBodies; i++) {
                float angle = ((float)rand()/RAND_MAX) * 2.0f*(float)M_PI;
                float r     = r_inner + (r_outer - r_inner) * (float)rand()/RAND_MAX;
                h_x[i] = r * std::cos(angle);
                h_y[i] = r * std::sin(angle);
                h_z[i] = ((float)rand()/RAND_MAX - 0.5f) * h_thick;
                float v = circVel(r, p);
                h_vx[i] =  std::sin(angle) * v;
                h_vy[i] = -std::cos(angle) * v;
                h_vz[i] = 0.0f;
            }
        }

        float randNorm() 
        {
            float u1 = std::max((float)rand()/RAND_MAX, 1e-7f);
            float u2 = (float)rand()/RAND_MAX;
            return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f*(float)M_PI * u2);
        }

        void generateGaussian(const SimParams& p)
        {
            float sigma = (float)p.spawnRange * 0.3f;
            for (int i = 1; i < p.numBodies; i++) {
                h_x[i] = randNorm() * sigma;
                h_y[i] = randNorm() * sigma * 0.25f;  // flattened disk shape
                h_z[i] = randNorm() * sigma;
                float rl = std::max(std::sqrt(h_x[i]*h_x[i]+h_z[i]*h_z[i]), 1.0f);
                float v  = circVel(rl, p) * 0.8f;
                // Rotate around Y axis
                h_vx[i] =  h_z[i]/rl * v;
                h_vy[i] = 0.0f;
                h_vz[i] = -h_x[i]/rl * v;
            }
        }

        void generatePlummer(const SimParams& p)
        {
            float a = (float)p.spawnRange * 0.15f;  // Plummer radius (core size)
            for (int i = 1; i < p.numBodies; i++) {
                // Inverse CDF sampling: r = a / sqrt(u^(-2/3) - 1)
                float u = (float)rand()/RAND_MAX;
                float r = a / std::sqrt(std::pow(u, -2.0f/3.0f) - 1.0f);
                r = std::min(r, (float)p.spawnRange * 2.0f);  // cap outliers

                // Uniform direction on sphere
                float cosTheta = 2.0f*(float)rand()/RAND_MAX - 1.0f;
                float phi      = 2.0f*(float)M_PI*(float)rand()/RAND_MAX;
                float sinTheta = std::sqrt(1.0f - cosTheta*cosTheta);
                h_x[i] = r * sinTheta * std::cos(phi);
                h_y[i] = r * cosTheta * 0.3f;   // flatten into disk
                h_z[i] = r * sinTheta * std::sin(phi);

                float rl = std::max(std::sqrt(h_x[i]*h_x[i]+h_z[i]*h_z[i]), 1.0f);
                float v  = circVel(rl, p) * 0.75f;
                h_vx[i] =  h_z[i]/rl * v;
                h_vy[i] = 0.0f;
                h_vz[i] = -h_x[i]/rl * v;
            }
        }

        void generateNFW(const SimParams& p)
        {
            float rs    = (float)p.spawnRange * 0.2f;   // scale radius
            float r_max = (float)p.spawnRange;
            // Peak of ρ(r)*r² (for shell-volume sampling) occurs at r = rs
            // We sample ρ(r)*r² and use rejection
            float peakVal = rs * rs / (rs * (1.0f + 1.0f) * (1.0f + 1.0f));

            for (int i = 1; i < p.numBodies; i++) {
                float r;
                // Rejection sampling loop — typically accepts in <3 trials
                for (;;) {
                    r = ((float)rand()/RAND_MAX) * r_max;
                    float x   = r / rs;
                    float pdf = r*r / (x * (1.0f + x) * (1.0f + x));
                    float y   = ((float)rand()/RAND_MAX) * peakVal * r_max * r_max;
                    if (y < pdf) break;
                }

                float cosTheta = 2.0f*(float)rand()/RAND_MAX - 1.0f;
                float phi      = 2.0f*(float)M_PI*(float)rand()/RAND_MAX;
                float sinTheta = std::sqrt(1.0f - cosTheta*cosTheta);
                h_x[i] = r * sinTheta * std::cos(phi);
                h_y[i] = r * cosTheta * 0.15f;  // very flat disk halo
                h_z[i] = r * sinTheta * std::sin(phi);

                float rl = std::max(std::sqrt(h_x[i]*h_x[i]+h_z[i]*h_z[i]), 1.0f);
                float v  = circVel(rl, p);
                h_vx[i] =  h_z[i]/rl * v;
                h_vy[i] = 0.0f;
                h_vz[i] = -h_x[i]/rl * v;
            }
        }

        void uploadBodies(SimParams& p)
        {
            auto up = [&](cl::Buffer& b, const std::vector<float>& v) 
            {
                queue.enqueueWriteBuffer(b, CL_TRUE, 0, p.numBodies*sizeof(float), v.data());
            };

            up(buf_x, h_x); 
            up(buf_y, h_y); 
            up(buf_z, h_z);
            up(buf_vx, h_vx); 
            up(buf_vy, h_vy); 
            up(buf_vz, h_vz);
            up(buf_mass, h_mass);

            std::vector<int> sorted(numNodes+1, 0);
            for (int i = 0; i < p.numBodies; i++)
            {
                sorted[i] = 0;
            }

            queue.enqueueWriteBuffer(buf_sorted, CL_TRUE, 0, (numNodes+1)*sizeof(int), sorted.data());

            std::vector<float> zeros(p.numBodies, 0.0f);
            queue.enqueueWriteBuffer(buf_accX, CL_TRUE, 0, p.numBodies*sizeof(float), zeros.data());
            queue.enqueueWriteBuffer(buf_accY, CL_TRUE, 0, p.numBodies*sizeof(float), zeros.data());
            queue.enqueueWriteBuffer(buf_accZ, CL_TRUE, 0, p.numBodies*sizeof(float), zeros.data());

            queue.finish();
        }
};


int main()
{
    srand((unsigned)time(nullptr));

    SimParams params;
    SimulationRender render;
    Simulation sim;

    render.init(&params, &sim.current);
    sim.init(render, params);

    glBindBuffer(GL_ARRAY_BUFFER, render.getMassVBO());
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*params.numBodies, sim.h_mass.data());

    render.loop([&]()
    { 
        sim.step(render, params); 
    });
    
    return 0;
}