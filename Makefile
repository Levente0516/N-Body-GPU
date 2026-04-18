CXX = g++
CXXFLAGS = 	-I. \
           	-Isrc \
           	-Ilib/OpenCL-SDK/external/OpenCL-Headers \
           	-Ilib/OpenCL-SDK/external/OpenCL-CLHPP/include \
           	-Ilib/glfw/include \
           	-Ilib/glad/include \
           	-Ilib/glm \
			-I"lib" \
            -I"lib\stb"

IMGUI_DIR = lib/imgui
IMGUI_SRC = $(IMGUI_DIR)/imgui.cpp \
            $(IMGUI_DIR)/imgui_draw.cpp \
            $(IMGUI_DIR)/imgui_tables.cpp \
            $(IMGUI_DIR)/imgui_widgets.cpp \
            $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp \
            $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp

CXXFLAGS += -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends

LDFLAGS = -L. -Llib -Llib/glfw/build/src -lglfw3 -lOpenCL -lopengl32 -lgdi32

n-body.exe: src/n-body.cpp lib/glad/src/glad.c $(IMGUI_SRC)
	$(CXX) src/n-body.cpp lib/glad/src/glad.c $(IMGUI_SRC) $(CXXFLAGS) $(LDFLAGS) -o n-body.exe

run: n-body.exe
	.\n-body.exe

clean:
	del n-body.exe vert.spv frag.spv