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

LDFLAGS = -L. -Llib -Llib/glfw/build/src -lglfw3 -lOpenCL -lopengl32 -lgdi32

n-body.exe: src/n-body.cpp lib/glad/src/glad.c
	$(CXX) src/n-body.cpp lib/glad/src/glad.c $(CXXFLAGS) $(LDFLAGS) -o n-body.exe

run: n-body.exe
	.\n-body.exe

clean:
	del n-body.exe vert.spv frag.spv