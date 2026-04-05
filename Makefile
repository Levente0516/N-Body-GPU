CXX = g++
CXXFLAGS = 	-I"OpenCL-SDK\external\OpenCL-Headers" \
			-I"glfw\include" \
			-I"glad\include" \
			-I"Vulkan-Headers\include" \
           	-DCL_TARGET_OPENCL_VERSION=300 
			-DCL_HPP_TARGET_OPENCL_VERSION=300

LDFLAGS = -L. -Lglfw/build/src -lglfw3 -lOpenCL -lopengl32 -lgdi32 -lvulkan-1

GLSLC = C:/VulkanSDK/1.3.268.0/Bin/glslc.exe

n-body.exe: n-body.cpp glad\src\glad.c
	$(CXX) n-body.cpp glad\src\glad.c $(CXXFLAGS) $(LDFLAGS) -o n-body.exe

run: n-body.exe
	.\n-body.exe

clean:
	del n-body.exe vert.spv frag.spv