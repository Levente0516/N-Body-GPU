CXX = g++
CXXFLAGS = -I"OpenCL-SDK\external\OpenCL-Headers" -I"glfw\include" -I"Vulkan-Headers\include" \
           -DCL_TARGET_OPENCL_VERSION=300 -DCL_HPP_TARGET_OPENCL_VERSION=300

LDFLAGS = -L. -Lglfw/build/src -lglfw3 -lOpenCL -lopengl32 -lgdi32 -lvulkan-1
#          ^^ -L. picks up libvulkan-1.a from current directory
#          Note: removed the VulkanSDK/Lib path since that only has the MSVC .lib

GLSLC = C:/VulkanSDK/1.3.268.0/Bin/glslc.exe

all: shaders n-body.exe

shaders:
	$(GLSLC) shader.vert -o vert.spv
	$(GLSLC) shader.frag -o frag.spv

n-body.exe: n-body.cpp
	$(CXX) n-body.cpp $(CXXFLAGS) $(LDFLAGS) -o n-body.exe

run: n-body.exe
	.\n-body.exe

clean:
	del n-body.exe vert.spv frag.spv