CXX = g++
CXXFLAGS = -I"OpenCL-SDK\external\OpenCL-Headers" -DCL_TARGET_OPENCL_VERSION=300 -DCL_HPP_TARGET_OPENCL_VERSION=300
LDFLAGS = -L. -lOpenCL

all: n-body.exe

n-body.exe: n-body.cpp
	$(CXX) n-body.cpp $(CXXFLAGS) $(LDFLAGS) -o n-body.exe

run: n-body.exe
	.\n-body.exe

clean:
	del n-body.exe