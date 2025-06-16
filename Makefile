# ... (existing compiler selection) ...

# CUDA Configuration (adjust paths if necessary)
CUDA_PATH ?= /usr/lib/cuda
NVCC := nvcc
CUDA_LIB_PATH := -L$(CUDA_PATH)/lib64
CUDA_LIBS := -lcudart -lcurand # Add other CUDA libs as needed (e.g., cufft, cublas)

# --- SLEEF Configuration (if still used for CPU parts or if you have a CUDA version) ---
# ... (your SLEEF config) ...

# General Compiler Flags
# For host code compiled by CXX
CXXFLAGS += -Wall -Wextra -Werror -march=native -O3 -fopenmp $(SLEEF_COMPILE_FLAGS)
# For device code compiled by NVCC (can also be set in NVCCFLAGS)
NVCCFLAGS := -O3 -std=c++17 --gpu-architecture=sm_70 # Replace sm_XX with your GPU's compute capability (e.g., sm_75)
NVCCFLAGS += -Xcompiler "$(CXXFLAGS)" # Pass CXXFLAGS to the host compiler part of nvcc
NVCCFLAGS += $(SLEEF_INCLUDE_PATH) # If SLEEF headers are needed by .cu files
NVCCFLAGS += -I./src # Project includes for .cu files

PROJECT_INCLUDE = -I./src
# Full command for C++ compilation
CXXCMD = $(CXX) $(MORE_CXXFLAGS) $(CXXFLAGS) $(PROJECT_INCLUDE) $(SLEEF_INCLUDE_PATH)
# Full command for CUDA compilation
NVCCCMD = $(NVCC) $(NVCCFLAGS) $(PROJECT_INCLUDE) $(SLEEF_INCLUDE_PATH)


# Source files
# Separate .cpp and .cu sources
CPP_SOURCES = $(filter-out ./src/spread_functions.cpp, $(wildcard ./src/*.cpp) $(wildcard graphics/*.cpp))
CU_SOURCES = $(wildcard ./src/*.cu)

# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

# All objects
OBJECTS = $(CPP_OBJECTS) $(CU_OBJECTS)

# ... (headers, mains) ...
mains = graphics/burned_probabilities_data graphics/fire_animation_data

all: $(mains)

# Rule to compile .cpp files
%.o: %.cpp $(headers)
	$(CXXCMD) -c $< -o $@

# Rule to compile .cu files
%.o: %.cu $(headers)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rule to link executables
$(mains): %: %.cpp $(filter-out graphics/%.o, $(CPP_OBJECTS)) $(CU_OBJECTS) $(headers)
	$(CXX) $(MORE_CXXFLAGS) $(CXXFLAGS) $(PROJECT_INCLUDE) $(SLEEF_INCLUDE_PATH) \
		$< $(filter-out $@.o graphics/%.o, $(OBJECTS)) \
		-o $@ $(SLEEF_LINK_FLAGS) -lm -fopenmp $(CUDA_LIB_PATH) $(CUDA_LIBS)

# ... (data, clean targets) ...
# Update clean target for .cu objects if needed
clean:
	rm -f $(OBJECTS) $(mains) graphics/*.png graphics/*.mp4 data.zip
	rm -rf data
	rm -f simulation_output.tmp.*
	rm -rf temp_frames.*