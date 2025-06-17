# Compiler selection
COMPILER ?= gcc

ifeq ($(COMPILER),gcc)
    CXX = g++
    CXXFLAGS = -std=c++17
else ifeq ($(COMPILER),clang)
    CXX = clang++
    CXXFLAGS = -std=c++17
else ifeq ($(COMPILER),icx)
    CXX = icpx
    CXXFLAGS = -std=c++17
else
    $(error Unsupported compiler: $(COMPILER))
endif

# CUDA Configuration
NVCC ?= nvcc
# NVCCFLAGS = -O3 -std=c++17 --gpu-architecture=sm_70 -Xcompiler="-fopenmp -march=native"
NVCCFLAGS = -O3 -std=c++17 -gencode arch=compute_61,code=sm_61 -Xcompiler="-fopenmp -march=native"

# General Flags
COMMON_FLAGS = -Wall -Wextra -Werror -march=native -ffast-math -mavx2 -O3 -ftree-vectorize -fopenmp
INCLUDE = -I./src

# Source and object files
cpp_sources := $(filter-out ./src/spread_functions.cpp, $(wildcard ./src/*.cpp))
cu_sources := ./src/spread_functions.cu
headers := $(wildcard ./src/*.hpp)

cpp_objects := $(cpp_sources:./src/%.cpp=./src/%.o)
cu_objects := $(cu_sources:./src/%.cu=./src/%.o)
objects := $(cpp_objects) $(cu_objects)

# Main executables
mains := graphics/burned_probabilities_data graphics/fire_animation_data

all: $(mains)

# Compile C++ source files
./src/%.o: ./src/%.cpp $(headers)
	$(CXX) $(CXXFLAGS) $(COMMON_FLAGS) $(INCLUDE) -c $< -o $@

# Compile CUDA source files
./src/%.o: ./src/%.cu $(headers)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@

# Link executables
$(mains): %: %.cpp $(objects) $(headers)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $< $(objects) -o $@ -Xcompiler="-fopenmp -march=native"

# Data file
data.zip:
	wget https://cs.famaf.unc.edu.ar/~nicolasw/data.zip

data: data.zip
	unzip data.zip

clean:
	rm -f $(objects) $(mains)

.PHONY: all clean data
