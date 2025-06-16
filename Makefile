# Compiler selection
COMPILER ?= gcc
CXXFLAGS =
MORE_CXXFLAGS =

ifeq ($(COMPILER),gcc)
    CXX = g++
    CXXFLAGS += -std=c++17
else ifeq ($(COMPILER),clang)
    CXX = clang++
    CXXFLAGS += -std=c++17
else ifeq ($(COMPILER),icx)
    CXX = icpx
    CXXFLAGS += -std=c++17
else
    $(error Unsupported compiler: $(COMPILER))
endif

# CUDA Configuration
NVCC ?= nvcc
NVCCFLAGS = -O3 -std=c++17 --gpu-architecture=sm_70 -Xcompiler="-fopenmp -march=native"

# Compiler flags
CXXFLAGS += -Wall -Wextra -Werror -march=native -ffast-math -mavx2 -O3 -ftree-vectorize -fopenmp
INCLUDE = -I./src
CXXCMD = $(CXX) ${MORE_CXXFLAGS} $(CXXFLAGS) $(INCLUDE)

# Sources and objects
cpp_sources := $(filter-out ./src/spread_functions.cpp, $(wildcard ./src/*.cpp)) # exclude if needed
cu_sources := $(wildcard ./src/*.cu)
headers := $(wildcard ./src/*.hpp)

cpp_objects := $(cpp_sources:./src/%.cpp=./src/%.o)
cu_objects := $(cu_sources:./src/%.cu=./src/%.o)
objects := $(cpp_objects) $(cu_objects)

# Mains
mains := graphics/burned_probabilities_data graphics/fire_animation_data

all: $(mains)

# Compile .cpp
./src/%.o: ./src/%.cpp $(headers)
	$(CXXCMD) -c $< -o $@

# Compile .cu
./src/%.o: ./src/%.cu $(headers)
	$(NVCC) $(NVCCFLAGS) -I./src -c $< -o $@

# Link binaries
$(mains): %: %.cpp $(objects) $(headers)
	$(CXXCMD) $(objects) -o $@ -fopenmp

# Data
data.zip:
	wget https://cs.famaf.unc.edu.ar/~nicolasw/data.zip

data: data.zip
	unzip data.zip

clean:
	rm -f $(objects) $(mains)

.PHONY: all clean data
