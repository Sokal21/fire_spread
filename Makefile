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

# General Compiler Flags
CXXFLAGS += -Wall -Wextra -Werror -march=native -ffast-math -mavx2 -O3 -fopt-info-vec-optimized -fopenmp
INCLUDE = -I./src
CXXCMD = $(CXX) ${MORE_CXXFLAGS} $(CXXFLAGS) $(INCLUDE)

# Sources and objects
cpp_sources := $(wildcard ./src/*.cpp)
cu_sources := $(wildcard ./src/*.cu)
sources := $(cpp_sources) $(cu_sources)

headers := $(wildcard ./src/*.hpp)
objects_names := $(sources:./src/%.cpp=%)
objects_names := $(objects_names:./src/%.cu=%)
objects := $(objects_names:%=./src/%.o)

# Main targets
mains = graphics/burned_probabilities_data graphics/fire_animation_data

all: $(mains)

# Compile .cpp to .o
./src/%.o: ./src/%.cpp $(headers)
	$(CXXCMD) -c $< -o $@

# Compile .cu to .o
./src/%.o: ./src/%.cu $(headers)
	$(NVCC) $(NVCCFLAGS) -I./src -c $< -o $@

# Link final executables
$(mains): %: %.cpp $(objects) $(headers)
	$(CXXCMD) $< $(objects) -o $@ -fopenmp

# Data download and extract
data.zip:
	wget https://cs.famaf.unc.edu.ar/~nicolasw/data.zip

data: data.zip
	unzip data.zip

clean:
	rm -f $(objects) $(mains)

.PHONY: all clean data
