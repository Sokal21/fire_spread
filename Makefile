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

# # SLEEF configuration
# SLEEF_CFLAGS = -DENABLE_AVX2 -DENABLE_AVX -DENABLE_SSE2 -DENABLE_SSE4 -DENABLE_FMA4 -DENABLE_FMA
# SLEEF_LDFLAGS = -lsleef

CXXFLAGS += -Wall -Wextra -Werror -march=native -ffast-math -mavx2 -O3 -ftree-vectorize -fopt-info-vec-optimized $(SLEEF_CFLAGS) -fopenmp # Add -fopenmp
INCLUDE = -I./src
CXXCMD = $(CXX) ${MORE_CXXFLAGS} $(CXXFLAGS) $(INCLUDE)

headers = $(wildcard ./src/*.hpp)
sources = $(wildcard ./src/*.cpp)
objects_names = $(sources:./src/%.cpp=%)
objects = $(objects_names:%=./src/%.o)

mains = graphics/burned_probabilities_data graphics/fire_animation_data

all: $(mains)

%.o: %.cpp $(headers)
	$(CXXCMD) -c $< -o $@

$(mains): %: %.cpp $(objects) $(headers)
	$(CXXCMD) $< $(objects) -o $@ $(SLEEF_LDFLAGS) -fopenmp # Add -fopenmp for linking

data.zip:
	wget https://cs.famaf.unc.edu.ar/~nicolasw/data.zip

data: data.zip
	unzip data.zip

clean:
	rm -f $(objects) $(mains)

.PHONY: all clean
