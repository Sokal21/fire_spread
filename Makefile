# Compiler selection
COMPILER ?= gcc
CXXFLAGS =
MORE_CXXFLAGS =

ifeq ($(COMPILER),gcc)
    CXX = g++
else ifeq ($(COMPILER),clang)
    CXX = clang++
    CXXFLAGS += -std=c++17
else ifeq ($(COMPILER),icx)
    CXX = icpx
    CXXFLAGS += -std=c++17
else
    $(error Unsupported compiler: $(COMPILER))
endif

CXXFLAGS += -Wall -Wextra -Werror
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
	$(CXXCMD) $< $(objects) -o $@

data.zip:
	wget https://cs.famaf.unc.edu.ar/~nicolasw/data.zip

data: data.zip
	unzip data.zip

clean:
	rm -f $(objects) $(mains)

.PHONY: all clean
