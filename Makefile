# CUDA paths - adjust CUDA_HOME if needed
CUDA_HOME ?= /usr/local/cuda
INC := -I$(CUDA_HOME)/include -I. -Iheaders
LIB := -L$(CUDA_HOME)/lib64 -lcudart -lcudpp

CXXFLAGS  := -O2 -std=c++17 -fPIC
# Adjust -arch for your GPU (sm_86 = RTX 30xx, sm_75 = RTX 20xx, sm_61 = GTX 10xx)
NVCCFLAGS := -O2 -lineinfo -arch=sm_86 --ptxas-options=-v --use_fast_math

CPU_SRCS := \
	cpu_graham.cpp \
	cpu_jarvis.cpp \
	cpu_quickhull.cpp

GPU_SRCS := \
	gpu_cuda_chain.cu \
	gpu_monotone_chain.cu \
	gpu_quickhull.cu

CPU_OBJS  := $(CPU_SRCS:.cpp=.o)
GPU_OBJS  := $(GPU_SRCS:.cu=.o)
MAIN_OBJ  := main.o

OBJS := $(MAIN_OBJ) $(CPU_OBJS) $(GPU_OBJS)

all: main

main: $(OBJS)
	g++ -o $@ $(OBJS) $(LIB)

%.o: %.cpp Makefile
	g++ $(CXXFLAGS) $(INC) -c $< -o $@

%.o: %.cu Makefile
	nvcc $(NVCCFLAGS) $(INC) -c $< -o $@

run: main
	./main

clean:
	rm -f main *.o

.PHONY: all clean run