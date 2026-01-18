CUDA_HOME ?= /usr/local/cuda
CCCL_HOME ?= /home/ctsadmin/studia/cccl

INC := -I. -Iheaders -I$(CCCL_HOME)/cub -I$(CCCL_HOME)/libcudacxx/include -I$(CCCL_HOME)/thrust -I$(CUDA_HOME)/include
LIB := -L$(CUDA_HOME)/lib64 -lcudart -ldl

CXXFLAGS  := -O2 -std=c++17 -fPIC
NVCCFLAGS := -O2 -std=c++17 -lineinfo -arch=sm_86 --ptxas-options=-v --use_fast_math

CPU_SRCS := \
	cpu_quickhull.cpp \
	cpu_monotone_chain.cpp

GPU_SRCS := \
	gpu_quickhull_slow.cu \
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