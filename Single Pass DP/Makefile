CFLAGS=-O3 -march=native -flto -Wall -Wextra -fdiagnostics-color=auto
CUDA_INC=/usr/local/cuda/include
CUDA_LINK=/usr/local/cuda/lib64
LINKER_OPTS=-lcuda -lcudart

all: dp_cuda.o
	g++-4.9 $(CFLAGS) -std=c++14 dp_cuda.cpp dp_cuda.o -o test -I$(CUDA_INC) -L$(CUDA_LINK) -Wl,-rpath=$(CUDA_LINK) $(LINKER_OPTS)
	rm dp_cuda.o
dp_cuda.o: dp_cuda.cu
	nvcc -O3 -c -Xptxas="-v" -Xcompiler "$(CFLAGS)" -arch=sm_21 dp_cuda.cu