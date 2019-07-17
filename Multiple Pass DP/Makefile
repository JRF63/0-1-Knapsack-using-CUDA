# nvcc dp_cuda.cu -Xptxas="-v" -arch=sm_20 -L/usr/lib/nvidia-319-updates -L/usr/lib/x86_64-linux-gnu -lcuda -lcudart -o vecadd_streams
dp_cuda.o: dp_cuda.cu
	nvcc -O3 -c -Xptxas="-v" -arch=sm_21 dp_cuda.cu
build: dp_cuda.o
	g++ -O3 -std=c++11 dp_cuda.cpp dp_cuda.o -o /home/rafael6676/Documents/knapsack/test4 -L/usr/lib/nvidia-319-updates -L/usr/lib/x86_64-linux-gnu -lcuda -lcudart
	rm dp_cuda.o