CC=nvcc
CFLAGS=-I/usr/local/cuda-5.5/include -I/cfs/zorn/nobackup/m/mircom/cuda/samples/common/inc
LDFLAGS=-L${CUDA_HOME}/lib64 

all: image_inpaint

image_inpaint: pa.cu
	$(CC) -o image_inpaint $(CFLAGS) -arch sm_30 $(LDFLAGS) pa.cu
	
clean: 
	rm image_inpaint
