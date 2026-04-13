.PHONY: build test test-gpu kernels bench lint clean

GO         ?= go
BINARY     := gollmgo
BUILD_DIR  := bin
KERNEL_DIR := kernels

# CUDA settings
NVCC       ?= nvcc
CUDA_ARCH  ?= native
CUDA_LIB   := $(KERNEL_DIR)/libgollmgo_backend.a

build: kernels
	CGO_ENABLED=1 $(GO) build -tags gpu -o $(BUILD_DIR)/$(BINARY) ./cmd/gollmgo

test:
	$(GO) test -race -count=1 ./...

test-gpu: kernels
	CGO_ENABLED=1 $(GO) test -race -count=1 -tags gpu ./...

kernels: $(CUDA_LIB)

$(CUDA_LIB): $(KERNEL_DIR)/gollmgo_backend.cu $(KERNEL_DIR)/gollmgo_backend.h
	$(NVCC) -arch=$(CUDA_ARCH) -c -o $(KERNEL_DIR)/gollmgo_backend.o $(KERNEL_DIR)/gollmgo_backend.cu
	ar rcs $(CUDA_LIB) $(KERNEL_DIR)/gollmgo_backend.o
	rm -f $(KERNEL_DIR)/gollmgo_backend.o

bench:
	$(GO) test -bench=. -benchmem ./...

lint:
	$(GO) vet ./...

clean:
	rm -rf $(BUILD_DIR) $(KERNEL_DIR)/*.o $(KERNEL_DIR)/*.a
