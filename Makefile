.PHONY: build test test-gpu kernels test-kernels test-paged-attn bench lint clean

GO         ?= go
BINARY     := gollmgo
BUILD_DIR  := bin
KERNEL_DIR := kernels

# CUDA settings
NVCC       ?= nvcc
CUDA_ARCH  ?= native
CUDA_LIBS  := -L/usr/local/cuda/targets/sbsa-linux/lib -lcudart -lcublas -lstdc++ -lm

# Static libraries
LIB_BACKEND    := $(KERNEL_DIR)/libgollmgo_backend.a
LIB_OPS        := $(KERNEL_DIR)/libgollmgo_ops.a
LIB_MODEL      := $(KERNEL_DIR)/libgollmgo_model.a
LIB_PAGED_ATTN := $(KERNEL_DIR)/libgollmgo_paged_attn.a
LIB_KVCACHE    := $(KERNEL_DIR)/libgollmgo_kvcache.a

build: kernels
	CGO_ENABLED=1 $(GO) build -tags gpu -o $(BUILD_DIR)/$(BINARY) ./cmd/gollmgo

test:
	$(GO) test -race -count=1 ./...

test-gpu: kernels
	CGO_ENABLED=1 $(GO) test -race -count=1 -tags gpu ./...

test-kernels: kernels
	$(NVCC) -arch=$(CUDA_ARCH) -o $(BUILD_DIR)/ops_test \
		$(KERNEL_DIR)/gollmgo_ops_test.cu $(KERNEL_DIR)/gollmgo_ops.cu \
		$(CUDA_LIBS)
	$(BUILD_DIR)/ops_test

test-paged-attn: kernels
	$(NVCC) -arch=$(CUDA_ARCH) -o $(BUILD_DIR)/paged_attn_test \
		$(KERNEL_DIR)/gollmgo_paged_attn_test.cu \
		$(KERNEL_DIR)/gollmgo_paged_attn.cu \
		$(KERNEL_DIR)/gollmgo_ops.cu \
		$(CUDA_LIBS)
	$(BUILD_DIR)/paged_attn_test

kernels: $(LIB_BACKEND) $(LIB_OPS) $(LIB_MODEL) $(LIB_PAGED_ATTN) $(LIB_KVCACHE)

$(LIB_BACKEND): $(KERNEL_DIR)/gollmgo_backend.cu $(KERNEL_DIR)/gollmgo_backend.h
	$(NVCC) -arch=$(CUDA_ARCH) -c -o $(KERNEL_DIR)/gollmgo_backend.o $<
	ar rcs $@ $(KERNEL_DIR)/gollmgo_backend.o
	rm -f $(KERNEL_DIR)/gollmgo_backend.o

$(LIB_OPS): $(KERNEL_DIR)/gollmgo_ops.cu $(KERNEL_DIR)/gollmgo_ops.cuh
	$(NVCC) -arch=$(CUDA_ARCH) -c -o $(KERNEL_DIR)/gollmgo_ops.o $<
	ar rcs $@ $(KERNEL_DIR)/gollmgo_ops.o
	rm -f $(KERNEL_DIR)/gollmgo_ops.o

$(LIB_MODEL): $(KERNEL_DIR)/gollmgo_model.cu $(KERNEL_DIR)/gollmgo_model.h $(KERNEL_DIR)/gollmgo_ops.cuh
	$(NVCC) -arch=$(CUDA_ARCH) -c -o $(KERNEL_DIR)/gollmgo_model.o $<
	ar rcs $@ $(KERNEL_DIR)/gollmgo_model.o
	rm -f $(KERNEL_DIR)/gollmgo_model.o

$(LIB_PAGED_ATTN): $(KERNEL_DIR)/gollmgo_paged_attn.cu $(KERNEL_DIR)/gollmgo_paged_attn.cuh
	$(NVCC) -arch=$(CUDA_ARCH) -c -o $(KERNEL_DIR)/gollmgo_paged_attn.o $<
	ar rcs $@ $(KERNEL_DIR)/gollmgo_paged_attn.o
	rm -f $(KERNEL_DIR)/gollmgo_paged_attn.o

$(LIB_KVCACHE): $(KERNEL_DIR)/gollmgo_kvcache.cu $(KERNEL_DIR)/gollmgo_kvcache.h $(KERNEL_DIR)/gollmgo_paged_attn.cuh
	$(NVCC) -arch=$(CUDA_ARCH) -c -o $(KERNEL_DIR)/gollmgo_kvcache.o $<
	ar rcs $@ $(KERNEL_DIR)/gollmgo_kvcache.o
	rm -f $(KERNEL_DIR)/gollmgo_kvcache.o

bench:
	$(GO) test -bench=. -benchmem ./...

lint:
	$(GO) vet ./...

clean:
	rm -rf $(BUILD_DIR) $(KERNEL_DIR)/*.o $(KERNEL_DIR)/*.a
