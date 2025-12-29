#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define IMG_DIM         784
#define HIDDEN_UNITS    256
#define NUM_CLASSES     47
#define NUM_TRAIN       10000
#define NUM_TEST        18800
#define MINI_BATCH      8
#define NUM_EPOCHS      10
#define LR              0.01f
#define TILE_SIZE       16

#define GPU_CHECK(stmt) \
    do { \
        cudaError_t err = stmt; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "GPU Error [%s:%d]: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            cudaDeviceReset(); \
            exit(1); \
        } \
    } while(0)

typedef struct {
    float *W1; 
    float *W2; 
    float *b1; 
    float *b2; 
    float *dW1;
    float *dW2;
    float *db1;
    float *db2;
} MLPModel;

typedef struct {
    double load_time;
    double fc1_matmul;
    double fc1_bias;
    double activation;
    double fc2_matmul;
    double fc2_bias;
    double softmax_time;
    double loss_compute;
    double grad_output;
    double grad_fc2_w;
    double grad_fc2_b;
    double grad_relu;
    double grad_fc1_w;
    double grad_fc1_b;
    double param_update;
    double elapsed;
} PerformanceMetrics;

static inline double elapsed_sec(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

void read_binary_images(const char *path, float *buf, int count) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open: %s\n", path);
        exit(1);
    }
    size_t n = fread(buf, sizeof(float), count, fp);
    if (n != (size_t)count) {
        fprintf(stderr, "Read error: got %zu, expected %d\n", n, count);
        exit(1);
    }
    fclose(fp);
}

void read_binary_labels(const char *path, int *buf, int count) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open: %s\n", path);
        exit(1);
    }
    size_t n = fread(buf, sizeof(int), count, fp);
    if (n != (size_t)count) {
        fprintf(stderr, "Read error: got %zu, expected %d\n", n, count);
        exit(1);
    }
    fclose(fp);
}

void standardize_inputs(float *arr, int len) {
    const float mu = 0.1307f;
    const float sigma = 0.3081f;
    for (int i = 0; i < len; i++) {
        arr[i] = (arr[i] - mu) / sigma;
    }
}

void init_he_weights(float *w, int fan_in, int fan_out) {
    float limit = sqrtf(2.0f / fan_in);
    for (int i = 0; i < fan_in * fan_out; i++) {
        w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * limit;
    }
}

void init_zero_bias(float *b, int len) {
    memset(b, 0, len * sizeof(float));
}


// Tiled Matrix multiply: C = A * B (no transpose)
// A: M x K, B: K x N, C: M x N
__global__ void gemm_nn_tiled_kernel(float *A, float *B, float *C, int M, int K, int N) {
    // Shared memory for tiles
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float acc = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        // Collaborative loading of tiles into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        // Load tile from A
        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B
        if (b_row < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Tiled Matrix multiply: C = A * B^T
// A: M x K, B: N x K (transposed), C: M x N
// C[i][j] = sum_k A[i][k] * B[j][k]
__global__ void gemm_nt_tiled_kernel(float *A, float *B, float *C, int M, int K, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float acc = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        // Load tile from A: tile_A[ty][tx] = A[row][t*TILE + tx]
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B^T: tile_B[ty][tx] = B[col][t*TILE + ty]
        // So that tile_B[k][tx] = B[col][t*TILE + k]
        int b_k = t * TILE_SIZE + threadIdx.y;
        if (col < N && b_k < K) {
            tile_B[threadIdx.y][threadIdx.x] = B[col * K + b_k];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Tiled Matrix multiply: C = A^T * B
// A: M x K (transposed access), B: M x N, C: K x N
// C[i][j] = sum_m A[m][i] * B[m][j]
__global__ void gemm_tn_tiled_kernel(float *A, float *B, float *C, int M, int K, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    // Output position in C (which is K x N)
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // row in C (0 to K-1)
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // col in C (0 to N-1)
    
    float acc = 0.0f;
    int num_tiles = (M + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        // For tile_A, we want tile_A[ty][tx] = A[t*TILE + ty][by*TILE + tx]
        // So that tile_A[k][ty] = A[t*TILE + k][row]
        int a_m = t * TILE_SIZE + threadIdx.y;
        int a_col = blockIdx.y * TILE_SIZE + threadIdx.x;
        
        if (a_m < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[a_m * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // For tile_B, we want tile_B[ty][tx] = B[t*TILE + ty][bx*TILE + tx]
        // So that tile_B[k][tx] = B[t*TILE + k][col]
        int b_m = t * TILE_SIZE + threadIdx.y;
        
        if (b_m < M && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_m * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // C[row][col] += sum_k A[t*TILE + k][row] * B[t*TILE + k][col]
        //             = sum_k tile_A[k][ty] * tile_B[k][tx]
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += tile_A[k][threadIdx.y] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < K && col < N) {
        C[row * N + col] = acc;
    }
}
// ReLU activation (in-place)
__global__ void activation_relu_kernel(float *z, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        z[idx] = fmaxf(0.0f, z[idx]);
    }
}

// Add bias to each row
__global__ void add_bias_kernel(float *z, float *bias, int batch, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / dim;
    int j = idx % dim;
    
    if (b < batch && j < dim) {
        z[idx] += bias[j];
    }
}

// Softmax (one block per sample)
__global__ void softmax_activation_kernel(float *logits, int batch, int dim) {
    int b = blockIdx.x;
    if (b >= batch) return;
    
    float *row = logits + b * dim;
    
    // Find max for numerical stability
    float max_val = row[0];
    for (int i = 1; i < dim; i++) {
        max_val = fmaxf(max_val, row[i]);
    }
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < dim; i++) {
        row[i] = expf(row[i] - max_val);
        sum_exp += row[i];
    }
    
    // Normalize
    for (int i = 0; i < dim; i++) {
        row[i] = fmaxf(row[i] / sum_exp, 1e-7f);
    }
}

// Zero out gradient buffer
__global__ void clear_gradients_kernel(float *g, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        g[idx] = 0.0f;
    }
}

// Compute output layer gradient (softmax - one_hot) / batch_size
__global__ void output_layer_grad_kernel(float *dout, float *probs, int *targets, int batch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    
    for (int c = 0; c < NUM_CLASSES; c++) {
        dout[b * NUM_CLASSES + c] = probs[b * NUM_CLASSES + c];
    }
    dout[b * NUM_CLASSES + targets[b]] -= 1.0f;
    
    for (int c = 0; c < NUM_CLASSES; c++) {
        dout[b * NUM_CLASSES + c] /= batch;
    }
}

// ReLU backward (element-wise)
__global__ void relu_derivative_kernel(float *dz, float *z, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        dz[idx] *= (z[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Accumulate bias gradients
__global__ void accumulate_bias_grad_kernel(float *db, float *dz, int batch, int dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= dim) return;
    
    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        sum += dz[b * dim + j];
    }
    db[j] = sum;
}

// SGD parameter update
__global__ void sgd_update_kernel(float *param, float *grad, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        param[idx] -= LR * grad[idx];
    }
}


float compute_ce_loss(float *probs, int *targets, int batch) {
    float loss = 0.0f;
    for (int b = 0; b < batch; b++) {
        loss -= logf(fmaxf(probs[b * NUM_CLASSES + targets[b]], 1e-7f));
    }
    return loss / batch;
}

void mlp_forward_pass(MLPModel *net, float *x, float *h, float *y, int batch, PerformanceMetrics *perf) {
    struct timespec t0, t1;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    // FC1: x @ W1 -> h
    clock_gettime(CLOCK_MONOTONIC, &t0);
    dim3 grid1((HIDDEN_UNITS + TILE_SIZE - 1) / TILE_SIZE, (batch + TILE_SIZE - 1) / TILE_SIZE);
    gemm_nn_tiled_kernel<<<grid1, threads>>>(x, net->W1, h, batch, IMG_DIM, HIDDEN_UNITS);
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->fc1_matmul += elapsed_sec(t0, t1);
    
    // Add bias1
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int total_h = batch * HIDDEN_UNITS;
    add_bias_kernel<<<(total_h + 255) / 256, 256>>>(h, net->b1, batch, HIDDEN_UNITS);
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->fc1_bias += elapsed_sec(t0, t1);
    
    // ReLU
    clock_gettime(CLOCK_MONOTONIC, &t0);
    activation_relu_kernel<<<(total_h + 255) / 256, 256>>>(h, total_h);
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->activation += elapsed_sec(t0, t1);
    
    // FC2: h @ W2 -> y
    clock_gettime(CLOCK_MONOTONIC, &t0);
    dim3 grid2((NUM_CLASSES + TILE_SIZE - 1) / TILE_SIZE, (batch + TILE_SIZE - 1) / TILE_SIZE);
    gemm_nn_tiled_kernel<<<grid2, threads>>>(h, net->W2, y, batch, HIDDEN_UNITS, NUM_CLASSES);
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->fc2_matmul += elapsed_sec(t0, t1);
    
    // Add bias2
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int total_y = batch * NUM_CLASSES;
    add_bias_kernel<<<(total_y + 255) / 256, 256>>>(y, net->b2, batch, NUM_CLASSES);
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->fc2_bias += elapsed_sec(t0, t1);
    
    // Softmax
    clock_gettime(CLOCK_MONOTONIC, &t0);
    softmax_activation_kernel<<<batch, 1>>>(y, batch, NUM_CLASSES);
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->softmax_time += elapsed_sec(t0, t1);
}

void mlp_backward_pass(MLPModel *net, float *x, float *h, float *y, int *targets, int batch, PerformanceMetrics *perf) {
    struct timespec t0, t1;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    // Clear gradient buffers
    clear_gradients_kernel<<<(IMG_DIM * HIDDEN_UNITS + 255) / 256, 256>>>(net->dW1, IMG_DIM * HIDDEN_UNITS);
    clear_gradients_kernel<<<(HIDDEN_UNITS * NUM_CLASSES + 255) / 256, 256>>>(net->dW2, HIDDEN_UNITS * NUM_CLASSES);
    clear_gradients_kernel<<<(HIDDEN_UNITS + 255) / 256, 256>>>(net->db1, HIDDEN_UNITS);
    clear_gradients_kernel<<<(NUM_CLASSES + 255) / 256, 256>>>(net->db2, NUM_CLASSES);
    
    // Allocate workspace
    float *dy, *dh, *dh_relu;
    GPU_CHECK(cudaMalloc(&dy, batch * NUM_CLASSES * sizeof(float)));
    GPU_CHECK(cudaMalloc(&dh, batch * HIDDEN_UNITS * sizeof(float)));
    GPU_CHECK(cudaMalloc(&dh_relu, batch * HIDDEN_UNITS * sizeof(float)));
    
    // Output gradient: dy = softmax - one_hot
    clock_gettime(CLOCK_MONOTONIC, &t0);
    output_layer_grad_kernel<<<(batch + 255) / 256, 256>>>(dy, y, targets, batch);
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->grad_output += elapsed_sec(t0, t1);
    
    // dW2 = h^T @ dy (using tiled kernel)
    clock_gettime(CLOCK_MONOTONIC, &t0);
    dim3 grid_w2((NUM_CLASSES + TILE_SIZE - 1) / TILE_SIZE, (HIDDEN_UNITS + TILE_SIZE - 1) / TILE_SIZE);
    gemm_tn_tiled_kernel<<<grid_w2, threads>>>(h, dy, net->dW2, batch, HIDDEN_UNITS, NUM_CLASSES);
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->grad_fc2_w += elapsed_sec(t0, t1);
    
    // db2 = sum(dy, axis=0)
    clock_gettime(CLOCK_MONOTONIC, &t0);
    accumulate_bias_grad_kernel<<<(NUM_CLASSES + 255) / 256, 256>>>(net->db2, dy, batch, NUM_CLASSES);
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->grad_fc2_b += elapsed_sec(t0, t1);
    
    // dh = dy @ W2^T (using tiled kernel)
    dim3 grid_h((HIDDEN_UNITS + TILE_SIZE - 1) / TILE_SIZE, (batch + TILE_SIZE - 1) / TILE_SIZE);
    gemm_nt_tiled_kernel<<<grid_h, threads>>>(dy, net->W2, dh, batch, NUM_CLASSES, HIDDEN_UNITS);
    
    // dh *= relu'(h)
    clock_gettime(CLOCK_MONOTONIC, &t0);
    relu_derivative_kernel<<<(batch * HIDDEN_UNITS + 255) / 256, 256>>>(dh, h, batch * HIDDEN_UNITS);
    GPU_CHECK(cudaMemcpy(dh_relu, dh, batch * HIDDEN_UNITS * sizeof(float), cudaMemcpyDeviceToDevice));
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->grad_relu += elapsed_sec(t0, t1);
    
    // dW1 = x^T @ dh_relu (using tiled kernel)
    clock_gettime(CLOCK_MONOTONIC, &t0);
    dim3 grid_w1((HIDDEN_UNITS + TILE_SIZE - 1) / TILE_SIZE, (IMG_DIM + TILE_SIZE - 1) / TILE_SIZE);
    gemm_tn_tiled_kernel<<<grid_w1, threads>>>(x, dh_relu, net->dW1, batch, IMG_DIM, HIDDEN_UNITS);
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->grad_fc1_w += elapsed_sec(t0, t1);
    
    // db1 = sum(dh_relu, axis=0)
    clock_gettime(CLOCK_MONOTONIC, &t0);
    accumulate_bias_grad_kernel<<<(HIDDEN_UNITS + 255) / 256, 256>>>(net->db1, dh_relu, batch, HIDDEN_UNITS);
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->grad_fc1_b += elapsed_sec(t0, t1);
    
    GPU_CHECK(cudaFree(dy));
    GPU_CHECK(cudaFree(dh));
    GPU_CHECK(cudaFree(dh_relu));
}

void apply_sgd_step(MLPModel *net, PerformanceMetrics *perf) {
    struct timespec t0, t1;
    
    clock_gettime(CLOCK_MONOTONIC, &t0);
    sgd_update_kernel<<<(IMG_DIM * HIDDEN_UNITS + 255) / 256, 256>>>(net->W1, net->dW1, IMG_DIM * HIDDEN_UNITS);
    sgd_update_kernel<<<(HIDDEN_UNITS * NUM_CLASSES + 255) / 256, 256>>>(net->W2, net->dW2, HIDDEN_UNITS * NUM_CLASSES);
    sgd_update_kernel<<<(HIDDEN_UNITS + 255) / 256, 256>>>(net->b1, net->db1, HIDDEN_UNITS);
    sgd_update_kernel<<<(NUM_CLASSES + 255) / 256, 256>>>(net->b2, net->db2, NUM_CLASSES);
    GPU_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    perf->param_update += elapsed_sec(t0, t1);
}


void run_training_loop(MLPModel *net, float *train_x, int *train_y) {
    // Device buffers
    float *d_x, *d_h, *d_y;
    int *d_labels;
    
    GPU_CHECK(cudaMalloc(&d_x, MINI_BATCH * IMG_DIM * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_h, MINI_BATCH * HIDDEN_UNITS * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_y, MINI_BATCH * NUM_CLASSES * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_labels, MINI_BATCH * sizeof(int)));
    
    // Host buffer for loss
    float *h_probs = (float *)malloc(MINI_BATCH * NUM_CLASSES * sizeof(float));
    
    int num_batches = NUM_TRAIN / MINI_BATCH;
    PerformanceMetrics perf = {0};
    
    struct timespec train_start, train_end, t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &train_start);
    
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int iter = 0; iter < num_batches; iter++) {
            int offset = iter * MINI_BATCH;
            
            // Transfer batch to GPU
            clock_gettime(CLOCK_MONOTONIC, &t0);
            GPU_CHECK(cudaMemcpy(d_x, train_x + offset * IMG_DIM, 
                                 MINI_BATCH * IMG_DIM * sizeof(float), cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_labels, train_y + offset, 
                                 MINI_BATCH * sizeof(int), cudaMemcpyHostToDevice));
            clock_gettime(CLOCK_MONOTONIC, &t1);
            perf.load_time += elapsed_sec(t0, t1);
            
            // Forward
            mlp_forward_pass(net, d_x, d_h, d_y, MINI_BATCH, &perf);
            
            // Compute loss on CPU
            GPU_CHECK(cudaMemcpy(h_probs, d_y, MINI_BATCH * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));
            
            clock_gettime(CLOCK_MONOTONIC, &t0);
            float batch_loss = compute_ce_loss(h_probs, train_y + offset, MINI_BATCH);
            epoch_loss += batch_loss;
            clock_gettime(CLOCK_MONOTONIC, &t1);
            perf.loss_compute += elapsed_sec(t0, t1);
            
            // Backward
            mlp_backward_pass(net, d_x, d_h, d_y, d_labels, MINI_BATCH, &perf);
            
            // Update
            apply_sgd_step(net, &perf);
        }
        
        printf("Epoch %d loss: %.4f\n", epoch, epoch_loss / num_batches);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &train_end);
    perf.elapsed = elapsed_sec(train_start, train_end);
    
    printf("\n=== EMNIST MLP CUDA PERFORMANCE ===\n");
    printf("Total training: %.1f seconds\n\n", perf.elapsed);
    
    double fwd = perf.fc1_matmul + perf.fc1_bias + perf.activation + perf.fc2_matmul + perf.fc2_bias + perf.softmax_time;
    double bwd = perf.grad_output + perf.grad_fc2_w + perf.grad_fc2_b + perf.grad_relu + perf.grad_fc1_w + perf.grad_fc1_b;
    
    printf("Breakdown:\n");
    printf("  Data transfer:  %6.3fs (%5.1f%%)\n", perf.load_time, 100.0 * perf.load_time / perf.elapsed);
    printf("  Forward pass:   %6.3fs (%5.1f%%)\n", fwd, 100.0 * fwd / perf.elapsed);
    printf("  Loss compute:   %6.3fs (%5.1f%%)\n", perf.loss_compute, 100.0 * perf.loss_compute / perf.elapsed);
    printf("  Backward pass:  %6.3fs (%5.1f%%)\n", bwd, 100.0 * bwd / perf.elapsed);
    printf("  Param update:   %6.3fs (%5.1f%%)\n", perf.param_update, 100.0 * perf.param_update / perf.elapsed);
    
    GPU_CHECK(cudaFree(d_x));
    GPU_CHECK(cudaFree(d_h));
    GPU_CHECK(cudaFree(d_y));
    GPU_CHECK(cudaFree(d_labels));
    free(h_probs);
}


float evaluate_accuracy(MLPModel *net, float *images, int *labels, int num_samples) {
    float *d_x, *d_h, *d_y;
    
    GPU_CHECK(cudaMalloc(&d_x, MINI_BATCH * IMG_DIM * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_h, MINI_BATCH * HIDDEN_UNITS * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_y, MINI_BATCH * NUM_CLASSES * sizeof(float)));
    
    float *h_probs = (float *)malloc(MINI_BATCH * NUM_CLASSES * sizeof(float));
    
    int correct = 0;
    int num_batches = num_samples / MINI_BATCH;
    PerformanceMetrics dummy = {0};
    
    for (int iter = 0; iter < num_batches; iter++) {
        int offset = iter * MINI_BATCH;
        
        GPU_CHECK(cudaMemcpy(d_x, images + offset * IMG_DIM, 
                             MINI_BATCH * IMG_DIM * sizeof(float), cudaMemcpyHostToDevice));
        
        mlp_forward_pass(net, d_x, d_h, d_y, MINI_BATCH, &dummy);
        
        GPU_CHECK(cudaMemcpy(h_probs, d_y, MINI_BATCH * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int b = 0; b < MINI_BATCH; b++) {
            int pred = 0;
            float max_prob = h_probs[b * NUM_CLASSES];
            for (int c = 1; c < NUM_CLASSES; c++) {
                if (h_probs[b * NUM_CLASSES + c] > max_prob) {
                    max_prob = h_probs[b * NUM_CLASSES + c];
                    pred = c;
                }
            }
            if (pred == labels[offset + b]) {
                correct++;
            }
        }
    }
    
    GPU_CHECK(cudaFree(d_x));
    GPU_CHECK(cudaFree(d_h));
    GPU_CHECK(cudaFree(d_y));
    free(h_probs);
    
    return (float)correct / (num_batches * MINI_BATCH) * 100.0f;
}


void create_mlp_model(MLPModel *net) {
    GPU_CHECK(cudaMalloc(&net->W1, IMG_DIM * HIDDEN_UNITS * sizeof(float)));
    GPU_CHECK(cudaMalloc(&net->W2, HIDDEN_UNITS * NUM_CLASSES * sizeof(float)));
    GPU_CHECK(cudaMalloc(&net->b1, HIDDEN_UNITS * sizeof(float)));
    GPU_CHECK(cudaMalloc(&net->b2, NUM_CLASSES * sizeof(float)));
    GPU_CHECK(cudaMalloc(&net->dW1, IMG_DIM * HIDDEN_UNITS * sizeof(float)));
    GPU_CHECK(cudaMalloc(&net->dW2, HIDDEN_UNITS * NUM_CLASSES * sizeof(float)));
    GPU_CHECK(cudaMalloc(&net->db1, HIDDEN_UNITS * sizeof(float)));
    GPU_CHECK(cudaMalloc(&net->db2, NUM_CLASSES * sizeof(float)));
    
    float *h_W1 = (float *)malloc(IMG_DIM * HIDDEN_UNITS * sizeof(float));
    float *h_W2 = (float *)malloc(HIDDEN_UNITS * NUM_CLASSES * sizeof(float));
    float *h_b1 = (float *)malloc(HIDDEN_UNITS * sizeof(float));
    float *h_b2 = (float *)malloc(NUM_CLASSES * sizeof(float));
    
    init_he_weights(h_W1, IMG_DIM, HIDDEN_UNITS);
    init_he_weights(h_W2, HIDDEN_UNITS, NUM_CLASSES);
    init_zero_bias(h_b1, HIDDEN_UNITS);
    init_zero_bias(h_b2, NUM_CLASSES);
    
    GPU_CHECK(cudaMemcpy(net->W1, h_W1, IMG_DIM * HIDDEN_UNITS * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(net->W2, h_W2, HIDDEN_UNITS * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(net->b1, h_b1, HIDDEN_UNITS * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(net->b2, h_b2, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_W1);
    free(h_W2);
    free(h_b1);
    free(h_b2);
}

void destroy_mlp_model(MLPModel *net) {
    GPU_CHECK(cudaFree(net->W1));
    GPU_CHECK(cudaFree(net->W2));
    GPU_CHECK(cudaFree(net->b1));
    GPU_CHECK(cudaFree(net->b2));
    GPU_CHECK(cudaFree(net->dW1));
    GPU_CHECK(cudaFree(net->dW2));
    GPU_CHECK(cudaFree(net->db1));
    GPU_CHECK(cudaFree(net->db2));
}


int main(int argc, char **argv) {
    srand((unsigned)time(NULL));
    
    float *train_images = (float *)malloc(NUM_TRAIN * IMG_DIM * sizeof(float));
    int *train_labels = (int *)malloc(NUM_TRAIN * sizeof(int));
    float *test_images = (float *)malloc(NUM_TEST * IMG_DIM * sizeof(float));
    int *test_labels = (int *)malloc(NUM_TEST * sizeof(int));
    
    read_binary_images("./data/X_train.bin", train_images, NUM_TRAIN * IMG_DIM);
    read_binary_labels("./data/y_train.bin", train_labels, NUM_TRAIN);
    read_binary_images("./data/X_test.bin", test_images, NUM_TEST * IMG_DIM);
    read_binary_labels("./data/y_test.bin", test_labels, NUM_TEST);
    
    standardize_inputs(train_images, NUM_TRAIN * IMG_DIM);
    standardize_inputs(test_images, NUM_TEST * IMG_DIM);
    
    MLPModel model;
    create_mlp_model(&model);
    
    run_training_loop(&model, train_images, train_labels);
    
    float test_acc = evaluate_accuracy(&model, test_images, test_labels, NUM_TEST);
    
    printf("Testing Accuracy:  %.2f%%\n", test_acc);
    
    destroy_mlp_model(&model);
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);
    
    return 0;
}

