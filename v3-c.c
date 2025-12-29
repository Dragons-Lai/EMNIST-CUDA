#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_DIM 784
#define HIDDEN_DIM 256
#define NUM_CLASSES 47
#define TRAINING_SAMPLES 10000
#define TESTING_SAMPLES 18800
#define BATCH_SZ 32
#define NUM_EPOCHS 10
#define LR 1e-2

typedef struct {
    double data_prep;
    double forward_pass;
    double loss_calc;
    double backprop;
    double param_update;
    double total_train_time;
} PerformanceMetrics;

typedef struct {
    float *w1;              
    float *w2;              
    float *b1;              
    float *b2;              
    float *grad_w1;         
    float *grad_w2;         
    float *grad_b1;         
    float *grad_b2;         
} CharacterClassifier;

double compute_time_diff(struct timespec t_start, struct timespec t_end) {
    return (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
}

void read_image_data(const char *filepath, float *buffer, int num_elements) {
    FILE *fp = fopen(filepath, "rb");
    size_t elements_read = fread(buffer, sizeof(float), num_elements, fp);
    fclose(fp);
}

void read_label_data(const char *filepath, int *buffer, int num_elements) {
    FILE *fp = fopen(filepath, "rb");
    size_t elements_read = fread(buffer, sizeof(int), num_elements, fp); 
    fclose(fp);
}

void apply_normalization(float *data, int num_elements) {
    const float MEAN_VAL = 0.1307f;
    const float STD_VAL = 0.3081f;
    for (int i = 0; i < num_elements; i++) {
        data[i] = (data[i] - MEAN_VAL) / STD_VAL;
    }
}

void init_layer_weights(float *weights, int fan_in, int fan_out) {
    float he_scale = sqrtf(2.0f / fan_in);
    for (int i = 0; i < fan_in * fan_out; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * he_scale - he_scale;
    }
}

void init_layer_bias(float *bias, int num_units) {
    for (int i = 0; i < num_units; i++) {
        bias[i] = 0.0f;
    }
}

void zero_gradients(float *grad_buffer, int num_elements) {
    memset(grad_buffer, 0, num_elements * sizeof(float));
}

void matmul_ab(float *mat_a, float *mat_b, float *result, int rows_a, int cols_a, int cols_b) {
    for (int i = 0; i < rows_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            result[i * cols_b + j] = 0.0f;
            for (int k = 0; k < cols_a; k++) {
                result[i * cols_b + j] += mat_a[i * cols_a + k] * mat_b[k * cols_b + j];
            }
        }
    }
}

// Matrix multiplication: result = mat_a.T @ mat_b
void matmul_atb(float *mat_a, float *mat_b, float *result, int rows_a, int cols_a, int cols_b) {
    for (int i = 0; i < cols_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            result[i * cols_b + j] = 0.0f;
            for (int k = 0; k < rows_a; k++) {
                result[i * cols_b + j] += mat_a[k * cols_a + i] * mat_b[k * cols_b + j];
            }
        }
    }
}

// Matrix multiplication: result = mat_a @ mat_b.T
void matmul_abt(float *mat_a, float *mat_b, float *result, int rows_a, int cols_a, int rows_b) {
    for (int i = 0; i < rows_a; i++) {
        for (int j = 0; j < rows_b; j++) {
            result[i * rows_b + j] = 0.0f;
            for (int k = 0; k < cols_a; k++) {
                result[i * rows_b + j] += mat_a[i * cols_a + k] * mat_b[j * cols_a + k];
            }
        }
    }
}

void add_bias(float *activations, float *bias_vector, int num_samples, int num_units) {
    for (int sample = 0; sample < num_samples; sample++) {
        for (int unit = 0; unit < num_units; unit++) {
            activations[sample * num_units + unit] += bias_vector[unit];
        }
    }
}

void relu_activation(float *activations, int num_elements) {
    for (int i = 0; i < num_elements; i++) {
        activations[i] = fmaxf(0.0f, activations[i]);
    }
}

void softmax_activation(float *logits, int num_samples, int num_classes) {
    for (int sample = 0; sample < num_samples; sample++) {
        float max_val = logits[sample * num_classes];
        for (int cls = 1; cls < num_classes; cls++) {
            if (logits[sample * num_classes + cls] > max_val) 
                max_val = logits[sample * num_classes + cls];
        }
        float sum_exp = 0.0f;
        for (int cls = 0; cls < num_classes; cls++) {
            logits[sample * num_classes + cls] = expf(logits[sample * num_classes + cls] - max_val);
            sum_exp += logits[sample * num_classes + cls];
        }
        for (int cls = 0; cls < num_classes; cls++) {
            logits[sample * num_classes + cls] = fmaxf(logits[sample * num_classes + cls] / sum_exp, 1e-7f);
        }
    }
}

float compute_cross_entropy(float *predictions, int *target_labels, int num_samples) {
    float total_loss = 0.0f;
    for (int sample = 0; sample < num_samples; sample++) {
        total_loss -= logf(fmaxf(predictions[sample * NUM_CLASSES + target_labels[sample]], 1e-7f));
    }
    return total_loss / num_samples;
}

void compute_loss_gradient(float *loss_grad, float *predictions, int *target_labels, int num_samples) {
    for (int sample = 0; sample < num_samples; sample++) {
        for (int cls = 0; cls < NUM_CLASSES; cls++) {
            loss_grad[sample * NUM_CLASSES + cls] = predictions[sample * NUM_CLASSES + cls];
        }
        loss_grad[sample * NUM_CLASSES + target_labels[sample]] -= 1.0f;
    }
    for (int i = 0; i < num_samples * NUM_CLASSES; i++) {
        loss_grad[i] /= num_samples;
    }
}

void compute_bias_gradient(float *grad_bias, float *grad_upstream, int num_samples, int num_units) {
    for (int unit = 0; unit < num_units; unit++) {
        grad_bias[unit] = 0.0f;
        for (int sample = 0; sample < num_samples; sample++) {
            grad_bias[unit] += grad_upstream[sample * num_units + unit];
        }
    }
}

void forward_propagation(CharacterClassifier *net, float *batch_input, float *hidden_cache, 
                        float *output_logits, int num_samples, PerformanceMetrics *metrics) {
    struct timespec t0, t1;
    
    clock_gettime(CLOCK_MONOTONIC, &t0);
    
    // Layer 1: input @ w1
    matmul_ab(batch_input, net->w1, hidden_cache, num_samples, INPUT_DIM, HIDDEN_DIM);
    
    // Add bias1
    add_bias(hidden_cache, net->b1, num_samples, HIDDEN_DIM);
    
    // Apply ReLU activation
    relu_activation(hidden_cache, num_samples * HIDDEN_DIM);
    
    // Layer 2: hidden @ w2
    matmul_ab(hidden_cache, net->w2, output_logits, num_samples, HIDDEN_DIM, NUM_CLASSES);
    
    // Add bias2
    add_bias(output_logits, net->b2, num_samples, NUM_CLASSES);
    
    // Apply softmax
    softmax_activation(output_logits, num_samples, NUM_CLASSES);
    
    clock_gettime(CLOCK_MONOTONIC, &t1);
    metrics->forward_pass += compute_time_diff(t0, t1);
}

void backward_propagation(CharacterClassifier *net, float *batch_input, float *hidden_cache,
                         float *output_logits, int *batch_labels, int num_samples, 
                         PerformanceMetrics *metrics) {
    struct timespec t0, t1;
    
    clock_gettime(CLOCK_MONOTONIC, &t0);
    
    // Initialize gradients
    zero_gradients(net->grad_w1, HIDDEN_DIM * INPUT_DIM);
    zero_gradients(net->grad_w2, NUM_CLASSES * HIDDEN_DIM);
    zero_gradients(net->grad_b1, HIDDEN_DIM);
    zero_gradients(net->grad_b2, NUM_CLASSES);
    
    // Compute gradient of loss w.r.t. output
    float *grad_output = malloc(num_samples * NUM_CLASSES * sizeof(float));
    compute_loss_gradient(grad_output, output_logits, batch_labels, num_samples);
    
    // Backprop through layer 2
    matmul_atb(hidden_cache, grad_output, net->grad_w2, num_samples, HIDDEN_DIM, NUM_CLASSES);
    compute_bias_gradient(net->grad_b2, grad_output, num_samples, NUM_CLASSES);
    
    // Compute gradient w.r.t. hidden layer
    float *grad_hidden = malloc(num_samples * HIDDEN_DIM * sizeof(float));
    matmul_abt(grad_output, net->w2, grad_hidden, num_samples, NUM_CLASSES, HIDDEN_DIM);
    
    // Backprop through ReLU
    for (int i = 0; i < num_samples * HIDDEN_DIM; i++) {
        grad_hidden[i] *= (hidden_cache[i] > 0);
    }
    
    // Backprop through layer 1
    matmul_atb(batch_input, grad_hidden, net->grad_w1, num_samples, INPUT_DIM, HIDDEN_DIM);
    compute_bias_gradient(net->grad_b1, grad_hidden, num_samples, HIDDEN_DIM);
    
    free(grad_output);
    free(grad_hidden);
    
    clock_gettime(CLOCK_MONOTONIC, &t1);
    metrics->backprop += compute_time_diff(t0, t1);
}

void apply_gradient_updates(CharacterClassifier *net, PerformanceMetrics *metrics) {
    struct timespec t0, t1;
    
    clock_gettime(CLOCK_MONOTONIC, &t0);
    
    for (int i = 0; i < HIDDEN_DIM * INPUT_DIM; i++) {
        net->w1[i] -= LR * net->grad_w1[i];
    }
    for (int i = 0; i < NUM_CLASSES * HIDDEN_DIM; i++) {
        net->w2[i] -= LR * net->grad_w2[i];
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        net->b1[i] -= LR * net->grad_b1[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        net->b2[i] -= LR * net->grad_b2[i];
    }
    
    clock_gettime(CLOCK_MONOTONIC, &t1);
    metrics->param_update += compute_time_diff(t0, t1);
}

void train_character_classifier(CharacterClassifier *net, float *training_images, int *training_targets) {
    float *hidden_cache = malloc(BATCH_SZ * HIDDEN_DIM * sizeof(float));
    float *output_logits = malloc(BATCH_SZ * NUM_CLASSES * sizeof(float));
    int steps_per_epoch = TRAINING_SAMPLES / BATCH_SZ;
    
    PerformanceMetrics metrics = {0};
    
    struct timespec train_start, train_end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &train_start);
    
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int step = 0; step < steps_per_epoch; step++) {
            int batch_start = step * BATCH_SZ;
            
            // Data preparation timing
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float *batch_images = &training_images[batch_start * INPUT_DIM];
            int *batch_labels = &training_targets[batch_start];
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            metrics.data_prep += compute_time_diff(step_start, step_end);
            
            // Forward pass
            forward_propagation(net, batch_images, hidden_cache, output_logits, BATCH_SZ, &metrics);
            
            // Loss calculation timing
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float loss_value = compute_cross_entropy(output_logits, batch_labels, BATCH_SZ);
            epoch_loss += loss_value;
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            metrics.loss_calc += compute_time_diff(step_start, step_end);
            
            // Backward pass
            backward_propagation(net, batch_images, hidden_cache, output_logits, batch_labels, BATCH_SZ, &metrics);
            
            // Parameter update
            apply_gradient_updates(net, &metrics);
        }
        
        printf("Epoch [%d/%d] - Loss: %.4f\n", epoch + 1, NUM_EPOCHS, epoch_loss / steps_per_epoch);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &train_end);
    metrics.total_train_time = compute_time_diff(train_start, train_end);
    
    printf("\nv3-c\n");
    printf("Total Training Time: %.1f seconds\n\n", metrics.total_train_time);
    
    printf("Time Breakdown by Phase:\n");
    printf("  Data Preparation:  %6.3fs  (%5.1f%%)\n", metrics.data_prep, 100.0 * metrics.data_prep / metrics.total_train_time);
    printf("  Forward Pass:      %6.3fs  (%5.1f%%)\n", metrics.forward_pass, 100.0 * metrics.forward_pass / metrics.total_train_time);
    printf("  Loss Calculation:  %6.3fs  (%5.1f%%)\n", metrics.loss_calc, 100.0 * metrics.loss_calc / metrics.total_train_time);
    printf("  Backward Pass:     %6.3fs  (%5.1f%%)\n", metrics.backprop, 100.0 * metrics.backprop / metrics.total_train_time);
    printf("  Parameter Update:  %6.3fs  (%5.1f%%)\n", metrics.param_update, 100.0 * metrics.param_update / metrics.total_train_time);
    printf("============================================================\n");
    
    free(hidden_cache);
    free(output_logits);
}

// Evaluate model accuracy on test set
void evaluate_test_accuracy(CharacterClassifier *net, float *testing_images, int *testing_targets) {
    float *hidden_cache = malloc(BATCH_SZ * HIDDEN_DIM * sizeof(float));
    float *output_logits = malloc(BATCH_SZ * NUM_CLASSES * sizeof(float));
    
    float total_accuracy = 0.0f;
    int num_batches = 0;
    int num_test_batches = TESTING_SAMPLES / BATCH_SZ;
    
    for (int batch_idx = 0; batch_idx < num_test_batches; batch_idx++) {
        int batch_start = batch_idx * BATCH_SZ;
        float *batch_images = &testing_images[batch_start * INPUT_DIM];
        int *batch_labels = &testing_targets[batch_start];
        
        // Forward pass
        matmul_ab(batch_images, net->w1, hidden_cache, BATCH_SZ, INPUT_DIM, HIDDEN_DIM);
        add_bias(hidden_cache, net->b1, BATCH_SZ, HIDDEN_DIM);
        relu_activation(hidden_cache, BATCH_SZ * HIDDEN_DIM);
        matmul_ab(hidden_cache, net->w2, output_logits, BATCH_SZ, HIDDEN_DIM, NUM_CLASSES);
        add_bias(output_logits, net->b2, BATCH_SZ, NUM_CLASSES);
        
        // Get predicted classes
        int num_correct = 0;
        for (int sample = 0; sample < BATCH_SZ; sample++) {
            int pred_class = 0;
            float max_logit = output_logits[sample * NUM_CLASSES];
            for (int cls = 1; cls < NUM_CLASSES; cls++) {
                if (output_logits[sample * NUM_CLASSES + cls] > max_logit) {
                    max_logit = output_logits[sample * NUM_CLASSES + cls];
                    pred_class = cls;
                }
            }
            if (pred_class == batch_labels[sample]) {
                num_correct++;
            }
        }
        
        float batch_acc = (float)num_correct / BATCH_SZ;
        total_accuracy += batch_acc;
        num_batches++;
    }
    
    float mean_accuracy = total_accuracy / num_batches;
    printf("Test Set Average Batch Accuracy: %.2f%%\n", mean_accuracy * 100.0f);
    
    free(hidden_cache);
    free(output_logits);
}

// Initialize the character classifier network
void init_character_classifier(CharacterClassifier *net) {
    net->w1 = malloc(INPUT_DIM * HIDDEN_DIM * sizeof(float));
    net->w2 = malloc(HIDDEN_DIM * NUM_CLASSES * sizeof(float));
    net->b1 = malloc(HIDDEN_DIM * sizeof(float));
    net->b2 = malloc(NUM_CLASSES * sizeof(float));
    net->grad_w1 = malloc(INPUT_DIM * HIDDEN_DIM * sizeof(float));
    net->grad_w2 = malloc(HIDDEN_DIM * NUM_CLASSES * sizeof(float));
    net->grad_b1 = malloc(HIDDEN_DIM * sizeof(float));
    net->grad_b2 = malloc(NUM_CLASSES * sizeof(float));
    
    init_layer_weights(net->w1, INPUT_DIM, HIDDEN_DIM);
    init_layer_weights(net->w2, HIDDEN_DIM, NUM_CLASSES);
    init_layer_bias(net->b1, HIDDEN_DIM);
    init_layer_bias(net->b2, NUM_CLASSES);
}

// Free network memory
void free_character_classifier(CharacterClassifier *net) {
    free(net->w1);
    free(net->w2);
    free(net->b1);
    free(net->b2);
    free(net->grad_w1);
    free(net->grad_w2);
    free(net->grad_b1);
    free(net->grad_b2);
}

int main() {
    // EMNIST Balanced dataset: 112,800 training samples, 18,800 test samples, 47 classes
    printf("Loading EMNIST Balanced dataset...\n");
    
    CharacterClassifier character_net;
    init_character_classifier(&character_net);
    
    float *training_images = malloc(TRAINING_SAMPLES * INPUT_DIM * sizeof(float));
    int *training_targets = malloc(TRAINING_SAMPLES * sizeof(int));
    float *testing_images = malloc(TESTING_SAMPLES * INPUT_DIM * sizeof(float));
    int *testing_targets = malloc(TESTING_SAMPLES * sizeof(int));
    
    read_image_data("./data/X_train.bin", training_images, TRAINING_SAMPLES * INPUT_DIM);
    apply_normalization(training_images, TRAINING_SAMPLES * INPUT_DIM);
    read_label_data("./data/y_train.bin", training_targets, TRAINING_SAMPLES);
    
    read_image_data("./data/X_test.bin", testing_images, TESTING_SAMPLES * INPUT_DIM);
    apply_normalization(testing_images, TESTING_SAMPLES * INPUT_DIM);
    read_label_data("./data/y_test.bin", testing_targets, TESTING_SAMPLES);
    
    train_character_classifier(&character_net, training_images, training_targets);
    
    printf("\nTraining complete! Evaluating model...\n");
    evaluate_test_accuracy(&character_net, testing_images, testing_targets);
    
    free_character_classifier(&character_net);
    free(training_images);
    free(training_targets);
    free(testing_images);
    free(testing_targets);
    
    return 0;
}

