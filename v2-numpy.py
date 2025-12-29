#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np

NUM_EPOCHS = 10
LR = 1e-2
BATCH_SZ = 32
TRAINING_SAMPLES = 10000  # Use 10,000 samples from 112,800 total training samples

def softmax_activation(logits):
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted_logits)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

def compute_cross_entropy(logits, target_labels):
    num_samples = logits.shape[0]
    probs = softmax_activation(logits)
    log_probs_correct = np.log(probs[np.arange(num_samples), target_labels])
    return -np.sum(log_probs_correct) / num_samples

def relu_activation(z):
    """Apply ReLU activation function"""
    return np.maximum(0, z)

def relu_grad(z):
    """Compute gradient of ReLU activation"""
    return (z > 0).astype(float)

# Weight initialization with He method
def init_layer_weights(fan_in, fan_out):
    """Initialize weights using He initialization for ReLU networks"""
    he_scale = np.sqrt(2.0 / fan_in) 
    return (np.random.rand(fan_in, fan_out) * 2.0 - 1.0) * he_scale

def init_layer_bias(num_units):
    """Initialize bias vector to zeros"""
    return np.zeros((1, num_units))

def fc_forward(input_data, weight_matrix, bias_vector):
    """Forward pass through a fully connected layer"""
    return input_data @ weight_matrix + bias_vector

def fc_backward(upstream_grad, layer_input, weight_matrix):
    weight_grad = layer_input.T @ upstream_grad
    bias_grad = np.sum(upstream_grad, axis=0, keepdims=True)
    input_grad = upstream_grad @ weight_matrix.T
    return input_grad, weight_grad, bias_grad


class CharacterClassifier:
    """Two-layer fully connected network for EMNIST character recognition"""
    
    def __init__(self, input_dim, hidden_dim, output_classes):
        # Layer 1: input -> hidden with He initialization
        self.w1 = init_layer_weights(input_dim, hidden_dim)
        self.b1 = init_layer_bias(hidden_dim)
        # Layer 2: hidden -> output with He initialization
        self.w2 = init_layer_weights(hidden_dim, output_classes)
        self.b2 = init_layer_bias(output_classes)

    def forward_pass(self, input_batch):
        """Execute forward propagation through the network"""
        num_samples = input_batch.shape[0]
        # Flatten image data to vector
        flattened_input = input_batch.reshape(num_samples, -1)
        # First layer computation
        hidden_preactivation = fc_forward(flattened_input, self.w1, self.b1)
        hidden_activation = relu_activation(hidden_preactivation)
        # Second layer computation
        output_logits = fc_forward(hidden_activation, self.w2, self.b2)
        # Store intermediate values for backpropagation
        forward_cache = (flattened_input, hidden_preactivation, hidden_activation)
        return output_logits, forward_cache

    def backward_pass(self, output_gradient, forward_cache):
        """Execute backward propagation to compute gradients"""
        flattened_input, hidden_preactivation, hidden_activation = forward_cache
        # Backprop through second layer
        grad_hidden, grad_w2, grad_b2 = fc_backward(output_gradient, hidden_activation, self.w2)
        # Backprop through ReLU activation
        grad_hidden_preactivation = grad_hidden * relu_grad(hidden_preactivation)
        # Backprop through first layer
        grad_input, grad_w1, grad_b1 = fc_backward(grad_hidden_preactivation, flattened_input, self.w1)
        return grad_w1, grad_b1, grad_w2, grad_b2

    def apply_gradients(self, grad_w1, grad_b1, grad_w2, grad_b2, step_size):
        """Update network parameters using computed gradients"""
        self.w1 -= step_size * grad_w1
        self.b1 -= step_size * grad_b1
        self.w2 -= step_size * grad_w2
        self.b2 -= step_size * grad_b2


def train_one_epoch(classifier, train_imgs, train_lbls, epoch_idx, metrics, losses_history):
    running_loss = 0.0
    steps = TRAINING_SAMPLES // BATCH_SZ
    
    for step_idx in range(steps):
        # Time: data preparation
        t0 = time.time()
        batch_start = step_idx * BATCH_SZ
        batch_end = (step_idx + 1) * BATCH_SZ
        batch_images = train_imgs[batch_start:batch_end]
        batch_labels = train_lbls[batch_start:batch_end]
        t1 = time.time()
        metrics['data_prep'] += t1 - t0
        
        # Time: forward propagation
        t2 = time.time()
        predictions, cache = classifier.forward_pass(batch_images)
        t3 = time.time()
        metrics['forward_pass'] += t3 - t2
        
        # Time: loss computation
        t4 = time.time()
        loss_value = compute_cross_entropy(predictions, batch_labels)
        running_loss += loss_value

        prob_predictions = softmax_activation(predictions)
        labels_one_hot = np.zeros_like(predictions)
        labels_one_hot[np.arange(len(batch_labels)), batch_labels] = 1
        loss_gradient = (prob_predictions - labels_one_hot) / len(batch_labels)
        t5 = time.time()
        metrics['loss_calc'] += t5 - t4
        
        # Time: backward propagation
        t6 = time.time()
        grad_w1, grad_b1, grad_w2, grad_b2 = classifier.backward_pass(loss_gradient, cache)
        t7 = time.time()
        metrics['backprop'] += t7 - t6
        
        # Time: parameter updates
        t8 = time.time()
        classifier.apply_gradients(grad_w1, grad_b1, grad_w2, grad_b2, LR)
        t9 = time.time()
        metrics['param_update'] += t9 - t8
    
    avg_loss = running_loss / steps
    losses_history.append(avg_loss)


def evaluate_test_accuracy(classifier, test_imgs, test_lbls):
    """Evaluate model accuracy on test set"""
    total_accuracy = 0.0
    num_batches = 0
    
    num_test_batches = len(test_imgs) // BATCH_SZ
    for batch_idx in range(num_test_batches):
        batch_start = batch_idx * BATCH_SZ
        batch_end = (batch_idx + 1) * BATCH_SZ
        imgs = test_imgs[batch_start:batch_end]
        lbls = test_lbls[batch_start:batch_end]
        
        # Forward pass without caching
        output_logits, _ = classifier.forward_pass(imgs)
        # Get predicted classes
        pred_classes = np.argmax(output_logits, axis=1)
        
        # Calculate batch accuracy
        num_correct = np.sum(pred_classes == lbls)
        batch_size = len(lbls)
        
        if batch_size > 0:
            batch_acc = num_correct / batch_size
            total_accuracy += batch_acc
            num_batches += 1
    
    mean_accuracy = total_accuracy / num_batches
    print(f"Test Set Average Batch Accuracy: {mean_accuracy * 100:.2f}%")


if __name__ == "__main__":
    # EMNIST Balanced dataset: 112,800 training samples, 18,800 test samples, 47 classes
    print("Loading EMNIST Balanced dataset...")
    train_imgs_raw = np.fromfile("data/X_train.bin", dtype=np.float32).reshape(112800, 784)
    train_lbls_raw = np.fromfile("data/y_train.bin", dtype=np.int32)
    test_imgs_raw = np.fromfile("data/X_test.bin", dtype=np.float32).reshape(18800, 784)
    test_lbls_raw = np.fromfile("data/y_test.bin", dtype=np.int32)
    
    MEAN_VAL, STD_VAL = 0.1307, 0.3081
    train_imgs_normalized = (train_imgs_raw - MEAN_VAL) / STD_VAL
    test_imgs_normalized = (test_imgs_raw - MEAN_VAL) / STD_VAL
    
    training_images = train_imgs_normalized[:TRAINING_SAMPLES].reshape(-1, 1, 28, 28)
    training_targets = train_lbls_raw[:TRAINING_SAMPLES]
    testing_images = test_imgs_normalized.reshape(-1, 1, 28, 28)
    testing_targets = test_lbls_raw
    
    input_dim = 784
    hidden_dim = 256
    output_classes = 47
    
    character_net = CharacterClassifier(input_dim, hidden_dim, output_classes)
    
    performance_metrics = {
        'data_prep': 0.0,
        'forward_pass': 0.0,
        'loss_calc': 0.0,
        'backprop': 0.0,
        'param_update': 0.0,
        'total_train_time': 0.0
    }
    epoch_loss_list = []
    
    print("Starting EMNIST character recognition model training...")
    train_start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(
            character_net,
            training_images,
            training_targets,
            epoch,
            performance_metrics,
            epoch_loss_list
        )
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {epoch_loss_list[epoch]:.4f}")
    
    train_end_time = time.time()
    performance_metrics['total_train_time'] = train_end_time - train_start_time
    
    print("\nv2-numpy")
    print(f"Total Training Time: {performance_metrics['total_train_time']:.1f} seconds\n")
    
    total_t = performance_metrics['total_train_time']
    print("Time Breakdown by Phase:")
    print(f"  Data Preparation:  {performance_metrics['data_prep']:6.3f}s  ({100.0 * performance_metrics['data_prep'] / total_t:5.1f}%)")
    print(f"  Forward Pass:      {performance_metrics['forward_pass']:6.3f}s  ({100.0 * performance_metrics['forward_pass'] / total_t:5.1f}%)")
    print(f"  Loss Calculation:  {performance_metrics['loss_calc']:6.3f}s  ({100.0 * performance_metrics['loss_calc'] / total_t:5.1f}%)")
    print(f"  Backward Pass:     {performance_metrics['backprop']:6.3f}s  ({100.0 * performance_metrics['backprop'] / total_t:5.1f}%)")
    print(f"  Parameter Update:  {performance_metrics['param_update']:6.3f}s  ({100.0 * performance_metrics['param_update'] / total_t:5.1f}%)")
    print("="*60)
    
    print("\nTraining complete! Evaluating model...")
    evaluate_test_accuracy(character_net, testing_images, testing_targets)

