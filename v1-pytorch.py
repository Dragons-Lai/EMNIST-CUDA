#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

NUM_EPOCHS = 10
LR = 1e-2
BATCH_SZ = 8
TRAINING_SAMPLES = 10000  # Number of samples to use from 112,800 total training samples

# Set matrix multiplication precision
torch.set_float32_matmul_precision("high")


class CharacterClassifier(nn.Module):
    """Two-layer fully connected network for character recognition"""
    
    def __init__(self, input_dim, hidden_dim, output_classes):
        super(CharacterClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_classes)
    
    def forward(self, inputs):
        # Flatten input images
        flattened = inputs.reshape(BATCH_SZ, 28 * 28)
        # First layer + activation
        hidden = self.layer1(flattened)
        activated = self.activation(hidden)
        # Output layer
        logits = self.layer2(activated)
        return logits


def train_one_epoch(net, loss_fn, opt, epoch_idx, metrics, losses_history):
    """Execute one epoch of training with timing for each phase"""
    net.train()
    running_loss = 0.0
    steps = TRAINING_SAMPLES // BATCH_SZ
    
    for step_idx in range(steps):
        t0 = time.time()
        batch_start = step_idx * BATCH_SZ
        batch_end = (step_idx + 1) * BATCH_SZ
        inputs_batch = training_images[batch_start:batch_end]
        labels_batch = training_targets[batch_start:batch_end]
        t1 = time.time()
        metrics['data_prep'] += t1 - t0
        
        opt.zero_grad()
        
        # Forward pass timing
        t2 = time.time()
        predictions = net(inputs_batch)
        t3 = time.time()
        metrics['forward_pass'] += t3 - t2
        
        # Loss calculation timing
        t4 = time.time()
        loss_value = loss_fn(predictions, labels_batch)
        running_loss += loss_value.item()
        t5 = time.time()
        metrics['loss_calc'] += t5 - t4
        
        # Backward pass timing
        t6 = time.time()
        loss_value.backward()
        t7 = time.time()
        metrics['backprop'] += t7 - t6
        
        # Parameter update timing
        t8 = time.time()
        opt.step()
        # opt.zero_grad()
        t9 = time.time()
        metrics['param_update'] += t9 - t8
    
    avg_loss = running_loss / steps
    losses_history.append(avg_loss)


def compute_test_accuracy(net, test_imgs, test_lbls):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    
    total_acc = torch.tensor(0.0, device=device)
    batch_count = 0
    
    with torch.no_grad():
        num_test_batches = len(test_imgs) // BATCH_SZ
        for batch_idx in range(num_test_batches):
            batch_start = batch_idx * BATCH_SZ
            batch_end = (batch_idx + 1) * BATCH_SZ
            imgs = test_imgs[batch_start:batch_end]
            lbls = test_lbls[batch_start:batch_end]
            
            output = net(imgs)
            _, pred_classes = torch.max(output.data, 1)
            
            num_correct = (pred_classes == lbls).sum().item()
            batch_size = lbls.size(0)
            
            if batch_size > 0:
                batch_acc = num_correct / batch_size
                total_acc += batch_acc
                batch_count += 1
    
    mean_accuracy = total_acc / batch_count
    print(f"Test Set Average Batch Accuracy: {mean_accuracy * 100:.2f}%")


if __name__ == "__main__":
    # EMNIST Balanced: 112,800 training samples, 18,800 test samples, 47 classes
    train_imgs_np = np.fromfile("data/X_train.bin", dtype=np.float32).reshape(112800, 784)
    train_lbls_np = np.fromfile("data/y_train.bin", dtype=np.int32)
    test_imgs_np = np.fromfile("data/X_test.bin", dtype=np.float32).reshape(18800, 784)
    test_lbls_np = np.fromfile("data/y_test.bin", dtype=np.int32)
    
    MEAN_VAL, STD_VAL = 0.1307, 0.3081
    train_imgs_np = (train_imgs_np - MEAN_VAL) / STD_VAL
    test_imgs_np = (test_imgs_np - MEAN_VAL) / STD_VAL
    
    training_images = torch.from_numpy(
        train_imgs_np[:TRAINING_SAMPLES].reshape(-1, 1, 28, 28)
    ).to("cuda")
    training_targets = torch.from_numpy(
        train_lbls_np[:TRAINING_SAMPLES]
    ).long().to("cuda")
    
    testing_images = torch.from_numpy(
        test_imgs_np.reshape(-1, 1, 28, 28)
    ).to("cuda")
    testing_targets = torch.from_numpy(test_lbls_np).long().to("cuda")
    
    character_net = CharacterClassifier(
        input_dim=784,
        hidden_dim=256,
        output_classes=47
    ).to("cuda")
    
    # Apply He initialization
    with torch.no_grad():
        fan_in_l1 = character_net.layer1.weight.size(1)
        init_scale_l1 = (2.0 / fan_in_l1) ** 0.5
        character_net.layer1.weight.uniform_(-init_scale_l1, init_scale_l1)
        character_net.layer1.bias.zero_()
        
        fan_in_l2 = character_net.layer2.weight.size(1)
        init_scale_l2 = (2.0 / fan_in_l2) ** 0.5
        character_net.layer2.weight.uniform_(-init_scale_l2, init_scale_l2)
        character_net.layer2.bias.zero_()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(character_net.parameters(), lr=LR)
    
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
            criterion,
            optimizer,
            epoch,
            performance_metrics,
            epoch_loss_list
        )
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {epoch_loss_list[epoch]:.4f}")
    
    train_end_time = time.time()
    performance_metrics['total_train_time'] = train_end_time - train_start_time
    
    print("v1-pytorch")
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
    compute_test_accuracy(character_net, testing_images, testing_targets)
